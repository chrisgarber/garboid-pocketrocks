from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest
from pocketrocks import BotDecision, DecisionContext

from garboid_pocketrocks.bots import BotSpec, BrainFactory, RandomBot
from garboid_pocketrocks.knowledge import RulesetKnowledge
from garboid_pocketrocks.promotion.analysis import PromotionAnalysis
from garboid_pocketrocks.promotion.corpus import PromotionCorpus, PromotionCorpusRecipe
from garboid_pocketrocks.promotion.runner import (
    PromotionRun,
    PromotionRunConfig,
    PromotionRunner,
)
from garboid_pocketrocks.simulator.errors import SimulationError
from garboid_pocketrocks.simulator.monte_carlo import MonteCarloResult

from .helpers import promotion_plan


class IllegalBidBrain:
    def choose_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
    ) -> BotDecision:
        del ruleset
        if context.decision_kind == "selectInfoToReveal":
            return BotDecision.select_info_to_reveal(context.revealable_count)
        assert context.legal_max_amount is not None
        return BotDecision.submit_bid(context.legal_max_amount + 1)


class RaisingDecisionBrain:
    def choose_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
    ) -> BotDecision:
        del context, ruleset
        raise RuntimeError("decision failed")


def illegal_bid_brain(seed: int | None) -> IllegalBidBrain:
    del seed
    return IllegalBidBrain()


def raising_decision_brain(seed: int | None) -> RaisingDecisionBrain:
    del seed
    return RaisingDecisionBrain()


def raising_construction_brain(seed: int | None) -> RaisingDecisionBrain:
    del seed
    raise RuntimeError("construction failed")


def _corpora(*, pair_count: int = 3) -> tuple[PromotionCorpus, PromotionCorpus]:
    held_out = promotion_plan(pair_count=pair_count).pairs
    held_out_corpus = PromotionCorpus(
        recipe=PromotionCorpusRecipe(
            schema_version=1,
            name="fixture-held-out-v1",
            purpose="held_out",
            root_seed=90_001,
            repetitions_per_seat_cell=1,
            charts=tuple(pair.case.chart for pair in held_out),
            player_counts=(3,),
            opponent_names=("opponent-a", "opponent-b"),
        ),
        cases=tuple(pair.case for pair in held_out),
        digest="e" * 64,
    )
    development_cases = tuple(
        replace(
            pair.case,
            case_id=pair.case.case_id.replace("held-out", "development"),
            engine_seed=pair.case.engine_seed + 1_000,
        )
        for pair in held_out
    )
    development_corpus = PromotionCorpus(
        recipe=PromotionCorpusRecipe(
            schema_version=1,
            name="fixture-development-v1",
            purpose="development",
            root_seed=8_001,
            repetitions_per_seat_cell=1,
            charts=tuple(case.chart for case in development_cases),
            player_counts=(3,),
            opponent_names=("opponent-a", "opponent-b"),
        ),
        cases=development_cases,
        digest="d" * 64,
    )
    return development_corpus, held_out_corpus


def _run_inputs(
    *,
    candidate: BotSpec | None = None,
    incumbent: BotSpec | None = None,
    pair_count: int = 3,
) -> tuple[PromotionRunConfig, dict[str, BotSpec]]:
    development, held_out = _corpora(pair_count=pair_count)
    candidate = candidate or BotSpec.for_simulation("candidate", RandomBot.build_brain)
    incumbent = incumbent or BotSpec.for_simulation("incumbent", RandomBot.build_brain)
    opponents = (
        BotSpec.for_simulation("opponent-a", RandomBot.build_brain),
        BotSpec.for_simulation("opponent-b", RandomBot.build_brain),
    )
    registry = {spec.name: spec for spec in (candidate, incumbent, *opponents)}
    return (
        PromotionRunConfig(
            candidate=candidate,
            incumbent=incumbent,
            development=development,
            held_out=held_out,
            bootstrap_samples=12,
            bootstrap_seed=7,
            batch_size=2,
        ),
        registry,
    )


def _run(
    tmp_path: Path,
    *,
    workers: int = 1,
    candidate: BotSpec | None = None,
    incumbent: BotSpec | None = None,
    pair_count: int = 3,
) -> PromotionRun:
    config, registry = _run_inputs(
        candidate=candidate,
        incumbent=incumbent,
        pair_count=pair_count,
    )
    return PromotionRunner.run(
        config,
        registry=registry,
        workers=workers,
        output_dir=tmp_path,
        repository_commit="test-commit",
    )


def test_real_simulator_is_deterministic_across_worker_counts(tmp_path: Path) -> None:
    serial = _run(tmp_path / "serial", workers=1)
    parallel = _run(tmp_path / "parallel", workers=2)
    serial_repeat = _run(tmp_path / "serial-repeat", workers=1)

    assert serial.plan == parallel.plan
    assert serial.monte_carlo_result == parallel.monte_carlo_result
    assert serial.report.analysis == parallel.report.analysis
    assert (
        serial.artifacts.paired_games_jsonl.read_bytes()
        == parallel.artifacts.paired_games_jsonl.read_bytes()
    )
    assert (
        serial.artifacts.corpus_snapshot_json.read_bytes()
        == parallel.artifacts.corpus_snapshot_json.read_bytes()
    )
    # Reports truthfully record their worker count, so cross-worker report bytes differ.
    assert serial.report == replace(parallel.report, workers=serial.report.workers)
    assert serial.artifacts.report_json.read_bytes() != parallel.artifacts.report_json.read_bytes()
    assert serial.report == serial_repeat.report
    assert (
        serial.artifacts.report_json.read_bytes()
        == serial_repeat.artifacts.report_json.read_bytes()
    )


def test_overlapping_corpus_seeds_write_a_nonpromotion_report(tmp_path: Path) -> None:
    config, registry = _run_inputs(pair_count=1)
    overlapping_development = replace(
        config.development,
        cases=(
            replace(
                config.development.cases[0],
                engine_seed=config.held_out.cases[0].engine_seed,
            ),
        ),
    )

    run = PromotionRunner.run(
        replace(config, development=overlapping_development),
        registry=registry,
        workers=1,
        output_dir=tmp_path,
        repository_commit="test-commit",
    )

    assert run.plan is None
    assert run.monte_carlo_result is None
    assert run.report.analysis.promoted is False
    assert [failure.code for failure in run.report.analysis.failures] == ["corpus_seed_overlap"]
    assert run.artifacts.report_json.is_file()
    assert run.artifacts.paired_games_jsonl.read_bytes() == b""


def test_candidate_and_incumbent_identity_collision_writes_a_report(tmp_path: Path) -> None:
    repeated = BotSpec.for_simulation("same-bot", RandomBot.build_brain)

    run = _run(
        tmp_path,
        candidate=repeated,
        incumbent=repeated,
        pair_count=1,
    )

    assert run.report.analysis.promoted is False
    assert [failure.code for failure in run.report.analysis.failures] == [
        "candidate_incumbent_identity_collision"
    ]
    assert run.artifacts.report_json.is_file()


@pytest.mark.parametrize(
    ("name", "brain_factory"),
    (
        ("illegal", illegal_bid_brain),
        ("raising-decision", raising_decision_brain),
        ("raising-construction", raising_construction_brain),
    ),
)
def test_bot_faults_write_nonpromotion_evidence(
    tmp_path: Path,
    name: str,
    brain_factory: BrainFactory,
) -> None:
    candidate = BotSpec.for_simulation(name, brain_factory)

    run = _run(tmp_path, candidate=candidate, pair_count=1)

    assert run.plan is not None
    assert run.monte_carlo_result is not None
    assert run.report.analysis.promoted is False
    assert "bot_fault" in {failure.code for failure in run.report.analysis.failures}
    assert dict(run.report.analysis.faults_by_identity)[candidate.bot_id] > 0
    assert run.artifacts.report_json.is_file()
    assert run.artifacts.paired_games_jsonl.read_bytes()


def test_simulation_error_writes_a_nonpromotion_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, registry = _run_inputs(pair_count=1)

    def fail_simulation(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise SimulationError("worker process failed")

    monkeypatch.setattr(
        "garboid_pocketrocks.promotion.runner.MonteCarloRunner.run_jobs",
        fail_simulation,
    )

    run = PromotionRunner.run(
        config,
        registry=registry,
        workers=1,
        output_dir=tmp_path,
        repository_commit="test-commit",
    )

    assert run.plan is not None
    assert run.monte_carlo_result is None
    assert [failure.code for failure in run.report.analysis.failures] == ["simulation_failed"]
    assert run.artifacts.report_json.is_file()


def test_nonfinite_analyzer_output_is_replaced_with_a_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, registry = _run_inputs(pair_count=1)

    def return_nonfinite_analysis(*args: object, **kwargs: object) -> PromotionAnalysis:
        del args, kwargs
        return PromotionAnalysis(
            requested_pairs=1,
            completed_pairs=1,
            requested_games=2,
            completed_games=2,
            rating_difference=math.nan,
            interval=None,
            bootstrap_requested=config.bootstrap_samples,
            bootstrap_converged=config.bootstrap_samples,
            faults_by_identity=(),
            warnings=(),
            failures=(),
            promoted=True,
        )

    monkeypatch.setattr(
        "garboid_pocketrocks.promotion.runner.analyze_promotion",
        return_nonfinite_analysis,
    )

    run = PromotionRunner.run(
        config,
        registry=registry,
        workers=1,
        output_dir=tmp_path,
        repository_commit="test-commit",
    )

    assert run.report.analysis.promoted is False
    assert run.report.analysis.rating_difference is None
    assert [failure.code for failure in run.report.analysis.failures] == ["nonfinite_analysis"]
    assert run.artifacts.report_json.is_file()


def test_missing_game_summaries_write_a_nonpromotion_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, registry = _run_inputs(pair_count=1)
    empty_result = MonteCarloResult(game_summaries=(), bot_statistics=(), replays=())
    monkeypatch.setattr(
        "garboid_pocketrocks.promotion.runner.MonteCarloRunner.run_jobs",
        lambda *args, **kwargs: empty_result,
    )

    run = PromotionRunner.run(
        config,
        registry=registry,
        workers=1,
        output_dir=tmp_path,
        repository_commit="test-commit",
    )

    assert run.monte_carlo_result == empty_result
    assert "missing_paired_game" in {failure.code for failure in run.report.analysis.failures}
    assert run.artifacts.report_json.is_file()


def test_output_directory_is_validated_before_simulation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, registry = _run_inputs(pair_count=1)
    (tmp_path / "keep.txt").write_text("keep", encoding="utf-8")

    def should_not_simulate(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("simulation started before output validation")

    monkeypatch.setattr(
        "garboid_pocketrocks.promotion.runner.MonteCarloRunner.run_jobs",
        should_not_simulate,
    )

    with pytest.raises(FileExistsError, match="not empty"):
        PromotionRunner.run(
            config,
            registry=registry,
            workers=1,
            output_dir=tmp_path,
            repository_commit="test-commit",
        )


def test_filesystem_errors_are_not_converted_to_domain_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, registry = _run_inputs(pair_count=1)

    def fail_write(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("disk failed")

    monkeypatch.setattr(
        "garboid_pocketrocks.promotion.runner.write_promotion_artifacts",
        fail_write,
    )

    with pytest.raises(OSError, match="disk failed"):
        PromotionRunner.run(
            config,
            registry=registry,
            workers=1,
            output_dir=tmp_path,
            repository_commit="test-commit",
        )
