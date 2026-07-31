from __future__ import annotations

import inspect
import json
import math
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from pocketrocks import BotDecision, DecisionContext

from garboid_pocketrocks.bots import BOT_SPECS_BY_NAME, BotSpec, BrainFactory, RandomBot
from garboid_pocketrocks.heuristics.frozen import FROZEN_CANDIDATES_BY_NAME
from garboid_pocketrocks.knowledge import RulesetKnowledge
from garboid_pocketrocks.promotion.analysis import PromotionAnalysis
from garboid_pocketrocks.promotion.candidates import resolve_promotion_candidate
from garboid_pocketrocks.promotion.corpus import (
    PromotionCorpus,
    PromotionCorpusRecipe,
    load_promotion_corpus,
    recompute_promotion_corpus_digest,
)
from garboid_pocketrocks.promotion.runner import (
    PromotionRun,
    PromotionRunConfig,
    PromotionRunner,
    _repository_commit,
)
from garboid_pocketrocks.simulator.errors import SimulationError
from garboid_pocketrocks.simulator.monte_carlo import MonteCarloResult, MonteCarloRunner

from .helpers import (
    EvilFactory,
    EvilString,
    FrozenCandidateFixture,
    evil_provenance,
    frozen_candidate_fixture,
    promotion_plan,
)


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


def _frozen_run_inputs() -> tuple[
    PromotionRunConfig,
    dict[str, BotSpec],
    dict[str, FrozenCandidateFixture],
]:
    config, registry = _run_inputs(
        incumbent=BOT_SPECS_BY_NAME["balanced-v2"],
        pair_count=1,
    )
    development = replace(
        config.development,
        digest=recompute_promotion_corpus_digest(config.development),
    )
    candidate = BotSpec.for_simulation(
        "balanced-v3-candidate-test",
        RandomBot.build_brain,
    )
    frozen = frozen_candidate_fixture(
        bot_spec=candidate,
        predecessor_name=config.incumbent.name,
        development=development,
    )
    catalog = {candidate.name: frozen}
    resolved = resolve_promotion_candidate(
        candidate.name,
        registry=registry,
        frozen_candidates=catalog,
    )
    return (
        replace(
            config,
            candidate=candidate,
            development=development,
            candidate_provenance=resolved.frozen_provenance,
        ),
        registry,
        catalog,
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
    payload = json.loads(run.artifacts.report_json.read_bytes())
    assert payload["execution"] == {
        "bot_ids": ["candidate", "incumbent", "opponent-a", "opponent-b"],
        "games": 2,
        "player_counts": [3],
        "value_charts": ["A"],
        "root_seed": 90_001,
        "objectives_enabled": [True],
        "fault_mode": "record_and_pass",
        "capture_replays": False,
        "workers": 1,
        "batch_size": 2,
    }


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


def test_runner_does_not_accept_an_alternate_frozen_candidate_catalog() -> None:
    fake_catalog = {"forged-candidate": object()}

    with pytest.raises(TypeError, match="unexpected keyword"):
        inspect.signature(PromotionRunner.run).bind_partial(
            frozen_candidates=fake_catalog,
        )


def test_runner_rejects_a_forged_frozen_identity_in_the_caller_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = next(iter(FROZEN_CANDIDATES_BY_NAME.values()))
    forged = BotSpec.for_simulation(frozen.bot_spec.name, RandomBot.build_brain)
    config, registry = _run_inputs(pair_count=1)
    registry = {**registry, forged.name: forged}

    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("simulation started with a forged frozen identity")

    monkeypatch.setattr(MonteCarloRunner, "run_jobs", forbidden)

    with pytest.raises(ValueError, match="provenance"):
        PromotionRunner.run(
            replace(config, candidate=forged),
            registry=registry,
            workers=1,
            output_dir=tmp_path,
            repository_commit="test-commit",
        )

    assert not tuple(tmp_path.iterdir())


def test_runner_rejects_a_forged_predecessor_for_a_real_frozen_candidate(
    tmp_path: Path,
) -> None:
    frozen = next(iter(FROZEN_CANDIDATES_BY_NAME.values()))
    resolved = resolve_promotion_candidate(
        frozen.bot_spec.name,
        registry=BOT_SPECS_BY_NAME,
        frozen_candidates=FROZEN_CANDIDATES_BY_NAME,
    )
    forged_predecessor = BotSpec.for_simulation(
        frozen.predecessor_name,
        RandomBot.build_brain,
    )
    config, registry = _run_inputs(
        candidate=frozen.bot_spec,
        incumbent=forged_predecessor,
        pair_count=1,
    )

    with pytest.raises(ValueError, match="exact canonical predecessor"):
        PromotionRunner.run(
            replace(config, candidate_provenance=resolved.frozen_provenance),
            registry={**registry, forged_predecessor.name: forged_predecessor},
            workers=1,
            output_dir=tmp_path,
            repository_commit="test-commit",
        )

    assert not tuple(tmp_path.iterdir())


def test_runner_rejects_a_different_equal_frozen_bot_spec(
    tmp_path: Path,
) -> None:
    frozen = next(iter(FROZEN_CANDIDATES_BY_NAME.values()))
    resolved = resolve_promotion_candidate(
        frozen.bot_spec.name,
        registry=BOT_SPECS_BY_NAME,
        frozen_candidates=FROZEN_CANDIDATES_BY_NAME,
    )
    forged = replace(frozen.bot_spec, brain_factory=EvilFactory())
    development = load_promotion_corpus(
        Path("configs/promotion/development-v1.json"),
        registry=BOT_SPECS_BY_NAME,
    )
    _, held_out = _corpora(pair_count=1)
    held_out = replace(
        held_out,
        recipe=replace(
            held_out.recipe,
            opponent_names=("random", "aggressive-v1"),
        ),
    )
    assert forged == frozen.bot_spec

    with pytest.raises(ValueError, match="trusted frozen candidate record"):
        PromotionRunner.run(
            PromotionRunConfig(
                candidate=forged,
                incumbent=BOT_SPECS_BY_NAME[frozen.predecessor_name],
                development=development,
                held_out=held_out,
                candidate_provenance=resolved.frozen_provenance,
            ),
            registry=BOT_SPECS_BY_NAME,
            workers=1,
            output_dir=tmp_path,
            repository_commit="test-commit",
        )

    assert not tuple(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("field_name", "forged_value"),
    (
        ("search_name", "forged-search-name"),
        ("profile_digest", "f" * 64),
    ),
)
def test_runner_rejects_lying_provenance_string_subclasses(
    tmp_path: Path,
    field_name: str,
    forged_value: str,
) -> None:
    frozen = next(iter(FROZEN_CANDIDATES_BY_NAME.values()))
    resolved = resolve_promotion_candidate(
        frozen.bot_spec.name,
        registry=BOT_SPECS_BY_NAME,
        frozen_candidates=FROZEN_CANDIDATES_BY_NAME,
    )
    assert resolved.frozen_provenance is not None
    lying_value = EvilString(forged_value)
    forged_provenance = replace(
        resolved.frozen_provenance,
        **{field_name: lying_value},
    )
    development = load_promotion_corpus(
        Path("configs/promotion/development-v1.json"),
        registry=BOT_SPECS_BY_NAME,
    )
    _, held_out = _corpora(pair_count=1)
    held_out = replace(
        held_out,
        recipe=replace(
            held_out.recipe,
            opponent_names=("random", "aggressive-v1"),
        ),
    )

    with pytest.raises(ValueError, match="built-in strings"):
        PromotionRunner.run(
            PromotionRunConfig(
                candidate=frozen.bot_spec,
                incumbent=BOT_SPECS_BY_NAME[frozen.predecessor_name],
                development=development,
                held_out=held_out,
                candidate_provenance=forged_provenance,
            ),
            registry=BOT_SPECS_BY_NAME,
            workers=1,
            output_dir=tmp_path,
            repository_commit="test-commit",
        )

    assert not tuple(tmp_path.iterdir())


def test_runner_rejects_a_lying_provenance_subclass(
    tmp_path: Path,
) -> None:
    frozen = next(iter(FROZEN_CANDIDATES_BY_NAME.values()))
    resolved = resolve_promotion_candidate(
        frozen.bot_spec.name,
        registry=BOT_SPECS_BY_NAME,
        frozen_candidates=FROZEN_CANDIDATES_BY_NAME,
    )
    assert resolved.frozen_provenance is not None
    forged_provenance = evil_provenance(resolved.frozen_provenance)
    development = load_promotion_corpus(
        Path("configs/promotion/development-v1.json"),
        registry=BOT_SPECS_BY_NAME,
    )
    _, held_out = _corpora(pair_count=1)
    held_out = replace(
        held_out,
        recipe=replace(
            held_out.recipe,
            opponent_names=("random", "aggressive-v1"),
        ),
    )

    with pytest.raises(ValueError, match="exact provenance type"):
        PromotionRunner.run(
            PromotionRunConfig(
                candidate=frozen.bot_spec,
                incumbent=BOT_SPECS_BY_NAME[frozen.predecessor_name],
                development=development,
                held_out=held_out,
                candidate_provenance=forged_provenance,
            ),
            registry=BOT_SPECS_BY_NAME,
            workers=1,
            output_dir=tmp_path,
            repository_commit="test-commit",
        )

    assert not tuple(tmp_path.iterdir())


def test_frozen_runner_rejects_a_forged_canonical_name_opponent(
    tmp_path: Path,
) -> None:
    frozen = next(iter(FROZEN_CANDIDATES_BY_NAME.values()))
    resolved = resolve_promotion_candidate(
        frozen.bot_spec.name,
        registry=BOT_SPECS_BY_NAME,
        frozen_candidates=FROZEN_CANDIDATES_BY_NAME,
    )
    development = load_promotion_corpus(
        Path("configs/promotion/development-v1.json"),
        registry=BOT_SPECS_BY_NAME,
    )
    _, held_out = _corpora(pair_count=1)
    held_out = replace(
        held_out,
        recipe=replace(
            held_out.recipe,
            opponent_names=("random", "aggressive-v1"),
        ),
    )
    forged_opponent = BotSpec.for_simulation("random", RandomBot.build_brain)
    registry = {
        **BOT_SPECS_BY_NAME,
        forged_opponent.name: forged_opponent,
    }
    config = PromotionRunConfig(
        candidate=frozen.bot_spec,
        incumbent=BOT_SPECS_BY_NAME[frozen.predecessor_name],
        development=development,
        held_out=held_out,
        candidate_provenance=resolved.frozen_provenance,
    )

    with pytest.raises(ValueError, match="exact canonical released opponent"):
        PromotionRunner.run(
            config,
            registry=registry,
            workers=1,
            output_dir=tmp_path,
            repository_commit="test-commit",
        )

    assert not tuple(tmp_path.iterdir())


@pytest.mark.parametrize("tampering", ("candidate", "provenance"))
def test_runner_rejects_identity_swap_and_fabricated_frozen_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tampering: str,
) -> None:
    config, registry, catalog = _frozen_run_inputs()
    monkeypatch.setattr(
        "garboid_pocketrocks.promotion.candidates.load_frozen_candidate_catalog",
        lambda: catalog,
    )
    assert config.candidate_provenance is not None
    provenance = config.candidate_provenance
    if tampering == "candidate":
        config = replace(
            config,
            candidate=BotSpec.for_simulation(
                config.candidate.name,
                illegal_bid_brain,
            ),
        )
    else:
        config = replace(
            config,
            candidate_provenance=replace(
                provenance,
                profile_digest="f" * 64,
            ),
        )

    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("simulation started with untrusted frozen provenance")

    monkeypatch.setattr(MonteCarloRunner, "run_jobs", forbidden)

    with pytest.raises(ValueError, match="trusted frozen candidate"):
        PromotionRunner.run(
            config,
            registry=registry,
            workers=1,
            output_dir=tmp_path,
            repository_commit="test-commit",
        )

    assert not tuple(tmp_path.iterdir())


@pytest.mark.parametrize(
    "corpus_tampering",
    ("name", "stored_digest", "content"),
)
def test_runner_recomputes_and_rebinds_the_development_corpus_before_simulation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corpus_tampering: str,
) -> None:
    config, registry, catalog = _frozen_run_inputs()
    monkeypatch.setattr(
        "garboid_pocketrocks.promotion.candidates.load_frozen_candidate_catalog",
        lambda: catalog,
    )
    if corpus_tampering == "name":
        development = replace(
            config.development,
            recipe=replace(config.development.recipe, name="other-development"),
        )
        development = replace(
            development,
            digest=recompute_promotion_corpus_digest(development),
        )
    elif corpus_tampering == "stored_digest":
        development = replace(config.development, digest="f" * 64)
    else:
        development = replace(config.development, cases=())

    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("simulation started with stale development evidence")

    monkeypatch.setattr(MonteCarloRunner, "run_jobs", forbidden)

    with pytest.raises(ValueError, match="development corpus"):
        PromotionRunner.run(
            replace(config, development=development),
            registry=registry,
            workers=1,
            output_dir=tmp_path,
            repository_commit="test-commit",
        )

    assert not tuple(tmp_path.iterdir())


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
    assert run.report.analysis.failures[0].message.endswith("worker process failed")
    assert run.artifacts.report_json.is_file()


@pytest.mark.parametrize(
    "simulator_error",
    (
        OSError("worker pipe closed"),
        RuntimeError("SDK execution failed"),
        ValueError("simulator result was invalid"),
    ),
)
def test_raw_simulator_exceptions_write_a_nonpromotion_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    simulator_error: Exception,
) -> None:
    config, registry = _run_inputs(pair_count=1)

    def fail_simulation(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise simulator_error

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

    failure = run.report.analysis.failures[0]
    assert run.plan is not None
    assert run.monte_carlo_result is None
    assert failure.code == "simulation_failed"
    assert type(simulator_error).__name__ in failure.message
    assert str(simulator_error) in failure.message
    assert run.artifacts.report_json.is_file()


@pytest.mark.parametrize("control_flow_error", (KeyboardInterrupt(), SystemExit(2)))
def test_simulator_preserves_process_control_exceptions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    control_flow_error: BaseException,
) -> None:
    config, registry = _run_inputs(pair_count=1)

    def interrupt_simulation(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise control_flow_error

    monkeypatch.setattr(
        "garboid_pocketrocks.promotion.runner.MonteCarloRunner.run_jobs",
        interrupt_simulation,
    )

    with pytest.raises(type(control_flow_error)) as captured:
        PromotionRunner.run(
            config,
            registry=registry,
            workers=1,
            output_dir=tmp_path,
            repository_commit="test-commit",
        )

    assert captured.value is control_flow_error


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
            unattributed_faults=0,
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


@pytest.mark.parametrize(
    "git_error",
    (
        subprocess.CalledProcessError(128, ("git", "rev-parse", "HEAD")),
        FileNotFoundError("git executable is unavailable"),
    ),
)
def test_repository_commit_wraps_git_failures_in_a_concise_operational_error(
    monkeypatch: pytest.MonkeyPatch,
    git_error: BaseException,
) -> None:
    def fail_git(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise git_error

    monkeypatch.setattr(
        "garboid_pocketrocks.promotion.runner.subprocess.run",
        fail_git,
    )

    with pytest.raises(
        RuntimeError,
        match="Could not determine the repository commit",
    ) as captured:
        _repository_commit()

    assert captured.value.__cause__ is git_error


@pytest.mark.parametrize("control_flow_error", (KeyboardInterrupt(), SystemExit(2)))
def test_repository_commit_preserves_process_control_exceptions(
    monkeypatch: pytest.MonkeyPatch,
    control_flow_error: BaseException,
) -> None:
    def interrupt_git(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise control_flow_error

    monkeypatch.setattr(
        "garboid_pocketrocks.promotion.runner.subprocess.run",
        interrupt_git,
    )

    with pytest.raises(type(control_flow_error)) as captured:
        _repository_commit()

    assert captured.value is control_flow_error
