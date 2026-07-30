from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from pocketrocks import BotDecision, DecisionContext

from garboid_pocketrocks.bots import BOT_SPECS_BY_NAME, BotSpec
from garboid_pocketrocks.knowledge import RulesetKnowledge
from garboid_pocketrocks.promotion.corpus import (
    PromotionCorpus,
    load_promotion_corpus,
)
from garboid_pocketrocks.promotion.runner import (
    PromotionRun,
    PromotionRunConfig,
    PromotionRunner,
)

_REPOSITORY_COMMIT = "integration-test-commit"
_OPPONENT_NAMES = ("random", "aggressive-v1", "balanced-v1", "passive-v1")


class _IllegalBidBrain:
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


def _build_illegal_bid_brain(seed: int | None) -> _IllegalBidBrain:
    del seed
    return _IllegalBidBrain()


def _write_corpus_recipe(
    path: Path,
    *,
    name: str,
    purpose: str,
    root_seed: int,
) -> None:
    payload = {
        "schema_version": 1,
        "name": name,
        "purpose": purpose,
        "root_seed": root_seed,
        "repetitions_per_seat_cell": 1,
        "charts": ["A"],
        "player_counts": [3, 4],
        "opponent_names": list(_OPPONENT_NAMES),
    }
    path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")


def _load_tiny_corpora(
    tmp_path: Path,
    *,
    held_out_root_seed: int = 22_007,
) -> tuple[PromotionCorpus, PromotionCorpus]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    development_path = tmp_path / "development-integration-v1.json"
    held_out_path = tmp_path / "held-out-integration-v1.json"
    _write_corpus_recipe(
        development_path,
        name="development-integration-v1",
        purpose="development",
        root_seed=11_003,
    )
    _write_corpus_recipe(
        held_out_path,
        name="held-out-integration-v1",
        purpose="held_out",
        root_seed=held_out_root_seed,
    )
    return (
        load_promotion_corpus(development_path, registry=BOT_SPECS_BY_NAME),
        load_promotion_corpus(held_out_path, registry=BOT_SPECS_BY_NAME),
    )


def _run_tiny_gate(
    output_dir: Path,
    *,
    development: PromotionCorpus,
    held_out: PromotionCorpus,
    workers: int,
    candidate: BotSpec | None = None,
) -> PromotionRun:
    return PromotionRunner.run(
        PromotionRunConfig(
            candidate=candidate or BOT_SPECS_BY_NAME["aggressive-v2"],
            incumbent=BOT_SPECS_BY_NAME["balanced-v2"],
            development=development,
            held_out=held_out,
            bootstrap_samples=10,
            bootstrap_seed=41,
            batch_size=4,
        ),
        registry=BOT_SPECS_BY_NAME,
        workers=workers,
        output_dir=output_dir,
        repository_commit=_REPOSITORY_COMMIT,
    )


def _artifact_bytes(run: PromotionRun) -> tuple[bytes, bytes, bytes]:
    return (
        run.artifacts.report_json.read_bytes(),
        run.artifacts.paired_games_jsonl.read_bytes(),
        run.artifacts.corpus_snapshot_json.read_bytes(),
    )


def test_real_paired_gate_is_deterministic_across_serial_parallel_and_repeat_runs(
    tmp_path: Path,
) -> None:
    development, held_out = _load_tiny_corpora(tmp_path)

    serial = _run_tiny_gate(
        tmp_path / "serial",
        development=development,
        held_out=held_out,
        workers=1,
    )
    parallel = _run_tiny_gate(
        tmp_path / "parallel",
        development=development,
        held_out=held_out,
        workers=2,
    )
    repeated = _run_tiny_gate(
        tmp_path / "repeated",
        development=development,
        held_out=held_out,
        workers=1,
    )

    assert serial.plan is not None
    assert serial.monte_carlo_result is not None
    assert serial.plan == parallel.plan == repeated.plan
    assert serial.monte_carlo_result == parallel.monte_carlo_result
    assert serial.report.analysis == parallel.report.analysis
    assert serial.report == replace(parallel.report, workers=serial.report.workers)

    serial_report, serial_games, serial_corpora = _artifact_bytes(serial)
    parallel_report, parallel_games, parallel_corpora = _artifact_bytes(parallel)
    assert serial_games == parallel_games
    assert serial_corpora == parallel_corpora
    assert serial_report != parallel_report
    assert json.loads(serial_report)["execution"]["workers"] == 1
    assert json.loads(parallel_report)["execution"]["workers"] == 2
    assert _artifact_bytes(serial) == _artifact_bytes(repeated)

    requested_cells = {
        (chart, player_count, focal_seat)
        for chart in held_out.recipe.charts
        for player_count in held_out.recipe.player_counts
        for focal_seat in range(player_count)
    }
    planned_cells = {
        (pair.case.chart, pair.case.player_count, pair.case.focal_seat)
        for pair in serial.plan.pairs
    }
    assert planned_cells == requested_cells
    for pair in serial.plan.pairs:
        assert pair.candidate_game.seed == pair.incumbent_game.seed == pair.case.engine_seed
        assert pair.candidate_game.value_chart == pair.incumbent_game.value_chart == pair.case.chart
        assert (
            pair.candidate_game.player_count
            == pair.incumbent_game.player_count
            == pair.case.player_count
        )
        assert (
            pair.candidate_game.lineup[: pair.case.focal_seat]
            + pair.candidate_game.lineup[pair.case.focal_seat + 1 :]
            == pair.incumbent_game.lineup[: pair.case.focal_seat]
            + pair.incumbent_game.lineup[pair.case.focal_seat + 1 :]
        )


def test_changing_the_held_out_root_seed_changes_the_digest_and_executed_seeds(
    tmp_path: Path,
) -> None:
    development, original_held_out = _load_tiny_corpora(tmp_path / "original")
    _, changed_held_out = _load_tiny_corpora(
        tmp_path / "changed",
        held_out_root_seed=22_008,
    )

    original = _run_tiny_gate(
        tmp_path / "original-results",
        development=development,
        held_out=original_held_out,
        workers=1,
    )
    changed = _run_tiny_gate(
        tmp_path / "changed-results",
        development=development,
        held_out=changed_held_out,
        workers=1,
    )

    assert original.monte_carlo_result is not None
    assert changed.monte_carlo_result is not None
    assert original_held_out.digest != changed_held_out.digest
    assert original_held_out.engine_seeds != changed_held_out.engine_seeds
    assert tuple(summary.seed for summary in original.monte_carlo_result.game_summaries) != tuple(
        summary.seed for summary in changed.monte_carlo_result.game_summaries
    )


def test_illegal_candidate_fails_closed_without_a_false_promotion_claim(
    tmp_path: Path,
) -> None:
    development, held_out = _load_tiny_corpora(tmp_path)
    illegal_candidate = BotSpec.for_simulation(
        "illegal-integration-candidate",
        _build_illegal_bid_brain,
    )

    run = _run_tiny_gate(
        tmp_path / "illegal-results",
        development=development,
        held_out=held_out,
        workers=2,
        candidate=illegal_candidate,
    )

    report_payload = json.loads(run.artifacts.report_json.read_bytes())
    assert run.report.analysis.promoted is False
    assert "bot_fault" in {failure.code for failure in run.report.analysis.failures}
    assert report_payload["promoted"] is False
    assert "bot_fault" in {failure["code"] for failure in report_payload["failures"]}
