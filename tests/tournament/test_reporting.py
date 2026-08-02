from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from diagnostics.test_analysis import _build as build_decision_report_fixture
from diagnostics.test_analysis import _game as decision_game_fixture
from diagnostics.test_analysis import _trace as decision_trace_fixture
from garboid_pocketrocks.diagnostics.trace import (
    FixedObjectiveOverlayV3BidExplanation,
    RecordedAction,
)
from garboid_pocketrocks.simulator.monte_carlo import MonteCarloResult
from garboid_pocketrocks.tournament.analysis import (
    BootstrapSummary,
    RatingInterval,
    TournamentAnalysis,
    analyze_tournament,
)
from garboid_pocketrocks.tournament.rating import (
    PlackettLuceFit,
    fit_plackett_luce,
    observations_from_games,
)
from garboid_pocketrocks.tournament.reporting import write_tournament_artifacts
from garboid_pocketrocks.tournament.schedule import (
    TournamentConfig,
    TournamentPlan,
    TournamentPlanner,
)

from .helpers import game_summary, random_specs

_DIAGNOSTIC_ARTIFACT_NAMES = (
    "ratings.csv",
    "summary.json",
    "report.html",
    "game-summaries.jsonl",
    "game-details.jsonl",
    "decision-traces.jsonl",
    "decision-slices.csv",
)


def _report_inputs() -> tuple[
    TournamentConfig,
    TournamentPlan,
    PlackettLuceFit,
    TournamentAnalysis,
    BootstrapSummary,
]:
    config = TournamentConfig(
        bot_specs=random_specs(3),
        games=1,
        player_counts=(3,),
        charts=("A",),
        bootstrap_samples=2,
        root_seed=42,
    )
    plan = TournamentPlanner.plan(config)
    game = replace(
        game_summary(
            ("random-0", "random-1", "random-2"),
            final_money=(30, 20, 10),
            ranks=(1, 2, 3),
        ),
        bot_names=("<alpha>", "beta & co", "gamma"),
    )
    result = MonteCarloResult((game,), (), ())
    fit = fit_plackett_luce(observations_from_games(result.game_summaries), game.bot_ids)
    analysis = analyze_tournament(result, fit)
    bootstrap = BootstrapSummary(
        requested=2,
        converged=2,
        intervals=tuple(
            RatingInterval(rating.bot_id, rating.rating - 10, rating.rating + 10)
            for rating in fit.ratings
        ),
        warnings=(),
    )
    return config, plan, fit, analysis, bootstrap


def test_artifacts_contain_machine_data_and_three_svg_charts(tmp_path: Path) -> None:
    config, plan, fit, analysis, bootstrap = _report_inputs()

    artifacts = write_tournament_artifacts(
        output_dir=tmp_path,
        overwrite=False,
        config=config,
        plan=plan,
        fit=fit,
        analysis=analysis,
        bootstrap=bootstrap,
    )

    assert artifacts.ratings_csv == tmp_path / "ratings.csv"
    payload = json.loads(artifacts.summary_json.read_text())
    assert payload["schema_version"] == 1
    assert payload["configuration"]["games"] == config.games
    assert payload["configuration"]["batch_size"] == config.batch_size
    assert payload["configuration"]["root_seed"] == 42
    assert payload["leaderboard"][0]["pl_rating"] == fit.ratings[0].rating
    assert payload["schedule"]["condition_quotas"] == [
        {"chart": "A", "games": 1, "player_count": 3}
    ]
    html = artifacts.report_html.read_text()
    assert html.count("<svg") == 3
    assert "PL rating leaderboard" in html
    assert "Rating versus mean winning money" in html
    assert "PL calibration" in html
    assert 'class="axis-label">PL rating</text>' in html
    assert 'class="axis-label">Mean winning final money</text>' in html
    assert 'class="axis-label">Predicted pairwise score</text>' in html
    assert 'class="axis-label">Observed pairwise score</text>' in html
    assert "&lt;alpha&gt;" in html
    assert "<alpha>" not in html
    assert "seed 42" in html
    assert "Decision diagnostics" not in html


def test_decision_report_adds_sanitized_artifacts_summary_and_html_links(tmp_path: Path) -> None:
    config, plan, fit, analysis, bootstrap = _report_inputs()
    game = decision_game_fixture(decision_counts=(1, 0, 0))
    trace = decision_trace_fixture(game, seat=0)
    decision_report = build_decision_report_fixture((trace,), (game,))

    artifacts = write_tournament_artifacts(
        output_dir=tmp_path,
        overwrite=False,
        config=config,
        plan=plan,
        fit=fit,
        analysis=analysis,
        bootstrap=bootstrap,
        decision_report=decision_report,
    )

    assert artifacts.game_summaries_jsonl == tmp_path / "game-summaries.jsonl"
    assert artifacts.game_details_jsonl == tmp_path / "game-details.jsonl"
    assert artifacts.decision_traces_jsonl == tmp_path / "decision-traces.jsonl"
    assert artifacts.decision_slices_csv == tmp_path / "decision-slices.csv"
    assert {path.name for path in tmp_path.iterdir()} == {
        "ratings.csv",
        "summary.json",
        "report.html",
        "game-summaries.jsonl",
        "game-details.jsonl",
        "decision-traces.jsonl",
        "decision-slices.csv",
    }

    payload = json.loads(artifacts.summary_json.read_text(encoding="utf-8"))
    assert payload["configuration"]["root_seed"] is None
    assert payload["decision_diagnostics"] == {
        "schema_version": 1,
        "seed_disclosure": "withheld_for_privacy",
        "reconciliation": {
            "game_count": 1,
            "game_seat_count": 3,
            "trace_decision_count": 1,
            "game_summary_decision_count": 1,
            "slice_decision_count": 1,
        },
    }
    assert payload["artifacts"] == {
        "ratings_csv": "ratings.csv",
        "summary_json": "summary.json",
        "report_html": "report.html",
        "game_summaries_jsonl": "game-summaries.jsonl",
        "game_details_jsonl": "game-details.jsonl",
        "decision_traces_jsonl": "decision-traces.jsonl",
        "decision_slices_csv": "decision-slices.csv",
    }

    html = artifacts.report_html.read_text(encoding="utf-8")
    assert "Seed withheld for decision-trace privacy" in html
    assert "seed 42" not in html
    assert '<a href="game-summaries.jsonl">Game summaries</a>' in html
    assert '<a href="game-details.jsonl">Game details</a>' in html
    assert '<a href="decision-traces.jsonl">Decision traces</a>' in html
    assert '<a href="decision-slices.csv">Decision slices</a>' in html


def test_decision_report_summarizes_v3_rule_applications(tmp_path: Path) -> None:
    config, plan, fit, analysis, bootstrap = _report_inputs()
    game = decision_game_fixture(decision_counts=(3, 0, 0))
    traces = (
        replace(
            decision_trace_fixture(game, seat=0, step_index=0),
            explanation=FixedObjectiveOverlayV3BidExplanation(
                rule="not_applicable",
                planned_bid=3,
                chosen_bid=3,
            ),
        ),
        replace(
            decision_trace_fixture(game, seat=0, step_index=1),
            explanation=FixedObjectiveOverlayV3BidExplanation(
                rule="baseline",
                planned_bid=3,
                chosen_bid=3,
            ),
        ),
        replace(
            decision_trace_fixture(
                game,
                seat=0,
                step_index=2,
                selected_action=RecordedAction("submitBid", 2),
            ),
            explanation=FixedObjectiveOverlayV3BidExplanation(
                rule="guaranteed_win",
                planned_bid=3,
                chosen_bid=2,
            ),
        ),
    )
    decision_report = build_decision_report_fixture(traces, (game,))

    artifacts = write_tournament_artifacts(
        output_dir=tmp_path,
        overwrite=False,
        config=config,
        plan=plan,
        fit=fit,
        analysis=analysis,
        bootstrap=bootstrap,
        decision_report=decision_report,
    )

    payload = json.loads(artifacts.summary_json.read_text(encoding="utf-8"))
    assert payload["decision_diagnostics"]["fixed_objective_overlay_v3_rules"] == {
        "bid_decisions": 3,
        "resource_auction_decisions": 2,
        "rule_counts": {
            "not_applicable": 1,
            "baseline": 1,
            "guaranteed_win": 1,
        },
        "rule_application_rate": pytest.approx(1 / 2),
        "adjusted_bid_decisions": 1,
        "adjusted_bid_rate": pytest.approx(1 / 2),
        "total_bid_reduction": 1,
    }
    html = artifacts.report_html.read_text(encoding="utf-8")
    assert "exact guarantee cap applied 1 times" in html
    assert "across 2 resource auctions" in html
    assert "reducing submitted bids by 1 units" in html


def test_csv_follows_rating_order_and_round_trips_values(tmp_path: Path) -> None:
    config, plan, fit, analysis, bootstrap = _report_inputs()

    artifacts = write_tournament_artifacts(
        output_dir=tmp_path,
        overwrite=False,
        config=config,
        plan=plan,
        fit=fit,
        analysis=analysis,
        bootstrap=bootstrap,
    )

    with artifacts.ratings_csv.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["bot_id"] for row in rows] == [item.bot_id for item in fit.ratings]
    assert float(rows[0]["pl_rating"]) == fit.ratings[0].rating
    assert float(rows[0]["rating_interval_lower"]) == pytest.approx(fit.ratings[0].rating - 10)


def test_summary_includes_zero_pair_exposure(tmp_path: Path) -> None:
    config, plan, fit, analysis, bootstrap = _report_inputs()
    zero_exposure = replace(plan.pair_exposures[0], games=0)

    artifacts = write_tournament_artifacts(
        output_dir=tmp_path,
        overwrite=False,
        config=config,
        plan=replace(plan, pair_exposures=(zero_exposure, *plan.pair_exposures[1:])),
        fit=fit,
        analysis=analysis,
        bootstrap=bootstrap,
    )

    payload = json.loads(artifacts.summary_json.read_text())
    assert payload["schedule"]["pair_exposure"]["minimum"] == 0


def test_nonempty_directory_requires_overwrite(tmp_path: Path) -> None:
    config, plan, fit, analysis, bootstrap = _report_inputs()
    (tmp_path / "unrelated.txt").write_text("keep")

    with pytest.raises(FileExistsError, match="not empty"):
        write_tournament_artifacts(
            output_dir=tmp_path,
            overwrite=False,
            config=config,
            plan=plan,
            fit=fit,
            analysis=analysis,
            bootstrap=bootstrap,
        )


def test_overwrite_preserves_unrelated_files_and_renders_na(tmp_path: Path) -> None:
    config, plan, fit, analysis, bootstrap = _report_inputs()
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("keep")
    rows = tuple(
        replace(row, mean_winning_money=None) if index == len(analysis.rows) - 1 else row
        for index, row in enumerate(analysis.rows)
    )

    artifacts = write_tournament_artifacts(
        output_dir=tmp_path,
        overwrite=True,
        config=config,
        plan=plan,
        fit=fit,
        analysis=replace(analysis, rows=rows),
        bootstrap=replace(
            bootstrap,
            intervals=(),
            warnings=("confidence intervals are unavailable",),
        ),
    )

    assert unrelated.read_text() == "keep"
    html = artifacts.report_html.read_text()
    assert "n/a" in html
    assert "confidence intervals are unavailable" in html


@pytest.mark.parametrize("failure_position", range(1, 8))
@pytest.mark.parametrize("existing_generation", (False, True))
def test_replace_failure_restores_the_complete_known_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_position: int,
    existing_generation: bool,
) -> None:
    config, plan, fit, analysis, bootstrap = _report_inputs()
    game = decision_game_fixture(decision_counts=(1, 0, 0))
    trace = decision_trace_fixture(game, seat=0)
    decision_report = build_decision_report_fixture((trace,), (game,))
    previous_bytes: dict[str, bytes] = {}
    unrelated = tmp_path / "notes.txt"
    if existing_generation:
        write_tournament_artifacts(
            output_dir=tmp_path,
            overwrite=False,
            config=config,
            plan=plan,
            fit=fit,
            analysis=analysis,
            bootstrap=bootstrap,
            decision_report=decision_report,
        )
        previous_bytes = {
            name: (tmp_path / name).read_bytes() for name in _DIAGNOSTIC_ARTIFACT_NAMES
        }
        unrelated.write_text("keep me", encoding="utf-8")

    real_replace = os.replace
    replacement_count = 0

    def fail_selected_replacement(source: Any, destination: Any) -> None:
        nonlocal replacement_count
        source_path = Path(source)
        destination_path = Path(destination)
        if destination_path.name in _DIAGNOSTIC_ARTIFACT_NAMES and ".staged." in source_path.name:
            replacement_count += 1
            if replacement_count == failure_position:
                raise OSError(f"simulated replacement failure {failure_position}")
        real_replace(source, destination)

    monkeypatch.setattr(
        "garboid_pocketrocks.tournament.reporting.os.replace",
        fail_selected_replacement,
    )

    with pytest.raises(OSError, match=f"simulated replacement failure {failure_position}"):
        write_tournament_artifacts(
            output_dir=tmp_path,
            overwrite=existing_generation,
            config=config,
            plan=plan,
            fit=fit,
            analysis=analysis,
            bootstrap=replace(bootstrap, warnings=("changed generation",)),
            decision_report=decision_report,
        )

    if existing_generation:
        assert {
            name: (tmp_path / name).read_bytes() for name in _DIAGNOSTIC_ARTIFACT_NAMES
        } == previous_bytes
        assert unrelated.read_text(encoding="utf-8") == "keep me"
    else:
        assert not any((tmp_path / name).exists() for name in _DIAGNOSTIC_ARTIFACT_NAMES)
    assert not tuple(path for path in tmp_path.iterdir() if path.name.startswith("."))


def test_failed_rollback_preserves_and_reports_the_recovery_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, plan, fit, analysis, bootstrap = _report_inputs()
    game = decision_game_fixture(decision_counts=(1, 0, 0))
    trace = decision_trace_fixture(game, seat=0)
    decision_report = build_decision_report_fixture((trace,), (game,))
    write_tournament_artifacts(
        output_dir=tmp_path,
        overwrite=False,
        config=config,
        plan=plan,
        fit=fit,
        analysis=analysis,
        bootstrap=bootstrap,
        decision_report=decision_report,
    )
    previous_ratings = (tmp_path / "ratings.csv").read_bytes()
    real_replace = os.replace

    def fail_forward_and_rollback(source: Any, destination: Any) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if ".staged." in source_path.name and destination_path.name == "summary.json":
            raise OSError("simulated forward replacement failure")
        if ".backup." in source_path.name and destination_path.name == "ratings.csv":
            raise OSError("simulated rollback restoration failure")
        real_replace(source, destination)

    monkeypatch.setattr(
        "garboid_pocketrocks.tournament.reporting.os.replace",
        fail_forward_and_rollback,
    )

    with pytest.raises(RuntimeError, match="could not be fully restored") as captured:
        write_tournament_artifacts(
            output_dir=tmp_path,
            overwrite=True,
            config=config,
            plan=plan,
            fit=fit,
            analysis=analysis,
            bootstrap=replace(bootstrap, warnings=("changed generation",)),
            decision_report=decision_report,
        )

    recovery_backups = tuple(tmp_path.glob(".ratings.csv.backup.*"))
    assert len(recovery_backups) == 1
    assert recovery_backups[0].read_bytes() == previous_ratings
    assert str(recovery_backups[0]) in str(captured.value)
    assert not tuple(tmp_path.glob(".*.staged.*"))


def test_overwrite_without_report_removes_stale_known_diagnostic_artifacts(
    tmp_path: Path,
) -> None:
    config, plan, fit, analysis, bootstrap = _report_inputs()
    game = decision_game_fixture(decision_counts=(1, 0, 0))
    trace = decision_trace_fixture(game, seat=0)
    decision_report = build_decision_report_fixture((trace,), (game,))
    write_tournament_artifacts(
        output_dir=tmp_path,
        overwrite=False,
        config=config,
        plan=plan,
        fit=fit,
        analysis=analysis,
        bootstrap=bootstrap,
        decision_report=decision_report,
    )
    unrelated = tmp_path / "notes.txt"
    unrelated.write_text("keep me", encoding="utf-8")

    artifacts = write_tournament_artifacts(
        output_dir=tmp_path,
        overwrite=True,
        config=config,
        plan=plan,
        fit=fit,
        analysis=analysis,
        bootstrap=bootstrap,
    )

    assert artifacts.game_summaries_jsonl is None
    assert artifacts.decision_traces_jsonl is None
    assert artifacts.decision_slices_csv is None
    assert {path.name for path in tmp_path.iterdir()} == {
        "ratings.csv",
        "summary.json",
        "report.html",
        "notes.txt",
    }
    assert unrelated.read_text(encoding="utf-8") == "keep me"


def test_nonfinite_summary_value_fails_before_any_artifact_is_written(
    tmp_path: Path,
) -> None:
    config, plan, fit, analysis, bootstrap = _report_inputs()
    nonfinite_row = replace(analysis.rows[0], worth=math.nan)

    with pytest.raises(ValueError, match="finite JSON"):
        write_tournament_artifacts(
            output_dir=tmp_path,
            overwrite=False,
            config=config,
            plan=plan,
            fit=fit,
            analysis=replace(analysis, rows=(nonfinite_row, *analysis.rows[1:])),
            bootstrap=bootstrap,
        )

    assert not any((tmp_path / name).exists() for name in _DIAGNOSTIC_ARTIFACT_NAMES)
