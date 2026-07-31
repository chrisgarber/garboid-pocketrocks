from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import garboid_pocketrocks.tournament.runner as runner_module
from garboid_pocketrocks.diagnostics.trace import RecordedAction
from garboid_pocketrocks.simulator.monte_carlo import (
    GameJob,
    MonteCarloConfig,
    MonteCarloResult,
    MonteCarloRunner,
)
from garboid_pocketrocks.tournament.runner import TournamentRunner
from garboid_pocketrocks.tournament.schedule import (
    TournamentConfig,
    TournamentPlanner,
)

from .helpers import random_specs


def test_decision_reports_are_opt_in_and_keep_the_ordinary_plan() -> None:
    ordinary_config = TournamentConfig(
        bot_specs=random_specs(),
        games=15,
        bootstrap_samples=0,
        root_seed=42,
    )
    diagnostic_config = replace(ordinary_config, decision_reports=True)

    ordinary = TournamentPlanner.plan(ordinary_config)
    diagnostic = TournamentPlanner.plan(diagnostic_config)

    assert ordinary_config.decision_reports is False
    assert ordinary.monte_carlo_config.capture_decision_traces is False
    assert diagnostic.monte_carlo_config.capture_decision_traces is True
    assert all(job.capture_decision_traces is False for job in ordinary.jobs)
    assert all(job.capture_decision_traces is True for job in diagnostic.jobs)
    assert (
        tuple(replace(job, capture_decision_traces=False) for job in diagnostic.jobs)
        == ordinary.jobs
    )


def test_decision_reports_preserve_ordinary_tournament_results_and_batching(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_sizes: list[int | None] = []
    original_run_jobs = MonteCarloRunner.run_jobs

    def record_batch_size(
        config: MonteCarloConfig,
        jobs: tuple[GameJob, ...],
        *,
        workers: int = 1,
        batch_size: int | None = None,
    ) -> MonteCarloResult:
        batch_sizes.append(batch_size)
        return original_run_jobs(config, jobs, workers=workers, batch_size=batch_size)

    monkeypatch.setattr(MonteCarloRunner, "run_jobs", record_batch_size)
    ordinary_config = TournamentConfig(
        bot_specs=random_specs(),
        games=15,
        batch_size=4,
        bootstrap_samples=4,
        root_seed=42,
    )
    diagnostic_config = replace(ordinary_config, decision_reports=True)

    ordinary = TournamentRunner.run(
        ordinary_config,
        workers=1,
        output_dir=tmp_path / "ordinary",
    )
    diagnostic = TournamentRunner.run(
        diagnostic_config,
        workers=1,
        output_dir=tmp_path / "diagnostic",
    )

    assert (
        tuple(replace(job, capture_decision_traces=False) for job in diagnostic.plan.jobs)
        == ordinary.plan.jobs
    )
    assert (
        replace(
            diagnostic.plan.monte_carlo_config,
            capture_decision_traces=False,
        )
        == ordinary.plan.monte_carlo_config
    )
    assert diagnostic.monte_carlo_result.decision_traces
    assert (
        replace(
            diagnostic.monte_carlo_result,
            decision_traces=(),
            game_details=(),
        )
        == ordinary.monte_carlo_result
    )
    replayed = original_run_jobs(
        replace(ordinary.plan.monte_carlo_config, capture_replays=True),
        ordinary.plan.jobs,
        workers=1,
        batch_size=ordinary.config.batch_size,
    )
    replayed_actions = {
        (replay.game_index, step_index, seat): RecordedAction.from_decision(decision)
        for replay in replayed.replays
        for step_index, decisions in replay.decisions
        for seat, decision in decisions
    }
    traced_actions = {
        (trace.game_index, trace.step_index, trace.seat): trace.selected_action
        for trace in diagnostic.monte_carlo_result.decision_traces
    }
    assert traced_actions == replayed_actions
    assert diagnostic.fit == ordinary.fit
    assert diagnostic.analysis == ordinary.analysis
    assert diagnostic.bootstrap == ordinary.bootstrap
    assert batch_sizes == [4, 4]
    assert {path.name for path in (tmp_path / "ordinary").iterdir()} == {
        "ratings.csv",
        "summary.json",
        "report.html",
    }
    assert diagnostic.artifacts.game_summaries_jsonl is not None
    assert diagnostic.artifacts.decision_traces_jsonl is not None
    assert diagnostic.artifacts.decision_slices_csv is not None


def test_invalid_decision_report_fails_before_artifacts_are_written(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer_called = False

    def fail_report(*_: object, **__: object) -> None:
        raise ValueError("diagnostic reconciliation failed")

    def record_writer(*_: object, **__: object) -> None:
        nonlocal writer_called
        writer_called = True

    monkeypatch.setattr(
        "garboid_pocketrocks.diagnostics.analysis.build_decision_report",
        fail_report,
    )
    monkeypatch.setattr(runner_module, "write_tournament_artifacts", record_writer)
    config = TournamentConfig(
        bot_specs=random_specs(),
        games=15,
        bootstrap_samples=0,
        decision_reports=True,
    )

    with pytest.raises(ValueError, match="diagnostic reconciliation"):
        TournamentRunner.run(config, workers=1, output_dir=tmp_path)

    assert writer_called is False
    assert not any(tmp_path.iterdir())


def test_runner_is_identical_across_worker_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_sizes: list[int | None] = []
    original_run_jobs = MonteCarloRunner.run_jobs

    def record_batch_size(
        config: MonteCarloConfig,
        jobs: tuple[GameJob, ...],
        *,
        workers: int = 1,
        batch_size: int | None = None,
    ) -> MonteCarloResult:
        batch_sizes.append(batch_size)
        return original_run_jobs(config, jobs, workers=workers, batch_size=batch_size)

    monkeypatch.setattr(MonteCarloRunner, "run_jobs", record_batch_size)
    config = TournamentConfig(
        bot_specs=random_specs(),
        games=15,
        batch_size=4,
        bootstrap_samples=4,
        root_seed=42,
    )

    serial = TournamentRunner.run(
        config,
        workers=1,
        output_dir=tmp_path / "serial",
    )
    parallel = TournamentRunner.run(
        config,
        workers=2,
        output_dir=tmp_path / "parallel",
    )

    assert serial.plan == parallel.plan
    assert serial.monte_carlo_result == parallel.monte_carlo_result
    assert serial.fit == parallel.fit
    assert serial.analysis == parallel.analysis
    assert serial.bootstrap == parallel.bootstrap
    assert len(serial.monte_carlo_result.game_summaries) == 15
    assert len({(job.value_chart, job.player_count) for job in serial.plan.jobs}) == 15
    assert serial.artifacts.ratings_csv.read_text() == parallel.artifacts.ratings_csv.read_text()
    assert serial.artifacts.summary_json.read_text() == parallel.artifacts.summary_json.read_text()
    assert serial.artifacts.report_html.read_text() == parallel.artifacts.report_html.read_text()
    assert batch_sizes == [4, 4]


def test_runner_writes_primary_artifacts_when_bootstrap_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_bootstrap(*_: object, **__: object) -> None:
        raise RuntimeError("uncertainty unavailable")

    monkeypatch.setattr(runner_module, "bootstrap_rating_intervals", fail_bootstrap)
    config = TournamentConfig(
        bot_specs=random_specs(),
        games=15,
        bootstrap_samples=4,
        root_seed=42,
    )

    run = TournamentRunner.run(
        config,
        workers=1,
        output_dir=tmp_path,
    )

    assert run.bootstrap.intervals == ()
    assert run.bootstrap.warnings == (
        "bootstrap failed with RuntimeError; confidence intervals are unavailable",
    )
    assert run.artifacts.ratings_csv.is_file()
    assert run.artifacts.summary_json.is_file()
    assert run.artifacts.report_html.is_file()
