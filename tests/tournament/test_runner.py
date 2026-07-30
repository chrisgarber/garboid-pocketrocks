from __future__ import annotations

from pathlib import Path

import pytest

import garboid_pocketrocks.tournament.runner as runner_module
from garboid_pocketrocks.simulator.monte_carlo import (
    GameJob,
    MonteCarloConfig,
    MonteCarloResult,
    MonteCarloRunner,
)
from garboid_pocketrocks.tournament.runner import TournamentRunner
from garboid_pocketrocks.tournament.schedule import TournamentConfig

from .helpers import random_specs


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
