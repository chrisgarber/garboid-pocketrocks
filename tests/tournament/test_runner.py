from __future__ import annotations

from pathlib import Path

from garboid_pocketrocks.tournament.runner import TournamentRunner
from garboid_pocketrocks.tournament.schedule import TournamentConfig

from .helpers import random_specs


def test_runner_is_identical_across_worker_counts(tmp_path: Path) -> None:
    config = TournamentConfig(
        bot_specs=random_specs(),
        games=15,
        bootstrap_samples=0,
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
    assert len(serial.monte_carlo_result.game_summaries) == 15
    assert len({(job.ruleset.name, job.player_count) for job in serial.plan.jobs}) == 15
    assert serial.artifacts.ratings_csv.read_text() == parallel.artifacts.ratings_csv.read_text()
    assert serial.artifacts.summary_json.read_text() == parallel.artifacts.summary_json.read_text()
    assert serial.artifacts.report_html.read_text() == parallel.artifacts.report_html.read_text()
