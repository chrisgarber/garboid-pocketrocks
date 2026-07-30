from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest

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
