from __future__ import annotations

import pytest

import garboid_pocketrocks.tournament.analysis as analysis_module
from garboid_pocketrocks.simulator.monte_carlo import GameSummary, MonteCarloResult
from garboid_pocketrocks.tournament.analysis import (
    analyze_tournament,
    bootstrap_rating_intervals,
)
from garboid_pocketrocks.tournament.rating import (
    PlackettLuceFit,
    RankingObservation,
    fit_plackett_luce,
)

from .helpers import game_summary


def _games() -> tuple[GameSummary, ...]:
    return (
        game_summary(
            ("a", "b", "c"),
            final_money=(30, 20, 10),
            ranks=(1, 2, 3),
            game_index=0,
        ),
        game_summary(
            ("b", "a", "c"),
            final_money=(25, 25, 5),
            ranks=(1, 1, 3),
            game_index=1,
            ruleset_name="live-B",
        ),
        game_summary(
            ("c", "b", "a"),
            final_money=(21, 20, 19),
            ranks=(1, 2, 3),
            game_index=2,
        ),
    )


def _fit() -> PlackettLuceFit:
    return fit_plackett_luce(
        (
            RankingObservation((("a",), ("b",), ("c",))),
            RankingObservation((("a", "b"), ("c",))),
            RankingObservation((("c",), ("b",), ("a",))),
        ),
        ("a", "b", "c"),
    )


def test_analysis_computes_finish_money_and_winning_money() -> None:
    result = MonteCarloResult(_games(), (), ())

    analysis = analyze_tournament(result, _fit())

    a = analysis.rows_by_id["a"]
    assert a.games == 3
    assert a.outright_wins == 1
    assert a.first_place_ties == 1
    assert a.mean_normalized_finish == pytest.approx((1.0 + 1.0 + 0.0) / 3)
    assert a.mean_final_money == pytest.approx((30 + 25 + 19) / 3)
    assert a.mean_winning_money == pytest.approx(27.5)
    assert a.faults == 0


def test_analysis_uses_none_when_a_bot_never_wins() -> None:
    result = MonteCarloResult(_games(), (), ())

    analysis = analyze_tournament(result, _fit())

    assert analysis.rows_by_id["c"].mean_winning_money == pytest.approx(21.0)
    no_win_games = (game_summary(("a", "b", "c"), final_money=(3, 2, 1), ranks=(1, 2, 3)),)
    no_win_fit = fit_plackett_luce(
        (RankingObservation((("a",), ("b",), ("c",))),),
        ("a", "b", "c"),
    )
    no_win = analyze_tournament(MonteCarloResult(no_win_games, (), ()), no_win_fit)
    assert no_win.rows_by_id["c"].mean_winning_money is None


def test_calibration_counts_every_pair_and_scores_ties_as_half() -> None:
    analysis = analyze_tournament(MonteCarloResult(_games(), (), ()), _fit())

    assert sum(bucket.count for bucket in analysis.calibration) == 9
    assert analysis.pair_outcomes == 9
    assert any(bucket.observed_score not in (0.0, 1.0) for bucket in analysis.calibration)


def test_condition_statistics_keep_chart_and_player_count() -> None:
    analysis = analyze_tournament(MonteCarloResult(_games(), (), ()), _fit())

    assert {(item.chart, item.player_count) for item in analysis.condition_statistics} == {
        ("A", 3),
        ("B", 3),
    }


def test_bootstrap_is_deterministic_and_resamples_whole_games() -> None:
    first = bootstrap_rating_intervals(
        _games(),
        ("a", "b", "c"),
        samples=20,
        root_seed=42,
        workers=1,
    )
    second = bootstrap_rating_intervals(
        _games(),
        ("a", "b", "c"),
        samples=20,
        root_seed=42,
        workers=2,
    )

    assert first == second
    assert first.requested == 20
    assert first.converged == 20
    assert tuple(interval.bot_id for interval in first.intervals) == ("a", "b", "c")
    assert all(interval.lower <= interval.upper for interval in first.intervals)


def test_bootstrap_zero_samples_returns_no_intervals() -> None:
    summary = bootstrap_rating_intervals(
        _games(),
        ("a", "b", "c"),
        samples=0,
        root_seed=42,
    )

    assert summary.requested == 0
    assert summary.converged == 0
    assert summary.intervals == ()
    assert summary.warnings == ()


def test_parallel_bootstrap_retries_serially_when_processes_are_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnavailableProcessPool:
        def __init__(self, **_: object) -> None:
            raise PermissionError("processes unavailable")

    monkeypatch.setattr(analysis_module, "ProcessPoolExecutor", UnavailableProcessPool)

    summary = bootstrap_rating_intervals(
        _games(),
        ("a", "b", "c"),
        samples=5,
        root_seed=42,
        workers=2,
    )

    assert summary.converged == 5
    assert len(summary.intervals) == 3
    assert summary.warnings == ("parallel bootstrap failed with PermissionError; retried serially",)


def test_bootstrap_rejects_invalid_bot_ids() -> None:
    with pytest.raises(ValueError, match="nonempty and unique"):
        bootstrap_rating_intervals(
            _games(),
            (),
            samples=1,
            root_seed=42,
        )
