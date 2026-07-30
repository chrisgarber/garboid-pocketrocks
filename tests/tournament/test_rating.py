from __future__ import annotations

import math
import statistics

import pytest

import garboid_pocketrocks.tournament.rating as rating_module
from garboid_pocketrocks.tournament.rating import (
    RankingObservation,
    TournamentRatingError,
    _ratings_from_worths,
    fit_plackett_luce,
    observations_from_games,
)

from .helpers import game_summary


def test_observations_preserve_multiplayer_ties() -> None:
    game = game_summary(
        ("a", "b", "c", "d"),
        final_money=(30, 20, 20, 10),
        ranks=(1, 2, 2, 4),
    )

    assert observations_from_games((game,)) == (RankingObservation((("a",), ("b", "c"), ("d",))),)


def test_rating_transform_maps_ten_to_one_worth_to_four_hundred_points() -> None:
    ratings = _ratings_from_worths({"strong": 10.0, "weak": 1.0})

    assert ratings["strong"] - ratings["weak"] == pytest.approx(400.0)
    assert statistics.mean(ratings.values()) == pytest.approx(1500.0)


def test_consistent_winner_has_highest_finite_rating() -> None:
    observations = (
        RankingObservation((("a",), ("b",), ("c",))),
        RankingObservation((("a",), ("c",), ("b",))),
        RankingObservation((("a",), ("b",), ("c",))),
    )

    fit = fit_plackett_luce(observations, ("a", "b", "c"))

    assert fit.ratings[0].bot_id == "a"
    assert all(math.isfinite(item.rating) and item.worth > 0 for item in fit.ratings)
    assert sum(item.worth for item in fit.ratings) == pytest.approx(1.0)
    assert statistics.mean(item.rating for item in fit.ratings) == pytest.approx(1500.0)


def test_tie_group_order_does_not_change_fit() -> None:
    first = fit_plackett_luce(
        (RankingObservation((("a", "b"), ("c",))),),
        ("a", "b", "c"),
    )
    second = fit_plackett_luce(
        (RankingObservation((("b", "a"), ("c",))),),
        ("a", "b", "c"),
    )

    assert first == second
    assert first.tie_prevalence[0].order == 2
    by_id = {item.bot_id: item for item in first.ratings}
    assert by_id["a"].rating == pytest.approx(by_id["b"].rating)


def test_later_tie_group_can_differ_from_fitted_bot_order() -> None:
    fit = fit_plackett_luce(
        (RankingObservation((("d",), ("b",), ("a", "z"))),),
        ("z", "b", "a", "d"),
    )

    assert {item.bot_id for item in fit.ratings} == {"z", "b", "a", "d"}


def test_duplicate_observations_are_combined_as_weights() -> None:
    ranking = (("a",), ("b",), ("c",))

    combined = rating_module._aggregate_observations(
        (
            RankingObservation(ranking),
            RankingObservation((("c",), ("b",), ("a",))),
            RankingObservation(ranking, weight=2.5),
        )
    )

    assert combined == (
        RankingObservation(ranking, weight=3.5),
        RankingObservation((("c",), ("b",), ("a",))),
    )


def test_weighted_observation_matches_repeated_observations() -> None:
    ranking = RankingObservation((("a",), ("b",), ("c",)))
    reverse = RankingObservation((("c",), ("b",), ("a",)))

    repeated = fit_plackett_luce((ranking, ranking, ranking, reverse), ("a", "b", "c"))
    weighted = fit_plackett_luce(
        (RankingObservation(ranking.rank_groups, weight=3.0), reverse),
        ("a", "b", "c"),
    )

    assert {item.bot_id: item.rating for item in repeated.ratings} == pytest.approx(
        {item.bot_id: item.rating for item in weighted.ratings}
    )


def test_higher_order_tie_is_fitted() -> None:
    fit = fit_plackett_luce(
        (
            RankingObservation((("a", "b", "c"), ("d",))),
            RankingObservation((("d",), ("a", "b", "c"))),
        ),
        ("a", "b", "c", "d"),
    )

    assert {item.order for item in fit.tie_prevalence} == {2, 3}
    assert all(item.prevalence >= 0 for item in fit.tie_prevalence)


def test_ghost_pseudorankings_keep_undefeated_and_winless_bots_finite() -> None:
    fit = fit_plackett_luce(
        (RankingObservation((("winner",), ("middle",), ("loser",))),) * 20,
        ("winner", "middle", "loser"),
    )

    assert all(math.isfinite(item.log_worth) for item in fit.ratings)
    assert fit.ratings[0].bot_id == "winner"
    assert fit.ratings[-1].bot_id == "loser"


def test_observation_rejects_duplicate_or_empty_rank_groups() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        RankingObservation((("a",), ()))
    with pytest.raises(ValueError, match="once"):
        RankingObservation((("a",), ("a",)))


def test_fit_rejects_unknown_bot_ids() -> None:
    with pytest.raises(TournamentRatingError, match="unknown"):
        fit_plackett_luce(
            (RankingObservation((("a",), ("missing",))),),
            ("a", "b"),
        )
