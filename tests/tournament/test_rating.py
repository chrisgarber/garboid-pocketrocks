from __future__ import annotations

import itertools
import math
import statistics

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.special import logsumexp  # type: ignore[import-untyped]

import garboid_pocketrocks.tournament.rating as rating_module
from garboid_pocketrocks.tournament.rating import (
    RankingObservation,
    TournamentRatingError,
    _ratings_from_worths,
    fit_plackett_luce,
    observations_from_games,
)

from .helpers import game_summary


def _reference_negative_log_likelihood(
    parameters: NDArray[np.float64],
    observations: tuple[rating_module._IndexedObservation, ...],
    *,
    bot_count: int,
    ghost_index: int,
    maximum_tie_order: int,
) -> tuple[float, NDArray[np.float64]]:
    value = 0.0
    gradient = np.zeros_like(parameters)
    tie_offset = bot_count

    def log_worth(index: int) -> float:
        return 0.0 if index == ghost_index else float(parameters[index])

    for observation in observations:
        remaining = tuple(sorted(index for group in observation.rank_groups for index in group))
        for chosen in observation.rank_groups:
            if len(remaining) == 1:
                break
            candidates = tuple(
                subset
                for order in range(1, min(maximum_tie_order, len(remaining)) + 1)
                for subset in itertools.combinations(remaining, order)
            )
            log_weights = np.asarray(
                [
                    (
                        0.0
                        if len(subset) == 1
                        else float(parameters[tie_offset + len(subset) - 2])
                    )
                    + sum(log_worth(index) for index in subset) / len(subset)
                    for subset in candidates
                ],
                dtype=np.float64,
            )
            chosen_index = candidates.index(tuple(sorted(chosen)))
            probabilities = np.exp(log_weights - logsumexp(log_weights))
            value += observation.weight * (
                float(logsumexp(log_weights)) - float(log_weights[chosen_index])
            )
            for probability, subset in zip(probabilities, candidates, strict=True):
                contribution = observation.weight * float(probability)
                for index in subset:
                    if index != ghost_index:
                        gradient[index] += contribution / len(subset)
                if len(subset) > 1:
                    gradient[tie_offset + len(subset) - 2] += contribution
            for index in chosen:
                if index != ghost_index:
                    gradient[index] -= observation.weight / len(chosen)
            if len(chosen) > 1:
                gradient[tie_offset + len(chosen) - 2] -= observation.weight
            chosen_set = set(chosen)
            remaining = tuple(index for index in remaining if index not in chosen_set)
    return value, gradient


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


def test_compiled_choice_sets_match_reference_likelihood_and_gradient() -> None:
    bot_count = 3
    ghost_index = 3
    maximum_tie_order = 2
    observations = (
        rating_module._IndexedObservation(((0,), (1, 2)), 2.5),
        rating_module._IndexedObservation(((1, 2), (0,)), 1.25),
        rating_module._IndexedObservation(((2,), (0,), (1,)), 0.75),
        rating_module._IndexedObservation(((0,), (ghost_index,)), 0.5),
    )
    parameters = np.asarray((0.2, -0.1, 0.05, math.log(0.7)), dtype=np.float64)
    problem = rating_module._FitProblem(
        choice_sets=rating_module._compile_choice_sets(
            observations,
            bot_count=bot_count,
            ghost_index=ghost_index,
            maximum_tie_order=maximum_tie_order,
        ),
        parameter_count=len(parameters),
    )

    compiled_value, compiled_gradient = rating_module._negative_log_likelihood(
        parameters,
        problem,
    )
    reference_value, reference_gradient = _reference_negative_log_likelihood(
        parameters,
        observations,
        bot_count=bot_count,
        ghost_index=ghost_index,
        maximum_tie_order=maximum_tie_order,
    )

    assert compiled_value == pytest.approx(reference_value, abs=1e-10)
    np.testing.assert_allclose(compiled_gradient, reference_gradient, atol=1e-10)


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
