from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize  # type: ignore[import-untyped]
from scipy.special import logsumexp  # type: ignore[import-untyped]

from garboid_pocketrocks.simulator.monte_carlo import GameSummary

_GHOST_ID = "__plackett_luce_ghost__"
_PSEUDO_WEIGHT = 0.5
_LOG_TIE_BOUND = 12.0


class TournamentRatingError(RuntimeError):
    """Raised when tournament rankings cannot produce a valid strength fit."""


@dataclass(frozen=True, slots=True)
class RankingObservation:
    rank_groups: tuple[tuple[str, ...], ...]
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.rank_groups or any(not group for group in self.rank_groups):
            raise ValueError("ranking observations require nonempty rank groups")
        normalized = tuple(tuple(sorted(group)) for group in self.rank_groups)
        bot_ids = tuple(bot_id for group in normalized for bot_id in group)
        if len(set(bot_ids)) != len(bot_ids):
            raise ValueError("each bot may appear only once in a ranking observation")
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("ranking observation weight must be finite and positive")
        object.__setattr__(self, "rank_groups", normalized)


@dataclass(frozen=True, slots=True)
class PLBotRating:
    bot_id: str
    worth: float
    log_worth: float
    rating: float


@dataclass(frozen=True, slots=True)
class TiePrevalence:
    order: int
    prevalence: float


@dataclass(frozen=True, slots=True)
class PLFitDiagnostics:
    objective: float
    iterations: int
    gradient_norm: float
    optimizer_message: str
    pseudo_weight: float


@dataclass(frozen=True, slots=True)
class PlackettLuceFit:
    ratings: tuple[PLBotRating, ...]
    tie_prevalence: tuple[TiePrevalence, ...]
    diagnostics: PLFitDiagnostics

    @property
    def ratings_by_id(self) -> dict[str, PLBotRating]:
        return {rating.bot_id: rating for rating in self.ratings}


@dataclass(frozen=True, slots=True)
class _IndexedObservation:
    rank_groups: tuple[tuple[int, ...], ...]
    weight: float


@dataclass(frozen=True, slots=True)
class _FitProblem:
    observations: tuple[_IndexedObservation, ...]
    bot_count: int
    ghost_index: int
    maximum_tie_order: int


def observations_from_games(
    games: tuple[GameSummary, ...],
) -> tuple[RankingObservation, ...]:
    observations: list[RankingObservation] = []
    for game in games:
        bot_id_by_seat = dict(enumerate(game.bot_ids))
        groups_by_rank: dict[int, list[str]] = {}
        for score in game.scores:
            groups_by_rank.setdefault(score.rank, []).append(bot_id_by_seat[score.seat])
        observations.append(
            RankingObservation(
                tuple(tuple(groups_by_rank[rank]) for rank in sorted(groups_by_rank))
            )
        )
    return tuple(observations)


def _aggregate_observations(
    observations: tuple[RankingObservation, ...],
) -> tuple[RankingObservation, ...]:
    weights: dict[tuple[tuple[str, ...], ...], float] = {}
    for observation in observations:
        weights[observation.rank_groups] = (
            weights.get(observation.rank_groups, 0.0) + observation.weight
        )
    return tuple(
        RankingObservation(rank_groups, weight=weight) for rank_groups, weight in weights.items()
    )


def fit_plackett_luce(
    observations: tuple[RankingObservation, ...],
    bot_ids: tuple[str, ...],
    *,
    pseudo_weight: float = _PSEUDO_WEIGHT,
) -> PlackettLuceFit:
    if not observations:
        raise TournamentRatingError("at least one ranking observation is required")
    observations = _aggregate_observations(observations)
    if not bot_ids or len(set(bot_ids)) != len(bot_ids):
        raise TournamentRatingError("fitted bot IDs must be nonempty and unique")
    if _GHOST_ID in bot_ids:
        raise TournamentRatingError("bot IDs must not use the reserved ghost identity")
    if not math.isfinite(pseudo_weight) or pseudo_weight <= 0:
        raise TournamentRatingError("pseudo-ranking weight must be finite and positive")

    index_by_id = {bot_id: index for index, bot_id in enumerate(bot_ids)}
    indexed: list[_IndexedObservation] = []
    maximum_tie_order = 1
    for observation in observations:
        unknown = {
            bot_id
            for group in observation.rank_groups
            for bot_id in group
            if bot_id not in index_by_id
        }
        if unknown:
            raise TournamentRatingError(
                f"ranking observations contain unknown bot IDs: {', '.join(sorted(unknown))}"
            )
        maximum_tie_order = max(
            maximum_tie_order,
            *(len(group) for group in observation.rank_groups),
        )
        indexed.append(
            _IndexedObservation(
                tuple(
                    tuple(index_by_id[bot_id] for bot_id in group)
                    for group in observation.rank_groups
                ),
                observation.weight,
            )
        )

    ghost_index = len(bot_ids)
    for bot_index in range(len(bot_ids)):
        indexed.extend(
            (
                _IndexedObservation(((bot_index,), (ghost_index,)), pseudo_weight),
                _IndexedObservation(((ghost_index,), (bot_index,)), pseudo_weight),
            )
        )

    problem = _FitProblem(
        observations=tuple(indexed),
        bot_count=len(bot_ids),
        ghost_index=ghost_index,
        maximum_tie_order=maximum_tie_order,
    )
    parameter_count = len(bot_ids) + maximum_tie_order - 1
    initial = np.zeros(parameter_count, dtype=np.float64)
    bounds = [(None, None)] * len(bot_ids) + [(-_LOG_TIE_BOUND, _LOG_TIE_BOUND)] * (
        maximum_tie_order - 1
    )
    result = minimize(
        _negative_log_likelihood,
        initial,
        args=(problem,),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": 1_000, "gtol": 1e-8},
    )
    gradient = np.asarray(result.jac, dtype=np.float64)
    if (
        not result.success
        or not math.isfinite(float(result.fun))
        or not np.all(np.isfinite(result.x))
        or not np.all(np.isfinite(gradient))
    ):
        raise TournamentRatingError(
            "Plackett-Luce optimization failed: "
            f"{result.message}; inspect bot faults and comparison coverage"
        )

    raw_log_worths = np.asarray(result.x[: len(bot_ids)], dtype=np.float64)
    shifted = raw_log_worths - np.max(raw_log_worths)
    raw_worths = np.exp(shifted)
    normalized_worths = raw_worths / np.sum(raw_worths)
    worths_by_id = {bot_id: float(normalized_worths[index]) for index, bot_id in enumerate(bot_ids)}
    display_ratings = _ratings_from_worths(worths_by_id)
    mean_log_worth = float(np.mean(np.log(normalized_worths)))
    ratings = tuple(
        sorted(
            (
                PLBotRating(
                    bot_id=bot_id,
                    worth=worths_by_id[bot_id],
                    log_worth=float(math.log(worths_by_id[bot_id]) - mean_log_worth),
                    rating=display_ratings[bot_id],
                )
                for bot_id in bot_ids
            ),
            key=lambda item: (-item.rating, item.bot_id),
        )
    )
    tie_prevalence = tuple(
        TiePrevalence(order=order, prevalence=float(math.exp(result.x[len(bot_ids) + order - 2])))
        for order in range(2, maximum_tie_order + 1)
    )
    return PlackettLuceFit(
        ratings=ratings,
        tie_prevalence=tie_prevalence,
        diagnostics=PLFitDiagnostics(
            objective=float(result.fun),
            iterations=int(result.nit),
            gradient_norm=float(np.linalg.norm(gradient)),
            optimizer_message=str(result.message),
            pseudo_weight=pseudo_weight,
        ),
    )


def _negative_log_likelihood(
    parameters: NDArray[np.float64],
    problem: _FitProblem,
) -> tuple[float, NDArray[np.float64]]:
    value = 0.0
    gradient = np.zeros_like(parameters)
    tie_offset = problem.bot_count

    def log_worth(index: int) -> float:
        if index == problem.ghost_index:
            return 0.0
        return float(parameters[index])

    for observation in problem.observations:
        remaining = tuple(sorted(index for group in observation.rank_groups for index in group))
        for chosen in observation.rank_groups:
            if len(remaining) == 1:
                break
            candidates = tuple(
                subset
                for order in range(
                    1,
                    min(problem.maximum_tie_order, len(remaining)) + 1,
                )
                for subset in itertools.combinations(remaining, order)
            )
            log_weights = np.asarray(
                [
                    (0.0 if len(subset) == 1 else float(parameters[tie_offset + len(subset) - 2]))
                    + sum(log_worth(index) for index in subset) / len(subset)
                    for subset in candidates
                ],
                dtype=np.float64,
            )
            chosen_tuple = tuple(sorted(chosen))
            try:
                chosen_index = candidates.index(chosen_tuple)
            except ValueError as error:
                raise TournamentRatingError(
                    "rank group is not a valid choice from remaining bots"
                ) from error
            probabilities = np.exp(log_weights - logsumexp(log_weights))
            value += observation.weight * (
                float(logsumexp(log_weights)) - float(log_weights[chosen_index])
            )
            for probability, subset in zip(probabilities, candidates, strict=True):
                contribution = observation.weight * float(probability)
                for index in subset:
                    if index != problem.ghost_index:
                        gradient[index] += contribution / len(subset)
                if len(subset) > 1:
                    gradient[tie_offset + len(subset) - 2] += contribution
            for index in chosen:
                if index != problem.ghost_index:
                    gradient[index] -= observation.weight / len(chosen)
            if len(chosen) > 1:
                gradient[tie_offset + len(chosen) - 2] -= observation.weight
            chosen_set = set(chosen)
            remaining = tuple(index for index in remaining if index not in chosen_set)
    return value, gradient


def _ratings_from_worths(worths: dict[str, float]) -> dict[str, float]:
    if not worths or any(not math.isfinite(worth) or worth <= 0 for worth in worths.values()):
        raise TournamentRatingError("worths must be finite and positive")
    mean_log_worth = sum(math.log(worth) for worth in worths.values()) / len(worths)
    return {
        bot_id: 1500.0 + (400.0 * (math.log(worth) - mean_log_worth) / math.log(10.0))
        for bot_id, worth in worths.items()
    }
