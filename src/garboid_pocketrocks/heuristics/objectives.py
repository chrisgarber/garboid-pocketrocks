from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache
from itertools import combinations

from pocketrocks import OBJECTIVES

_SUIT_COUNT = 5


@dataclass(frozen=True, slots=True)
class ObjectiveValue:
    objective_id: int
    completion_value: float
    progress_value: float

    @property
    def total(self) -> float:
        return self.completion_value + self.progress_value


@cache
def requirement_vectors(objective_id: int) -> tuple[tuple[int, ...], ...]:
    """Return every minimal suit-count vector that satisfies an objective."""
    objective = OBJECTIVES[objective_id]
    if objective.pattern == "same2":
        return tuple(
            tuple(2 if index == suit else 0 for index in range(_SUIT_COUNT))
            for suit in range(_SUIT_COUNT)
        )
    if objective.pattern == "same3":
        return tuple(
            tuple(3 if index == suit else 0 for index in range(_SUIT_COUNT))
            for suit in range(_SUIT_COUNT)
        )
    if objective.pattern in {"different3", "different4"}:
        required_suits = int(objective.pattern[-1])
        return tuple(
            tuple(1 if index in suit_set else 0 for index in range(_SUIT_COUNT))
            for suit_set in combinations(range(_SUIT_COUNT), required_suits)
        )
    if objective.pattern == "twoPairs4":
        return tuple(
            tuple(2 if index in suit_pair else 0 for index in range(_SUIT_COUNT))
            for suit_pair in combinations(range(_SUIT_COUNT), 2)
        )
    assert objective.requirement is not None
    return (tuple(objective.requirement),)


def objective_distance(objective_id: int, counts: tuple[int, ...]) -> int:
    """Return the fewest additional resources needed to satisfy an objective."""
    return min(
        sum(max(required - owned, 0) for owned, required in zip(counts, vector, strict=True))
        for vector in requirement_vectors(objective_id)
    )


def objective_is_met(objective_id: int, counts: tuple[int, ...]) -> bool:
    return objective_distance(objective_id, counts) == 0


def _progress(counts: tuple[int, ...], vectors: tuple[tuple[int, ...], ...]) -> float:
    return max(
        1.0
        - (
            sum(max(required - owned, 0) for owned, required in zip(counts, vector, strict=True))
            / sum(vector)
        )
        for vector in vectors
    )


def evaluate_objectives(
    *,
    active_objective_ids: tuple[int, ...],
    owned_objective_ids_by_seat: tuple[tuple[int, ...], ...],
    won_resource_counts_by_seat: tuple[tuple[int, ...], ...],
    bot_seat: int,
    offered_counts: tuple[int, ...],
    horizon: float,
    progress_weight: float,
) -> tuple[ObjectiveValue, ...]:
    """Value active, unowned objectives for receiving an offered resource bundle."""
    if not math.isfinite(horizon) or not 0 <= horizon <= 1:
        raise ValueError("horizon must be finite and between zero and one")
    if not math.isfinite(progress_weight) or not 0 <= progress_weight <= 1:
        raise ValueError("progress weight must be finite and between zero and one")

    owned_objective_ids = {
        objective_id
        for objective_ids in owned_objective_ids_by_seat
        for objective_id in objective_ids
    }
    bot_counts = won_resource_counts_by_seat[bot_seat]
    post_win_counts = tuple(
        owned + offered for owned, offered in zip(bot_counts, offered_counts, strict=True)
    )
    values: list[ObjectiveValue] = []
    for objective_id in active_objective_ids:
        if objective_id in owned_objective_ids:
            continue
        vectors = requirement_vectors(objective_id)
        if not objective_is_met(objective_id, bot_counts) and objective_is_met(
            objective_id, post_win_counts
        ):
            values.append(
                ObjectiveValue(
                    objective_id=objective_id,
                    completion_value=float(OBJECTIVES[objective_id].payout),
                    progress_value=0.0,
                )
            )
            continue

        before_progress = _progress(bot_counts, vectors)
        after_progress = _progress(post_win_counts, vectors)
        post_win_distance = objective_distance(objective_id, post_win_counts)
        opponents_at_or_closer = sum(
            objective_distance(objective_id, counts) <= post_win_distance
            for seat, counts in enumerate(won_resource_counts_by_seat)
            if seat != bot_seat
        )
        contest_factor = 1 / (1 + opponents_at_or_closer)
        progress_value = (
            OBJECTIVES[objective_id].payout
            * progress_weight
            * max(after_progress**2 - before_progress**2, 0.0)
            * horizon
            * contest_factor
        )
        values.append(
            ObjectiveValue(
                objective_id=objective_id,
                completion_value=0.0,
                progress_value=progress_value,
            )
        )
    return tuple(values)
