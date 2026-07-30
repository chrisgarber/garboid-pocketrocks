from __future__ import annotations

import pytest
from pocketrocks import OBJECTIVES

from garboid_pocketrocks.heuristics.objectives import (
    evaluate_objectives,
    requirement_vectors,
)


def test_every_sdk_objective_has_requirement_vectors() -> None:
    for objective_id in OBJECTIVES:
        vectors = requirement_vectors(objective_id)
        assert vectors
        assert all(len(vector) == 5 and sum(vector) > 0 for vector in vectors)


def test_immediate_completion_has_full_payout_and_no_progress() -> None:
    values = evaluate_objectives(
        active_objective_ids=(6,),
        owned_objective_ids_by_seat=((), (), ()),
        won_resource_counts_by_seat=((1, 0, 0, 0, 0),) * 3,
        bot_seat=0,
        offered_counts=(1, 0, 0, 0, 0),
        horizon=0.5,
        progress_weight=0.2,
    )
    assert values[0].completion_value == OBJECTIVES[6].payout
    assert values[0].progress_value == 0.0


def test_near_completion_progress_exceeds_initial_progress() -> None:
    initial = evaluate_objectives(
        active_objective_ids=(2,),
        owned_objective_ids_by_seat=((), (), ()),
        won_resource_counts_by_seat=((0, 0, 0, 0, 0),) * 3,
        bot_seat=0,
        offered_counts=(1, 0, 0, 0, 0),
        horizon=1.0,
        progress_weight=0.2,
    )[0]
    near = evaluate_objectives(
        active_objective_ids=(2,),
        owned_objective_ids_by_seat=((), (), ()),
        won_resource_counts_by_seat=(
            (1, 0, 0, 0, 0),
            (0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0),
        ),
        bot_seat=0,
        offered_counts=(1, 0, 0, 0, 0),
        horizon=1.0,
        progress_weight=0.2,
    )[0]
    assert near.progress_value > initial.progress_value


@pytest.mark.parametrize("weight", (-0.1, 1.1, float("nan"), float("inf")))
def test_evaluate_objectives_rejects_invalid_progress_weight(weight: float) -> None:
    with pytest.raises(ValueError, match="progress weight"):
        evaluate_objectives(
            active_objective_ids=(),
            owned_objective_ids_by_seat=((),),
            won_resource_counts_by_seat=((0, 0, 0, 0, 0),),
            bot_seat=0,
            offered_counts=(0, 0, 0, 0, 0),
            horizon=0.0,
            progress_weight=weight,
        )


@pytest.mark.parametrize("horizon", (-0.1, 1.1, float("nan"), float("inf")))
def test_evaluate_objectives_rejects_invalid_horizon(horizon: float) -> None:
    with pytest.raises(ValueError, match="horizon"):
        evaluate_objectives(
            active_objective_ids=(),
            owned_objective_ids_by_seat=((),),
            won_resource_counts_by_seat=((0, 0, 0, 0, 0),),
            bot_seat=0,
            offered_counts=(0, 0, 0, 0, 0),
            horizon=horizon,
            progress_weight=0.2,
        )
