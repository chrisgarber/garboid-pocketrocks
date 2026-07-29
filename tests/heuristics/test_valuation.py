from __future__ import annotations

import math
from dataclasses import replace

import pytest
from pocketrocks import ActionId, DecisionContext

from garboid_pocketrocks.heuristics.errors import HeuristicInputError
from garboid_pocketrocks.heuristics.profiles import (
    AGGRESSIVE_PROFILE,
    BALANCED_PROFILE,
    PASSIVE_PROFILE,
    HeuristicProfile,
)
from garboid_pocketrocks.heuristics.valuation import HeuristicValuator

from .helpers import make_context, make_knowledge

NO_LIQUIDITY = HeuristicProfile("test", 0.0, 0.0, 0.0)


def test_constant_ten_dollar_resource_has_ten_dollar_reservation() -> None:
    chart = (10, 10, 10, 10, 10, 10)
    context = make_context(value_chart=chart, legal_max=30)
    result = HeuristicValuator(NO_LIQUIDITY).evaluate_bid(
        context,
        make_knowledge(value_chart=chart),
    )

    assert result.reservation_bid == 10
    assert result.chosen_bid == 10
    assert result.points[10].breakdown.resource == 10.0
    assert result.points[10].win_delta == 0.0


def test_two_identical_resources_are_valued_as_two_cards() -> None:
    chart = (7, 7, 7, 7, 7, 7)
    result = HeuristicValuator(NO_LIQUIDITY).evaluate_bid(
        make_context(
            action_id=ActionId.AUCTION2,
            current_resources=(1, 1),
            value_chart=chart,
            legal_max=20,
        ),
        make_knowledge(value_chart=chart),
    )

    assert result.points[0].breakdown.resource == 14.0
    assert result.reservation_bid == 14


def test_newly_completed_objective_adds_its_full_payout() -> None:
    context = make_context(
        won=((1, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
        objectives=(6,),
    )
    knowledge = replace(
        make_knowledge(),
        active_objective_count=1,
        objectives_enabled=True,
    )

    result = HeuristicValuator(NO_LIQUIDITY).evaluate_bid(context, knowledge)

    assert result.points[0].breakdown.resource == 0.0
    assert result.points[0].breakdown.objective_completion == 5.0
    assert result.points[0].breakdown.objective_progress == 0.0
    assert result.reservation_bid == 5


def test_zero_horizon_loan_passes_and_investment_can_lock_all_cash() -> None:
    won = ((2, 2, 2, 2, 2), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0))
    loan_context = make_context(
        action_id=ActionId.LOAN10,
        current_resources=(0, 0),
        won=won,
        cash=(20, 20, 20),
        legal_max=30,
    )
    investment_context = make_context(
        action_id=ActionId.INVEST5,
        current_resources=(0, 0),
        won=won,
        cash=(20, 20, 20),
        legal_max=20,
    )
    evaluator = HeuristicValuator(NO_LIQUIDITY)

    assert evaluator.evaluate_bid(loan_context, make_knowledge()).chosen_bid == 0
    assert evaluator.evaluate_bid(investment_context, make_knowledge()).chosen_bid == 20


@pytest.mark.parametrize(
    "profile",
    (AGGRESSIVE_PROFILE, BALANCED_PROFILE, PASSIVE_PROFILE),
)
@pytest.mark.parametrize(
    ("action_id", "resources", "legal_max"),
    (
        (ActionId.AUCTION1, (1, 0), 30),
        (ActionId.LOAN10, (0, 0), 40),
        (ActionId.INVEST5, (0, 0), 30),
    ),
)
def test_profile_shading_never_exceeds_reservation_or_legal_maximum(
    profile: HeuristicProfile,
    action_id: ActionId,
    resources: tuple[int, int],
    legal_max: int,
) -> None:
    result = HeuristicValuator(profile).evaluate_bid(
        make_context(
            action_id=action_id,
            current_resources=resources,
            legal_max=legal_max,
        ),
        make_knowledge(),
    )

    assert 0 <= result.chosen_bid <= result.reservation_bid <= legal_max


def test_chosen_bid_is_floor_of_shaded_reservation() -> None:
    chart = (10, 10, 10, 10, 10, 10)
    profile = HeuristicProfile("half", 0.0, 0.0, 0.25)
    result = HeuristicValuator(profile).evaluate_bid(
        make_context(value_chart=chart, legal_max=30),
        make_knowledge(value_chart=chart),
    )

    assert result.reservation_bid == 10
    assert result.chosen_bid == 7


def test_breakdown_components_sum_to_each_point_delta() -> None:
    result = HeuristicValuator(NO_LIQUIDITY).evaluate_bid(
        make_context(),
        make_knowledge(),
    )

    for point in result.points:
        breakdown = point.breakdown
        expected = (
            breakdown.resource
            + breakdown.objective_completion
            + breakdown.objective_progress
            + breakdown.terminal_cash
            + breakdown.liquidity
        )
        assert breakdown.total == expected == point.win_delta
        assert all(
            math.isfinite(component)
            for component in (
                breakdown.resource,
                breakdown.objective_completion,
                breakdown.objective_progress,
                breakdown.terminal_cash,
                breakdown.liquidity,
                breakdown.total,
            )
        )


def test_financial_actions_have_no_resource_or_objective_value() -> None:
    result = HeuristicValuator(NO_LIQUIDITY).evaluate_bid(
        make_context(
            action_id=ActionId.INVEST10,
            current_resources=(0, 0),
            legal_max=30,
        ),
        make_knowledge(),
    )

    assert all(
        point.breakdown.resource == 0.0
        and point.breakdown.objective_completion == 0.0
        and point.breakdown.objective_progress == 0.0
        for point in result.points
    )


@pytest.mark.parametrize(
    ("context", "message"),
    (
        (
            make_context(
                decision_kind="selectInfoToReveal",
                action_id=ActionId.AUCTION1,
                current_resources=(1, 0),
                hand=(2,),
                legal_max=None,
            ),
            "bid",
        ),
        (make_context(action_id=None), "action"),
        (make_context(legal_max=None), "legal maximum"),
        (
            make_context(
                action_id=ActionId.INVEST5,
                current_resources=(0, 0),
                cash=(10, 30, 30),
                legal_max=11,
            ),
            "legal_max",
        ),
    ),
)
def test_invalid_bid_contexts_raise_heuristic_input_error(
    context: DecisionContext,
    message: str,
) -> None:
    with pytest.raises(HeuristicInputError, match=message):
        HeuristicValuator(NO_LIQUIDITY).evaluate_bid(context, make_knowledge())


def test_context_contradictions_from_belief_remain_heuristic_input_errors() -> None:
    context = make_context()
    knowledge = replace(make_knowledge(), player_count=4)

    with pytest.raises(HeuristicInputError, match="player count"):
        HeuristicValuator(NO_LIQUIDITY).evaluate_bid(context, knowledge)
