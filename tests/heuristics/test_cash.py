from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pocketrocks import ActionId

from garboid_pocketrocks.heuristics.cash import (
    cash_option_value,
    evaluate_action_curve,
)


def test_option_value_is_zero_at_zero_horizon() -> None:
    assert cash_option_value(30, horizon=0.0, starting_cash=30, strength=0.75) == 0.0


def test_option_value_is_increasing_and_concave() -> None:
    values = [
        cash_option_value(cash, horizon=1.0, starting_cash=30, strength=0.75)
        for cash in (0, 10, 20, 30)
    ]
    assert values == sorted(values)
    assert values[1] - values[0] > values[2] - values[1] > values[3] - values[2]


def test_finite_but_overflowing_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="cash option value"):
        cash_option_value(30, horizon=1.0, starting_cash=30, strength=1e308)
    with pytest.raises(ValueError, match="action economics"):
        evaluate_action_curve(
            action_id=ActionId.LOAN10,
            cash=30,
            legal_max=0,
            horizon=1.0,
            starting_cash=30,
            liquidity_strength=8e306,
            gross_value=1.6e308,
        )


def test_zero_horizon_action_accounting_is_exact() -> None:
    loan = evaluate_action_curve(
        action_id=ActionId.LOAN10,
        cash=20,
        legal_max=30,
        horizon=0.0,
        starting_cash=30,
        liquidity_strength=0.75,
        gross_value=0.0,
    )
    investment = evaluate_action_curve(
        action_id=ActionId.INVEST5,
        cash=20,
        legal_max=20,
        horizon=0.0,
        starting_cash=30,
        liquidity_strength=0.75,
        gross_value=0.0,
    )
    assert [point.win_delta for point in loan[:3]] == [0.0, -1.0, -2.0]
    assert all(math.isclose(point.win_delta, 5.0) for point in investment)


@pytest.mark.parametrize(
    ("action_id", "legal_max"),
    (
        (ActionId.AUCTION1, 20),
        (ActionId.LOAN10, 30),
        (ActionId.INVEST5, 20),
    ),
)
def test_action_curve_is_nonincreasing_in_bid(
    action_id: ActionId,
    legal_max: int,
) -> None:
    curve = evaluate_action_curve(
        action_id=action_id,
        cash=20,
        legal_max=legal_max,
        horizon=0.7,
        starting_cash=30,
        liquidity_strength=0.75,
        gross_value=4.0,
    )
    assert [point.bid for point in curve] == list(range(legal_max + 1))
    assert all(
        left.win_delta >= right.win_delta
        for left, right in zip(curve, curve[1:], strict=False)
    )


def test_loan_benefit_increases_with_horizon() -> None:
    early = evaluate_action_curve(
        action_id=ActionId.LOAN10,
        cash=20,
        legal_max=0,
        horizon=0.2,
        starting_cash=30,
        liquidity_strength=0.75,
        gross_value=0.0,
    )[0]
    late = evaluate_action_curve(
        action_id=ActionId.LOAN10,
        cash=20,
        legal_max=0,
        horizon=0.8,
        starting_cash=30,
        liquidity_strength=0.75,
        gross_value=0.0,
    )[0]
    assert late.liquidity > early.liquidity


def test_investment_lock_cost_increases_with_horizon() -> None:
    early = evaluate_action_curve(
        action_id=ActionId.INVEST5,
        cash=20,
        legal_max=10,
        horizon=0.2,
        starting_cash=30,
        liquidity_strength=0.75,
        gross_value=0.0,
    )[-1]
    late = evaluate_action_curve(
        action_id=ActionId.INVEST5,
        cash=20,
        legal_max=10,
        horizon=0.8,
        starting_cash=30,
        liquidity_strength=0.75,
        gross_value=0.0,
    )[-1]
    assert late.liquidity < early.liquidity


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"cash": -1}, "cash"),
        ({"horizon": float("nan")}, "horizon"),
        ({"starting_cash": 0}, "starting_cash"),
        ({"strength": float("inf")}, "strength"),
    ),
)
def test_cash_option_value_rejects_invalid_numeric_inputs(
    kwargs: dict[str, float | int],
    message: str,
) -> None:
    arguments: dict[str, float | int] = {
        "cash": 20,
        "horizon": 0.5,
        "starting_cash": 30,
        "strength": 0.75,
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        cash_option_value(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"cash": -1}, "cash"),
        ({"legal_max": -1}, "legal_max"),
        ({"horizon": 1.1}, "horizon"),
        ({"starting_cash": 0}, "starting_cash"),
        ({"liquidity_strength": float("nan")}, "liquidity_strength"),
        ({"gross_value": float("inf")}, "gross_value"),
    ),
)
def test_action_curve_rejects_invalid_numeric_inputs(
    kwargs: dict[str, float | int],
    message: str,
) -> None:
    arguments: dict[str, ActionId | float | int] = {
        "action_id": ActionId.AUCTION1,
        "cash": 20,
        "legal_max": 20,
        "horizon": 0.5,
        "starting_cash": 30,
        "liquidity_strength": 0.75,
        "gross_value": 4.0,
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        evaluate_action_curve(**arguments)  # type: ignore[arg-type]


@given(
    cash=st.integers(min_value=0, max_value=200),
    starting_cash=st.integers(min_value=1, max_value=200),
    strength=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
    horizon=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
def test_option_value_is_finite_and_monotone(
    cash: int,
    starting_cash: int,
    strength: float,
    horizon: float,
) -> None:
    value = cash_option_value(cash, horizon=horizon, starting_cash=starting_cash, strength=strength)
    next_value = cash_option_value(
        cash + 1,
        horizon=horizon,
        starting_cash=starting_cash,
        strength=strength,
    )
    assert math.isfinite(value)
    assert value >= 0.0
    assert next_value >= value


@given(
    action_id=st.sampled_from(
        (
            ActionId.AUCTION1,
            ActionId.AUCTION2,
            ActionId.LOAN10,
            ActionId.LOAN20,
            ActionId.INVEST5,
            ActionId.INVEST10,
        )
    ),
    cash=st.integers(min_value=0, max_value=200),
    starting_cash=st.integers(min_value=1, max_value=200),
    legal_max=st.integers(min_value=0, max_value=200),
    strength=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
    horizon=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
def test_action_curve_is_finite_and_monotone_for_every_legal_bid(
    action_id: ActionId,
    cash: int,
    starting_cash: int,
    legal_max: int,
    strength: float,
    horizon: float,
) -> None:
    if action_id is ActionId.LOAN10:
        legal_max = min(legal_max, cash + 10)
    elif action_id is ActionId.LOAN20:
        legal_max = min(legal_max, cash + 20)
    else:
        legal_max = min(legal_max, cash)
    curve = evaluate_action_curve(
        action_id=action_id,
        cash=cash,
        legal_max=legal_max,
        horizon=horizon,
        starting_cash=starting_cash,
        liquidity_strength=strength,
        gross_value=3.0,
    )
    assert [point.bid for point in curve] == list(range(legal_max + 1))
    assert all(
        math.isfinite(component)
        for point in curve
        for component in (point.terminal_cash, point.liquidity, point.win_delta)
    )
    assert all(
        left.win_delta >= right.win_delta
        for left, right in zip(curve, curve[1:], strict=False)
    )


@given(
    cash=st.integers(min_value=0, max_value=200),
    starting_cash=st.integers(min_value=1, max_value=200),
    strength=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
    bid=st.integers(min_value=0, max_value=200),
)
def test_cash_effects_move_in_the_required_horizon_direction(
    cash: int,
    starting_cash: int,
    strength: float,
    bid: int,
) -> None:
    loan_bid = min(bid, 10)
    investment_bid = min(bid, cash)
    early_loan = evaluate_action_curve(
        action_id=ActionId.LOAN10,
        cash=cash,
        legal_max=loan_bid,
        horizon=0.2,
        starting_cash=starting_cash,
        liquidity_strength=strength,
        gross_value=0.0,
    )[-1]
    late_loan = evaluate_action_curve(
        action_id=ActionId.LOAN10,
        cash=cash,
        legal_max=loan_bid,
        horizon=0.8,
        starting_cash=starting_cash,
        liquidity_strength=strength,
        gross_value=0.0,
    )[-1]
    early_investment = evaluate_action_curve(
        action_id=ActionId.INVEST5,
        cash=cash,
        legal_max=investment_bid,
        horizon=0.2,
        starting_cash=starting_cash,
        liquidity_strength=strength,
        gross_value=0.0,
    )[-1]
    late_investment = evaluate_action_curve(
        action_id=ActionId.INVEST5,
        cash=cash,
        legal_max=investment_bid,
        horizon=0.8,
        starting_cash=starting_cash,
        liquidity_strength=strength,
        gross_value=0.0,
    )[-1]
    assert late_loan.liquidity >= early_loan.liquidity
    assert late_investment.liquidity <= early_investment.liquidity
