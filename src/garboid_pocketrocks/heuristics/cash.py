from __future__ import annotations

import math
from dataclasses import dataclass

from pocketrocks import ActionId


@dataclass(frozen=True, slots=True)
class ActionEconomics:
    bid: int
    terminal_cash: float
    liquidity: float
    win_delta: float


def _require_nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _require_positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_finite_number(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def cash_option_value(
    cash: int,
    *,
    horizon: float,
    starting_cash: int,
    strength: float,
) -> float:
    """Return the finite-horizon utility retained by immediately available cash."""
    cash = _require_nonnegative_integer(cash, "cash")
    horizon = _require_finite_number(horizon, "horizon")
    starting_cash = _require_positive_integer(starting_cash, "starting_cash")
    strength = _require_finite_number(strength, "strength")
    if not 0.0 <= horizon <= 1.0:
        raise ValueError("horizon must be between zero and one")
    if strength < 0.0:
        raise ValueError("strength must be nonnegative")
    try:
        kappa = max(starting_cash / 2.0, 1.0)
        value = strength * horizon * kappa * math.log1p(cash / kappa)
    except OverflowError as error:
        raise ValueError("cash option value must be finite") from error
    if not math.isfinite(value):
        raise ValueError("cash option value must be finite")
    return value


def _action_terms(action_id: ActionId, bid: int, cash: int) -> tuple[float, int]:
    if action_id in (ActionId.AUCTION1, ActionId.AUCTION2):
        return -float(bid), cash - bid
    if action_id is ActionId.LOAN10:
        return -float(bid), cash + 10 - bid
    if action_id is ActionId.LOAN20:
        return -float(bid), cash + 20 - bid
    if action_id is ActionId.INVEST5:
        return 5.0, cash - bid
    if action_id is ActionId.INVEST10:
        return 10.0, cash - bid
    raise ValueError("action_id must be a bidding action")


def evaluate_action_curve(
    *,
    action_id: ActionId,
    cash: int,
    legal_max: int,
    horizon: float,
    starting_cash: int,
    liquidity_strength: float,
    gross_value: float,
) -> tuple[ActionEconomics, ...]:
    """Evaluate every legal integer bid with component-wise cash accounting."""
    if not isinstance(action_id, ActionId):
        raise ValueError("action_id must be an ActionId")
    cash = _require_nonnegative_integer(cash, "cash")
    legal_max = _require_nonnegative_integer(legal_max, "legal_max")
    horizon = _require_finite_number(horizon, "horizon")
    starting_cash = _require_positive_integer(starting_cash, "starting_cash")
    liquidity_strength = _require_finite_number(liquidity_strength, "liquidity_strength")
    gross_value = _require_finite_number(gross_value, "gross_value")
    if not 0.0 <= horizon <= 1.0:
        raise ValueError("horizon must be between zero and one")
    if liquidity_strength < 0.0:
        raise ValueError("liquidity_strength must be nonnegative")

    _, final_cash = _action_terms(action_id, legal_max, cash)
    if final_cash < 0:
        raise ValueError("legal_max exceeds available cash for action")

    current_option = cash_option_value(
        cash,
        horizon=horizon,
        starting_cash=starting_cash,
        strength=liquidity_strength,
    )
    curve: list[ActionEconomics] = []
    for bid in range(legal_max + 1):
        try:
            terminal_cash, post_cash = _action_terms(action_id, bid, cash)
        except OverflowError as error:
            raise ValueError("action economics must be finite") from error
        liquidity = (
            cash_option_value(
                post_cash,
                horizon=horizon,
                starting_cash=starting_cash,
                strength=liquidity_strength,
            )
            - current_option
        )
        win_delta = gross_value + terminal_cash + liquidity
        if not all(math.isfinite(value) for value in (terminal_cash, liquidity, win_delta)):
            raise ValueError("action economics must be finite")
        curve.append(
            ActionEconomics(
                bid=bid,
                terminal_cash=terminal_cash,
                liquidity=liquidity,
                win_delta=win_delta,
            )
        )
    return tuple(curve)
