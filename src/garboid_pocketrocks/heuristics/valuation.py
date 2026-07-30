from __future__ import annotations

import math
from dataclasses import dataclass

from pocketrocks import ActionId, DecisionContext

from garboid_pocketrocks.heuristics.belief import (
    BeliefState,
    build_belief,
    offered_resource_counts,
)
from garboid_pocketrocks.heuristics.cash import evaluate_action_curve
from garboid_pocketrocks.heuristics.errors import HeuristicInputError
from garboid_pocketrocks.heuristics.objectives import evaluate_objectives
from garboid_pocketrocks.heuristics.profiles import HeuristicProfile
from garboid_pocketrocks.heuristics.reveals import choose_reveal
from garboid_pocketrocks.knowledge import RulesetKnowledge

_AUCTIONS = (ActionId.AUCTION1, ActionId.AUCTION2)


@dataclass(frozen=True, slots=True)
class ValueBreakdown:
    """Auditable dollar-equivalent components for winning at one bid."""

    resource: float
    objective_completion: float
    objective_progress: float
    terminal_cash: float
    liquidity: float
    future_cash: float
    total: float


@dataclass(frozen=True, slots=True)
class BidPoint:
    """The value of winning the current action at one legal integer bid."""

    bid: int
    win_delta: float
    breakdown: ValueBreakdown


@dataclass(frozen=True, slots=True)
class BidEvaluation:
    """A complete action curve and its reservation and shaded bids."""

    belief: BeliefState
    points: tuple[BidPoint, ...]
    reservation_bid: int
    chosen_bid: int


class HeuristicValuator:
    """Compose public information into deterministic action valuations."""

    def __init__(self, profile: HeuristicProfile) -> None:
        self.profile = profile

    def evaluate_bid(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
    ) -> BidEvaluation:
        """Evaluate every legal bid and choose a profile-shaded submission."""
        if context.decision_kind != "submitBid":
            raise HeuristicInputError("evaluate_bid requires a bid decision context")
        if context.current_action_id is None:
            raise HeuristicInputError("bid decision requires a current action")
        if context.legal_max_amount is None:
            raise HeuristicInputError("bid decision requires a legal maximum")

        belief = build_belief(context, ruleset)
        try:
            action = ActionId(context.current_action_id)
            offered_counts = offered_resource_counts(context, action)
            resource_value = self._resource_value(
                belief=belief,
                offered_counts=offered_counts,
                action=action,
            )
            objective_completion, objective_progress = self._objective_value(
                context=context,
                belief=belief,
                offered_counts=offered_counts,
                action=action,
            )
            gross_value = resource_value + objective_completion + objective_progress
            if not math.isfinite(gross_value):
                raise ValueError("gross action value must be finite")
            economics = evaluate_action_curve(
                action_id=action,
                cash=context.cash_by_seat[context.bot_seat],
                legal_max=context.legal_max_amount,
                horizon=belief.normalized_horizon,
                starting_cash=context.starting_cash,
                liquidity_strength=self.profile.liquidity_strength,
                future_cash_weight=self.profile.future_cash_weight,
                gross_value=gross_value,
            )
        except HeuristicInputError:
            raise
        except (IndexError, KeyError, OverflowError, TypeError, ValueError) as error:
            raise HeuristicInputError(str(error)) from error

        points = tuple(
            self._bid_point(
                bid=point.bid,
                resource=resource_value,
                objective_completion=objective_completion,
                objective_progress=objective_progress,
                terminal_cash=point.terminal_cash,
                liquidity=point.liquidity,
                future_cash=point.future_cash,
            )
            for point in economics
        )
        reservation_bid = max(
            (point.bid for point in points if point.win_delta >= 0.0),
            default=0,
        )
        chosen_bid = math.floor(reservation_bid * (1.0 - self.profile.bid_shading))
        chosen_bid = min(max(chosen_bid, 0), context.legal_max_amount)
        if chosen_bid > reservation_bid:
            raise HeuristicInputError("chosen bid exceeds its reservation bid")
        return BidEvaluation(
            belief=belief,
            points=points,
            reservation_bid=reservation_bid,
            chosen_bid=chosen_bid,
        )

    def choose_reveal(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
    ) -> int:
        """Choose which private-card index to reveal."""
        return choose_reveal(context, ruleset)

    @staticmethod
    def _resource_value(
        *,
        belief: BeliefState,
        offered_counts: tuple[int, ...],
        action: ActionId,
    ) -> float:
        if action not in _AUCTIONS:
            return 0.0
        return sum(
            count * suit.expected_terminal_price
            for count, suit in zip(offered_counts, belief.suits, strict=True)
        )

    def _objective_value(
        self,
        *,
        context: DecisionContext,
        belief: BeliefState,
        offered_counts: tuple[int, ...],
        action: ActionId,
    ) -> tuple[float, float]:
        if action not in _AUCTIONS:
            return 0.0, 0.0
        objective_values = evaluate_objectives(
            active_objective_ids=context.objective_ids,
            owned_objective_ids_by_seat=context.owned_objective_ids_by_seat,
            won_resource_counts_by_seat=context.won_resource_counts_by_seat,
            bot_seat=context.bot_seat,
            offered_counts=offered_counts,
            horizon=belief.normalized_horizon,
            progress_weight=self.profile.objective_progress_weight,
        )
        return (
            sum(value.completion_value for value in objective_values),
            sum(value.progress_value for value in objective_values),
        )

    @staticmethod
    def _bid_point(
        *,
        bid: int,
        resource: float,
        objective_completion: float,
        objective_progress: float,
        terminal_cash: float,
        liquidity: float,
        future_cash: float,
    ) -> BidPoint:
        total = (
            resource
            + objective_completion
            + objective_progress
            + terminal_cash
            + liquidity
            + future_cash
        )
        if not all(
            math.isfinite(value)
            for value in (
                resource,
                objective_completion,
                objective_progress,
                terminal_cash,
                liquidity,
                future_cash,
                total,
            )
        ):
            raise HeuristicInputError("valuation breakdown must be finite")
        breakdown = ValueBreakdown(
            resource=resource,
            objective_completion=objective_completion,
            objective_progress=objective_progress,
            terminal_cash=terminal_cash,
            liquidity=liquidity,
            future_cash=future_cash,
            total=total,
        )
        return BidPoint(bid=bid, win_delta=total, breakdown=breakdown)
