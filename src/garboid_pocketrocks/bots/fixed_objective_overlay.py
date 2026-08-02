from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

from pocketrocks import ActionId, BotDecision, DecisionContext

from garboid_pocketrocks.adapters.public_history import PublicHistory
from garboid_pocketrocks.bots.base import BotSpec
from garboid_pocketrocks.bots.fixed_bid import (
    FIXED_BID_PROFILE,
    FIXED_BID_TUNED_V1_PROFILE,
    FixedBidProfile,
    ProfiledFixedBidBotBrain,
)
from garboid_pocketrocks.diagnostics.trace import (
    ExplainedBotDecision,
    FixedObjectiveOverlayV3BidExplanation,
    FixedObjectiveOverlayV3Rule,
)
from garboid_pocketrocks.heuristics.belief import (
    build_belief,
    offered_resource_counts,
    terminal_price_pmf,
)
from garboid_pocketrocks.heuristics.errors import HeuristicInputError
from garboid_pocketrocks.heuristics.objectives import evaluate_objectives
from garboid_pocketrocks.knowledge import RulesetKnowledge

_AUCTIONS = (ActionId.AUCTION1, ActionId.AUCTION2)
_SUIT_COUNT = 5


@dataclass(frozen=True, slots=True)
class FixedObjectiveOverlayConfig:
    """Frozen coefficients for one bounded fixed-bid overlay generation."""

    objective_progress_weight: float
    private_edge_weight: float
    positive_threshold: float
    strong_positive_threshold: float
    negative_threshold: float
    strong_negative_threshold: float
    max_adjustment: int

    def __post_init__(self) -> None:
        coefficients = (
            self.objective_progress_weight,
            self.private_edge_weight,
            self.positive_threshold,
            self.strong_positive_threshold,
            self.negative_threshold,
            self.strong_negative_threshold,
        )
        if not all(math.isfinite(value) for value in coefficients):
            raise ValueError("fixed-objective overlay coefficients must be finite")
        if not 0 <= self.objective_progress_weight <= 1:
            raise ValueError("objective progress weight must be between zero and one")
        if self.private_edge_weight < 0:
            raise ValueError("private edge weight must be nonnegative")
        if not 0 < self.positive_threshold < self.strong_positive_threshold:
            raise ValueError("positive thresholds must be ordered above zero")
        if not self.strong_negative_threshold < self.negative_threshold < 0:
            raise ValueError("negative thresholds must be ordered below zero")
        if self.max_adjustment < 1:
            raise ValueError("maximum adjustment must be positive")


FIXED_OBJECTIVE_OVERLAY_V1_CONFIG = FixedObjectiveOverlayConfig(
    objective_progress_weight=0.20,
    private_edge_weight=1.75,
    positive_threshold=1.0,
    strong_positive_threshold=5.0,
    negative_threshold=-0.75,
    strong_negative_threshold=-4.0,
    max_adjustment=2,
)
FIXED_OBJECTIVE_OVERLAY_V2_CONFIG = FIXED_OBJECTIVE_OVERLAY_V1_CONFIG
FIXED_OBJECTIVE_OVERLAY_V3_PROFILE = FixedBidProfile((5, 10, 3, 6, 4, 7))


def _count_resources(resource_ids: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(
        sum(resource_id == suit_id for resource_id in resource_ids)
        for suit_id in range(1, _SUIT_COUNT + 1)
    )


def _public_only_expected_prices(
    context: DecisionContext,
    ruleset: RulesetKnowledge,
) -> tuple[float, ...]:
    """Return price beliefs before observing the acting bot's hidden suits."""

    revealed = tuple(
        sum(row[suit] for row in context.revealed_info_counts_by_seat)
        for suit in range(_SUIT_COUNT)
    )
    won = tuple(
        sum(row[suit] for row in context.won_resource_counts_by_seat) for suit in range(_SUIT_COUNT)
    )
    visible = _count_resources(context.current_resource_ids)
    unseen = tuple(
        total - public - owned - on_table
        for total, public, owned, on_table in zip(
            ruleset.resource_counts,
            revealed,
            won,
            visible,
            strict=True,
        )
    )
    if any(count < 0 for count in unseen):
        raise HeuristicInputError("public card counts exceed ruleset resources")

    population = sum(unseen)
    hidden_slots = (context.player_count * ruleset.private_cards_per_player) - sum(revealed)
    if hidden_slots < 0 or hidden_slots > population:
        raise HeuristicInputError("public hidden slots exceed unseen resources")

    prices: list[float] = []
    for known, unseen_suit_count in zip(revealed, unseen, strict=True):
        pmf = terminal_price_pmf(
            known_reveals=known,
            unseen_population=population,
            unseen_suit_count=unseen_suit_count,
            opponent_hidden_slots=hidden_slots,
        )
        prices.append(
            sum(
                probability * price
                for probability, price in zip(pmf, context.value_chart, strict=True)
            )
        )
    if not all(math.isfinite(price) for price in prices):
        raise HeuristicInputError("public expected prices must be finite")
    return tuple(prices)


class FixedObjectiveOverlayBrain(ProfiledFixedBidBotBrain):
    """Shared fixed-bid engine with a bounded objective and information nudge."""

    CONFIG: ClassVar[FixedObjectiveOverlayConfig]

    def choose_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
        history: PublicHistory = (),
    ) -> BotDecision:
        if context.current_action_id is None:
            return super().choose_decision(context, ruleset, history)
        try:
            action = ActionId(context.current_action_id)
        except TypeError, ValueError:
            return super().choose_decision(context, ruleset, history)
        if context.decision_kind != "submitBid" or action not in _AUCTIONS:
            return super().choose_decision(context, ruleset, history)

        legal_max = context.legal_max_amount
        target = self.PROFILE.target_bid(context.current_action_id)
        if legal_max is None or legal_max <= 0 or target is None:
            return super().choose_decision(context, ruleset, history)
        try:
            adjustment = self._auction_adjustment(context, ruleset, action)
        except (
            ArithmeticError,
            HeuristicInputError,
            IndexError,
            KeyError,
            TypeError,
            ValueError,
        ):
            adjustment = 0
        bid = min(max(target + adjustment, 1), legal_max)
        return BotDecision.submit_bid(bid)

    def _auction_adjustment(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
        action: ActionId,
    ) -> int:
        belief = build_belief(context, ruleset)
        offered = offered_resource_counts(context, action)
        offered_count = sum(offered)
        private_prices = tuple(suit.expected_terminal_price for suit in belief.suits)
        offered_value = sum(
            count * price for count, price in zip(offered, private_prices, strict=True)
        )
        average_bundle_value = offered_count * sum(private_prices) / len(private_prices)
        relative_resource_edge = offered_value - average_bundle_value

        public_prices = _public_only_expected_prices(context, ruleset)
        public_value = sum(
            count * price for count, price in zip(offered, public_prices, strict=True)
        )
        private_edge = offered_value - public_value

        objective_values = evaluate_objectives(
            active_objective_ids=context.objective_ids,
            owned_objective_ids_by_seat=context.owned_objective_ids_by_seat,
            won_resource_counts_by_seat=context.won_resource_counts_by_seat,
            bot_seat=context.bot_seat,
            offered_counts=offered,
            horizon=belief.normalized_horizon,
            progress_weight=self.CONFIG.objective_progress_weight,
        )
        objective_edge = sum(value.total for value in objective_values)
        signal = (
            objective_edge
            + relative_resource_edge
            + (self.CONFIG.private_edge_weight * private_edge)
        )

        if signal >= self.CONFIG.strong_positive_threshold:
            return self.CONFIG.max_adjustment
        if signal >= self.CONFIG.positive_threshold:
            return 1
        if signal <= self.CONFIG.strong_negative_threshold:
            return -self.CONFIG.max_adjustment
        if signal <= self.CONFIG.negative_threshold:
            return -1
        return 0


class FixedObjectiveOverlayV1Brain(FixedObjectiveOverlayBrain):
    """Original fixed-base overlay generation."""

    PROFILE = FIXED_BID_PROFILE
    CONFIG = FIXED_OBJECTIVE_OVERLAY_V1_CONFIG


class FixedObjectiveOverlayV2Brain(FixedObjectiveOverlayBrain):
    """Tuned-fixed-base overlay generation."""

    PROFILE = FIXED_BID_TUNED_V1_PROFILE
    CONFIG = FIXED_OBJECTIVE_OVERLAY_V2_CONFIG


@dataclass(frozen=True, slots=True)
class _FixedObjectiveOverlayV3Choice:
    decision: BotDecision
    rule: FixedObjectiveOverlayV3Rule | None
    planned_bid: int | None


class FixedObjectiveOverlayV3Brain(FixedObjectiveOverlayV2Brain):
    """Tuned overlay with an exact opponent-cash guarantee cap."""

    PROFILE = FIXED_OBJECTIVE_OVERLAY_V3_PROFILE

    def choose_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
        history: PublicHistory = (),
    ) -> BotDecision:
        return self._choose(context, ruleset, history).decision

    def choose_explained_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
        history: PublicHistory,
    ) -> ExplainedBotDecision:
        """Choose once and expose the public rule classification for diagnostics."""

        choice = self._choose(context, ruleset, history)
        if choice.rule is None or choice.planned_bid is None:
            return ExplainedBotDecision(decision=choice.decision)
        chosen_bid = choice.decision.value if choice.decision.action_kind == "submitBid" else 0
        if chosen_bid is None:
            raise AssertionError("a v3 bid decision always has an effective amount")
        return ExplainedBotDecision(
            decision=choice.decision,
            explanation=FixedObjectiveOverlayV3BidExplanation(
                rule=choice.rule,
                planned_bid=choice.planned_bid,
                chosen_bid=chosen_bid,
            ),
        )

    def _choose(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
        history: PublicHistory,
    ) -> _FixedObjectiveOverlayV3Choice:
        baseline = super().choose_decision(context, ruleset, history)
        planned_bid = baseline.value if baseline.action_kind == "submitBid" else 0
        if (
            context.decision_kind != "submitBid"
            or context.current_action_id is None
            or context.legal_max_amount is None
            or planned_bid is None
        ):
            return _FixedObjectiveOverlayV3Choice(baseline, None, None)
        if planned_bid <= 0:
            return _FixedObjectiveOverlayV3Choice(baseline, "not_applicable", planned_bid)
        try:
            action = ActionId(context.current_action_id)
        except TypeError, ValueError:
            return _FixedObjectiveOverlayV3Choice(baseline, "not_applicable", planned_bid)
        if action not in _AUCTIONS:
            return _FixedObjectiveOverlayV3Choice(baseline, "not_applicable", planned_bid)

        guaranteed_bid = _guaranteed_bid_cap(context, planned_bid)
        if guaranteed_bid is None:
            return _FixedObjectiveOverlayV3Choice(baseline, "baseline", planned_bid)
        return _FixedObjectiveOverlayV3Choice(
            _bid_decision(guaranteed_bid),
            "guaranteed_win",
            planned_bid,
        )


def _bid_decision(bid: int) -> BotDecision:
    return BotDecision.pass_turn() if bid == 0 else BotDecision.submit_bid(bid)


def _guaranteed_bid_cap(context: DecisionContext, planned_bid: int) -> int | None:
    opponent_cash = tuple(
        (seat, cash) for seat, cash in enumerate(context.cash_by_seat) if seat != context.bot_seat
    )
    if not opponent_cash:
        return None
    maximum_cash = max(cash for _, cash in opponent_cash)
    if maximum_cash >= planned_bid:
        return None

    first_tiebreak_seat = (context.tiebreak_seat + 1) % context.player_count

    def tiebreak_order(seat: int) -> int:
        return (seat - first_tiebreak_seat) % context.player_count

    own_order = tiebreak_order(context.bot_seat)
    wins_maximum_tie = all(
        own_order < tiebreak_order(seat) for seat, cash in opponent_cash if cash == maximum_cash
    )
    return maximum_cash if wins_maximum_tie else maximum_cash + 1


FIXED_OBJECTIVE_OVERLAY_V1_BOT_SPEC = BotSpec.for_simulation(
    "fixed-objective-overlay-v1",
    FixedObjectiveOverlayV1Brain,
)
FIXED_OBJECTIVE_OVERLAY_V2_BOT_SPEC = BotSpec.for_simulation(
    "fixed-objective-overlay-v2",
    FixedObjectiveOverlayV2Brain,
)
FIXED_OBJECTIVE_OVERLAY_V3_BOT_SPEC = BotSpec.for_simulation(
    "fixed-objective-overlay-v3",
    FixedObjectiveOverlayV3Brain,
)
