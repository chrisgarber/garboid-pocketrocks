from __future__ import annotations

import math
from dataclasses import dataclass

from pocketrocks import ActionId, BotDecision, DecisionContext

from garboid_pocketrocks.heuristics.belief import build_belief, offered_resource_counts
from garboid_pocketrocks.heuristics.objectives import evaluate_objectives
from garboid_pocketrocks.knowledge import RulesetKnowledge

BASE_TARGETS = {
    ActionId.AUCTION1: 5,
    ActionId.AUCTION2: 10,
    ActionId.LOAN10: 2,
    ActionId.LOAN20: 4,
    ActionId.INVEST5: 4,
    ActionId.INVEST10: 9,
}


@dataclass(frozen=True, slots=True)
class OverlayConfig:
    name: str
    progress_weight: float
    private_edge_weight: float
    positive_step: float
    strong_positive_step: float
    negative_step: float
    strong_negative_step: float
    max_adjustment: int


NARROW = OverlayConfig("narrow", 0.20, 1.25, 2.0, 7.0, -1.0, -5.0, 1)
STANDARD = OverlayConfig("standard", 0.25, 1.35, 1.5, 6.0, -1.0, -5.0, 2)
PRIVATE_HEAVY = OverlayConfig("private-heavy", 0.20, 1.75, 1.0, 5.0, -0.75, -4.0, 2)


def _target(action_id: int | None) -> tuple[ActionId | None, int | None]:
    if action_id is None:
        return None, None
    try:
        action = ActionId(action_id)
    except ValueError:
        return None, None
    return action, BASE_TARGETS.get(action)


def _count_resources(resource_ids: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(
        sum(resource_id == suit_id for resource_id in resource_ids) for suit_id in range(1, 6)
    )


def _hypergeom_probability(
    population: int,
    successes: int,
    draws: int,
    selected: int,
) -> float:
    if (
        selected < 0
        or selected > successes
        or selected > draws
        or draws - selected > population - successes
    ):
        return 0.0
    return (
        math.comb(successes, selected)
        * math.comb(population - successes, draws - selected)
        / math.comb(population, draws)
    )


def _public_only_expected_prices(
    context: DecisionContext,
    ruleset: RulesetKnowledge,
) -> tuple[float, ...]:
    """Price expectations before using the acting bot's hidden hand identities."""
    revealed = tuple(
        sum(row[suit] for row in context.revealed_info_counts_by_seat) for suit in range(5)
    )
    won = tuple(sum(row[suit] for row in context.won_resource_counts_by_seat) for suit in range(5))
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
    population = sum(unseen)
    hidden_slots = (context.player_count * ruleset.private_cards_per_player) - sum(revealed)
    prices: list[float] = []
    for known, successes in zip(revealed, unseen, strict=True):
        pmf = [0.0] * len(context.value_chart)
        for selected in range(hidden_slots + 1):
            probability = _hypergeom_probability(population, successes, hidden_slots, selected)
            pmf[min(known + selected, len(pmf) - 1)] += probability
        prices.append(
            sum(
                probability * price
                for probability, price in zip(pmf, context.value_chart, strict=True)
            )
        )
    return tuple(prices)


class FixedObjectiveOverlayBrain:
    """A fixed-bid policy with a bounded objective/information auction overlay."""

    config = STANDARD

    def __init__(self, seed: int | None = None) -> None:
        del seed

    def choose_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
    ) -> BotDecision:
        if context.decision_kind == "selectInfoToReveal":
            if context.revealable_count <= 0:
                return BotDecision.pass_turn()
            return BotDecision.select_info_to_reveal(0)

        legal_max = context.legal_max_amount
        action, target = _target(context.current_action_id)
        if legal_max is None or legal_max <= 0 or target is None or action is None:
            return BotDecision.pass_turn()

        adjustment = 0
        if action in (ActionId.AUCTION1, ActionId.AUCTION2):
            try:
                adjustment = self._auction_adjustment(context, ruleset, action)
            except IndexError, KeyError, OverflowError, TypeError, ValueError:
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
            progress_weight=self.config.progress_weight,
        )
        objective_edge = sum(value.total for value in objective_values)
        signal = (
            objective_edge
            + relative_resource_edge
            + (self.config.private_edge_weight * private_edge)
        )

        if signal >= self.config.strong_positive_step:
            return self.config.max_adjustment
        if signal >= self.config.positive_step:
            return 1
        if signal <= self.config.strong_negative_step:
            return -self.config.max_adjustment
        if signal <= self.config.negative_step:
            return -1
        return 0


class FixedObjectiveOverlayNarrowV1Brain(FixedObjectiveOverlayBrain):
    config = NARROW


class FixedObjectiveOverlayStandardV1Brain(FixedObjectiveOverlayBrain):
    config = STANDARD


class FixedObjectiveOverlayPrivateHeavyV1Brain(FixedObjectiveOverlayBrain):
    config = PRIVATE_HEAVY
