from __future__ import annotations

from dataclasses import dataclass

from pocketrocks import BotDecision, DecisionContext

from garboid_pocketrocks.adapters.public_history import PublicHistory
from garboid_pocketrocks.bots.base import BotSpec, PocketRocksFastBot
from garboid_pocketrocks.diagnostics.trace import (
    ExplainedBotDecision,
    HeuristicBidExplanation,
)
from garboid_pocketrocks.heuristics.errors import HeuristicInputError
from garboid_pocketrocks.heuristics.profiles import (
    HEURISTIC_V1,
    HEURISTIC_V2,
    HeuristicProfile,
)
from garboid_pocketrocks.heuristics.valuation import BidEvaluation, HeuristicValuator
from garboid_pocketrocks.knowledge import RulesetKnowledge


@dataclass(frozen=True, slots=True)
class _HeuristicChoice:
    decision: BotDecision
    bid_evaluation: BidEvaluation | None


class HeuristicBotBrain:
    """Synchronous adapter from a heuristic profile to SDK decisions."""

    def __init__(self, profile: HeuristicProfile) -> None:
        self.valuator = HeuristicValuator(profile)

    def choose_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
    ) -> BotDecision:
        return self._choose_raw(context, ruleset).decision

    def choose_explained_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
        history: PublicHistory,
    ) -> ExplainedBotDecision:
        """Choose once and retain the valuation that produced a heuristic bid."""

        del history
        choice = self._choose_raw(context, ruleset)
        evaluation = choice.bid_evaluation
        if evaluation is None:
            return ExplainedBotDecision(decision=choice.decision)
        bid = evaluation.chosen_bid
        point = evaluation.points[bid]
        return ExplainedBotDecision(
            decision=choice.decision,
            explanation=HeuristicBidExplanation(
                resource_value=point.breakdown.resource,
                objective_completion_value=point.breakdown.objective_completion,
                objective_progress_value=point.breakdown.objective_progress,
                terminal_cash_value=point.breakdown.terminal_cash,
                liquidity_value=point.breakdown.liquidity,
                future_cash_value=point.breakdown.future_cash,
                total_value=point.breakdown.total,
                reservation_bid=evaluation.reservation_bid,
                chosen_bid=bid,
            ),
        )

    def _choose_raw(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
    ) -> _HeuristicChoice:
        """Choose once without constructing diagnostic records."""

        try:
            if context.decision_kind == "selectInfoToReveal":
                return _HeuristicChoice(
                    BotDecision.select_info_to_reveal(
                        self.valuator.choose_reveal(context, ruleset),
                    ),
                    None,
                )
            evaluation = self.valuator.evaluate_bid(context, ruleset)
            bid = evaluation.chosen_bid
            return _HeuristicChoice(
                BotDecision.pass_turn() if bid == 0 else BotDecision.submit_bid(bid),
                evaluation,
            )
        except HeuristicInputError:
            return _HeuristicChoice(BotDecision.pass_turn(), None)


class AggressiveHeuristicV1Brain(HeuristicBotBrain):
    def __init__(self, seed: int | None = None) -> None:
        del seed
        super().__init__(HEURISTIC_V1.aggressive)


class BalancedHeuristicV1Brain(HeuristicBotBrain):
    def __init__(self, seed: int | None = None) -> None:
        del seed
        super().__init__(HEURISTIC_V1.balanced)


class PassiveHeuristicV1Brain(HeuristicBotBrain):
    def __init__(self, seed: int | None = None) -> None:
        del seed
        super().__init__(HEURISTIC_V1.passive)


class AggressiveHeuristicV2Brain(HeuristicBotBrain):
    def __init__(self, seed: int | None = None) -> None:
        del seed
        super().__init__(HEURISTIC_V2.aggressive)


class BalancedHeuristicV2Brain(HeuristicBotBrain):
    def __init__(self, seed: int | None = None) -> None:
        del seed
        super().__init__(HEURISTIC_V2.balanced)


class PassiveHeuristicV2Brain(HeuristicBotBrain):
    def __init__(self, seed: int | None = None) -> None:
        del seed
        super().__init__(HEURISTIC_V2.passive)


class AggressiveHeuristicBrain(AggressiveHeuristicV2Brain):
    """Latest aggressive heuristic brain."""


class BalancedHeuristicBrain(BalancedHeuristicV2Brain):
    """Latest balanced heuristic brain."""


class PassiveHeuristicBrain(PassiveHeuristicV2Brain):
    """Latest passive heuristic brain."""


class AggressiveHeuristicBot(PocketRocksFastBot):
    """Live wrapper for the aggressive heuristic."""

    BOT_ID = "bot_386b81bb-14df-477a-8d4c-0231cf1b3b1a"
    BOT_NAME = "aggressive"

    @classmethod
    def build_brain(cls, seed: int | None) -> AggressiveHeuristicBrain:
        del seed
        return AggressiveHeuristicBrain()


class BalancedHeuristicBot(PocketRocksFastBot):
    """Live wrapper for the balanced heuristic."""

    BOT_ID = "bot_265c84aa-f28e-4a35-b4de-a4f4ee406415"
    BOT_NAME = "balanced"

    @classmethod
    def build_brain(cls, seed: int | None) -> BalancedHeuristicBrain:
        del seed
        return BalancedHeuristicBrain()


class PassiveHeuristicBot(PocketRocksFastBot):
    """Live wrapper for the passive heuristic."""

    BOT_ID = "bot_9d33c9de-4d90-4608-9a58-d2c77d93e0bd"
    BOT_NAME = "passive"

    @classmethod
    def build_brain(cls, seed: int | None) -> PassiveHeuristicBrain:
        del seed
        return PassiveHeuristicBrain()


AGGRESSIVE_HEURISTIC_BOT_SPEC = BotSpec.from_bot_class(AggressiveHeuristicBot)
BALANCED_HEURISTIC_BOT_SPEC = BotSpec.from_bot_class(BalancedHeuristicBot)
PASSIVE_HEURISTIC_BOT_SPEC = BotSpec.from_bot_class(PassiveHeuristicBot)

AGGRESSIVE_HEURISTIC_V1_BOT_SPEC = BotSpec.for_simulation(
    "aggressive-v1",
    AggressiveHeuristicV1Brain,
)
BALANCED_HEURISTIC_V1_BOT_SPEC = BotSpec.for_simulation(
    "balanced-v1",
    BalancedHeuristicV1Brain,
)
PASSIVE_HEURISTIC_V1_BOT_SPEC = BotSpec.for_simulation(
    "passive-v1",
    PassiveHeuristicV1Brain,
)

AGGRESSIVE_HEURISTIC_V2_BOT_SPEC = BotSpec.for_simulation(
    "aggressive-v2",
    AggressiveHeuristicV2Brain,
)
BALANCED_HEURISTIC_V2_BOT_SPEC = BotSpec.for_simulation(
    "balanced-v2",
    BalancedHeuristicV2Brain,
)
PASSIVE_HEURISTIC_V2_BOT_SPEC = BotSpec.for_simulation(
    "passive-v2",
    PassiveHeuristicV2Brain,
)
