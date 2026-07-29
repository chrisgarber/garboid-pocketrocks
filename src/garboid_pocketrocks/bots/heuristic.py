from __future__ import annotations

from pocketrocks import BotDecision, DecisionContext

from garboid_pocketrocks.bots.base import BotSpec, PocketRocksFastBot
from garboid_pocketrocks.heuristics.errors import HeuristicInputError
from garboid_pocketrocks.heuristics.profiles import (
    AGGRESSIVE_PROFILE,
    BALANCED_PROFILE,
    PASSIVE_PROFILE,
    HeuristicProfile,
)
from garboid_pocketrocks.heuristics.valuation import HeuristicValuator
from garboid_pocketrocks.rules import RulesetKnowledge


class HeuristicBotBrain:
    """Synchronous adapter from a heuristic profile to SDK decisions."""

    def __init__(self, profile: HeuristicProfile) -> None:
        self.valuator = HeuristicValuator(profile)

    def choose_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
    ) -> BotDecision:
        try:
            if context.decision_kind == "selectInfoToReveal":
                return BotDecision.select_info_to_reveal(
                    self.valuator.choose_reveal(context, ruleset)
                )
            bid = self.valuator.evaluate_bid(context, ruleset).chosen_bid
            return (
                BotDecision.pass_turn()
                if bid == 0
                else BotDecision.submit_bid(bid)
            )
        except HeuristicInputError:
            return BotDecision.pass_turn()


class AggressiveHeuristicBrain(HeuristicBotBrain):
    def __init__(self) -> None:
        super().__init__(AGGRESSIVE_PROFILE)


class BalancedHeuristicBrain(HeuristicBotBrain):
    def __init__(self) -> None:
        super().__init__(BALANCED_PROFILE)


class PassiveHeuristicBrain(HeuristicBotBrain):
    def __init__(self) -> None:
        super().__init__(PASSIVE_PROFILE)


class AggressiveHeuristicBot(PocketRocksFastBot):
    """Development-only live wrapper for the aggressive heuristic."""

    BOT_ID = "bot_00000000-0000-4000-8000-00000000000a"
    BOT_NAME = "aggressive"

    @classmethod
    def build_brain(cls, seed: int | None) -> AggressiveHeuristicBrain:
        del seed
        return AggressiveHeuristicBrain()


class BalancedHeuristicBot(PocketRocksFastBot):
    """Development-only live wrapper for the balanced heuristic."""

    BOT_ID = "bot_00000000-0000-4000-8000-00000000000b"
    BOT_NAME = "balanced"

    @classmethod
    def build_brain(cls, seed: int | None) -> BalancedHeuristicBrain:
        del seed
        return BalancedHeuristicBrain()


class PassiveHeuristicBot(PocketRocksFastBot):
    """Development-only live wrapper for the passive heuristic."""

    BOT_ID = "bot_00000000-0000-4000-8000-00000000000c"
    BOT_NAME = "passive"

    @classmethod
    def build_brain(cls, seed: int | None) -> PassiveHeuristicBrain:
        del seed
        return PassiveHeuristicBrain()


AGGRESSIVE_HEURISTIC_BOT_SPEC = BotSpec.from_bot_class(AggressiveHeuristicBot)
BALANCED_HEURISTIC_BOT_SPEC = BotSpec.from_bot_class(BalancedHeuristicBot)
PASSIVE_HEURISTIC_BOT_SPEC = BotSpec.from_bot_class(PassiveHeuristicBot)
