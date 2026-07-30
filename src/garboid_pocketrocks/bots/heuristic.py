from __future__ import annotations

from pocketrocks import BotDecision, DecisionContext

from garboid_pocketrocks.bots.base import BotSpec, PocketRocksFastBot
from garboid_pocketrocks.heuristics.errors import HeuristicInputError
from garboid_pocketrocks.heuristics.profiles import (
    HEURISTIC_V1,
    HEURISTIC_V2,
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
            return BotDecision.pass_turn() if bid == 0 else BotDecision.submit_bid(bid)
        except HeuristicInputError:
            return BotDecision.pass_turn()


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
    """Development-only live wrapper for the latest aggressive heuristic."""

    BOT_ID = "bot_00000000-0000-4000-8000-00000000000a"
    BOT_NAME = "aggressive"

    @classmethod
    def build_brain(cls, seed: int | None) -> AggressiveHeuristicBrain:
        del seed
        return AggressiveHeuristicBrain()


class BalancedHeuristicBot(PocketRocksFastBot):
    """Development-only live wrapper for the latest balanced heuristic."""

    BOT_ID = "bot_00000000-0000-4000-8000-00000000000b"
    BOT_NAME = "balanced"

    @classmethod
    def build_brain(cls, seed: int | None) -> BalancedHeuristicBrain:
        del seed
        return BalancedHeuristicBrain()


class PassiveHeuristicBot(PocketRocksFastBot):
    """Development-only live wrapper for the latest passive heuristic."""

    BOT_ID = "bot_00000000-0000-4000-8000-00000000000c"
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
