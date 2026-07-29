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
    def __init__(self) -> None:
        super().__init__(HEURISTIC_V1.aggressive)


class BalancedHeuristicV1Brain(HeuristicBotBrain):
    def __init__(self) -> None:
        super().__init__(HEURISTIC_V1.balanced)


class PassiveHeuristicV1Brain(HeuristicBotBrain):
    def __init__(self) -> None:
        super().__init__(HEURISTIC_V1.passive)


class AggressiveHeuristicV2Brain(HeuristicBotBrain):
    def __init__(self) -> None:
        super().__init__(HEURISTIC_V2.aggressive)


class BalancedHeuristicV2Brain(HeuristicBotBrain):
    def __init__(self) -> None:
        super().__init__(HEURISTIC_V2.balanced)


class PassiveHeuristicV2Brain(HeuristicBotBrain):
    def __init__(self) -> None:
        super().__init__(HEURISTIC_V2.passive)


class AggressiveHeuristicBrain(AggressiveHeuristicV2Brain):
    """Latest aggressive heuristic brain."""


class BalancedHeuristicBrain(BalancedHeuristicV2Brain):
    """Latest balanced heuristic brain."""


class PassiveHeuristicBrain(PassiveHeuristicV2Brain):
    """Latest passive heuristic brain."""


class _VersionedHeuristicBot(PocketRocksFastBot):
    """Shared type for explicit and latest heuristic wrappers."""


class AggressiveHeuristicV1Bot(_VersionedHeuristicBot):
    """Simulation wrapper for the frozen v1 aggressive heuristic."""

    BOT_ID = "bot_10000000-0000-4000-8000-000000000001"
    BOT_NAME = "aggressive-v1"

    @classmethod
    def build_brain(cls, seed: int | None) -> AggressiveHeuristicV1Brain:
        del seed
        return AggressiveHeuristicV1Brain()


class BalancedHeuristicV1Bot(_VersionedHeuristicBot):
    """Simulation wrapper for the frozen v1 balanced heuristic."""

    BOT_ID = "bot_10000000-0000-4000-8000-000000000002"
    BOT_NAME = "balanced-v1"

    @classmethod
    def build_brain(cls, seed: int | None) -> BalancedHeuristicV1Brain:
        del seed
        return BalancedHeuristicV1Brain()


class PassiveHeuristicV1Bot(_VersionedHeuristicBot):
    """Simulation wrapper for the frozen v1 passive heuristic."""

    BOT_ID = "bot_10000000-0000-4000-8000-000000000003"
    BOT_NAME = "passive-v1"

    @classmethod
    def build_brain(cls, seed: int | None) -> PassiveHeuristicV1Brain:
        del seed
        return PassiveHeuristicV1Brain()


class AggressiveHeuristicV2Bot(_VersionedHeuristicBot):
    """Simulation wrapper for the frozen v2 aggressive heuristic."""

    BOT_ID = "bot_20000000-0000-4000-8000-000000000001"
    BOT_NAME = "aggressive-v2"

    @classmethod
    def build_brain(cls, seed: int | None) -> AggressiveHeuristicV2Brain:
        del seed
        return AggressiveHeuristicV2Brain()


class BalancedHeuristicV2Bot(_VersionedHeuristicBot):
    """Simulation wrapper for the frozen v2 balanced heuristic."""

    BOT_ID = "bot_20000000-0000-4000-8000-000000000002"
    BOT_NAME = "balanced-v2"

    @classmethod
    def build_brain(cls, seed: int | None) -> BalancedHeuristicV2Brain:
        del seed
        return BalancedHeuristicV2Brain()


class PassiveHeuristicV2Bot(_VersionedHeuristicBot):
    """Simulation wrapper for the frozen v2 passive heuristic."""

    BOT_ID = "bot_20000000-0000-4000-8000-000000000003"
    BOT_NAME = "passive-v2"

    @classmethod
    def build_brain(cls, seed: int | None) -> PassiveHeuristicV2Brain:
        del seed
        return PassiveHeuristicV2Brain()


class AggressiveHeuristicBot(_VersionedHeuristicBot):
    """Development-only live wrapper for the latest aggressive heuristic."""

    BOT_ID = "bot_00000000-0000-4000-8000-00000000000a"
    BOT_NAME = "aggressive"

    @classmethod
    def build_brain(cls, seed: int | None) -> AggressiveHeuristicBrain:
        del seed
        return AggressiveHeuristicBrain()


class BalancedHeuristicBot(_VersionedHeuristicBot):
    """Development-only live wrapper for the latest balanced heuristic."""

    BOT_ID = "bot_00000000-0000-4000-8000-00000000000b"
    BOT_NAME = "balanced"

    @classmethod
    def build_brain(cls, seed: int | None) -> BalancedHeuristicBrain:
        del seed
        return BalancedHeuristicBrain()


class PassiveHeuristicBot(_VersionedHeuristicBot):
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

AGGRESSIVE_HEURISTIC_V1_BOT_SPEC = BotSpec.from_bot_class(AggressiveHeuristicV1Bot)
BALANCED_HEURISTIC_V1_BOT_SPEC = BotSpec.from_bot_class(BalancedHeuristicV1Bot)
PASSIVE_HEURISTIC_V1_BOT_SPEC = BotSpec.from_bot_class(PassiveHeuristicV1Bot)

AGGRESSIVE_HEURISTIC_V2_BOT_SPEC = BotSpec.from_bot_class(AggressiveHeuristicV2Bot)
BALANCED_HEURISTIC_V2_BOT_SPEC = BotSpec.from_bot_class(BalancedHeuristicV2Bot)
PASSIVE_HEURISTIC_V2_BOT_SPEC = BotSpec.from_bot_class(PassiveHeuristicV2Bot)
