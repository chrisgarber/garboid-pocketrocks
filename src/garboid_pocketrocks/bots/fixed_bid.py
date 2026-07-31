from __future__ import annotations

import random
from dataclasses import dataclass
from typing import ClassVar

from pocketrocks import ActionId, BotDecision, DecisionContext

from garboid_pocketrocks.bots.base import BotSpec
from garboid_pocketrocks.knowledge import RulesetKnowledge

_ACTIONS = tuple(ActionId)


@dataclass(frozen=True, slots=True)
class FixedBidProfile:
    """Immutable bids for the six PocketRocks actions in wire-ID order."""

    target_bids: tuple[int, int, int, int, int, int]

    def __post_init__(self) -> None:
        if len(self.target_bids) != len(_ACTIONS):
            raise ValueError("fixed-bid profile must define all six actions")
        if any(type(target) is not int or target < 0 for target in self.target_bids):
            raise ValueError("fixed-bid targets must be nonnegative integers")

    def target_bid(self, action_id: int | None) -> int | None:
        if action_id is None:
            return None
        try:
            action = ActionId(action_id)
        except ValueError:
            return None
        return self.target_bids[int(action) - 1]


FIXED_BID_PROFILE = FixedBidProfile((5, 10, 2, 4, 4, 9))
FIXED_BID_TUNED_V1_PROFILE = FixedBidProfile((5, 10, 2, 5, 4, 7))
FIXED_BID_DIVERSE_V1_PROFILE = FixedBidProfile((4, 9, 2, 5, 4, 7))


def _target_bid(action_id: int | None) -> int | None:
    return FIXED_BID_PROFILE.target_bid(action_id)


class ProfiledFixedBidBotBrain:
    """Shared deterministic engine for immutable fixed-bid generations."""

    PROFILE: ClassVar[FixedBidProfile]

    def __init__(self, seed: int | None = None) -> None:
        del seed

    def choose_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
    ) -> BotDecision:
        del ruleset
        if context.decision_kind == "selectInfoToReveal":
            if context.revealable_count <= 0:
                return BotDecision.pass_turn()
            return BotDecision.select_info_to_reveal(0)

        legal_max = context.legal_max_amount
        target = self.PROFILE.target_bid(context.current_action_id)
        if legal_max is None or legal_max <= 0 or target is None:
            return BotDecision.pass_turn()
        adjusted_target = self._adjust_target(target)
        if adjusted_target <= 0:
            return BotDecision.pass_turn()
        return BotDecision.submit_bid(min(adjusted_target, legal_max))

    def _adjust_target(self, target: int) -> int:
        return target


class FixedBidBotBrain(ProfiledFixedBidBotBrain):
    """Original deterministic fixed-bid baseline."""

    PROFILE = FIXED_BID_PROFILE


class FixedBidTunedV1Brain(ProfiledFixedBidBotBrain):
    """Search finalist favoring the strongest local changes."""

    PROFILE = FIXED_BID_TUNED_V1_PROFILE


class FixedBidDiverseV1Brain(ProfiledFixedBidBotBrain):
    """Search finalist favoring different auction values when strength is close."""

    PROFILE = FIXED_BID_DIVERSE_V1_PROFILE


class FixedBidTunedNormalV1Brain(ProfiledFixedBidBotBrain):
    """Tuned fixed bids plus independent rounded standard-normal offsets."""

    PROFILE = FIXED_BID_TUNED_V1_PROFILE

    def __init__(self, seed: int | None = None) -> None:
        self._random = random.Random(seed)

    def _sample_offset(self) -> float:
        return self._random.normalvariate(0.0, 1.0)

    def _adjust_target(self, target: int) -> int:
        return target + round(self._sample_offset())


FIXED_BID_BOT_SPEC = BotSpec.for_simulation("fixed-bid", FixedBidBotBrain)
FIXED_BID_TUNED_V1_BOT_SPEC = BotSpec.for_simulation(
    "fixed-bid-tuned-v1",
    FixedBidTunedV1Brain,
)
FIXED_BID_DIVERSE_V1_BOT_SPEC = BotSpec.for_simulation(
    "fixed-bid-diverse-v1",
    FixedBidDiverseV1Brain,
)
FIXED_BID_TUNED_NORMAL_V1_BOT_SPEC = BotSpec.for_simulation(
    "fixed-bid-tuned-normal-v1",
    FixedBidTunedNormalV1Brain,
)
