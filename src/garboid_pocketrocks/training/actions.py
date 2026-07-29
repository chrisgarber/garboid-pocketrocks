from __future__ import annotations

import gymnasium as gym
import numpy as np
from pocketrocks import BotDecision, DecisionContext

from garboid_pocketrocks.training.bounds import EnvironmentBounds


class ActionCodec:
    """Maps a fixed Gymnasium action space to SDK decisions."""

    def __init__(self, bounds: EnvironmentBounds) -> None:
        self.bounds = bounds
        self.size = 1 + bounds.max_bid + bounds.max_hand_size
        self.action_space: gym.spaces.Discrete[np.int32] = gym.spaces.Discrete(
            self.size,
            dtype=np.int32,
        )

    def encode(self, decision: BotDecision) -> int:
        if decision.action_kind == "pass":
            return 0
        if decision.action_kind == "submitBid":
            amount = decision.value
            if amount == 0:
                return 0
            if amount is None or not 1 <= amount <= self.bounds.max_bid:
                raise ValueError("bid is outside the fixed action space")
            return amount
        if decision.action_kind == "selectInfoToReveal":
            index = decision.value
            if index is None or not 0 <= index < self.bounds.max_hand_size:
                raise ValueError("reveal index is outside the fixed action space")
            return self.bounds.max_bid + 1 + index
        raise ValueError(f"unsupported decision kind {decision.action_kind!r}")

    def decode(self, action: int) -> BotDecision:
        if not 0 <= action < self.size:
            raise ValueError("action is outside the fixed action space")
        if action == 0:
            return BotDecision.pass_turn()
        if action <= self.bounds.max_bid:
            return BotDecision.submit_bid(action)
        return BotDecision.select_info_to_reveal(action - self.bounds.max_bid - 1)

    def mask(self, context: DecisionContext) -> np.ndarray[tuple[int], np.dtype[np.int8]]:
        mask = np.zeros(self.size, dtype=np.int8)
        mask[0] = 1
        if context.decision_kind == "submitBid":
            legal_max = context.legal_max_amount
            if legal_max is None or legal_max < 0:
                raise ValueError("bid context requires a nonnegative legal maximum")
            if legal_max > self.bounds.max_bid:
                raise ValueError("legal maximum exceeds environment bounds")
            mask[1 : legal_max + 1] = 1
        elif context.decision_kind == "selectInfoToReveal":
            if not 0 <= context.revealable_count <= self.bounds.max_hand_size:
                raise ValueError("revealable count exceeds environment bounds")
            start = self.bounds.max_bid + 1
            mask[start : start + context.revealable_count] = 1
        else:
            raise ValueError(f"unsupported decision kind {context.decision_kind!r}")
        return mask
