"""Live SDK wrapper for the current public large-teen PPO policy."""

from __future__ import annotations

from pocketrocks import BotDecision, DecisionContext

from garboid_pocketrocks.adapters.public_history import public_history_from_sdk_frame
from garboid_pocketrocks.bots.base import HistoryAwareBotBrain, PocketRocksFastBot
from garboid_pocketrocks.neural.tournament_bot import VectorPpoLargeV1G350kBrain


class PpoLargeTeenBot(PocketRocksFastBot):
    """Public latest alias backed by the frozen 349,860-game large policy."""

    BOT_ID = "bot_dd7807c1-93bc-4f70-80c8-a2f2d7d26429"
    BOT_NAME = "ppo-large-teen"

    @classmethod
    def build_brain(cls, seed: int | None) -> VectorPpoLargeV1G350kBrain:
        return VectorPpoLargeV1G350kBrain(seed=seed)

    async def choose_raw_decision(
        self,
        frame: object,
        context: DecisionContext,
    ) -> BotDecision:
        if not isinstance(self._brain, HistoryAwareBotBrain):
            raise TypeError("ppo-large-teen requires a history-aware brain")
        return self._brain.choose_decision_with_history(
            context,
            self._knowledge_for_context(context),
            public_history_from_sdk_frame(frame),
        )
