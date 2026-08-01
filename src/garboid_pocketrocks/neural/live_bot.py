"""Live SDK wrapper for the current public large-teen PPO policy."""

from __future__ import annotations

from garboid_pocketrocks.bots.base import PocketRocksFastBot
from garboid_pocketrocks.neural.tournament_bot import VectorPpoLargeV1G350kBrain


class PpoLargeTeenBot(PocketRocksFastBot):
    """Public latest alias backed by the frozen 349,860-game large policy."""

    BOT_ID = "bot_dd7807c1-93bc-4f70-80c8-a2f2d7d26429"
    BOT_NAME = "ppo-large-teen"

    @classmethod
    def build_brain(cls, seed: int | None) -> VectorPpoLargeV1G350kBrain:
        return VectorPpoLargeV1G350kBrain(seed=seed)
