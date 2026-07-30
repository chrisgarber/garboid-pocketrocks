from __future__ import annotations

import argparse
import random

from pocketrocks import BotDecision, DecisionContext

from garboid_pocketrocks.bots.base import PocketRocksFastBot
from garboid_pocketrocks.knowledge import RulesetKnowledge


class RandomBotBrain:
    def __init__(self, *, seed: int | None = None) -> None:
        self._random = random.Random(seed)

    def choose_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
    ) -> BotDecision:
        del ruleset
        if context.decision_kind == "submitBid":
            max_amount = context.legal_max_amount
            if max_amount is None or max_amount <= 0:
                return BotDecision.pass_turn()
            amount = self._random.randint(0, max_amount)
            return BotDecision.pass_turn() if amount == 0 else BotDecision.submit_bid(amount)

        if context.revealable_count <= 0:
            return BotDecision.pass_turn()
        index = self._random.randrange(context.revealable_count)
        return BotDecision.select_info_to_reveal(index)


class RandomBot(PocketRocksFastBot):
    """PocketRocks bot that samples uniformly from the legal action range."""

    BOT_ID = "bot_e0e2c541-1615-4f47-983c-224e7d888d89"
    BOT_NAME = "random"

    @classmethod
    def build_brain(cls, seed: int | None) -> RandomBotBrain:
        return RandomBotBrain(seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Garboid random PocketRocks bot")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed the bot's random decisions for reproducibility",
    )
    args = parser.parse_args()
    RandomBot(seed=args.seed).run()
