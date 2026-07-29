from __future__ import annotations

import argparse
import os
import random
from typing import Any

from dotenv import find_dotenv, load_dotenv
from pocketrocks import BotDecision, DecisionContext, PocketRocksBot


class RandomBot(PocketRocksBot):
    """PocketRocks bot that samples uniformly from the legal action range."""

    def __init__(self, *, seed: int | None = None, **sdk_options: Any) -> None:
        super().__init__(**sdk_options)
        self._random = random.Random(seed)

    async def choose_decision(self, context: DecisionContext) -> BotDecision:
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


def _random_bot_id() -> str | None:
    load_dotenv(find_dotenv(usecwd=True), override=False)
    return os.getenv("RANDOM_BOT_ID")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Garboid random PocketRocks bot")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed the bot's random decisions for reproducibility",
    )
    args = parser.parse_args()
    RandomBot(seed=args.seed, bot_id=_random_bot_id()).run()
