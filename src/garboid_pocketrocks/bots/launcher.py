from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Protocol

from garboid_pocketrocks.bots.heuristic import (
    AggressiveHeuristicBot,
    BalancedHeuristicBot,
    PassiveHeuristicBot,
)
from garboid_pocketrocks.bots.random_bot import RandomBot


class RunnableBot(Protocol):
    async def run_async(self) -> None:
        """Run one bot until cancellation or a terminal SDK outcome."""


BotFactory = Callable[[], RunnableBot]

BOT_REGISTRY: dict[str, BotFactory] = {
    RandomBot.BOT_NAME: RandomBot,
    AggressiveHeuristicBot.BOT_NAME: AggressiveHeuristicBot,
    BalancedHeuristicBot.BOT_NAME: BalancedHeuristicBot,
    PassiveHeuristicBot.BOT_NAME: PassiveHeuristicBot,
}


def _bot_names(value: str) -> tuple[str, ...]:
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    if not names:
        raise argparse.ArgumentTypeError("at least one bot name is required")
    duplicates = tuple(name for index, name in enumerate(names) if name in names[:index])
    if duplicates:
        duplicate_names = ", ".join(dict.fromkeys(duplicates))
        raise argparse.ArgumentTypeError(f"duplicate bot name(s): {duplicate_names}")
    unknown = tuple(name for name in names if name not in BOT_REGISTRY)
    if unknown:
        unknown_names = ", ".join(dict.fromkeys(unknown))
        raise argparse.ArgumentTypeError(f"unknown bot name(s): {unknown_names}")
    return names


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Garboid PocketRocks bots against the live service"
    )
    parser.add_argument(
        "--bots",
        type=_bot_names,
        default=tuple(BOT_REGISTRY),
        help=("comma-separated bot names; defaults to " f"{','.join(BOT_REGISTRY)}"),
    )
    return parser
