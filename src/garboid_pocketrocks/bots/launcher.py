from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

from pocketrocks._logging import install_default_logging

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


class BotRuntimeStopped(RuntimeError):
    def __init__(self, bot_name: str) -> None:
        self.bot_name = bot_name
        super().__init__(f"bot runtime stopped: {bot_name}")


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


async def _run_bot(name: str, bot: RunnableBot) -> None:
    await bot.run_async()
    raise BotRuntimeStopped(name)


async def run_bots(
    names: Sequence[str],
    *,
    registry: Mapping[str, BotFactory] = BOT_REGISTRY,
) -> None:
    bots = tuple((name, registry[name]()) for name in names)
    try:
        async with asyncio.TaskGroup() as group:
            for name, bot in bots:
                group.create_task(_run_bot(name, bot), name=name)
    except BaseExceptionGroup as errors:
        if errors.exceptions and all(
            isinstance(error, BotRuntimeStopped) for error in errors.exceptions
        ):
            stopped = errors.exceptions[0]
            assert isinstance(stopped, BotRuntimeStopped)
            raise stopped from None
        raise


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    install_default_logging()
    try:
        asyncio.run(run_bots(args.bots))
    except KeyboardInterrupt:
        return
    except BotRuntimeStopped as error:
        parser.exit(1, f"{parser.prog}: error: {error}\n")
