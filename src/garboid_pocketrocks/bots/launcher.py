from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from typing import Protocol

from garboid_pocketrocks.bots.heuristic import (
    AggressiveHeuristicBot,
    BalancedHeuristicBot,
    PassiveHeuristicBot,
)
from garboid_pocketrocks.bots.random_bot import RandomBot


class RunnableBot(Protocol):
    def run(self) -> None: ...


BotFactory = Callable[[], RunnableBot]

BOT_REGISTRY: dict[str, BotFactory] = {
    RandomBot.BOT_NAME: RandomBot,
    AggressiveHeuristicBot.BOT_NAME: AggressiveHeuristicBot,
    BalancedHeuristicBot.BOT_NAME: BalancedHeuristicBot,
    PassiveHeuristicBot.BOT_NAME: PassiveHeuristicBot,
}


def run_bots(*, registry: Mapping[str, BotFactory] = BOT_REGISTRY) -> None:
    bots = tuple((name, factory()) for name, factory in registry.items())
    threads = tuple(
        threading.Thread(
            target=bot.run,
            name=f"garboid-{name}",
            daemon=True,
        )
        for name, bot in bots
    )
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def main() -> None:
    try:
        run_bots()
    except KeyboardInterrupt:
        pass
