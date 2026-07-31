from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from typing import Protocol

from garboid_pocketrocks.bots import (
    AggressiveHeuristicBot,
    BalancedHeuristicBot,
    PassiveHeuristicBot,
    RandomBot,
)
from garboid_pocketrocks.bots.launcher import BOT_REGISTRY, run_bots
from garboid_pocketrocks.neural.live_bot import PpoLargeTeenBot


class _RunnableBot(Protocol):
    def run(self) -> None: ...


def test_registry_contains_every_live_wrapper_in_stable_order() -> None:
    assert tuple(BOT_REGISTRY) == (
        "random",
        "aggressive",
        "balanced",
        "passive",
        "ppo-large-teen",
    )
    assert tuple(BOT_REGISTRY.values()) == (
        RandomBot,
        AggressiveHeuristicBot,
        BalancedHeuristicBot,
        PassiveHeuristicBot,
        PpoLargeTeenBot,
    )


def test_run_bots_starts_one_thread_per_bot() -> None:
    barrier = threading.Barrier(2)
    constructions: list[str] = []
    thread_names: list[str] = []

    def factory(name: str) -> Callable[[], _RunnableBot]:
        class ControlledBot:
            def __init__(self) -> None:
                constructions.append(name)

            def run(self) -> None:
                thread_names.append(threading.current_thread().name)
                barrier.wait(timeout=1)

        return ControlledBot

    registry: Mapping[str, Callable[[], _RunnableBot]] = {
        "first": factory("first"),
        "second": factory("second"),
    }

    run_bots(registry=registry)

    assert constructions == ["first", "second"]
    assert set(thread_names) == {"garboid-first", "garboid-second"}
