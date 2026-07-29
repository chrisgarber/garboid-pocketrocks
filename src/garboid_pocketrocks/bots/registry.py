from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from garboid_pocketrocks.bots.base import BotSpec
from garboid_pocketrocks.bots.heuristic import (
    AggressiveHeuristicBot,
    BalancedHeuristicBot,
    PassiveHeuristicBot,
)
from garboid_pocketrocks.bots.random_bot import RandomBot

BOT_SPECS = tuple(
    BotSpec.from_bot_class(bot_class)
    for bot_class in (
        RandomBot,
        AggressiveHeuristicBot,
        BalancedHeuristicBot,
        PassiveHeuristicBot,
    )
)


def _index_specs(specs: tuple[BotSpec, ...]) -> Mapping[str, BotSpec]:
    names = tuple(spec.name for spec in specs)
    bot_ids = tuple(spec.bot_id for spec in specs)
    if len(set(names)) != len(names):
        raise ValueError("registered bot names must be unique")
    if len(set(bot_ids)) != len(bot_ids):
        raise ValueError("registered bot IDs must be unique")
    return MappingProxyType({spec.name: spec for spec in specs})


BOT_SPECS_BY_NAME = _index_specs(BOT_SPECS)


def registered_bot_specs() -> tuple[BotSpec, ...]:
    return BOT_SPECS
