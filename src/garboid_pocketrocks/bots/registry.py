from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from garboid_pocketrocks.bots.base import BotSpec
from garboid_pocketrocks.bots.heuristic import (
    AGGRESSIVE_HEURISTIC_BOT_SPEC,
    AGGRESSIVE_HEURISTIC_V1_BOT_SPEC,
    AGGRESSIVE_HEURISTIC_V2_BOT_SPEC,
    BALANCED_HEURISTIC_BOT_SPEC,
    BALANCED_HEURISTIC_V1_BOT_SPEC,
    BALANCED_HEURISTIC_V2_BOT_SPEC,
    PASSIVE_HEURISTIC_BOT_SPEC,
    PASSIVE_HEURISTIC_V1_BOT_SPEC,
    PASSIVE_HEURISTIC_V2_BOT_SPEC,
)
from garboid_pocketrocks.bots.random_bot import RandomBot

BOT_SPECS = (
    BotSpec.from_bot_class(RandomBot),
    AGGRESSIVE_HEURISTIC_BOT_SPEC,
    BALANCED_HEURISTIC_BOT_SPEC,
    PASSIVE_HEURISTIC_BOT_SPEC,
    AGGRESSIVE_HEURISTIC_V1_BOT_SPEC,
    BALANCED_HEURISTIC_V1_BOT_SPEC,
    PASSIVE_HEURISTIC_V1_BOT_SPEC,
    AGGRESSIVE_HEURISTIC_V2_BOT_SPEC,
    BALANCED_HEURISTIC_V2_BOT_SPEC,
    PASSIVE_HEURISTIC_V2_BOT_SPEC,
)

DEFAULT_TOURNAMENT_BOT_SPECS = (
    BOT_SPECS[0],
    AGGRESSIVE_HEURISTIC_V1_BOT_SPEC,
    BALANCED_HEURISTIC_V1_BOT_SPEC,
    PASSIVE_HEURISTIC_V1_BOT_SPEC,
    AGGRESSIVE_HEURISTIC_V2_BOT_SPEC,
    BALANCED_HEURISTIC_V2_BOT_SPEC,
    PASSIVE_HEURISTIC_V2_BOT_SPEC,
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
