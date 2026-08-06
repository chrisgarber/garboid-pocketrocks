"""Named and file-backed tournament bot fields."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import NoReturn

from garboid_pocketrocks.bots.base import BotSpec
from garboid_pocketrocks.bots.registry import (
    DEFAULT_TOURNAMENT_BOT_SPECS,
    registered_bot_specs,
)

_FIELD_CONFIG_KEYS = {"schema_version", "name", "bot_names"}
_BUILTIN_FIELD_NAMES = frozenset({"default", "all"})


def builtin_tournament_field_names() -> tuple[str, ...]:
    return ("default", "all")


def load_tournament_field_bot_names(field: str) -> tuple[str, ...]:
    """Resolve a built-in field name or a JSON field config path to bot names."""

    if field in _BUILTIN_FIELD_NAMES:
        specs = (
            DEFAULT_TOURNAMENT_BOT_SPECS
            if field == "default"
            else registered_bot_specs()
        )
        return tuple(spec.name for spec in specs)

    path = Path(field)
    if not path.is_file():
        _field_error(
            f"unknown tournament field {field!r}; "
            f"use one of {', '.join(builtin_tournament_field_names())} "
            "or a path to a field config JSON file"
        )
    return _load_field_config(path)


def resolve_tournament_field_specs(
    field: str,
    *,
    registry: Mapping[str, BotSpec],
) -> tuple[BotSpec, ...]:
    """Resolve a field to registered bot specs in configured order."""

    names = load_tournament_field_bot_names(field)
    unknown = set(names) - set(registry)
    if unknown:
        _field_error(f"unknown bot name(s) in field config: {', '.join(sorted(unknown))}")
    return tuple(registry[name] for name in names)


def _load_field_config(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        _field_error(f"{path} must contain a JSON object")
    unexpected = set(payload) - _FIELD_CONFIG_KEYS
    if unexpected:
        _field_error(
            f"{path} has unexpected key(s): {', '.join(sorted(unexpected))}"
        )
    if payload.get("schema_version") != 1:
        _field_error(f"{path} must set schema_version to 1")
    name = payload.get("name")
    if not isinstance(name, str) or not name:
        _field_error(f"{path} must set a nonempty string name")
    bot_names = payload.get("bot_names")
    if not isinstance(bot_names, list) or not bot_names:
        _field_error(f"{path} must set a nonempty bot_names array")
    if not all(isinstance(item, str) and item for item in bot_names):
        _field_error(f"{path} bot_names must contain only nonempty strings")
    if len(set(bot_names)) != len(bot_names):
        _field_error(f"{path} bot_names must be unique")
    return tuple(bot_names)


def _field_error(message: str) -> NoReturn:
    raise ValueError(message)
