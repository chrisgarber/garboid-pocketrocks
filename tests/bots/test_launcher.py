from __future__ import annotations

import argparse

import pytest

from garboid_pocketrocks.bots import (
    AggressiveHeuristicBot,
    BalancedHeuristicBot,
    PassiveHeuristicBot,
    RandomBot,
)
from garboid_pocketrocks.bots.launcher import BOT_REGISTRY, _bot_names, _parser


def test_registry_contains_every_live_wrapper_in_stable_order() -> None:
    assert tuple(BOT_REGISTRY) == ("random", "aggressive", "balanced", "passive")
    assert tuple(BOT_REGISTRY.values()) == (
        RandomBot,
        AggressiveHeuristicBot,
        BalancedHeuristicBot,
        PassiveHeuristicBot,
    )


def test_parser_selects_all_registered_bots_by_default() -> None:
    assert _parser().parse_args([]).bots == tuple(BOT_REGISTRY)


def test_bot_names_accepts_a_trimmed_subset_in_requested_order() -> None:
    assert _bot_names(" passive, aggressive ") == ("passive", "aggressive")


@pytest.mark.parametrize(
    ("value", "message"),
    (
        ("", "at least one bot name is required"),
        ("  ", "at least one bot name is required"),
        ("random,random", "duplicate bot name"),
        ("random,missing", "unknown bot name"),
    ),
)
def test_bot_names_rejects_invalid_selections(value: str, message: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match=message):
        _bot_names(value)
