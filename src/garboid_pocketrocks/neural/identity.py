"""Durable local identities for trained neural bots."""

from __future__ import annotations

import re

from garboid_pocketrocks.neural.config import ModelProfile

_RESEARCH_STRATEGY = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def trained_neural_bot_id(
    profile: ModelProfile,
    completed_games: int,
    generation: int = 1,
) -> str:
    """Return a lossless local ID for one trained neural checkpoint."""

    if profile not in ("small", "medium", "large"):
        raise ValueError("profile must be small, medium, or large")
    if (
        not isinstance(completed_games, int)
        or isinstance(completed_games, bool)
        or completed_games < 0
    ):
        raise ValueError("completed_games must be a nonnegative integer")
    if not isinstance(generation, int) or isinstance(generation, bool) or generation < 1:
        raise ValueError("generation must be a positive integer")

    return f"vector_ppo_{profile}_v{generation}_g{completed_games}"


def experimental_neural_bot_id(
    profile: ModelProfile,
    *,
    strategy: str,
    root_seed: int,
    completed_games: int,
    config_digest: str,
    generation: int = 2,
) -> str:
    """Name one immutable research checkpoint without implying promotion."""

    base_identity = trained_neural_bot_id(
        profile,
        completed_games,
        generation,
    )
    if not isinstance(strategy, str) or _RESEARCH_STRATEGY.fullmatch(strategy) is None:
        raise ValueError("strategy must be a lowercase hyphenated name")
    if not isinstance(root_seed, int) or isinstance(root_seed, bool) or root_seed < 0:
        raise ValueError("root_seed must be a nonnegative integer")
    if not isinstance(config_digest, str) or _SHA256.fullmatch(config_digest) is None:
        raise ValueError("config_digest must be a lowercase SHA-256 digest")
    return f"{base_identity}_{strategy}_s{root_seed}_c{config_digest[:12]}"
