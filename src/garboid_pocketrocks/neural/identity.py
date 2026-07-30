"""Durable local identities for trained neural bots."""

from __future__ import annotations

from garboid_pocketrocks.neural.config import ModelProfile


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
