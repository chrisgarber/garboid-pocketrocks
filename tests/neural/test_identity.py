from __future__ import annotations

import re
from typing import cast

import pytest

from garboid_pocketrocks.neural.config import ModelProfile
from garboid_pocketrocks.neural.identity import (
    experimental_neural_bot_id,
    trained_neural_bot_id,
)


@pytest.mark.parametrize(
    ("profile", "completed_games", "generation", "expected"),
    (
        ("small", 0, 1, "vector_ppo_small_v1_g0"),
        ("medium", 1_500, 1, "vector_ppo_medium_v1_g1500"),
        ("large", 100_000, 2, "vector_ppo_large_v2_g100000"),
        ("large", 1_000_000, 1, "vector_ppo_large_v1_g1000000"),
        ("large", 1_000_001, 12, "vector_ppo_large_v12_g1000001"),
    ),
)
def test_trained_neural_bot_id_has_exact_deterministic_components(
    profile: ModelProfile,
    completed_games: int,
    generation: int,
    expected: str,
) -> None:
    assert trained_neural_bot_id(profile, completed_games, generation) == expected
    assert trained_neural_bot_id(profile, completed_games, generation) == expected


def test_trained_neural_bot_id_defaults_to_generation_one() -> None:
    assert trained_neural_bot_id("small", 1) == "vector_ppo_small_v1_g1"


def test_exact_game_counts_prevent_rounding_collisions() -> None:
    checkpoint_ids = {
        trained_neural_bot_id("medium", completed_games)
        for completed_games in (1_499, 1_500, 1_501, 999_999, 1_000_000, 1_000_001)
    }

    assert len(checkpoint_ids) == 6


def test_trained_neural_bot_id_is_a_compact_local_bot_id() -> None:
    bot_id = trained_neural_bot_id("large", 1_000_000)

    assert re.fullmatch(r"[a-z0-9_]+", bot_id)
    assert len(bot_id) <= 32


@pytest.mark.parametrize("profile", ("", "tiny", "Large"))
def test_trained_neural_bot_id_rejects_unknown_profiles(profile: str) -> None:
    with pytest.raises(ValueError, match="profile must be small, medium, or large"):
        trained_neural_bot_id(cast(ModelProfile, profile), 0)


@pytest.mark.parametrize(
    "completed_games",
    (-1, True, 1.5, "1500"),
)
def test_trained_neural_bot_id_rejects_invalid_completed_games(
    completed_games: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="completed_games must be a nonnegative integer",
    ):
        trained_neural_bot_id("small", cast(int, completed_games))


@pytest.mark.parametrize(
    "generation",
    (0, -1, True, 1.5, "1"),
)
def test_trained_neural_bot_id_rejects_invalid_generations(
    generation: object,
) -> None:
    with pytest.raises(ValueError, match="generation must be a positive integer"):
        trained_neural_bot_id("small", 0, cast(int, generation))


def test_experimental_identity_records_every_selection_input() -> None:
    assert experimental_neural_bot_id(
        "large",
        strategy="behavior-cloning-balanced-v3",
        root_seed=42,
        completed_games=349_860,
        config_digest="a" * 64,
    ) == ("vector_ppo_large_v2_g349860_behavior-cloning-balanced-v3_s42_caaaaaaaaaaaa")


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("strategy", "Behavior Cloning"),
        ("root_seed", -1),
        ("config_digest", "not-a-digest"),
    ),
)
def test_experimental_identity_rejects_ambiguous_values(
    field: str,
    value: object,
) -> None:
    arguments: dict[str, object] = {
        "strategy": "behavior-cloning",
        "root_seed": 42,
        "completed_games": 10,
        "config_digest": "a" * 64,
    }
    arguments[field] = value
    with pytest.raises(ValueError):
        experimental_neural_bot_id("large", **arguments)  # type: ignore[arg-type]
