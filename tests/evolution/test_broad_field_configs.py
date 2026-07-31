from __future__ import annotations

from pathlib import Path

import pytest

from garboid_pocketrocks.bots import BOT_SPECS_BY_NAME
from garboid_pocketrocks.evolution.manifest import load_search_manifest
from garboid_pocketrocks.promotion.corpus import (
    load_promotion_corpus,
    validate_corpus_separation,
)

REPOSITORY_ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize("personality", ("aggressive", "balanced", "passive"))
def test_v3_search_uses_a_separate_broad_field_for_each_personality(
    personality: str,
) -> None:
    development = load_promotion_corpus(
        REPOSITORY_ROOT / f"configs/promotion/development-{personality}-v3-broad-v1.json",
        registry=BOT_SPECS_BY_NAME,
    )
    held_out = load_promotion_corpus(
        REPOSITORY_ROOT / f"configs/promotion/held-out-{personality}-v3-broad-v1.json",
        registry=BOT_SPECS_BY_NAME,
    )
    manifest = load_search_manifest(
        REPOSITORY_ROOT / f"configs/evolution/{personality}-v3-search-v1.json",
        development_corpus=development,
    )

    assert len(development.cases) == 240
    assert len(held_out.cases) == 480
    assert len(development.recipe.opponent_names) == 8
    assert len(held_out.recipe.opponent_names) == 7
    assert manifest.development_corpus.name == development.recipe.name
    assert manifest.development_corpus.digest == development.digest
    assert manifest.predecessor_name not in development.recipe.opponent_names
    assert manifest.predecessor_name not in held_out.recipe.opponent_names
    assert {
        "random",
        "fixed-bid",
        "sdk-greedy-value-v1",
        "aggressive-v1",
        "balanced-v1",
        "passive-v1",
    }.issubset(development.recipe.opponent_names)
    assert {
        "vector_ppo_small_v1_g1500",
        "vector_ppo_large_v1_g350k",
    }.issubset(held_out.recipe.opponent_names)
    assert not {
        "vector_ppo_small_v1_g1500",
        "vector_ppo_large_v1_g350k",
    }.intersection(development.recipe.opponent_names)
    assert set(development.recipe.opponent_names) == {
        name
        for case in development.cases
        for name in case.opponent_names_by_seat
        if name is not None
    }
    validate_corpus_separation(development, held_out)
