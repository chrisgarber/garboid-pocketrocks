from __future__ import annotations

import garboid_pocketrocks.bots.registry as registry_module
from garboid_pocketrocks.bots.registry import (
    BOT_SPECS_BY_NAME,
    registered_bot_specs,
)


def test_registered_bot_specs_have_unique_names_and_ids() -> None:
    specs = registered_bot_specs()

    assert tuple(spec.name for spec in specs) == (
        "random",
        "aggressive",
        "balanced",
        "passive",
        "aggressive-v1",
        "balanced-v1",
        "passive-v1",
        "aggressive-v2",
        "balanced-v2",
        "passive-v2",
        "aggressive-v3",
        "balanced-v3",
        "passive-v3",
        "vector_ppo_small_v1_g1500",
        "vector_ppo_large_v1_g350k",
    )
    assert len({spec.name for spec in specs}) == len(specs)
    assert len({spec.bot_id for spec in specs}) == len(specs)
    assert BOT_SPECS_BY_NAME == {spec.name: spec for spec in specs}
    assert all("-candidate-" not in spec.name for spec in specs)


def test_default_tournament_specs_include_baseline_and_versioned_bots_only() -> None:
    specs = registry_module.DEFAULT_TOURNAMENT_BOT_SPECS

    assert tuple(spec.name for spec in specs) == (
        "random",
        "aggressive-v1",
        "balanced-v1",
        "passive-v1",
        "aggressive-v2",
        "balanced-v2",
        "passive-v2",
        "aggressive-v3",
        "balanced-v3",
        "passive-v3",
        "vector_ppo_small_v1_g1500",
        "vector_ppo_large_v1_g350k",
    )
