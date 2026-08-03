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
        "fixed-bid",
        "fixed-bid-tuned-v1",
        "fixed-bid-diverse-v1",
        "fixed-bid-tuned-normal-v1",
        "fixed-objective-overlay-v1",
        "fixed-objective-overlay-v2",
        "fixed-objective-overlay-v3",
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
        "sdk-greedy-value-v1",
        "surplus",
        "surplus-v1",
        "surplus-v2",
        "surplus-v3",
        "surplus-v4",
        "surplus-v5",
        "surplus-v6",
        "surplus-v7",
        "surplus-v8",
        "surplus-v9",
        "surplus-v10",
        "surplus-v11",
        "vector_ppo_small_v1_g1500",
        "vector_ppo_large_v1_g350k",
    )
    assert len({spec.name for spec in specs}) == len(specs)
    assert len({spec.bot_id for spec in specs}) == len(specs)
    assert BOT_SPECS_BY_NAME == {spec.name: spec for spec in specs}
    assert all("-candidate-" not in spec.name for spec in specs)


def test_default_tournament_specs_include_curated_field_and_v3_personalities() -> None:
    specs = registry_module.DEFAULT_TOURNAMENT_BOT_SPECS

    assert tuple(spec.name for spec in specs) == (
        "fixed-objective-overlay-v3",
        "fixed-objective-overlay-v2",
        "fixed-objective-overlay-v1",
        "fixed-bid-tuned-v1",
        "aggressive-v2",
        "fixed-bid-diverse-v1",
        "balanced-v2",
        "fixed-bid",
        "vector_ppo_large_v1_g350k",
        "passive-v2",
        "passive-v1",
        "aggressive-v3",
        "balanced-v3",
        "passive-v3",
    )


def test_fixed_bid_brain_is_exported_from_bots_package() -> None:
    from garboid_pocketrocks.bots import FixedBidBotBrain
    from garboid_pocketrocks.bots.fixed_bid import FixedBidBotBrain as DefinedBrain

    assert FixedBidBotBrain is DefinedBrain


def test_new_fixed_family_brains_are_exported_from_bots_package() -> None:
    from garboid_pocketrocks.bots import (
        FixedBidDiverseV1Brain,
        FixedBidTunedNormalV1Brain,
        FixedBidTunedV1Brain,
        FixedObjectiveOverlayBrain,
        FixedObjectiveOverlayV1Brain,
        FixedObjectiveOverlayV2Brain,
        FixedObjectiveOverlayV3Brain,
    )
    from garboid_pocketrocks.bots.fixed_bid import (
        FixedBidDiverseV1Brain as DefinedDiverse,
    )
    from garboid_pocketrocks.bots.fixed_bid import (
        FixedBidTunedNormalV1Brain as DefinedTunedNormal,
    )
    from garboid_pocketrocks.bots.fixed_bid import FixedBidTunedV1Brain as DefinedTuned
    from garboid_pocketrocks.bots.fixed_objective_overlay import (
        FixedObjectiveOverlayBrain as DefinedOverlayEngine,
    )
    from garboid_pocketrocks.bots.fixed_objective_overlay import (
        FixedObjectiveOverlayV1Brain as DefinedOverlay,
    )
    from garboid_pocketrocks.bots.fixed_objective_overlay import (
        FixedObjectiveOverlayV2Brain as DefinedOverlayV2,
    )
    from garboid_pocketrocks.bots.fixed_objective_overlay import (
        FixedObjectiveOverlayV3Brain as DefinedOverlayV3,
    )

    assert FixedBidTunedV1Brain is DefinedTuned
    assert FixedBidDiverseV1Brain is DefinedDiverse
    assert FixedBidTunedNormalV1Brain is DefinedTunedNormal
    assert FixedObjectiveOverlayBrain is DefinedOverlayEngine
    assert FixedObjectiveOverlayV1Brain is DefinedOverlay
    assert FixedObjectiveOverlayV2Brain is DefinedOverlayV2
    assert FixedObjectiveOverlayV3Brain is DefinedOverlayV3


def test_bottom_five_remain_registered_for_explicit_runs() -> None:
    assert {
        "balanced-v1",
        "aggressive-v1",
        "vector_ppo_small_v1_g1500",
        "sdk-greedy-value-v1",
        "random",
    } <= set(BOT_SPECS_BY_NAME)
