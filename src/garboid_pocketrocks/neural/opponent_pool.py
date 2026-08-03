"""Versioned fixed-opponent pools for neural training."""

from __future__ import annotations

from garboid_pocketrocks.bots.base import BotSpec
from garboid_pocketrocks.bots.fixed_bid import FIXED_BID_TUNED_V1_BOT_SPEC
from garboid_pocketrocks.bots.fixed_objective_overlay import (
    FIXED_OBJECTIVE_OVERLAY_V3_BOT_SPEC,
)
from garboid_pocketrocks.bots.heuristic import (
    AGGRESSIVE_HEURISTIC_V2_BOT_SPEC,
    BALANCED_HEURISTIC_V3_BOT_SPEC,
    PASSIVE_HEURISTIC_V3_BOT_SPEC,
)
from garboid_pocketrocks.bots.surplus import SURPLUS_V10_BOT_SPEC

STRONG_FIELD_POOL_V1_IDENTITY = "strong-field-pool-v1"

# Repeated entries are deliberate sampling weights. The two tournament leaders
# occupy half of the slots while four distinct styles preserve strategic breadth.
STRONG_FIELD_POOL_V1: tuple[BotSpec, ...] = (
    SURPLUS_V10_BOT_SPEC,
    SURPLUS_V10_BOT_SPEC,
    FIXED_OBJECTIVE_OVERLAY_V3_BOT_SPEC,
    FIXED_OBJECTIVE_OVERLAY_V3_BOT_SPEC,
    AGGRESSIVE_HEURISTIC_V2_BOT_SPEC,
    BALANCED_HEURISTIC_V3_BOT_SPEC,
    PASSIVE_HEURISTIC_V3_BOT_SPEC,
    FIXED_BID_TUNED_V1_BOT_SPEC,
)

FIXED_TRAINING_BOT_SPECS_BY_NAME = {
    spec.name: spec for spec in STRONG_FIELD_POOL_V1
}
