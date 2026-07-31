"""Immutable released heuristic teachers used by neural research only."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal

from garboid_pocketrocks.bots.heuristic import (
    AggressiveHeuristicV3Brain,
    BalancedHeuristicV3Brain,
    HeuristicBotBrain,
    PassiveHeuristicV3Brain,
)
from garboid_pocketrocks.heuristics.profiles import HEURISTIC_V3, HeuristicProfile

RELEASED_HEURISTIC_V3_IDENTITIES = (
    "aggressive-v3",
    "balanced-v3",
    "passive-v3",
)
BALANCED_V3_TEACHER_IDENTITY = "balanced-v3"
BALANCED_V3_PROFILE_DIGEST = "e3971899626ca3f651b2992d0cc429dc3ffd57fcdbb7cfac8249e6f0f9d9b03e"

_PROFILE_BY_IDENTITY = {
    "aggressive-v3": HEURISTIC_V3.aggressive,
    "balanced-v3": HEURISTIC_V3.balanced,
    "passive-v3": HEURISTIC_V3.passive,
}
_BRAIN_BY_IDENTITY = {
    "aggressive-v3": AggressiveHeuristicV3Brain,
    "balanced-v3": BalancedHeuristicV3Brain,
    "passive-v3": PassiveHeuristicV3Brain,
}


def build_released_v3_brain(identity: str) -> HeuristicBotBrain:
    """Construct one explicit released teacher without resolving an alias."""

    try:
        brain_type = _BRAIN_BY_IDENTITY[identity]
    except KeyError as error:
        raise ValueError(f"unknown released v3 heuristic identity: {identity!r}") from error
    return brain_type()


def released_v3_profile_digest(identity: str) -> str:
    """Recompute the profile digest used by frozen promotion evidence."""

    try:
        profile = _PROFILE_BY_IDENTITY[identity]
    except KeyError as error:
        raise ValueError(f"unknown released v3 heuristic identity: {identity!r}") from error
    return _profile_digest(profile)


def _profile_digest(profile: HeuristicProfile) -> str:
    payload = {
        name: _canonical_decimal_text(Decimal(str(getattr(profile, name))))
        for name in (
            "bid_shading",
            "future_cash_weight",
            "liquidity_strength",
            "objective_progress_weight",
        )
    }
    encoded = (
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _canonical_decimal_text(value: Decimal) -> str:
    normalized = format(value.normalize(), "f")
    return "0" if normalized in ("-0", "") else normalized
