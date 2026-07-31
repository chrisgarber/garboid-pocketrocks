from __future__ import annotations

import pytest

pytest.importorskip("torch")

from garboid_pocketrocks.neural.heuristic_teachers import (  # noqa: E402
    BALANCED_V3_PROFILE_DIGEST,
    RELEASED_HEURISTIC_V3_IDENTITIES,
    build_released_v3_brain,
    released_v3_profile_digest,
)


def test_balanced_teacher_digest_matches_frozen_promotion_evidence() -> None:
    assert released_v3_profile_digest("balanced-v3") == BALANCED_V3_PROFILE_DIGEST


def test_every_teacher_identity_resolves_explicitly() -> None:
    assert tuple(
        type(build_released_v3_brain(identity)).__name__
        for identity in RELEASED_HEURISTIC_V3_IDENTITIES
    ) == (
        "AggressiveHeuristicV3Brain",
        "BalancedHeuristicV3Brain",
        "PassiveHeuristicV3Brain",
    )

    with pytest.raises(ValueError, match="unknown released"):
        build_released_v3_brain("balanced")
