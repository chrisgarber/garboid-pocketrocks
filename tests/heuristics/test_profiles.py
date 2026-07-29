import math
from dataclasses import FrozenInstanceError

import pytest

from garboid_pocketrocks.heuristics.errors import HeuristicInputError
from garboid_pocketrocks.heuristics.profiles import (
    AGGRESSIVE_PROFILE,
    BALANCED_PROFILE,
    PASSIVE_PROFILE,
    HeuristicProfile,
)


def test_heuristic_input_error_is_a_value_error() -> None:
    assert issubclass(HeuristicInputError, ValueError)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("liquidity_strength", -0.1),
        ("liquidity_strength", float("nan")),
        ("liquidity_strength", float("inf")),
        ("objective_progress_weight", -0.1),
        ("objective_progress_weight", 1.1),
        ("objective_progress_weight", float("-inf")),
        ("bid_shading", -0.1),
        ("bid_shading", 1.1),
        ("bid_shading", float("nan")),
    ),
)
def test_profile_rejects_invalid_coefficient(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        HeuristicProfile(
            name="invalid",
            liquidity_strength=value if field == "liquidity_strength" else 0.4,
            objective_progress_weight=(value if field == "objective_progress_weight" else 0.2),
            bid_shading=value if field == "bid_shading" else 0.25,
        )


def test_profile_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="name"):
        HeuristicProfile("", 0.4, 0.2, 0.25)


def test_named_profiles_have_exact_constants() -> None:
    assert AGGRESSIVE_PROFILE == HeuristicProfile("aggressive", 0.75, 0.25, 0.05)
    assert BALANCED_PROFILE == HeuristicProfile("balanced", 0.40, 0.20, 0.25)
    assert PASSIVE_PROFILE == HeuristicProfile("passive", 0.15, 0.15, 0.50)


def test_named_profiles_have_expected_ordering() -> None:
    assert (
        AGGRESSIVE_PROFILE.liquidity_strength
        > BALANCED_PROFILE.liquidity_strength
        > PASSIVE_PROFILE.liquidity_strength
    )
    assert (
        AGGRESSIVE_PROFILE.bid_shading < BALANCED_PROFILE.bid_shading < PASSIVE_PROFILE.bid_shading
    )


def test_profiles_are_frozen_and_coefficients_are_finite() -> None:
    assert all(
        math.isfinite(coefficient)
        for profile in (AGGRESSIVE_PROFILE, BALANCED_PROFILE, PASSIVE_PROFILE)
        for coefficient in (
            profile.liquidity_strength,
            profile.objective_progress_weight,
            profile.bid_shading,
        )
    )
    with pytest.raises(FrozenInstanceError):
        AGGRESSIVE_PROFILE.bid_shading = 1.0  # type: ignore[misc]
