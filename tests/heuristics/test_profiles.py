import math
from dataclasses import FrozenInstanceError

import pytest

import garboid_pocketrocks.heuristics.profiles as profiles_module
from garboid_pocketrocks.heuristics.errors import HeuristicInputError
from garboid_pocketrocks.heuristics.profiles import (
    AGGRESSIVE_PROFILE,
    BALANCED_PROFILE,
    HEURISTIC_V1,
    HEURISTIC_V2,
    HEURISTIC_V3,
    LATEST_HEURISTICS,
    PASSIVE_PROFILE,
    HeuristicProfile,
    HeuristicProfileSet,
)


def test_heuristic_input_error_is_a_value_error() -> None:
    assert issubclass(HeuristicInputError, ValueError)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("liquidity_strength", -0.1),
        ("liquidity_strength", float("nan")),
        ("liquidity_strength", float("inf")),
        ("future_cash_weight", -0.1),
        ("future_cash_weight", float("nan")),
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
            future_cash_weight=value if field == "future_cash_weight" else 0.5,
            objective_progress_weight=(value if field == "objective_progress_weight" else 0.2),
            bid_shading=value if field == "bid_shading" else 0.25,
        )


def test_profile_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="name"):
        HeuristicProfile("", 0.4, 0.5, 0.2, 0.25)


def test_versioned_profiles_have_exact_constants() -> None:
    assert HEURISTIC_V1.version == "v1"
    assert HEURISTIC_V1.aggressive == HeuristicProfile("aggressive", 0.75, 0.0, 0.25, 0.05)
    assert HEURISTIC_V1.balanced == HeuristicProfile("balanced", 0.40, 0.0, 0.20, 0.25)
    assert HEURISTIC_V1.passive == HeuristicProfile("passive", 0.15, 0.0, 0.15, 0.50)

    assert HEURISTIC_V2.version == "v2"
    assert HEURISTIC_V2.aggressive == HeuristicProfile("aggressive", 0.75, 1.50, 0.25, 0.05)
    assert HEURISTIC_V2.balanced == HeuristicProfile("balanced", 0.40, 0.75, 0.20, 0.25)
    assert HEURISTIC_V2.passive == HeuristicProfile("passive", 0.15, 0.60, 0.15, 0.30)

    assert HEURISTIC_V3.version == "v3"
    assert HEURISTIC_V3.aggressive == HeuristicProfile("aggressive", 1.00, 1.95, 0.15, 0.40)
    assert HEURISTIC_V3.balanced == HeuristicProfile("balanced", 0.25, 1.55, 0.30, 0.35)
    assert HEURISTIC_V3.passive == HeuristicProfile("passive", 1.50, 1.80, 0.95, 0.45)


def test_unversioned_profiles_alias_latest_generation() -> None:
    assert LATEST_HEURISTICS is HEURISTIC_V3
    assert AGGRESSIVE_PROFILE is HEURISTIC_V3.aggressive
    assert BALANCED_PROFILE is HEURISTIC_V3.balanced
    assert PASSIVE_PROFILE is HEURISTIC_V3.passive


@pytest.mark.parametrize("version", ("", "1", "version-1", "v0", "v-1", "V1"))
def test_profile_set_rejects_noncanonical_version(version: str) -> None:
    with pytest.raises(ValueError, match="version"):
        HeuristicProfileSet(
            version,
            HeuristicProfile("aggressive", 0.75, 0.0, 0.25, 0.05),
            HeuristicProfile("balanced", 0.40, 0.0, 0.20, 0.25),
            HeuristicProfile("passive", 0.15, 0.0, 0.15, 0.50),
        )


def test_profile_set_rejects_mismatched_personality_names() -> None:
    with pytest.raises(ValueError, match="personalities"):
        HeuristicProfileSet(
            "v3",
            HeuristicProfile("balanced", 0.75, 0.0, 0.25, 0.05),
            HeuristicProfile("aggressive", 0.40, 0.0, 0.20, 0.25),
            HeuristicProfile("passive", 0.15, 0.0, 0.15, 0.50),
        )


def test_each_released_generation_has_three_distinct_canonical_profiles() -> None:
    for profile_set in (HEURISTIC_V1, HEURISTIC_V2, HEURISTIC_V3):
        profiles = (
            profile_set.aggressive,
            profile_set.balanced,
            profile_set.passive,
        )
        assert tuple(profile.name for profile in profiles) == (
            "aggressive",
            "balanced",
            "passive",
        )
        assert len(set(profiles)) == 3


def test_profiles_are_frozen_and_coefficients_are_finite() -> None:
    assert all(
        math.isfinite(coefficient)
        for profile_set in (HEURISTIC_V1, HEURISTIC_V2, HEURISTIC_V3)
        for profile in (
            profile_set.aggressive,
            profile_set.balanced,
            profile_set.passive,
        )
        for coefficient in (
            profile.liquidity_strength,
            profile.future_cash_weight,
            profile.objective_progress_weight,
            profile.bid_shading,
        )
    )
    with pytest.raises(FrozenInstanceError):
        AGGRESSIVE_PROFILE.bid_shading = 1.0  # type: ignore[misc]


def test_phase_aware_profile_returns_the_expert_for_each_phase() -> None:
    profile_class = profiles_module.PhaseAwareHeuristicProfile
    early = HeuristicProfile("balanced", 0.1, 0.2, 0.3, 0.4)
    middle = HeuristicProfile("balanced", 0.2, 0.3, 0.4, 0.5)
    late = HeuristicProfile("balanced", 0.3, 0.4, 0.5, 0.6)

    profile = profile_class("balanced", early, middle, late)

    assert profile.profile_for_phase("early") is early
    assert profile.profile_for_phase("middle") is middle
    assert profile.profile_for_phase("late") is late
    assert profile.phase_selector == "public-resource-horizon-v1"
    with pytest.raises(FrozenInstanceError):
        profile.early = middle  # type: ignore[misc]


@pytest.mark.parametrize("name", ("", "custom", "Balanced"))
def test_phase_aware_profile_rejects_noncanonical_personality(name: str) -> None:
    profile_class = profiles_module.PhaseAwareHeuristicProfile
    expert = HeuristicProfile("balanced", 0.1, 0.2, 0.3, 0.4)

    with pytest.raises(ValueError, match="canonical personality"):
        profile_class(name, expert, expert, expert)


@pytest.mark.parametrize("missing_phase", ("early", "middle", "late"))
def test_phase_aware_profile_requires_matching_experts_for_every_phase(
    missing_phase: str,
) -> None:
    profile_class = profiles_module.PhaseAwareHeuristicProfile
    balanced = HeuristicProfile("balanced", 0.1, 0.2, 0.3, 0.4)
    aggressive = HeuristicProfile("aggressive", 0.1, 0.2, 0.3, 0.4)
    experts = {
        "early": balanced,
        "middle": balanced,
        "late": balanced,
    }
    experts[missing_phase] = aggressive

    with pytest.raises(ValueError, match="matching canonical personality"):
        profile_class("balanced", experts["early"], experts["middle"], experts["late"])


def test_phase_aware_profile_rejects_an_unknown_phase() -> None:
    profile_class = profiles_module.PhaseAwareHeuristicProfile
    expert = HeuristicProfile("balanced", 0.1, 0.2, 0.3, 0.4)
    profile = profile_class("balanced", expert, expert, expert)

    with pytest.raises(ValueError, match="phase"):
        profile.profile_for_phase("endgame")  # type: ignore[arg-type]


def test_phase_aware_profile_rejects_an_alternate_selector() -> None:
    profile_class = profiles_module.PhaseAwareHeuristicProfile
    expert = HeuristicProfile("balanced", 0.1, 0.2, 0.3, 0.4)

    with pytest.raises(ValueError, match="selector"):
        profile_class(
            "balanced",
            expert,
            expert,
            expert,
            phase_selector="turn-number-v1",
        )
