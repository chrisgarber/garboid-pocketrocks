from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HeuristicProfile:
    """Validated coefficients controlling a heuristic bot's behavior."""

    name: str
    liquidity_strength: float
    future_cash_weight: float
    objective_progress_weight: float
    bid_shading: float

    def __post_init__(self) -> None:
        coefficients = (
            self.liquidity_strength,
            self.future_cash_weight,
            self.objective_progress_weight,
            self.bid_shading,
        )
        if not self.name:
            raise ValueError("profile name must be nonempty")
        if not all(math.isfinite(value) for value in coefficients):
            raise ValueError("profile coefficients must be finite")
        if self.liquidity_strength < 0:
            raise ValueError("liquidity strength must be nonnegative")
        if self.future_cash_weight < 0:
            raise ValueError("future cash weight must be nonnegative")
        if not 0 <= self.objective_progress_weight <= 1:
            raise ValueError("objective progress weight must be between zero and one")
        if not 0 <= self.bid_shading <= 1:
            raise ValueError("bid shading must be between zero and one")


@dataclass(frozen=True, slots=True)
class HeuristicProfileSet:
    """One immutable generation of the three heuristic personalities."""

    version: str
    aggressive: HeuristicProfile
    balanced: HeuristicProfile
    passive: HeuristicProfile

    def __post_init__(self) -> None:
        if (
            len(self.version) < 2
            or self.version[0] != "v"
            or not self.version[1:].isdigit()
            or int(self.version[1:]) < 1
        ):
            raise ValueError("heuristic version must use canonical vN form")
        names = (
            self.aggressive.name,
            self.balanced.name,
            self.passive.name,
        )
        if names != ("aggressive", "balanced", "passive"):
            raise ValueError("profile set must contain canonical personalities")


HEURISTIC_V1 = HeuristicProfileSet(
    "v1",
    HeuristicProfile("aggressive", 0.75, 0.00, 0.25, 0.05),
    HeuristicProfile("balanced", 0.40, 0.00, 0.20, 0.25),
    HeuristicProfile("passive", 0.15, 0.00, 0.15, 0.50),
)
HEURISTIC_V2 = HeuristicProfileSet(
    "v2",
    HeuristicProfile("aggressive", 0.75, 1.50, 0.25, 0.05),
    HeuristicProfile("balanced", 0.40, 0.75, 0.20, 0.25),
    HeuristicProfile("passive", 0.15, 0.60, 0.15, 0.30),
)
HEURISTIC_V3 = HeuristicProfileSet(
    "v3",
    HeuristicProfile("aggressive", 1.40, 1.05, 0.70, 0.30),
    HeuristicProfile("balanced", 0.25, 1.55, 0.30, 0.35),
    HeuristicProfile("passive", 1.40, 1.10, 0.40, 0.30),
)
LATEST_HEURISTICS = HEURISTIC_V3

AGGRESSIVE_PROFILE = LATEST_HEURISTICS.aggressive
BALANCED_PROFILE = LATEST_HEURISTICS.balanced
PASSIVE_PROFILE = LATEST_HEURISTICS.passive
