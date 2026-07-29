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


AGGRESSIVE_PROFILE = HeuristicProfile("aggressive", 0.75, 1.50, 0.25, 0.05)
BALANCED_PROFILE = HeuristicProfile("balanced", 0.40, 0.75, 0.20, 0.25)
PASSIVE_PROFILE = HeuristicProfile("passive", 0.15, 0.60, 0.15, 0.30)
