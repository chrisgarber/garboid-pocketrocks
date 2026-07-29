from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EnvironmentBounds:
    """Fixed action and observation limits shared by an environment family."""

    max_bid: int
    max_hand_size: int

    def __post_init__(self) -> None:
        if self.max_bid < 0 or self.max_hand_size < 0:
            raise ValueError("environment bounds must be nonnegative")
