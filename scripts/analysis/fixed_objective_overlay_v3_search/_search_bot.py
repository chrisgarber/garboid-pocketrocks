from __future__ import annotations

import os

from garboid_pocketrocks.bots.fixed_bid import FixedBidProfile
from garboid_pocketrocks.bots.fixed_objective_overlay import FixedObjectiveOverlayV3Brain


class SearchFixedObjectiveOverlayV3Brain(FixedObjectiveOverlayV3Brain):
    """Process-safe fixed-profile candidate configured by the search driver."""

    def __init__(self, seed: int | None = None) -> None:
        super().__init__(seed)
        raw = os.environ["GARBOID_FIXED_OBJECTIVE_OVERLAY_V3_SEARCH_VALUES"]
        values = tuple(int(value) for value in raw.split(","))
        self.PROFILE = FixedBidProfile(values)
