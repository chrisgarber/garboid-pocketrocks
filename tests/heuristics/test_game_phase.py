from __future__ import annotations

import pytest

from garboid_pocketrocks.heuristics.game_phase import game_phase_for_turn_index


@pytest.mark.parametrize(
    ("turn_index", "expected"),
    ((0, "early"), (4, "early"), (5, "middle"), (11, "middle"), (12, "late")),
)
def test_public_game_phase_uses_documented_turn_boundaries(
    turn_index: int,
    expected: str,
) -> None:
    assert game_phase_for_turn_index(turn_index) == expected


@pytest.mark.parametrize("turn_index", (-1, True, 1.5))
def test_public_game_phase_rejects_invalid_turn_indexes(turn_index: object) -> None:
    with pytest.raises(ValueError, match="nonnegative integer"):
        game_phase_for_turn_index(turn_index)  # type: ignore[arg-type]
