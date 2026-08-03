"""Name the public game phase from the zero-based turn number."""

from __future__ import annotations

from typing import Literal

type GamePhase = Literal["early", "middle", "late"]


def game_phase_for_turn_index(turn_index: int) -> GamePhase:
    """Apply the turn boundaries used by public decision reports."""

    if not isinstance(turn_index, int) or isinstance(turn_index, bool) or turn_index < 0:
        raise ValueError("turn index must be a nonnegative integer")
    one_based_turn = turn_index + 1
    if one_based_turn <= 5:
        return "early"
    if one_based_turn <= 12:
        return "middle"
    return "late"
