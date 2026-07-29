from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pocketrocks import ActionId

from garboid_pocketrocks.simulator.model import Score, Seat


class EventKind(StrEnum):
    GAME_SETUP = "game_setup"
    TURN_OPENED = "turn_opened"
    DECISION_SUBMITTED = "decision_submitted"
    AUCTION_RESOLVED = "auction_resolved"
    RESOURCES_AWARDED = "resources_awarded"
    LOAN_ACQUIRED = "loan_acquired"
    INVESTMENT_ACQUIRED = "investment_acquired"
    OBJECTIVE_CLAIMED = "objective_claimed"
    INFORMATION_REVEALED = "information_revealed"
    GAME_ENDED = "game_ended"
    SCORES_CALCULATED = "scores_calculated"
    BOT_FAULT = "bot_fault"
    FALLBACK_APPLIED = "fallback_applied"


@dataclass(frozen=True, slots=True)
class GameEvent:
    kind: EventKind
    turn_index: int | None = None
    seat: Seat | None = None
    action_id: ActionId | None = None
    amount: int | None = None
    resource_ids: tuple[int, ...] | None = None
    objective_ids: tuple[int, ...] | None = None
    scores: tuple[Score, ...] | None = None
    automatic: bool | None = None
