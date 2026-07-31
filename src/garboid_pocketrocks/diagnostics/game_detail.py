"""Public, seed-free game detail used by diagnostic visualizations."""

from __future__ import annotations

from dataclasses import dataclass

from pocketrocks.sim import ScoreRow, TurnRecord


@dataclass(frozen=True, slots=True)
class PublicTurnDetail:
    """One resolved public auction without hidden cards or engine seeds."""

    turn_index: int
    action: str
    raw_bids: tuple[int, ...]
    effective_bids: tuple[int, ...]
    winner_seat: int
    paid: int
    bundle_suits: tuple[int, ...]
    claimed_objective_ids: tuple[int, ...]

    @classmethod
    def from_sdk(cls, turn: TurnRecord) -> PublicTurnDetail:
        return cls(
            turn_index=turn.turn_index,
            action=turn.action,
            raw_bids=tuple(turn.raw_bids),
            effective_bids=tuple(turn.effective_bids),
            winner_seat=turn.winner_seat,
            paid=turn.paid,
            bundle_suits=tuple(turn.bundle_suits),
            claimed_objective_ids=tuple(turn.claimed_objective_wire_ids),
        )


@dataclass(frozen=True, slots=True)
class PublicScoreDetail:
    """Public terminal score components for one seat."""

    seat: int
    cash: int
    items_value: int
    objectives_value: int
    investments_value: int
    loans_value: int
    total: int

    @classmethod
    def from_sdk(cls, row: ScoreRow) -> PublicScoreDetail:
        return cls(
            seat=row.seat,
            cash=row.cash,
            items_value=row.items_value,
            objectives_value=row.objectives_value,
            investments_value=row.investments_value,
            loans_value=row.loans_value,
            total=row.total,
        )


@dataclass(frozen=True, slots=True)
class PublicGameDetail:
    """The public turn ledger and score breakdown for one tournament game."""

    game_index: int
    chart: str
    player_count: int
    bot_names: tuple[str, ...]
    bot_ids: tuple[str, ...]
    value_chart: tuple[int, ...]
    turns: tuple[PublicTurnDetail, ...]
    scores: tuple[PublicScoreDetail, ...]
