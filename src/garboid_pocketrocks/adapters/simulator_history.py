from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pocketrocks import ActionId, Suit

from garboid_pocketrocks.adapters.public_history import (
    PublicAuctionResolved,
    PublicEvent,
    PublicEventKind,
    PublicGameSetup,
    PublicHistory,
    PublicInformationRevealed,
    PublicTurnOpened,
)
from garboid_pocketrocks.simulator.engine import EngineTransition
from garboid_pocketrocks.simulator.events import EventKind, GameEvent


class SimulatorHistoryError(ValueError):
    """Raised when engine events cannot form canonical public history."""


@dataclass(slots=True)
class SimulatorPublicHistoryAdapter:
    _player_count: int
    _priority_seat: int
    _events: list[PublicEvent]

    @classmethod
    def from_initial_transition(
        cls,
        transition: EngineTransition,
    ) -> SimulatorPublicHistoryAdapter:
        state = transition.state
        setup_events = tuple(
            event for event in transition.events if event.kind is EventKind.GAME_SETUP
        )
        if len(setup_events) != 1:
            raise SimulatorHistoryError("initial transition must contain one game setup event")
        setup = PublicGameSetup(
            kind=PublicEventKind.GAME_SETUP,
            player_count=state.player_count,
            starting_cash=state.ruleset.setup_for(state.player_count).starting_cash,
            value_chart=state.ruleset.value_chart,
            initial_tiebreak_seat=state.priority_seat,
            objective_ids=state.active_objective_ids,
        )
        adapter = cls(
            _player_count=state.player_count,
            _priority_seat=state.priority_seat,
            _events=[setup],
        )
        adapter.append(transition.events)
        return adapter

    @property
    def history(self) -> PublicHistory:
        return tuple(self._events)

    def append(self, events: Sequence[GameEvent]) -> None:
        batch = tuple(events)
        resolutions = tuple(event for event in batch if event.kind is EventKind.AUCTION_RESOLVED)
        if len(resolutions) > 1:
            raise SimulatorHistoryError(
                "transition batch must not contain multiple auction resolutions"
            )
        if resolutions:
            self._append_resolved_bids(batch, resolutions[0])

        for event in batch:
            if event.kind is EventKind.INFORMATION_REVEALED:
                self._append_reveal(event)
            elif event.kind is EventKind.TURN_OPENED:
                self._append_turn(event)

    def _append_resolved_bids(
        self,
        batch: tuple[GameEvent, ...],
        resolution: GameEvent,
    ) -> None:
        bids_by_seat: dict[int, int] = {}
        for event in batch:
            if event.kind is not EventKind.DECISION_SUBMITTED:
                continue
            seat = event.seat
            if seat is None or not 0 <= seat < self._player_count:
                raise SimulatorHistoryError("submitted bid seat is outside player count")
            if seat in bids_by_seat:
                raise SimulatorHistoryError(f"duplicate submitted bid for seat {seat}")
            amount = 0 if event.amount is None else event.amount
            if amount < 0:
                raise SimulatorHistoryError("submitted bids must be nonnegative")
            bids_by_seat[seat] = amount

        expected_seats = set(range(self._player_count))
        if set(bids_by_seat) != expected_seats:
            raise SimulatorHistoryError(
                "auction resolution requires exactly one submitted bid per seat"
            )
        winner = resolution.seat
        if winner is None or not 0 <= winner < self._player_count:
            raise SimulatorHistoryError("auction winner seat is outside player count")
        self._events.append(
            PublicAuctionResolved(
                kind=PublicEventKind.AUCTION_RESOLVED,
                bids_by_seat=tuple(bids_by_seat[seat] for seat in range(self._player_count)),
            )
        )
        self._priority_seat = winner

    def _append_reveal(self, event: GameEvent) -> None:
        seat = event.seat
        if seat is None or not 0 <= seat < self._player_count:
            raise SimulatorHistoryError("revealed information seat is outside player count")
        resource_ids = event.resource_ids
        if resource_ids is None or len(resource_ids) != 1:
            raise SimulatorHistoryError("revealed information event must contain exactly one suit")
        suit_id = resource_ids[0]
        if not 1 <= suit_id <= len(Suit):
            raise SimulatorHistoryError("revealed information suit is unknown")
        self._events.append(
            PublicInformationRevealed(
                kind=PublicEventKind.INFORMATION_REVEALED,
                seat=seat,
                suit_id=suit_id,
            )
        )

    def _append_turn(self, event: GameEvent) -> None:
        action = event.action_id
        if action is None:
            raise SimulatorHistoryError("turn-opened event requires an action ID")
        resources = event.resource_ids or ()
        if action is ActionId.AUCTION1:
            if not resources:
                raise SimulatorHistoryError("one-card auction requires an offered resource")
            offered = (resources[0], 0)
        elif action is ActionId.AUCTION2:
            if not resources:
                raise SimulatorHistoryError("two-card auction requires an offered resource")
            offered = (
                resources[0],
                resources[1] if len(resources) > 1 else 0,
            )
        else:
            offered = (0, 0)
        if any(not 0 <= resource_id <= len(Suit) for resource_id in offered):
            raise SimulatorHistoryError("turn-opened resource ID is outside the known range")
        self._events.append(
            PublicTurnOpened(
                kind=PublicEventKind.TURN_OPENED,
                action_id=int(action),
                resource_ids=offered,
            )
        )
