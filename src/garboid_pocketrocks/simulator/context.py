from __future__ import annotations

from dataclasses import dataclass

from pocketrocks import ActionId, DecisionContext, Suit

from garboid_pocketrocks.simulator.model import (
    GameState,
    Phase,
    ResourceCard,
    Seat,
)


@dataclass(frozen=True, slots=True)
class DecisionBatch:
    phase: Phase
    contexts: tuple[tuple[Seat, DecisionContext], ...]

    @property
    def contexts_by_seat(self) -> dict[Seat, DecisionContext]:
        return dict(self.contexts)

    @property
    def acting_seats(self) -> tuple[Seat, ...]:
        return tuple(seat for seat, _ in self.contexts)


def _counts_by_suit(cards: tuple[ResourceCard, ...]) -> tuple[int, ...]:
    return tuple(sum(card.suit is suit for card in cards) for suit in Suit)


def _current_resources(state: GameState) -> tuple[int, int]:
    action = state.current_action
    if action is None or action.action_id not in (
        ActionId.AUCTION1,
        ActionId.AUCTION2,
    ):
        return (0, 0)
    count = 1 if action.action_id is ActionId.AUCTION1 else 2
    suit_ids = tuple(int(card.suit) for card in state.visible_resources[:count])
    first = suit_ids[0] if suit_ids else 0
    second = suit_ids[1] if len(suit_ids) > 1 else 0
    return (first, second)


def _legal_max(state: GameState, seat: Seat) -> int:
    action = state.current_action
    cash = state.players[seat].cash
    if action is None:
        return cash
    if action.action_id is ActionId.LOAN10:
        return cash + 10
    if action.action_id is ActionId.LOAN20:
        return cash + 20
    return cash


def _context_for(state: GameState, seat: Seat) -> DecisionContext:
    is_reveal = state.phase is Phase.REVEAL
    player = state.players[seat]
    return DecisionContext(
        request_id=f"sim-{state.seed}-{state.turn_index}-{state.phase.value}-{seat}",
        deadline_at=2**63 - 1,
        received_at=0,
        decision_kind="selectInfoToReveal" if is_reveal else "submitBid",
        player_count=state.player_count,
        starting_cash=state.ruleset.setup_for(state.player_count).starting_cash,
        value_chart=state.ruleset.value_chart,
        objective_ids=state.active_objective_ids,
        current_action_id=(
            int(state.current_action.action_id)
            if state.current_action is not None
            else None
        ),
        current_resource_ids=_current_resources(state),
        cash_by_seat=tuple(other.cash for other in state.players),
        tiebreak_seat=state.priority_seat,
        won_resource_counts_by_seat=tuple(
            _counts_by_suit(other.won_resources) for other in state.players
        ),
        revealed_info_counts_by_seat=tuple(
            _counts_by_suit(other.revealed_info) for other in state.players
        ),
        owned_objective_ids_by_seat=tuple(
            other.owned_objective_ids for other in state.players
        ),
        bot_seat=seat,
        current_hand_suit_ids=tuple(int(card.suit) for card in player.private_hand),
        legal_max_amount=None if is_reveal else _legal_max(state, seat),
        revealable_count=len(player.private_hand) if is_reveal else 0,
    )


def build_decision_batch(state: GameState) -> DecisionBatch:
    if state.phase is Phase.BIDDING:
        seats = tuple(range(state.player_count))
    elif state.phase is Phase.REVEAL and state.reveal_seat is not None:
        seats = (state.reveal_seat,)
    else:
        raise ValueError(f"state phase {state.phase.value!r} has no pending decisions")
    return DecisionBatch(
        phase=state.phase,
        contexts=tuple((seat, _context_for(state, seat)) for seat in seats),
    )
