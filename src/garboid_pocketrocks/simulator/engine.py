from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from pocketrocks import OBJECTIVES, ActionId, BotDecision, Suit
from pocketrocks.exceptions import InvalidBotDecision

from garboid_pocketrocks.rules import Ruleset
from garboid_pocketrocks.simulator.context import (
    DecisionBatch,
    build_decision_batch,
)
from garboid_pocketrocks.simulator.errors import (
    ActingSeatsError,
    ActionDeckExhaustedError,
    IllegalDecisionError,
    InvalidPhaseError,
)
from garboid_pocketrocks.simulator.events import EventKind, GameEvent
from garboid_pocketrocks.simulator.model import (
    GameResult,
    GameState,
    InvestmentPosition,
    LoanPosition,
    Phase,
    PlayerState,
    ResourceCard,
    Score,
    Seat,
    offered_resource_ids,
)
from garboid_pocketrocks.simulator.setup import build_setup


@dataclass(frozen=True, slots=True)
class EngineTransition:
    state: GameState
    events: tuple[GameEvent, ...]
    pending: DecisionBatch | None
    result: GameResult | None

    @property
    def terminated(self) -> bool:
        return self.result is not None


class GameEngine:
    @staticmethod
    def start(
        ruleset: Ruleset,
        *,
        player_count: int,
        seed: int,
    ) -> EngineTransition:
        setup = build_setup(ruleset, player_count=player_count, seed=seed)
        return EngineTransition(
            state=setup.state,
            events=setup.events,
            pending=build_decision_batch(setup.state),
            result=None,
        )

    @staticmethod
    def resume(state: GameState) -> EngineTransition:
        if state.phase is Phase.TERMINAL:
            raise InvalidPhaseError("terminal state has no pending decision")
        return EngineTransition(
            state=state,
            events=(),
            pending=build_decision_batch(state),
            result=None,
        )

    @staticmethod
    def step(
        state: GameState,
        decisions_by_seat: Mapping[Seat, BotDecision],
    ) -> EngineTransition:
        if state.phase is Phase.BIDDING:
            return _resolve_bidding(state, decisions_by_seat)
        if state.phase is Phase.REVEAL:
            return _resolve_reveal(state, decisions_by_seat)
        raise InvalidPhaseError("cannot step terminal phase")


def _validate_decisions(
    state: GameState,
    decisions_by_seat: Mapping[Seat, BotDecision],
) -> DecisionBatch:
    batch = build_decision_batch(state)
    actual = set(decisions_by_seat)
    expected = set(batch.acting_seats)
    if actual != expected:
        raise ActingSeatsError(
            f"phase={state.phase.value} expected seats={sorted(expected)} "
            f"received seats={sorted(actual)}"
        )
    for seat, context in batch.contexts:
        decision = decisions_by_seat[seat]
        try:
            context.validate(decision)
        except InvalidBotDecision as error:
            raise IllegalDecisionError(
                f"phase={state.phase.value} seat={seat} decision={decision}: {error}"
            ) from error
    return batch


def _bid_amount(decision: BotDecision) -> int:
    if decision.action_kind == "submitBid":
        assert decision.value is not None
        return decision.value
    return 0


def _winning_seat(
    bids: tuple[int, ...],
    *,
    priority_seat: Seat,
) -> Seat:
    highest = max(bids)
    for offset in range(1, len(bids) + 1):
        seat = (priority_seat + offset) % len(bids)
        if bids[seat] == highest:
            return seat
    raise AssertionError("at least one seat must have the maximum bid")


def _decision_events(
    state: GameState,
    decisions_by_seat: Mapping[Seat, BotDecision],
) -> tuple[GameEvent, ...]:
    return tuple(
        GameEvent(
            EventKind.DECISION_SUBMITTED,
            turn_index=state.turn_index,
            seat=seat,
            action_id=(
                state.current_action.action_id
                if state.current_action is not None
                else None
            ),
            amount=decision.value,
        )
        for seat, decision in sorted(decisions_by_seat.items())
    )


def _resolve_bidding(
    state: GameState,
    decisions_by_seat: Mapping[Seat, BotDecision],
) -> EngineTransition:
    _validate_decisions(state, decisions_by_seat)
    if state.current_action is None:
        raise InvalidPhaseError("bidding phase requires a current action")
    bids = tuple(_bid_amount(decisions_by_seat[seat]) for seat in range(state.player_count))
    winner = _winning_seat(bids, priority_seat=state.priority_seat)
    winning_bid = bids[winner]
    action_id = state.current_action.action_id
    events = [
        *_decision_events(state, decisions_by_seat),
        GameEvent(
            EventKind.AUCTION_RESOLVED,
            turn_index=state.turn_index,
            seat=winner,
            action_id=action_id,
            amount=winning_bid,
        ),
    ]

    players = list(state.players)
    winner_state = players[winner]
    visible_resources = state.visible_resources
    awarded: tuple[ResourceCard, ...] = ()

    if action_id in (ActionId.AUCTION1, ActionId.AUCTION2):
        award_count = 1 if action_id is ActionId.AUCTION1 else 2
        awarded = visible_resources[:award_count]
        visible_resources = visible_resources[len(awarded) :]
        winner_state = replace(
            winner_state,
            cash=winner_state.cash - winning_bid,
            won_resources=(*winner_state.won_resources, *awarded),
        )
        events.append(
            GameEvent(
                EventKind.RESOURCES_AWARDED,
                turn_index=state.turn_index,
                seat=winner,
                action_id=action_id,
                amount=winning_bid,
                resource_ids=tuple(int(card.suit) for card in awarded),
            )
        )
    elif action_id in (ActionId.LOAN10, ActionId.LOAN20):
        principal = 10 if action_id is ActionId.LOAN10 else 20
        winner_state = replace(
            winner_state,
            cash=winner_state.cash - winning_bid + principal,
            loans=(
                *winner_state.loans,
                LoanPosition(principal=principal, winning_bid=winning_bid),
            ),
        )
        events.append(
            GameEvent(
                EventKind.LOAN_ACQUIRED,
                turn_index=state.turn_index,
                seat=winner,
                action_id=action_id,
                amount=principal,
            )
        )
    else:
        payout = 5 if action_id is ActionId.INVEST5 else 10
        winner_state = replace(
            winner_state,
            cash=winner_state.cash - winning_bid,
            investments=(
                *winner_state.investments,
                InvestmentPosition(locked=winning_bid, payout=payout),
            ),
        )
        events.append(
            GameEvent(
                EventKind.INVESTMENT_ACQUIRED,
                turn_index=state.turn_index,
                seat=winner,
                action_id=action_id,
                amount=payout,
            )
        )

    players[winner] = winner_state
    if awarded:
        players, objective_events = _claim_objectives(state, tuple(players), winner)
        events.extend(objective_events)

    winner_state = players[winner]
    next_state = replace(
        state,
        players=tuple(players),
        visible_resources=visible_resources,
        priority_seat=winner,
        phase=Phase.REVEAL if winner_state.private_hand else Phase.BIDDING,
        reveal_seat=winner if winner_state.private_hand else None,
        current_resource_ids=offered_resource_ids(action_id, state.visible_resources),
    )
    if winner_state.private_hand:
        return EngineTransition(
            state=next_state,
            events=tuple(events),
            pending=build_decision_batch(next_state),
            result=None,
        )
    return _advance_turn(next_state, events)


def _claim_objectives(
    old_state: GameState,
    players: tuple[PlayerState, ...],
    winner: Seat,
) -> tuple[list[PlayerState], tuple[GameEvent, ...]]:
    mutable_players = list(players)
    winner_state = mutable_players[winner]
    counts = [0] * len(Suit)
    for card in winner_state.won_resources:
        counts[int(card.suit) - 1] += 1
    owned = {
        objective_id
        for player in old_state.players
        for objective_id in player.owned_objective_ids
    }
    claimed = tuple(
        objective_id
        for objective_id in old_state.active_objective_ids
        if objective_id not in owned and _objective_is_met(objective_id, counts)
    )
    if not claimed:
        return mutable_players, ()
    mutable_players[winner] = replace(
        winner_state,
        owned_objective_ids=(*winner_state.owned_objective_ids, *claimed),
    )
    return mutable_players, tuple(
        GameEvent(
            EventKind.OBJECTIVE_CLAIMED,
            turn_index=old_state.turn_index,
            seat=winner,
            objective_ids=(objective_id,),
        )
        for objective_id in claimed
    )


def _objective_is_met(objective_id: int, counts: list[int]) -> bool:
    objective = OBJECTIVES[objective_id]
    if objective.pattern == "same2":
        return any(count >= 2 for count in counts)
    if objective.pattern == "same3":
        return any(count >= 3 for count in counts)
    if objective.pattern == "different3":
        return sum(count > 0 for count in counts) >= 3
    if objective.pattern == "different4":
        return sum(count > 0 for count in counts) >= 4
    if objective.pattern == "twoPairs4":
        return sum(count >= 2 for count in counts) >= 2
    assert objective.requirement is not None
    return all(
        count >= required
        for count, required in zip(counts, objective.requirement, strict=True)
    )


def _resolve_reveal(
    state: GameState,
    decisions_by_seat: Mapping[Seat, BotDecision],
) -> EngineTransition:
    _validate_decisions(state, decisions_by_seat)
    if state.reveal_seat is None:
        raise InvalidPhaseError("reveal phase requires a reveal seat")
    seat = state.reveal_seat
    decision = decisions_by_seat[seat]
    automatic = decision.action_kind == "pass"
    index = 0 if automatic else decision.value
    assert index is not None
    player = state.players[seat]
    revealed_card = player.private_hand[index]
    players = list(state.players)
    players[seat] = replace(
        player,
        private_hand=player.private_hand[:index] + player.private_hand[index + 1 :],
        revealed_info=(*player.revealed_info, revealed_card),
    )
    next_state = replace(
        state,
        players=tuple(players),
        phase=Phase.BIDDING,
        reveal_seat=None,
    )
    events = [
        *_decision_events(state, decisions_by_seat),
        GameEvent(
            EventKind.INFORMATION_REVEALED,
            turn_index=state.turn_index,
            seat=seat,
            resource_ids=(int(revealed_card.suit),),
            automatic=automatic,
        ),
    ]
    return _advance_turn(next_state, events)


def _advance_turn(
    state: GameState,
    events: list[GameEvent],
) -> EngineTransition:
    visible = list(state.visible_resources)
    deck = list(state.resource_deck)
    while len(visible) < 2 and deck:
        visible.append(deck.pop(0))
    next_turn = state.turn_index + 1
    advanced = replace(
        state,
        turn_index=next_turn,
        resource_deck=tuple(deck),
        visible_resources=tuple(visible),
    )
    if not visible and not deck:
        return _finish_game(advanced, events)
    if not state.action_deck:
        raise ActionDeckExhaustedError(
            f"turn={next_turn} resources remain but action deck is exhausted"
        )
    current_action = state.action_deck[0]
    advanced = replace(
        advanced,
        phase=Phase.BIDDING,
        current_action=current_action,
        action_deck=state.action_deck[1:],
        reveal_seat=None,
        current_resource_ids=offered_resource_ids(
            current_action.action_id,
            advanced.visible_resources,
        ),
    )
    events.append(
        GameEvent(
            EventKind.TURN_OPENED,
            turn_index=next_turn,
            action_id=current_action.action_id,
            resource_ids=tuple(int(card.suit) for card in advanced.visible_resources),
        )
    )
    return EngineTransition(
        state=advanced,
        events=tuple(events),
        pending=build_decision_batch(advanced),
        result=None,
    )


def _finish_game(
    state: GameState,
    events: list[GameEvent],
) -> EngineTransition:
    players: list[PlayerState] = []
    for player in state.players:
        players.append(
            replace(
                player,
                private_hand=(),
                revealed_info=(*player.revealed_info, *player.private_hand),
            )
        )
        events.extend(
            GameEvent(
                EventKind.INFORMATION_REVEALED,
                turn_index=state.turn_index,
                seat=player.seat,
                resource_ids=(int(card.suit),),
                automatic=True,
            )
            for card in player.private_hand
        )
    terminal_state = replace(
        state,
        phase=Phase.TERMINAL,
        players=tuple(players),
        current_action=None,
        reveal_seat=None,
        current_resource_ids=(0, 0),
    )
    result = _score(terminal_state)
    events.extend(
        (
            GameEvent(EventKind.GAME_ENDED, turn_index=state.turn_index),
            GameEvent(
                EventKind.SCORES_CALCULATED,
                turn_index=state.turn_index,
                scores=result.scores,
            ),
        )
    )
    return EngineTransition(
        state=terminal_state,
        events=tuple(events),
        pending=None,
        result=result,
    )


def _score(state: GameState) -> GameResult:
    revealed_by_suit = tuple(
        sum(card.suit is suit for player in state.players for card in player.revealed_info)
        for suit in Suit
    )
    money_by_seat: list[int] = []
    for player in state.players:
        owned_by_suit = tuple(
            sum(card.suit is suit for card in player.won_resources) for suit in Suit
        )
        resource_value = sum(
            count
            * state.ruleset.value_chart[min(revealed_by_suit[int(suit) - 1], 5)]
            for suit, count in zip(Suit, owned_by_suit, strict=True)
        )
        objective_value = sum(
            OBJECTIVES[objective_id].payout
            for objective_id in player.owned_objective_ids
        )
        final_money = (
            player.cash
            + resource_value
            + objective_value
            + sum(
                investment.locked + investment.payout
                for investment in player.investments
            )
            - sum(loan.principal for loan in player.loans)
        )
        money_by_seat.append(final_money)
    return GameResult(
        scores=tuple(
            Score(
                seat=seat,
                final_money=money,
                rank=1 + sum(other > money for other in money_by_seat),
            )
            for seat, money in enumerate(money_by_seat)
        )
    )
