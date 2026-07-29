from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from pocketrocks import ActionId, BotDecision, Suit
from pocketrocks.internal.bot_wire_v2 import decode_frame
from pocketrocks.testing import scenario

from garboid_pocketrocks.adapters.public_history import (
    PublicAuctionResolved,
    PublicEventKind,
    PublicGameSetup,
    PublicHistoryCompatibilityError,
    PublicInformationRevealed,
    PublicTurnOpened,
    public_history_from_sdk_frame,
)
from garboid_pocketrocks.adapters.simulator_history import (
    SimulatorHistoryError,
    SimulatorPublicHistoryAdapter,
)
from garboid_pocketrocks.rules import LIVE_RULESET
from garboid_pocketrocks.simulator.engine import GameEngine
from garboid_pocketrocks.simulator.events import EventKind, GameEvent
from garboid_pocketrocks.simulator.model import Phase


def _decoded_history_frame(
    *,
    initial_tiebreak_seat: int = 0,
    bids: tuple[int, int, int] = (2, 5, 1),
) -> object:
    return decode_frame(
        scenario(
            players=3,
            starting_cash=30,
            initial_tiebreak_seat=initial_tiebreak_seat,
            objective_ids=(1, 2, 3, 4),
        )
        .turn(ActionId.AUCTION1, resources=(Suit.BRICK, 0))
        .auction(bids)
        .reveal(Suit.ORE)
        .turn(ActionId.LOAN10)
        .deciding(seat=2, hand=(Suit.WOOD,))
        .to_bytes(deadline_at=1_000)
    )


def test_sdk_frame_becomes_immutable_public_history() -> None:
    history = public_history_from_sdk_frame(_decoded_history_frame())

    assert history == (
        PublicGameSetup(
            kind=PublicEventKind.GAME_SETUP,
            player_count=3,
            starting_cash=30,
            value_chart=(0, 4, 8, 12, 16, 20),
            initial_tiebreak_seat=0,
            objective_ids=(1, 2, 3, 4),
        ),
        PublicTurnOpened(
            kind=PublicEventKind.TURN_OPENED,
            action_id=int(ActionId.AUCTION1),
            resource_ids=(int(Suit.BRICK), 0),
        ),
        PublicAuctionResolved(
            kind=PublicEventKind.AUCTION_RESOLVED,
            bids_by_seat=(2, 5, 1),
        ),
        PublicInformationRevealed(
            kind=PublicEventKind.INFORMATION_REVEALED,
            seat=1,
            suit_id=int(Suit.ORE),
        ),
        PublicTurnOpened(
            kind=PublicEventKind.TURN_OPENED,
            action_id=int(ActionId.LOAN10),
            resource_ids=(0, 0),
        ),
    )
    assert isinstance(history, tuple)
    setup = history[0]
    with pytest.raises(FrozenInstanceError):
        setup.starting_cash = 99  # type: ignore[misc]


@pytest.mark.parametrize(
    ("initial_tiebreak_seat", "bids", "expected_winner"),
    (
        (0, (5, 5, 1), 1),
        (1, (5, 5, 1), 0),
        (2, (5, 5, 1), 0),
    ),
)
def test_sdk_adapter_derives_reveal_seat_using_public_tiebreak_order(
    initial_tiebreak_seat: int,
    bids: tuple[int, int, int],
    expected_winner: int,
) -> None:
    history = public_history_from_sdk_frame(
        _decoded_history_frame(
            initial_tiebreak_seat=initial_tiebreak_seat,
            bids=bids,
        )
    )

    revealed = cast(PublicInformationRevealed, history[3])
    assert revealed.seat == expected_winner


@pytest.mark.parametrize(
    "frame",
    (
        object(),
        SimpleNamespace(common_events=()),
        SimpleNamespace(
            common_events=(
                SimpleNamespace(
                    kind="gameSetup",
                    player_count=3,
                ),
            )
        ),
        SimpleNamespace(
            common_events=(
                SimpleNamespace(
                    kind="turnOpened",
                    action_id=1,
                    resource_ids=(1, 0),
                ),
            )
        ),
        SimpleNamespace(
            common_events=(
                SimpleNamespace(
                    kind="gameSetup",
                    player_count=3,
                    starting_cash=30,
                    value_chart=(0, 4, 8, 12, 16, 20),
                    initial_tiebreak_seat=0,
                    objective_ids=(1, 2, 3, 4),
                ),
                SimpleNamespace(kind="newServerEvent"),
            )
        ),
        SimpleNamespace(
            common_events=(
                SimpleNamespace(
                    kind="gameSetup",
                    player_count=3,
                    starting_cash=30,
                    value_chart=(0, 4, 8, 12, 16, 20),
                    initial_tiebreak_seat=0,
                    objective_ids=(1, 2, 3, 4),
                ),
                SimpleNamespace(kind="infoRevealed", suit_id=1),
            )
        ),
        SimpleNamespace(
            common_events=(
                SimpleNamespace(
                    kind="gameSetup",
                    player_count=3,
                    starting_cash=30,
                    value_chart=(0, 4, 8, 12, 16, 20),
                    initial_tiebreak_seat=0,
                    objective_ids=(1, 2, 3, 4),
                ),
                SimpleNamespace(
                    kind="turnOpened",
                    action_id=1,
                    resource_ids=(1, 0),
                ),
                SimpleNamespace(
                    kind="auctionResolved",
                    bids_by_seat=(1, 2),
                ),
            )
        ),
    ),
)
def test_sdk_adapter_fails_closed_on_missing_or_malformed_history(
    frame: object,
) -> None:
    with pytest.raises(PublicHistoryCompatibilityError):
        public_history_from_sdk_frame(frame)


def test_production_adapter_does_not_import_sdk_internals() -> None:
    import garboid_pocketrocks.adapters.public_history as public_history_module

    source = Path(public_history_module.__file__).read_text(encoding="utf-8")

    assert "pocketrocks.internal" not in source


def test_simulator_history_matches_equivalent_sdk_history() -> None:
    started = GameEngine.start(LIVE_RULESET, player_count=3, seed=60)
    assert started.state.current_action is not None
    assert started.pending is not None
    adapter = SimulatorPublicHistoryAdapter.from_initial_transition(started)
    bids = {
        0: BotDecision.submit_bid(2),
        1: BotDecision.submit_bid(5),
        2: BotDecision.submit_bid(1),
    }
    auction = GameEngine.step(started.state, bids)
    adapter.append(auction.events)
    assert auction.pending is not None
    winner = auction.pending.acting_seats[0]
    reveal_context = auction.pending.contexts_by_seat[winner]
    revealed_suit = reveal_context.current_hand_suit_ids[0]
    revealed = GameEngine.step(
        auction.state,
        {winner: BotDecision.select_info_to_reveal(0)},
    )
    adapter.append(revealed.events)
    assert revealed.pending is not None
    next_context = revealed.pending.contexts[0][1]

    frame = decode_frame(
        scenario(
            players=3,
            starting_cash=30,
            value_chart=cast(
                tuple[int, int, int, int, int, int],
                LIVE_RULESET.value_chart,
            ),
            initial_tiebreak_seat=started.state.priority_seat,
            objective_ids=started.state.active_objective_ids,
        )
        .turn(
            started.state.current_action.action_id,
            resources=started.pending.contexts[0][1].current_resource_ids,
        )
        .auction((2, 5, 1))
        .reveal(revealed_suit)
        .turn(
            cast(int, next_context.current_action_id),
            resources=next_context.current_resource_ids,
        )
        .deciding(
            seat=next_context.bot_seat,
            hand=next_context.current_hand_suit_ids,
        )
        .to_bytes(deadline_at=1_000)
    )

    assert adapter.history == public_history_from_sdk_frame(frame)


@pytest.mark.parametrize(
    ("reveal_decision", "revealed_index"),
    (
        (BotDecision.select_info_to_reveal(1), 1),
        (BotDecision.pass_turn(), 0),
    ),
)
def test_simulator_classifies_bid_and_reveal_transition_batches_separately(
    reveal_decision: BotDecision,
    revealed_index: int,
) -> None:
    started = GameEngine.start(LIVE_RULESET, player_count=3, seed=60)
    adapter = SimulatorPublicHistoryAdapter.from_initial_transition(started)
    auction = GameEngine.step(
        started.state,
        {
            0: BotDecision.submit_bid(5),
            1: BotDecision.pass_turn(),
            2: BotDecision.submit_bid(1),
        },
    )

    adapter.append(auction.events)

    resolved = cast(PublicAuctionResolved, adapter.history[-1])
    assert resolved.bids_by_seat == (5, 0, 1)
    assert auction.pending is not None
    assert auction.pending.acting_seats == (0,)
    expected_suit = int(auction.state.players[0].private_hand[revealed_index].suit)
    before_reveal = adapter.history
    reveal = GameEngine.step(
        auction.state,
        {0: reveal_decision},
    )

    adapter.append(reveal.events)

    assert adapter.history[: len(before_reveal)] == before_reveal
    added = adapter.history[len(before_reveal) :]
    information = tuple(
        event for event in added if event.kind is PublicEventKind.INFORMATION_REVEALED
    )
    assert information == (
        PublicInformationRevealed(
            kind=PublicEventKind.INFORMATION_REVEALED,
            seat=0,
            suit_id=expected_suit,
        ),
    )
    assert not any(event.kind is PublicEventKind.AUCTION_RESOLVED for event in added)
    assert sum(event.kind is PublicEventKind.AUCTION_RESOLVED for event in adapter.history) == 1
    assert adapter.history[-1].kind is PublicEventKind.TURN_OPENED


def test_terminal_automatic_reveal_batch_keeps_public_seats_and_suits() -> None:
    started = GameEngine.start(LIVE_RULESET, player_count=3, seed=60)
    adapter = SimulatorPublicHistoryAdapter.from_initial_transition(started)
    terminal_reveal_state = replace(
        started.state,
        phase=Phase.REVEAL,
        priority_seat=0,
        reveal_seat=0,
        visible_resources=(),
        resource_deck=(),
    )

    terminal = GameEngine.step(
        terminal_reveal_state,
        {0: BotDecision.pass_turn()},
    )
    expected = tuple(
        (event.seat, cast(tuple[int, ...], event.resource_ids)[0])
        for event in terminal.events
        if event.kind is EventKind.INFORMATION_REVEALED
    )
    before = adapter.history

    adapter.append(terminal.events)

    assert terminal.result is not None
    added = adapter.history[len(before) :]
    actual = tuple(
        (event.seat, event.suit_id)
        for event in added
        if isinstance(event, PublicInformationRevealed)
    )
    assert actual == expected
    assert len(actual) == sum(len(player.private_hand) for player in terminal_reveal_state.players)
    assert not any(event.kind is PublicEventKind.AUCTION_RESOLVED for event in added)


@pytest.mark.parametrize(
    "events",
    (
        (
            GameEvent(EventKind.DECISION_SUBMITTED, seat=0, amount=2),
            GameEvent(EventKind.DECISION_SUBMITTED, seat=1, amount=5),
            GameEvent(EventKind.AUCTION_RESOLVED, seat=1, amount=5),
        ),
        (
            GameEvent(EventKind.DECISION_SUBMITTED, seat=0, amount=2),
            GameEvent(EventKind.DECISION_SUBMITTED, seat=0, amount=5),
            GameEvent(EventKind.DECISION_SUBMITTED, seat=2, amount=1),
            GameEvent(EventKind.AUCTION_RESOLVED, seat=0, amount=5),
        ),
    ),
)
def test_simulator_rejects_missing_or_duplicate_bid_seats(
    events: tuple[GameEvent, ...],
) -> None:
    started = GameEngine.start(LIVE_RULESET, player_count=3, seed=60)
    adapter = SimulatorPublicHistoryAdapter.from_initial_transition(started)

    with pytest.raises(SimulatorHistoryError, match="bid"):
        adapter.append(events)
