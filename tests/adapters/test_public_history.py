from __future__ import annotations

from collections.abc import Callable
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
    PublicHistory,
    PublicHistoryCompatibilityError,
    PublicInformationRevealed,
    PublicTurnOpened,
    public_history_from_sdk_frame,
    validate_public_history,
)
from garboid_pocketrocks.simulator.session import SdkGameSession


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
    with pytest.raises(FrozenInstanceError):
        history[0].starting_cash = 99  # type: ignore[misc]


@pytest.mark.parametrize(
    ("initial_tiebreak_seat", "bids", "expected_winner"),
    ((0, (5, 5, 1), 1), (1, (5, 5, 1), 0), (2, (5, 5, 1), 0)),
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

    assert cast(PublicInformationRevealed, history[3]).seat == expected_winner


@pytest.mark.parametrize(
    "frame",
    (
        object(),
        SimpleNamespace(common_events=()),
        SimpleNamespace(common_events=(SimpleNamespace(kind="gameSetup", player_count=3),)),
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
    ),
)
def test_sdk_adapter_fails_closed_on_missing_or_malformed_history(frame: object) -> None:
    with pytest.raises(PublicHistoryCompatibilityError):
        public_history_from_sdk_frame(frame)


def test_production_adapter_does_not_import_sdk_internals() -> None:
    import garboid_pocketrocks.adapters.public_history as public_history_module

    source = Path(public_history_module.__file__).read_text(encoding="utf-8")

    assert "pocketrocks.internal" not in source


def test_shared_validator_derives_the_exact_open_turn_state() -> None:
    history = public_history_from_sdk_frame(_decoded_history_frame())

    state = validate_public_history(history)

    assert state.phase == "turn_open"
    assert state.latest_turn == history[-1]
    assert state.tiebreak_seat == 1


@pytest.mark.parametrize(
    "mutate",
    (
        lambda history: (
            history[0],
            replace(history[1], kind=PublicEventKind.GAME_SETUP),
            *history[2:],
        ),
        lambda history: (
            history[0],
            replace(history[1], resource_ids=(1,)),
            *history[2:],
        ),
        lambda history: (
            *history[:2],
            replace(history[2], bids_by_seat=(2, 5)),
            *history[3:],
        ),
        lambda history: (
            *history[:3],
            replace(history[3], seat=2),
            *history[4:],
        ),
        lambda history: (
            *history[:3],
            history[4],
        ),
    ),
)
def test_shared_validator_rejects_malformed_events_and_state_transitions(
    mutate: Callable[[PublicHistory], PublicHistory],
) -> None:
    history = public_history_from_sdk_frame(_decoded_history_frame())
    malformed = mutate(history)

    with pytest.raises(PublicHistoryCompatibilityError):
        validate_public_history(malformed)


@pytest.mark.parametrize("player_count", (3, 4, 5))
def test_shared_validator_accepts_complete_sdk_games_after_players_exhaust_their_hands(
    player_count: int,
) -> None:
    session = SdkGameSession.start(player_count=player_count, seed=104_729)

    while not session.terminated:
        decisions = {
            seat: (
                BotDecision.submit_bid(1)
                if session.pending.decision_kind == "submitBid" and seat == 0
                else BotDecision.pass_turn()
            )
            for seat in session.pending.acting_seats
        }
        session.step(decisions)

    history = public_history_from_sdk_frame(SimpleNamespace(common_events=session.events))

    assert validate_public_history(history).setup.player_count == player_count
    assert any(
        isinstance(current, PublicAuctionResolved) and isinstance(following, PublicTurnOpened)
        for current, following in zip(history, history[1:], strict=False)
    )
