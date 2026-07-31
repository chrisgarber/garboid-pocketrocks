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
        .turn(ActionId.AUCTION1, resources=(Suit.BRICK, Suit.WOOD))
        .auction(bids)
        .reveal(Suit.ORE)
        .turn(ActionId.LOAN10, resources=(Suit.WOOD, Suit.ORE))
        .deciding(
            seat=2,
            hand=(Suit.BRICK, Suit.WOOD, Suit.ORE, Suit.SHEEP, Suit.WHEAT),
        )
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
            resource_ids=(int(Suit.BRICK), int(Suit.WOOD)),
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
            resource_ids=(int(Suit.WOOD), int(Suit.ORE)),
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
    state = validate_public_history(history)

    assert state.setup.player_count == player_count
    assert state.biddable_resource_budget == {3: 15, 4: 14, 5: 15}[player_count]
    assert state.public_biddable_resource_count == state.biddable_resource_budget
    assert any(
        isinstance(current, PublicAuctionResolved) and isinstance(following, PublicTurnOpened)
        for current, following in zip(history, history[1:], strict=False)
    )


@pytest.mark.parametrize(
    ("player_count", "starting_cash", "expected_budget"), ((4, 25, 14), (5, 20, 15))
)
def test_shared_validator_derives_the_sdk_biddable_resource_budget(
    player_count: int,
    starting_cash: int,
    expected_budget: int,
) -> None:
    history = (
        PublicGameSetup(
            PublicEventKind.GAME_SETUP,
            player_count,
            starting_cash,
            (0, 4, 8, 12, 16, 20),
            0,
            (),
        ),
        PublicTurnOpened(PublicEventKind.TURN_OPENED, int(ActionId.AUCTION2), (1, 2)),
    )

    state = validate_public_history(history)

    assert state.biddable_resource_budget == expected_budget
    assert state.public_biddable_resource_count == 2


def test_eighth_three_player_auction_two_must_be_one_card_and_terminal() -> None:
    setup = PublicGameSetup(
        PublicEventKind.GAME_SETUP,
        3,
        30,
        (0, 4, 8, 12, 16, 20),
        0,
        (),
    )
    offers = ((1, 1), (1, 2), (2, 2), (3, 3), (3, 4), (4, 4), (5, 5), (5, 0))
    reveal_suits = (1, 2, 3, 4, 5, 1, 2, 3)
    history_events: list[object] = [setup]
    tiebreak_seat = 0
    history_before_eighth_turn: PublicHistory | None = None
    for turn_number, (offer, reveal_suit) in enumerate(
        zip(offers, reveal_suits, strict=True),
        start=1,
    ):
        if turn_number == 8:
            history_before_eighth_turn = cast(PublicHistory, tuple(history_events))
        winner = (tiebreak_seat + 1) % 3
        history_events.extend(
            (
                PublicTurnOpened(PublicEventKind.TURN_OPENED, int(ActionId.AUCTION2), offer),
                PublicAuctionResolved(PublicEventKind.AUCTION_RESOLVED, (0, 0, 0)),
                PublicInformationRevealed(
                    PublicEventKind.INFORMATION_REVEALED,
                    winner,
                    reveal_suit,
                ),
            )
        )
        tiebreak_seat = winner
    history = cast(PublicHistory, tuple(history_events))
    if history_before_eighth_turn is None:
        raise AssertionError("the fixed eight-turn history must reach its final turn")

    state = validate_public_history(history)

    assert state.resolved_turn_count == 8
    assert state.seen_action_counts[int(ActionId.AUCTION2) - 1] == 8
    assert state.biddable_resource_budget == 15
    assert state.public_biddable_resource_count == 15
    assert sum(sum(row) for row in state.won_resource_counts_by_seat) == 15

    two_card_eighth_turn = (
        *history_before_eighth_turn,
        PublicTurnOpened(PublicEventKind.TURN_OPENED, int(ActionId.AUCTION2), (5, 5)),
    )
    with pytest.raises(PublicHistoryCompatibilityError, match="biddable-card budget"):
        validate_public_history(two_card_eighth_turn)

    later_turn = (
        *history,
        PublicTurnOpened(PublicEventKind.TURN_OPENED, int(ActionId.LOAN10), (4, 5)),
    )
    with pytest.raises(PublicHistoryCompatibilityError, match="terminal one-card"):
        validate_public_history(later_turn)


def test_shared_validator_reconstructs_cash_assets_and_objectives() -> None:
    history: PublicHistory = (
        PublicGameSetup(
            kind=PublicEventKind.GAME_SETUP,
            player_count=3,
            starting_cash=30,
            value_chart=(0, 4, 8, 12, 16, 20),
            initial_tiebreak_seat=1,
            objective_ids=(1, 2, 3, 4),
        ),
        PublicTurnOpened(PublicEventKind.TURN_OPENED, int(ActionId.LOAN20), (1, 2)),
        PublicAuctionResolved(PublicEventKind.AUCTION_RESOLVED, (35, 0, 0)),
        PublicInformationRevealed(PublicEventKind.INFORMATION_REVEALED, 0, 5),
        PublicTurnOpened(PublicEventKind.TURN_OPENED, int(ActionId.INVEST5), (1, 2)),
        PublicAuctionResolved(PublicEventKind.AUCTION_RESOLVED, (0, 7, 0)),
        PublicInformationRevealed(PublicEventKind.INFORMATION_REVEALED, 1, 4),
        PublicTurnOpened(PublicEventKind.TURN_OPENED, int(ActionId.AUCTION2), (1, 2)),
        PublicAuctionResolved(PublicEventKind.AUCTION_RESOLVED, (0, 0, 3)),
        PublicInformationRevealed(PublicEventKind.INFORMATION_REVEALED, 2, 3),
        PublicTurnOpened(PublicEventKind.TURN_OPENED, int(ActionId.AUCTION1), (3, 4)),
        PublicAuctionResolved(PublicEventKind.AUCTION_RESOLVED, (0, 0, 2)),
    )

    state = validate_public_history(history)

    assert state.phase == "reveal_pending"
    assert state.cash_by_seat == (15, 23, 25)
    assert state.won_resource_counts_by_seat == (
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
        (1, 1, 1, 0, 0),
    )
    assert state.revealed_info_counts_by_seat == (
        (0, 0, 0, 0, 1),
        (0, 0, 0, 1, 0),
        (0, 0, 1, 0, 0),
    )
    assert state.owned_objective_ids_by_seat == ((), (), (3,))
    assert state.loan_principal_by_seat == (20, 0, 0)
    assert state.investment_value_by_seat == (0, 12, 0)
    assert state.legal_max_bid_by_seat is None
    assert state.resolved_turn_count == 4
    assert state.seen_action_counts == (1, 1, 0, 1, 1, 0)


def test_shared_validator_includes_loan_credit_in_open_turn_legal_maxima() -> None:
    history = (
        PublicGameSetup(
            PublicEventKind.GAME_SETUP,
            3,
            30,
            (0, 4, 8, 12, 16, 20),
            0,
            (),
        ),
        PublicTurnOpened(PublicEventKind.TURN_OPENED, int(ActionId.LOAN10), (1, 2)),
    )

    assert validate_public_history(history).legal_max_bid_by_seat == (40, 40, 40)


def test_shared_validator_rejects_bids_above_replayed_cash_and_action_credit() -> None:
    history = (
        PublicGameSetup(
            PublicEventKind.GAME_SETUP,
            3,
            30,
            (0, 4, 8, 12, 16, 20),
            0,
            (),
        ),
        PublicTurnOpened(PublicEventKind.TURN_OPENED, int(ActionId.AUCTION1), (1, 2)),
        PublicAuctionResolved(PublicEventKind.AUCTION_RESOLVED, (31, 0, 0)),
    )

    with pytest.raises(PublicHistoryCompatibilityError, match="legal maximum"):
        validate_public_history(history)


def test_shared_validator_rejects_resource_carry_and_terminal_turn_violations() -> None:
    setup = PublicGameSetup(
        PublicEventKind.GAME_SETUP,
        3,
        30,
        (0, 4, 8, 12, 16, 20),
        0,
        (),
    )
    skipped_carry = (
        setup,
        PublicTurnOpened(PublicEventKind.TURN_OPENED, int(ActionId.AUCTION1), (1, 2)),
        PublicAuctionResolved(PublicEventKind.AUCTION_RESOLVED, (1, 0, 0)),
        PublicInformationRevealed(PublicEventKind.INFORMATION_REVEALED, 0, 3),
        PublicTurnOpened(PublicEventKind.TURN_OPENED, int(ActionId.LOAN10), (4, 5)),
    )
    continued_terminal = (
        setup,
        PublicTurnOpened(PublicEventKind.TURN_OPENED, int(ActionId.AUCTION2), (1, 0)),
        PublicAuctionResolved(PublicEventKind.AUCTION_RESOLVED, (1, 0, 0)),
        PublicInformationRevealed(PublicEventKind.INFORMATION_REVEALED, 0, 3),
        PublicTurnOpened(PublicEventKind.TURN_OPENED, int(ActionId.LOAN10), (4, 5)),
    )

    with pytest.raises(PublicHistoryCompatibilityError, match="carry"):
        validate_public_history(skipped_carry)
    with pytest.raises(PublicHistoryCompatibilityError, match="terminal one-card"):
        validate_public_history(continued_terminal)


def test_shared_validator_enforces_finite_action_and_resource_decks() -> None:
    setup = PublicGameSetup(
        PublicEventKind.GAME_SETUP,
        3,
        30,
        (0, 4, 8, 12, 16, 20),
        0,
        (),
    )
    too_many_loan20_actions: list[object] = [setup]
    for winner, reveal_suit in ((1, 2), (2, 3)):
        too_many_loan20_actions.extend(
            (
                PublicTurnOpened(
                    PublicEventKind.TURN_OPENED,
                    int(ActionId.LOAN20),
                    (4, 5),
                ),
                PublicAuctionResolved(
                    PublicEventKind.AUCTION_RESOLVED,
                    tuple(1 if seat == winner else 0 for seat in range(3)),
                ),
                PublicInformationRevealed(
                    PublicEventKind.INFORMATION_REVEALED,
                    winner,
                    reveal_suit,
                ),
            )
        )
    too_many_loan20_actions.append(
        PublicTurnOpened(PublicEventKind.TURN_OPENED, int(ActionId.LOAN20), (4, 5))
    )

    too_many_public_bricks: list[object] = [setup]
    actions = (
        ActionId.LOAN10,
        ActionId.LOAN10,
        ActionId.LOAN10,
        ActionId.LOAN20,
        ActionId.LOAN20,
        ActionId.INVEST5,
        ActionId.INVEST5,
    )
    tiebreak = 0
    for action in actions:
        winner = (tiebreak + 1) % 3
        too_many_public_bricks.extend(
            (
                PublicTurnOpened(PublicEventKind.TURN_OPENED, int(action), (4, 5)),
                PublicAuctionResolved(PublicEventKind.AUCTION_RESOLVED, (0, 0, 0)),
                PublicInformationRevealed(
                    PublicEventKind.INFORMATION_REVEALED,
                    winner,
                    int(Suit.BRICK),
                ),
            )
        )
        tiebreak = winner

    with pytest.raises(PublicHistoryCompatibilityError, match="action appears"):
        validate_public_history(cast(PublicHistory, tuple(too_many_loan20_actions)))
    with pytest.raises(PublicHistoryCompatibilityError, match="finite SDK deck"):
        validate_public_history(cast(PublicHistory, tuple(too_many_public_bricks)))
