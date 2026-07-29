from dataclasses import replace

import pytest
from pocketrocks import ActionId, BotDecision, Suit

from garboid_pocketrocks.rules import LIVE_RULESET
from garboid_pocketrocks.simulator.engine import GameEngine
from garboid_pocketrocks.simulator.errors import (
    ActingSeatsError,
    IllegalDecisionError,
)
from garboid_pocketrocks.simulator.model import ActionCard, GameState, Phase, ResourceCard
from garboid_pocketrocks.simulator.setup import build_setup


def _bidding_state(action_id: ActionId, *, priority_seat: int = 0) -> GameState:
    state = build_setup(LIVE_RULESET, player_count=3, seed=23).state
    return replace(
        state,
        current_action=ActionCard(card_id=10_001, action_id=action_id),
        priority_seat=priority_seat,
    )


def test_all_pass_tie_awards_clockwise_after_priority() -> None:
    state = _bidding_state(ActionId.AUCTION1, priority_seat=0)
    awarded = state.visible_resources[0]

    transition = GameEngine.step(
        state,
        {
            0: BotDecision.pass_turn(),
            1: BotDecision.pass_turn(),
            2: BotDecision.pass_turn(),
        },
    )

    assert transition.state.priority_seat == 1
    assert transition.state.phase is Phase.REVEAL
    assert awarded in transition.state.players[1].won_resources
    assert transition.state.players[1].cash == 30
    assert transition.pending is not None
    assert transition.pending.acting_seats == (1,)


def test_investment_records_locked_bid_and_payout() -> None:
    state = _bidding_state(ActionId.INVEST5)

    transition = GameEngine.step(
        state,
        {
            0: BotDecision.pass_turn(),
            1: BotDecision.submit_bid(3),
            2: BotDecision.submit_bid(4),
        },
    )

    winner = transition.state.players[2]
    assert winner.cash == 26
    assert len(winner.investments) == 1
    assert winner.investments[0].locked == 4
    assert winner.investments[0].payout == 5


def test_auction_two_awards_both_visible_resources() -> None:
    state = _bidding_state(ActionId.AUCTION2, priority_seat=2)
    offered = state.visible_resources

    transition = GameEngine.step(
        state,
        {
            0: BotDecision.submit_bid(7),
            1: BotDecision.pass_turn(),
            2: BotDecision.pass_turn(),
        },
    )

    assert transition.state.players[0].won_resources[-2:] == offered
    assert transition.state.visible_resources == ()
    assert transition.state.players[0].cash == 23


@pytest.mark.parametrize(
    ("action_id", "principal"),
    [(ActionId.LOAN10, 10), (ActionId.LOAN20, 20)],
)
def test_loan_adds_cash_and_records_principal(
    action_id: ActionId,
    principal: int,
) -> None:
    state = _bidding_state(action_id, priority_seat=2)

    transition = GameEngine.step(
        state,
        {
            0: BotDecision.submit_bid(8),
            1: BotDecision.pass_turn(),
            2: BotDecision.pass_turn(),
        },
    )

    winner = transition.state.players[0]
    assert winner.cash == 30 - 8 + principal
    assert winner.loans[-1].principal == principal
    assert winner.loans[-1].winning_bid == 8


def test_newly_satisfied_objective_is_claimed_once() -> None:
    state = _bidding_state(ActionId.AUCTION1, priority_seat=2)
    players = list(state.players)
    players[0] = replace(
        players[0],
        won_resources=(ResourceCard(card_id=9_998, suit=Suit.BRICK),),
    )
    state = replace(
        state,
        players=tuple(players),
        visible_resources=(ResourceCard(card_id=9_999, suit=Suit.BRICK),),
        active_objective_ids=(1,),
    )

    transition = GameEngine.step(
        state,
        {
            0: BotDecision.pass_turn(),
            1: BotDecision.pass_turn(),
            2: BotDecision.pass_turn(),
        },
    )

    assert transition.state.players[0].owned_objective_ids == (1,)


def test_bidding_requires_exactly_every_seat() -> None:
    state = _bidding_state(ActionId.AUCTION1)

    with pytest.raises(ActingSeatsError, match=r"expected seats=\[0, 1, 2\]"):
        GameEngine.step(
            state,
            {
                0: BotDecision.pass_turn(),
                1: BotDecision.pass_turn(),
            },
        )


def test_illegal_bid_is_rejected_with_seat_context() -> None:
    state = _bidding_state(ActionId.AUCTION1)

    with pytest.raises(IllegalDecisionError, match="seat=0"):
        GameEngine.step(
            state,
            {
                0: BotDecision.submit_bid(31),
                1: BotDecision.pass_turn(),
                2: BotDecision.pass_turn(),
            },
        )


def test_pass_during_reveal_auto_reveals_first_card() -> None:
    state = _bidding_state(ActionId.LOAN10)
    won = GameEngine.step(
        state,
        {
            0: BotDecision.submit_bid(1),
            1: BotDecision.pass_turn(),
            2: BotDecision.pass_turn(),
        },
    )
    winner_seat = won.state.priority_seat
    first_card = won.state.players[winner_seat].private_hand[0]

    revealed = GameEngine.step(
        won.state,
        {winner_seat: BotDecision.pass_turn()},
    )

    player = revealed.state.players[winner_seat]
    assert first_card not in player.private_hand
    assert first_card in player.revealed_info
    assert revealed.state.turn_index == state.turn_index + 1
