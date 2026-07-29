from dataclasses import replace

from pocketrocks import ActionId

from garboid_pocketrocks.rules import LIVE_RULESET
from garboid_pocketrocks.simulator.engine import GameEngine
from garboid_pocketrocks.simulator.model import ActionCard, GameState, PlayerState
from garboid_pocketrocks.simulator.setup import build_setup


def _state_with_action(
    action_id: ActionId,
    *,
    cash: tuple[int, ...],
) -> GameState:
    state = build_setup(
        LIVE_RULESET,
        player_count=len(cash),
        seed=11,
    ).state
    players = tuple(
        replace(player, cash=seat_cash)
        for player, seat_cash in zip(state.players, cash, strict=True)
    )
    return replace(
        state,
        players=players,
        current_action=ActionCard(card_id=10_000, action_id=action_id),
    )


def test_bidding_context_matches_live_loan_limit() -> None:
    transition = GameEngine.resume(
        _state_with_action(ActionId.LOAN20, cash=(7, 12, 30))
    )

    assert transition.pending is not None
    contexts = transition.pending.contexts_by_seat
    assert contexts[0].legal_max_amount == 27
    assert contexts[1].legal_max_amount == 32
    assert contexts[2].legal_max_amount == 50


def test_context_contains_only_public_state_and_own_hand() -> None:
    state = _state_with_action(ActionId.AUCTION1, cash=(30, 30, 30))

    transition = GameEngine.resume(state)

    assert transition.pending is not None
    contexts = transition.pending.contexts_by_seat
    assert contexts[0].current_hand_suit_ids == tuple(
        int(card.suit) for card in state.players[0].private_hand
    )
    assert contexts[0].won_resource_counts_by_seat == (
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
    )
    assert not hasattr(contexts[0], "resource_deck")
    assert not hasattr(contexts[0], "opponent_hands")


def test_resume_does_not_mutate_existing_state() -> None:
    state = _state_with_action(ActionId.INVEST5, cash=(30, 30, 30))

    transition = GameEngine.resume(state)

    assert transition.state is state
    assert transition.events == ()
    assert transition.result is None


def test_state_helper_preserves_player_seats() -> None:
    state = _state_with_action(ActionId.LOAN10, cash=(7, 12, 30))

    assert tuple(player.seat for player in state.players) == (0, 1, 2)
    assert all(isinstance(player, PlayerState) for player in state.players)
