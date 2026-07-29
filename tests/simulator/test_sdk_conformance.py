from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import cast

import pytest
from pocketrocks import ActionId, BotDecision, Suit
from pocketrocks.testing import Scenario, scenario

from garboid_pocketrocks.rules import LIVE_RULESET
from garboid_pocketrocks.simulator.engine import EngineTransition, GameEngine
from garboid_pocketrocks.simulator.events import EventKind, GameEvent
from garboid_pocketrocks.simulator.model import ActionCard, ResourceCard

CONFORMANT_FIELDS = (
    "player_count",
    "starting_cash",
    "value_chart",
    "objective_ids",
    "current_action_id",
    "current_resource_ids",
    "cash_by_seat",
    "tiebreak_seat",
    "won_resource_counts_by_seat",
    "revealed_info_counts_by_seat",
    "owned_objective_ids_by_seat",
    "bot_seat",
    "current_hand_suit_ids",
    "legal_max_amount",
    "revealable_count",
)


def _offered_resources(event: GameEvent) -> tuple[int, int]:
    if event.action_id is ActionId.AUCTION1:
        resources = event.resource_ids or ()
        return (resources[0], 0)
    if event.action_id is ActionId.AUCTION2:
        resources = event.resource_ids or ()
        return (resources[0], resources[1])
    return (0, 0)


def _append_events(narration: Scenario, events: Sequence[GameEvent]) -> None:
    bids_by_seat: dict[int, int] = {}
    for event in events:
        if event.kind is EventKind.DECISION_SUBMITTED and event.seat is not None:
            bids_by_seat[event.seat] = event.amount or 0
        elif event.kind is EventKind.AUCTION_RESOLVED:
            narration.auction(bids_by_seat)
            bids_by_seat.clear()
        elif event.kind is EventKind.INFORMATION_REVEALED:
            assert event.resource_ids is not None
            narration.reveal(event.resource_ids[0])
        elif event.kind is EventKind.TURN_OPENED:
            assert event.action_id is not None
            narration.turn(
                event.action_id,
                resources=_offered_resources(event),
            )


def _assert_batch_conforms(
    transition: EngineTransition,
    narration: Scenario,
) -> None:
    assert transition.pending is not None
    for seat, actual in transition.pending.contexts:
        hand = tuple(int(card.suit) for card in transition.state.players[seat].private_hand)
        expected = narration.deciding(
            seat=seat,
            hand=hand,
            kind=actual.decision_kind,
        ).to_context(received_at=0)
        for field in CONFORMANT_FIELDS:
            assert getattr(actual, field) == getattr(expected, field), field


def _step(
    transition: EngineTransition,
    narration: Scenario,
    decisions_by_seat: Mapping[int, BotDecision],
) -> EngineTransition:
    next_transition = GameEngine.step(transition.state, decisions_by_seat)
    _append_events(narration, next_transition.events)
    _assert_batch_conforms(next_transition, narration)
    return next_transition


def _reveal_first(
    transition: EngineTransition,
    narration: Scenario,
) -> EngineTransition:
    assert transition.pending is not None
    seat = transition.pending.acting_seats[0]
    return _step(
        transition,
        narration,
        {seat: BotDecision.select_info_to_reveal(0)},
    )


def test_bidding_revealable_count_matches_sdk_hand_size() -> None:
    transition = GameEngine.start(
        LIVE_RULESET,
        player_count=3,
        seed=60,
    )

    assert transition.pending is not None
    actual = transition.pending.contexts_by_seat[0]
    expected = scenario(
        players=3,
        starting_cash=30,
    ).deciding(
        seat=0,
        hand=actual.current_hand_suit_ids,
    ).to_context(received_at=0)

    assert actual.revealable_count == expected.revealable_count


def test_reveal_context_preserves_resolved_auction_resources() -> None:
    started = GameEngine.start(
        LIVE_RULESET,
        player_count=3,
        seed=60,
    )
    assert started.pending is not None
    offered = started.pending.contexts_by_seat[0].current_resource_ids

    transition = GameEngine.step(
        started.state,
        {
            0: BotDecision.submit_bid(3),
            1: BotDecision.submit_bid(7),
            2: BotDecision.submit_bid(5),
        },
    )

    assert transition.pending is not None
    actual = transition.pending.contexts_by_seat[1]
    assert actual.current_resource_ids == offered


@pytest.mark.parametrize(
    "invalid_resource_ids",
    (
        (1,),
        (0, 1),
        (len(Suit) + 1, 0),
    ),
)
def test_game_state_rejects_invalid_current_resource_ids(
    invalid_resource_ids: tuple[int, ...],
) -> None:
    state = GameEngine.start(
        LIVE_RULESET,
        player_count=3,
        seed=60,
    ).state

    with pytest.raises(ValueError, match="current resource IDs"):
        replace(
            state,
            current_resource_ids=cast(tuple[int, int], invalid_resource_ids),
        )


def test_one_card_auction_two_reveal_does_not_include_previously_won_resource() -> None:
    started = GameEngine.start(
        LIVE_RULESET,
        player_count=3,
        seed=60,
    )
    offered_card = started.state.visible_resources[0]
    players = list(started.state.players)
    players[0] = replace(
        players[0],
        won_resources=(ResourceCard(card_id=10_000, suit=Suit.BRICK),),
    )
    state = replace(
        started.state,
        players=tuple(players),
        resource_deck=(),
        visible_resources=(offered_card,),
        current_action=ActionCard(card_id=10_001, action_id=ActionId.AUCTION2),
    )

    transition = GameEngine.step(
        state,
        {
            0: BotDecision.submit_bid(3),
            1: BotDecision.submit_bid(2),
            2: BotDecision.submit_bid(1),
        },
    )

    assert transition.pending is not None
    actual = transition.pending.contexts_by_seat[0]
    assert actual.current_resource_ids == (int(offered_card.suit), 0)


def test_seeded_engine_history_conforms_to_sdk_scenario() -> None:
    transition = GameEngine.start(
        LIVE_RULESET,
        player_count=3,
        seed=60,
    )
    narration = scenario(
        players=transition.state.player_count,
        starting_cash=LIVE_RULESET.setup_for(3).starting_cash,
        value_chart=cast(
            tuple[int, int, int, int, int, int],
            LIVE_RULESET.value_chart,
        ),
        initial_tiebreak_seat=transition.state.priority_seat,
        objective_ids=transition.state.active_objective_ids,
    )
    _append_events(narration, transition.events)
    _assert_batch_conforms(transition, narration)

    assert transition.state.current_action is not None
    assert transition.state.current_action.action_id is ActionId.AUCTION2
    transition = _step(
        transition,
        narration,
        {
            0: BotDecision.submit_bid(3),
            1: BotDecision.submit_bid(7),
            2: BotDecision.submit_bid(5),
        },
    )
    transition = _reveal_first(transition, narration)

    assert transition.state.current_action is not None
    assert transition.state.current_action.action_id is ActionId.LOAN10
    transition = _step(
        transition,
        narration,
        {
            0: BotDecision.submit_bid(8),
            1: BotDecision.submit_bid(10),
            2: BotDecision.submit_bid(9),
        },
    )
    transition = _reveal_first(transition, narration)

    assert transition.state.current_action is not None
    assert transition.state.current_action.action_id is ActionId.AUCTION1
    _step(
        transition,
        narration,
        {
            0: BotDecision.submit_bid(6),
            1: BotDecision.submit_bid(4),
            2: BotDecision.submit_bid(5),
        },
    )
