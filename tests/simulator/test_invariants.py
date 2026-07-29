from collections import Counter

from garboid_pocketrocks.rules import LIVE_RULESET
from garboid_pocketrocks.simulator.model import GameState
from garboid_pocketrocks.simulator.setup import build_setup


def assert_resource_conservation(state: GameState) -> None:
    all_cards = [
        *state.resource_deck,
        *state.visible_resources,
        *(
            card
            for player in state.players
            for card in (
                *player.private_hand,
                *player.revealed_info,
                *player.won_resources,
            )
        ),
    ]
    expected_counts = dict(
        zip(
            type(all_cards[0].suit),
            state.ruleset.resource_counts,
            strict=True,
        )
    )

    assert len(all_cards) == sum(state.ruleset.resource_counts)
    assert len({card.card_id for card in all_cards}) == len(all_cards)
    assert Counter(card.suit for card in all_cards) == expected_counts


def assert_objective_ownership(state: GameState) -> None:
    active_objectives = set(state.active_objective_ids)
    owned_objectives = [
        objective_id
        for player in state.players
        for objective_id in player.owned_objective_ids
    ]

    assert len(active_objectives) == len(state.active_objective_ids)
    assert set(owned_objectives) <= active_objectives
    assert all(count == 1 for count in Counter(owned_objectives).values())


def test_initial_states_preserve_resources_and_objective_ownership() -> None:
    for player_count in range(3, 6):
        for seed in range(10):
            state = build_setup(
                LIVE_RULESET,
                player_count=player_count,
                seed=seed,
            ).state

            assert_resource_conservation(state)
            assert_objective_ownership(state)
