import random

import pytest
from helpers import (
    assert_objective_ownership,
    assert_resource_conservation,
    assert_transition_invariants,
)
from hypothesis import given, settings
from hypothesis import strategies as st

from garboid_pocketrocks.bots.random_bot import RandomBotBrain
from garboid_pocketrocks.rules import LIVE_RULESET
from garboid_pocketrocks.simulator.engine import GameEngine
from garboid_pocketrocks.simulator.setup import build_setup


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


@pytest.mark.parametrize("player_count", (3, 4, 5))
@settings(max_examples=40, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_random_brain_games_preserve_transition_invariants(
    player_count: int,
    seed: int,
) -> None:
    brain_rng = random.Random(seed)
    brains = tuple(RandomBotBrain(seed=brain_rng.randrange(2**63)) for _ in range(player_count))
    knowledge = LIVE_RULESET.knowledge(player_count)
    transition = GameEngine.start(
        LIVE_RULESET,
        player_count=player_count,
        seed=seed,
    )

    while True:
        assert_transition_invariants(transition)
        if transition.result is not None:
            break
        assert transition.pending is not None
        decisions_by_seat = {
            seat: brains[seat].choose_decision(context, knowledge)
            for seat, context in transition.pending.contexts
        }
        transition = GameEngine.step(transition.state, decisions_by_seat)
