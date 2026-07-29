from __future__ import annotations

from dataclasses import replace

import pytest
from pocketrocks import ActionId, BotDecision

from garboid_pocketrocks.rules import LIVE_RULESET
from garboid_pocketrocks.simulator.engine import EngineTransition, GameEngine
from garboid_pocketrocks.simulator.events import EventKind, GameEvent
from garboid_pocketrocks.simulator.model import ActionCard, GameResult, Phase, Score
from garboid_pocketrocks.training.rewards import RewardConfig, RewardTracker


def _transition_for(action_id: ActionId) -> EngineTransition:
    started = GameEngine.start(LIVE_RULESET, player_count=3, seed=60)
    state = replace(
        started.state,
        current_action=ActionCard(card_id=10_000, action_id=action_id),
        phase=Phase.BIDDING,
    )
    return GameEngine.step(
        state,
        {
            0: BotDecision.submit_bid(4),
            1: BotDecision.pass_turn(),
            2: BotDecision.pass_turn(),
        },
    )


@pytest.mark.parametrize(
    ("action_id", "expected"),
    (
        (ActionId.AUCTION1, -4 / 30),
        (ActionId.LOAN10, -4 / 30),
        (ActionId.INVEST5, 5 / 30),
    ),
)
def test_accounting_potential_matches_each_action_type(
    action_id: ActionId,
    expected: float,
) -> None:
    transition = _transition_for(action_id)
    tracker = RewardTracker()
    tracker.reset(transition.state)

    # Reset from the pre-transition state to capture the public potential delta.
    before = replace(
        transition.state,
        players=GameEngine.start(LIVE_RULESET, player_count=3, seed=60).state.players,
    )
    tracker.reset(before)
    rewards = tracker.update(transition)

    assert rewards[0].accounting == pytest.approx(expected)


def test_terminal_resource_value_supplies_the_residual_final_money_delta() -> None:
    started = GameEngine.start(LIVE_RULESET, player_count=3, seed=60)
    terminal = replace(started.state, phase=Phase.TERMINAL)
    transition = EngineTransition(
        state=terminal,
        events=(),
        pending=None,
        result=GameResult(
            scores=(Score(0, 35, 1), Score(1, 30, 2), Score(2, 30, 2)),
        ),
    )
    tracker = RewardTracker()
    tracker.reset(started.state)

    rewards = tracker.update(transition)

    assert rewards[0].terminal_resource == pytest.approx(5 / 30)
    assert rewards[0].accounting == 0


def test_tied_winners_split_the_configured_win_bonus_once() -> None:
    started = GameEngine.start(LIVE_RULESET, player_count=3, seed=60)
    terminal = replace(started.state, phase=Phase.TERMINAL)
    transition = EngineTransition(
        state=terminal,
        events=(),
        pending=None,
        result=GameResult(
            scores=(Score(0, 30, 1), Score(1, 30, 1), Score(2, 25, 3)),
        ),
    )
    tracker = RewardTracker(RewardConfig(win_bonus=2.0, placement_bonuses=(1.0, 0.5)))
    tracker.reset(started.state)

    rewards = tracker.update(transition)
    second = tracker.update(transition)

    assert rewards[0].placement == pytest.approx(2.0)
    assert rewards[1].placement == pytest.approx(2.0)
    assert rewards[2].placement == 0
    assert second[0].placement == 0


def test_event_shaping_is_auditable_and_config_rejects_invalid_event_configuration() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        RewardConfig(event_bonuses=((EventKind.RESOURCES_AWARDED.value, 1.0),) * 2)
    with pytest.raises(ValueError, match="unknown"):
        RewardConfig(event_bonuses=(("not_an_event", 1.0),))

    started = GameEngine.start(LIVE_RULESET, player_count=3, seed=60)
    tracker = RewardTracker(
        RewardConfig(event_bonuses=((EventKind.RESOURCES_AWARDED.value, 0.25),))
    )
    tracker.reset(started.state)
    transition = EngineTransition(
        state=started.state,
        events=(GameEvent(EventKind.RESOURCES_AWARDED, seat=1),),
        pending=started.pending,
        result=None,
    )

    rewards = tracker.update(transition)

    assert rewards[1].shaping == 0.25
    assert rewards[0].shaping == 0
