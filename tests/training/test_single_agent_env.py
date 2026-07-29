from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from garboid_pocketrocks.bots.base import BotSpec
from garboid_pocketrocks.bots.random_bot import RandomBot
from garboid_pocketrocks.rules import LIVE_RULESET
from garboid_pocketrocks.simulator.sampling import FixedRulesetSampler
from garboid_pocketrocks.training import EnvironmentBounds, RewardConfig
from garboid_pocketrocks.training.single_agent_env import (
    InvalidActionMode,
    PocketRocksEnv,
)


def _opponents() -> tuple[BotSpec, ...]:
    return (BotSpec.from_bot_class(RandomBot), BotSpec.from_bot_class(RandomBot))


def _make_env(
    *,
    learner_seat: int | None = 0,
    invalid_action_mode: InvalidActionMode = InvalidActionMode.RAISE,
) -> PocketRocksEnv:
    return PocketRocksEnv(
        opponent_specs=_opponents(),
        ruleset_sampler=FixedRulesetSampler(LIVE_RULESET),
        player_count=3,
        bounds=EnvironmentBounds(max_bid=100, max_hand_size=5),
        learner_seat=learner_seat,
        invalid_action_mode=invalid_action_mode,
        reward_config=RewardConfig(invalid_action_penalty=0.5),
    )


def test_reset_seed_reproducibly_selects_learner_seat() -> None:
    env = _make_env(learner_seat=None)
    env.reset(seed=17)
    first = env.learner_seat
    env.reset(seed=17)
    assert env.learner_seat == first


def test_fixed_learner_seat_overrides_randomization() -> None:
    env = _make_env(learner_seat=2)
    env.reset(seed=17)
    assert env.learner_seat == 2


def test_learner_bid_runs_opponents_and_resolves_joint_batch() -> None:
    env = _make_env()
    observation, _ = env.reset(seed=3)
    assert env.transition is not None
    before = env.transition.state

    action = int(np.flatnonzero(observation["action_mask"])[0])
    observation, _, terminated, truncated, _ = env.step(action)

    assert env.transition is not None
    assert env.transition.state != before
    assert not truncated
    assert not terminated or observation["action_mask"][0] == 1


def test_invalid_masked_action_raises_by_default() -> None:
    env = _make_env()
    observation, _ = env.reset(seed=3)
    invalid = next(index for index, enabled in enumerate(observation["action_mask"]) if not enabled)

    with pytest.raises(ValueError, match="not legal"):
        env.step(invalid)


def test_penalty_and_pass_records_penalty_and_advances() -> None:
    env = _make_env(invalid_action_mode=InvalidActionMode.PENALIZE_AND_PASS)
    observation, _ = env.reset(seed=3)
    invalid = next(index for index, enabled in enumerate(observation["action_mask"]) if not enabled)

    _, reward, _, truncated, info = env.step(invalid)

    assert not truncated
    assert info["reward_breakdown"]["penalty"] == -0.5
    assert reward == pytest.approx(sum(info["reward_breakdown"].values()))


def test_gymnasium_contract() -> None:
    check_env(_make_env(), skip_render_check=True)
