from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from garboid_pocketrocks.adapters import PublicEventKind
from garboid_pocketrocks.bots.base import BotBrain, BotSpec
from garboid_pocketrocks.bots.random_bot import RandomBot, RandomBotBrain
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
    opponent_specs: tuple[BotSpec, ...] | None = None,
) -> PocketRocksEnv:
    return PocketRocksEnv(
        opponent_specs=_opponents() if opponent_specs is None else opponent_specs,
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


def test_reset_exposes_current_public_learner_inputs_and_history() -> None:
    env = _make_env()
    with pytest.raises(RuntimeError, match="reset"):
        _ = env.learner_context
    with pytest.raises(RuntimeError, match="reset"):
        _ = env.ruleset_knowledge
    with pytest.raises(RuntimeError, match="reset"):
        _ = env.public_history

    observation, _ = env.reset(seed=3, options={"opponent_seed": 91})

    assert env.learner_context.bot_seat == env.learner_seat
    assert env.ruleset_knowledge == LIVE_RULESET.knowledge(3)
    assert env.public_history[0].kind is PublicEventKind.GAME_SETUP
    before = env.public_history
    action = int(np.flatnonzero(observation["action_mask"])[0])
    env.step(action)
    assert env.public_history[: len(before)] == before
    assert len(env.public_history) > len(before)


@pytest.mark.parametrize(
    "options",
    (
        {"unexpected": 1},
        {"opponent_seed": "91"},
        {"opponent_seed": True},
        {"opponent_seed": None},
    ),
)
def test_reset_rejects_unknown_or_noninteger_opponent_seed(
    options: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="option|opponent_seed"):
        _make_env().reset(seed=3, options=options)


@dataclass(slots=True)
class RecordingBrainFactory:
    seeds: list[int] = field(default_factory=list)

    def __call__(self, seed: int | None) -> BotBrain:
        assert seed is not None
        self.seeds.append(seed)
        return RandomBotBrain(seed=seed)


def _reset_with_recorded_opponent_seeds(
    *,
    environment_seed: int,
    opponent_seed: int,
) -> tuple[object, tuple[int, ...]]:
    factory = RecordingBrainFactory()
    spec = BotSpec(name="recording", bot_id="recording", brain_factory=factory)
    env = _make_env(opponent_specs=(spec, spec))

    env.reset(seed=environment_seed, options={"opponent_seed": opponent_seed})

    assert env.transition is not None
    return env.transition.state, tuple(factory.seeds)


def test_opponent_seed_is_independent_from_environment_seed() -> None:
    first_state, first_seeds = _reset_with_recorded_opponent_seeds(
        environment_seed=17,
        opponent_seed=91,
    )
    repeated_state, repeated_seeds = _reset_with_recorded_opponent_seeds(
        environment_seed=17,
        opponent_seed=91,
    )
    changed_state, changed_seeds = _reset_with_recorded_opponent_seeds(
        environment_seed=17,
        opponent_seed=92,
    )

    assert first_state == repeated_state == changed_state
    assert first_seeds == repeated_seeds
    assert first_seeds != changed_seeds


def _assert_observations_equal(
    first: dict[str, object],
    second: dict[str, object],
) -> None:
    assert first.keys() == second.keys()
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_same_environment_and_opponent_seeds_reproduce_complete_trajectory() -> None:
    first = _make_env()
    second = _make_env()
    first_observation, first_info = first.reset(
        seed=17,
        options={"opponent_seed": 91},
    )
    second_observation, second_info = second.reset(
        seed=17,
        options={"opponent_seed": 91},
    )

    _assert_observations_equal(first_observation, second_observation)
    assert first_info == second_info
    assert first.learner_context == second.learner_context
    assert first.public_history == second.public_history

    for _ in range(200):
        legal_actions = np.flatnonzero(first_observation["action_mask"])
        action = int(legal_actions[-1])
        assert second_observation["action_mask"][action]

        first_step = first.step(action)
        second_step = second.step(action)
        (
            first_observation,
            first_reward,
            first_terminated,
            first_truncated,
            first_info,
        ) = first_step
        (
            second_observation,
            second_reward,
            second_terminated,
            second_truncated,
            second_info,
        ) = second_step

        _assert_observations_equal(first_observation, second_observation)
        assert first_reward == second_reward
        assert first_info["reward_breakdown"] == second_info["reward_breakdown"]
        assert (first_terminated, first_truncated) == (
            second_terminated,
            second_truncated,
        )
        assert first.learner_context == second.learner_context
        assert first.public_history == second.public_history
        if first_terminated:
            break
    else:
        raise AssertionError("seeded environments did not terminate")


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
