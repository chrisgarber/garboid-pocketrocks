from __future__ import annotations

import numpy as np

from garboid_pocketrocks.bots import BotSpec, RandomBot
from garboid_pocketrocks.rules import live_ruleset
from garboid_pocketrocks.simulator.runner import MatchRunner
from garboid_pocketrocks.simulator.sampling import FixedRulesetSampler
from garboid_pocketrocks.training import (
    EnvironmentBounds,
    PocketRocksAECEnv,
    PocketRocksEnv,
)


def _random_specs(count: int) -> tuple[BotSpec, ...]:
    return tuple(BotSpec.from_bot_class(RandomBot) for _ in range(count))


def test_every_live_chart_and_player_count_runs_to_completion() -> None:
    for chart in "ABCDE":
        ruleset = live_ruleset(chart)
        for player_count in (3, 4, 5):
            match = MatchRunner.run(
                _random_specs(player_count),
                ruleset=ruleset,
                player_count=player_count,
                seed=(ord(chart) * 10) + player_count,
            )

            assert len(match.result.scores) == player_count
            assert {score.seat for score in match.result.scores} == set(range(player_count))


def test_single_agent_environment_runs_to_termination_with_masked_actions() -> None:
    ruleset = live_ruleset("E")
    env = PocketRocksEnv(
        opponent_specs=_random_specs(2),
        ruleset_sampler=FixedRulesetSampler(ruleset),
        player_count=3,
        bounds=EnvironmentBounds(max_bid=100, max_hand_size=5),
        learner_seat=1,
    )
    observation, _ = env.reset(seed=77)

    for _ in range(200):
        action = int(np.flatnonzero(observation["action_mask"])[-1])
        observation, _, terminated, truncated, _ = env.step(action)
        assert not truncated
        if terminated:
            break
    else:
        raise AssertionError("single-agent environment did not terminate")


def test_multi_agent_environment_runs_to_termination_with_masked_actions() -> None:
    env = PocketRocksAECEnv(
        ruleset_sampler=FixedRulesetSampler(live_ruleset("C")),
        player_count=3,
        bounds=EnvironmentBounds(max_bid=100, max_hand_size=5),
    )
    env.reset(seed=88)

    for _ in range(500):
        agent = env.agent_selection
        if env.terminations[agent]:
            break
        observation = env.observe(agent)
        action = int(np.flatnonzero(observation["action_mask"])[-1])
        env.step(action)
    else:
        raise AssertionError("multi-agent environment did not terminate")

    assert all(env.terminations.values())
