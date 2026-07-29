from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from garboid_pocketrocks.bots import (
    AGGRESSIVE_HEURISTIC_BOT_SPEC,
    BALANCED_HEURISTIC_BOT_SPEC,
    PASSIVE_HEURISTIC_BOT_SPEC,
    BotSpec,
    RandomBot,
)
from garboid_pocketrocks.rules import live_ruleset
from garboid_pocketrocks.simulator.monte_carlo import (
    MonteCarloConfig,
    MonteCarloRunner,
)
from garboid_pocketrocks.simulator.sampling import FixedRulesetSampler
from garboid_pocketrocks.training import (
    EnvironmentBounds,
    PocketRocksAECEnv,
    PocketRocksEnv,
)


def _random_specs(count: int) -> tuple[BotSpec, ...]:
    return tuple(BotSpec.from_bot_class(RandomBot) for _ in range(count))


def _heuristic_smoke_lineup(player_count: int) -> tuple[BotSpec, ...]:
    return (
        AGGRESSIVE_HEURISTIC_BOT_SPEC,
        BALANCED_HEURISTIC_BOT_SPEC,
        PASSIVE_HEURISTIC_BOT_SPEC,
        *_random_specs(player_count - 3),
    )


@pytest.mark.parametrize("chart", tuple("ABCDE"))
@pytest.mark.parametrize("player_count", (3, 4, 5))
def test_heuristic_profiles_make_only_legal_decisions_across_live_games(
    chart: str,
    player_count: int,
) -> None:
    lineup = _heuristic_smoke_lineup(player_count)
    root_seed = 42 + 100 * (ord(chart) - ord("A")) + player_count
    result = MonteCarloRunner.run(
        MonteCarloConfig(
            bot_specs=lineup,
            games=15,
            player_counts=(player_count,),
            ruleset_sampler=FixedRulesetSampler(live_ruleset(chart)),
            root_seed=root_seed,
        )
    )

    assert tuple(spec.name for spec in lineup) == (
        "aggressive",
        "balanced",
        "passive",
        *("random",) * (player_count - 3),
    )
    assert len(result.game_summaries) == 15
    assert all(summary.root_seed == root_seed for summary in result.game_summaries)
    assert all(
        Counter(summary.bot_names) == Counter(spec.name for spec in lineup)
        for summary in result.game_summaries
    )
    assert all(not any(summary.fault_counts) for summary in result.game_summaries)
    assert all(statistics.faults == 0 for statistics in result.bot_statistics)
    assert all(statistics.decision_count > 0 for statistics in result.bot_statistics)
    assert all(
        statistics.games == 15
        for statistics in result.bot_statistics
        if statistics.bot_name != "random"
    )


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
