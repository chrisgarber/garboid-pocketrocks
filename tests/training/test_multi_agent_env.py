from __future__ import annotations

from pettingzoo.test import api_test  # type: ignore[import-untyped]

from garboid_pocketrocks.training import EnvironmentBounds
from garboid_pocketrocks.training.multi_agent_env import PocketRocksAECEnv


def make_small_aec_env() -> PocketRocksAECEnv:
    return PocketRocksAECEnv(
        value_charts=("A",),
        player_count=3,
        bounds=EnvironmentBounds(max_bid=100, max_hand_size=5),
    )


def test_sealed_bids_resolve_only_after_every_seat_acts() -> None:
    env = make_small_aec_env()
    env.reset(seed=3)

    assert env.agent_selection == "seat_0"
    before = env.unwrapped.transition.snapshot
    env.step(0)
    assert env.agent_selection == "seat_1"
    assert env.unwrapped.transition.snapshot == before
    env.step(0)
    assert env.agent_selection == "seat_2"
    assert env.unwrapped.transition.snapshot == before
    env.step(0)

    assert env.unwrapped.transition.snapshot != before
    assert env.unwrapped.transition.pending is not None
    winner = env.unwrapped.transition.pending.acting_seats[0]
    assert env.agent_selection == f"seat_{winner}"
    assert all("sealed_bids" not in env.observe(agent) for agent in env.agents)


def test_winner_alone_reveals_then_bidding_restarts_at_seat_zero() -> None:
    env = make_small_aec_env()
    env.reset(seed=3)

    env.step(0)
    env.step(0)
    env.step(1)

    assert env.agent_selection == "seat_2"
    assert env.unwrapped.transition.pending is not None
    assert env.unwrapped.transition.pending.acting_seats == (2,)
    env.step(0)
    assert env.agent_selection == "seat_0"


def test_terminal_infos_include_every_agent_score() -> None:
    env = make_small_aec_env()
    env.reset(seed=3)

    for _ in range(1_000):
        agent = env.agent_selection
        if env.terminations[agent]:
            break
        action = int(env.action_space(agent).sample(env.observe(agent)["action_mask"]))
        env.step(action)

    assert all(env.terminations.values())
    assert {info["score"]["seat"] for info in env.infos.values()} == {0, 1, 2}


def test_aec_contract() -> None:
    api_test(make_small_aec_env(), num_cycles=200, verbose_progress=False)
