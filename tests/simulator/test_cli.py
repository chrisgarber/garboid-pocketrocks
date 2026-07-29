from __future__ import annotations

import json
import subprocess
from pathlib import Path

from garboid_pocketrocks.bots import BotSpec, RandomBot
from garboid_pocketrocks.rules import LIVE_RULESET
from garboid_pocketrocks.simulator.cli import _bot_names, _table
from garboid_pocketrocks.simulator.monte_carlo import (
    MonteCarloConfig,
    MonteCarloRunner,
)
from garboid_pocketrocks.simulator.replay import load_replay, replay_match
from garboid_pocketrocks.simulator.sampling import FixedRulesetSampler


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", "garboid-simulate", *arguments],
        check=True,
        text=True,
        capture_output=True,
    )


def test_simulate_cli_accepts_explicit_heuristic_generations() -> None:
    assert _bot_names(
        "aggressive-v1,balanced-v1,passive-v1,aggressive-v2,balanced-v2,passive-v2"
    ) == (
        "aggressive-v1",
        "balanced-v1",
        "passive-v1",
        "aggressive-v2",
        "balanced-v2",
        "passive-v2",
    )


def test_simulate_cli_runs_mixed_generations_without_faults() -> None:
    completed = _run_cli(
        "--bots",
        "balanced-v1,balanced-v2,passive-v2",
        "--games",
        "6",
        "--players",
        "3",
        "--seed",
        "20260729",
        "--workers",
        "2",
        "--format",
        "json",
    )

    statistics = json.loads(completed.stdout)["result"]["bot_statistics"]

    assert {entry["bot_name"] for entry in statistics} == {
        "balanced-v1",
        "balanced-v2",
        "passive-v2",
    }
    assert all(entry["faults"] == 0 for entry in statistics)


def test_simulate_cli_runs_heuristic_bots_identically_across_worker_counts() -> None:
    arguments = (
        "--bots",
        "aggressive,balanced,passive",
        "--games",
        "6",
        "--players",
        "3",
        "--seed",
        "42",
        "--format",
        "json",
    )

    serial = json.loads(_run_cli(*arguments, "--workers", "1").stdout)
    parallel = json.loads(_run_cli(*arguments, "--workers", "2").stdout)

    assert serial["result"] == parallel["result"]
    assert serial["configuration"]["workers"] == 1
    assert parallel["configuration"]["workers"] == 2


def test_simulate_cli_json_includes_exact_raw_behavior_shape() -> None:
    completed = _run_cli(
        "--bots",
        "random,random,random",
        "--games",
        "1",
        "--players",
        "3",
        "--seed",
        "42",
        "--format",
        "json",
    )

    payload = json.loads(completed.stdout)
    behavior = payload["result"]["bot_statistics"][0]["behavior"]

    assert set(behavior) == {
        "bidding_requests",
        "passes",
        "nonzero_bids",
        "reveal_choices",
        "wins_by_action",
        "resource_cards_won",
        "objectives_claimed",
    }
    assert type(behavior["bidding_requests"]) is int
    assert type(behavior["passes"]) is int
    assert all(type(bid) is int for bid in behavior["nonzero_bids"])
    assert all(type(choice) is int for choice in behavior["reveal_choices"])
    assert len(behavior["wins_by_action"]) == 6
    assert all(type(wins) is int for wins in behavior["wins_by_action"])
    assert type(behavior["resource_cards_won"]) is int
    assert type(behavior["objectives_claimed"]) is int


def test_table_includes_formatted_behavior_columns() -> None:
    repeated = BotSpec.from_bot_class(RandomBot)
    result = MonteCarloRunner.run(
        MonteCarloConfig(
            bot_specs=(repeated, repeated, repeated),
            games=1,
            player_counts=(3,),
            ruleset_sampler=FixedRulesetSampler(LIVE_RULESET),
            root_seed=42,
        )
    )

    table = _table(result)
    headings = table.splitlines()[0].split()
    values = table.splitlines()[2].split()

    assert headings[-4:] == [
        "pass_rate",
        "mean_bid",
        "resource_wins",
        "objectives",
    ]
    assert values[-4] == f"{result.bot_statistics[0].behavior.pass_rate():.3f}"
    assert values[-3] == (f"{result.bot_statistics[0].behavior.mean_nonzero_bid():.3f}")
    assert values[-2:] == [
        str(result.bot_statistics[0].behavior.resource_cards_won),
        str(result.bot_statistics[0].behavior.objectives_claimed),
    ]


def test_simulate_cli_writes_replay_files(tmp_path: Path) -> None:
    _run_cli(
        "--bots",
        "random,random,random",
        "--games",
        "2",
        "--players",
        "3",
        "--seed",
        "9",
        "--replay-dir",
        str(tmp_path),
    )

    replay_paths = sorted(tmp_path.glob("*.json"))
    assert [path.name for path in replay_paths] == [
        "game-000000.json",
        "game-000001.json",
    ]
    for path in replay_paths:
        replay = load_replay(path)
        assert replay_match(replay).result.scores
