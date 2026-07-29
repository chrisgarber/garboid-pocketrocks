from __future__ import annotations

import json
import subprocess
from pathlib import Path

from garboid_pocketrocks.simulator.replay import load_replay, replay_match


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", "garboid-simulate", *arguments],
        check=True,
        text=True,
        capture_output=True,
    )


def test_simulate_cli_emits_reproducible_json() -> None:
    arguments = (
        "--bots",
        "random,random,random",
        "--games",
        "6",
        "--players",
        "3",
        "--seed",
        "42",
        "--format",
        "json",
    )

    left = _run_cli(*arguments)
    right = _run_cli(*arguments)

    assert json.loads(left.stdout) == json.loads(right.stdout)


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
