from pathlib import Path

from garboid_pocketrocks.bots import BotSpec, RandomBot
from garboid_pocketrocks.rules import LIVE_RULESET
from garboid_pocketrocks.simulator.replay import (
    load_replay,
    replay_match,
    save_replay,
)
from garboid_pocketrocks.simulator.runner import MatchResult, MatchRunner


def _run_random_match(seed: int) -> MatchResult:
    lineup = tuple(BotSpec.from_bot_class(RandomBot) for _ in range(3))
    return MatchRunner.run(
        lineup,
        ruleset=LIVE_RULESET,
        player_count=3,
        seed=seed,
    )


def test_replay_reproduces_events_and_result() -> None:
    match = _run_random_match(seed=17)

    replayed = replay_match(match.replay)

    assert replayed.events == match.events
    assert replayed.result == match.result


def test_replay_json_round_trip_is_lossless(tmp_path: Path) -> None:
    original = _run_random_match(seed=18).replay
    path = tmp_path / "match.json"

    save_replay(original, path)

    assert load_replay(path) == original
