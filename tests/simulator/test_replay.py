from pathlib import Path

import pytest
from pocketrocks import BotDecision

from garboid_pocketrocks.bots import BotSpec, RandomBot
from garboid_pocketrocks.simulator.replay import (
    MatchReplay,
    ReplayDivergence,
    load_replay,
    replay_match,
    save_replay,
)
from garboid_pocketrocks.simulator.runner import MatchResult, MatchRunner


def _run_random_match(seed: int) -> MatchResult:
    lineup = tuple(BotSpec.from_bot_class(RandomBot) for _ in range(3))
    return MatchRunner.run(
        lineup,
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


def test_replay_uses_schema_version_two() -> None:
    replay = _run_random_match(seed=19).replay

    payload = replay.to_dict()

    assert payload["schema_version"] == 2
    assert MatchReplay.from_dict(payload) == replay


def test_replay_rejects_removed_engine_schema() -> None:
    with pytest.raises(ReplayDivergence, match="schema version 1"):
        MatchReplay.from_dict({"schema_version": 1})


def test_replay_detects_changed_decision() -> None:
    replay = _run_random_match(seed=20).replay
    step, decisions = replay.decisions[0]
    seat, _decision = decisions[0]
    changed = (
        (step, ((seat, BotDecision.submit_bid(1)), *decisions[1:])),
        *replay.decisions[1:],
    )

    with pytest.raises(ReplayDivergence, match="decision step|differs"):
        replay_match(
            MatchReplay(
                schema_version=replay.schema_version,
                player_count=replay.player_count,
                seed=replay.seed,
                value_chart=replay.value_chart,
                objectives_enabled=replay.objectives_enabled,
                root_seed=replay.root_seed,
                game_index=replay.game_index,
                bot_names=replay.bot_names,
                decisions=changed,
                turns=replay.turns,
                result=replay.result,
            )
        )
