from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

pytest.importorskip("torch")

from garboid_pocketrocks.neural.config import training_encoder_config  # noqa: E402
from garboid_pocketrocks.neural.planning import (  # noqa: E402
    SeatPolicy,
    decision_seed,
    plan_mirror_episodes,
)
from garboid_pocketrocks.neural.run_config import (  # noqa: E402
    ParallelConfig,
    TrainingRunConfig,
)
from garboid_pocketrocks.rules import live_ruleset  # noqa: E402


def test_training_encoder_covers_every_live_chart_and_player_count() -> None:
    config = training_encoder_config()

    assert config.supported_ruleset_names == (
        "live-A",
        "live-B",
        "live-C",
        "live-D",
        "live-E",
    )
    assert config.supported_player_counts == (3, 4, 5)
    for chart in "ABCDE":
        ruleset = live_ruleset(chart)
        for player_count in (3, 4, 5):
            required = (
                1
                + (2 * sum(ruleset.action_counts))
                + (
                    player_count
                    * ruleset.setup_for(player_count).private_cards_per_player
                )
            )
            assert required <= config.max_history_events


def test_mirror_plan_has_one_hundred_games_in_every_cell() -> None:
    plans = plan_mirror_episodes(
        root_seed=42,
        update_index=0,
        games_per_cell=100,
        policy_identity="current",
    )

    assert len(plans) == 1_500
    counts = Counter((plan.ruleset_name, plan.player_count) for plan in plans)
    assert set(counts.values()) == {100}
    assert set(counts) == {
        (f"live-{chart}", players)
        for chart in "ABCDE"
        for players in (3, 4, 5)
    }
    assert all(
        len(plan.seat_sampling_seeds) == plan.player_count
        and plan.seat_policies
        == (SeatPolicy(identity="current", trainable=True),) * plan.player_count
        for plan in plans
    )


def test_plans_and_decision_seeds_are_named_stable() -> None:
    first = plan_mirror_episodes(
        root_seed=42,
        update_index=7,
        games_per_cell=2,
        policy_identity="candidate",
    )
    second = plan_mirror_episodes(
        root_seed=42,
        update_index=7,
        games_per_cell=2,
        policy_identity="candidate",
    )

    assert first == second
    assert len(
        {
            decision_seed(plan, seat, 0)
            for plan in first
            for seat in range(plan.player_count)
        }
    ) == sum(plan.player_count for plan in first)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "root_seed": 42,
                "update_index": 0,
                "games_per_cell": 0,
                "policy_identity": "current",
            },
            "games_per_cell",
        ),
        (
            {
                "root_seed": 42,
                "update_index": -1,
                "games_per_cell": 1,
                "policy_identity": "current",
            },
            "update_index",
        ),
        (
            {
                "root_seed": 42,
                "update_index": 0,
                "games_per_cell": 1,
                "policy_identity": "",
            },
            "policy_identity",
        ),
    ],
)
def test_mirror_plan_rejects_invalid_inputs(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        plan_mirror_episodes(**kwargs)  # type: ignore[arg-type]


def test_run_config_round_trips_exact_json(tmp_path: Path) -> None:
    config = TrainingRunConfig(
        device="cpu",
        games_per_cell=3,
        max_updates=2,
        parallel=ParallelConfig(
            workers=2,
            active_games_per_worker=4,
            max_inference_batch=64,
            max_queue_delay_ms=0.5,
        ),
    )
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(config.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    assert TrainingRunConfig.from_json(path) == config


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"unknown": 1}, "unknown"),
        (
            {"games_per_cell": 1, "target_decisions_per_update": 100},
            "exactly one",
        ),
        (
            {"games_per_cell": None, "target_decisions_per_update": None},
            "exactly one",
        ),
        ({"games_per_cell": True}, "games_per_cell"),
        ({"device": "tpu"}, "device"),
        ({"league_fraction": 1.0}, "league_fraction"),
        ({"parallel": {"workers": 0}}, "workers"),
        ({"parallel": {"extra": 1}}, "unknown"),
    ],
)
def test_run_config_rejects_invalid_json(
    tmp_path: Path,
    payload: dict[str, object],
    message: str,
) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        TrainingRunConfig.from_json(path)
