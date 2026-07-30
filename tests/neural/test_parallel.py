from __future__ import annotations

import multiprocessing
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")

from garboid_pocketrocks.neural.collector import collect_self_play  # noqa: E402
from garboid_pocketrocks.neural.config import (  # noqa: E402
    training_encoder_config,
    training_model_config,
)
from garboid_pocketrocks.neural.model import NeuralPolicy  # noqa: E402
from garboid_pocketrocks.neural.parallel import (  # noqa: E402
    _spawn_plan_workers,
    collect_self_play_parallel,
)
from garboid_pocketrocks.neural.planning import plan_mirror_episodes  # noqa: E402
from garboid_pocketrocks.neural.rollout import RolloutBatch  # noqa: E402
from garboid_pocketrocks.training.rewards import RewardConfig  # noqa: E402


def _records(rollout: RolloutBatch) -> tuple[tuple[int, int, int, float], ...]:
    return tuple(
        (
            transition.metadata.environment_seed,
            transition.metadata.learner_seat,
            transition.action,
            transition.reward,
        )
        for transition in rollout.transitions
    )


def test_plan_worker_partial_startup_failure_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parents = (Mock(), Mock())
    children = (Mock(), Mock())
    first_process = Mock()
    first_process.is_alive.side_effect = (True, False)
    startup_error = RuntimeError("second worker failed to start")
    second_process = Mock()
    second_process.start.side_effect = startup_error
    context = Mock()
    context.Pipe.side_effect = tuple(zip(parents, children, strict=True))
    context.Process.side_effect = (first_process, second_process)
    monkeypatch.setattr(
        multiprocessing,
        "get_context",
        lambda method: context,
    )

    with pytest.raises(RuntimeError) as raised:
        _spawn_plan_workers(
            ((), ()),
            encoder_config=training_encoder_config(),
            reward_config=RewardConfig(),
            active_games_per_worker=1,
        )

    assert raised.value is startup_error
    first_process.terminate.assert_called_once_with()
    first_process.join.assert_called_once_with(timeout=5.0)
    second_process.is_alive.assert_not_called()
    for endpoint in (*parents, *children):
        endpoint.close.assert_called_once_with()


def test_spawned_workers_match_serial_seeded_rollout() -> None:
    torch.manual_seed(93)
    config = training_encoder_config()
    model = NeuralPolicy(config, training_model_config("small"))
    plans = plan_mirror_episodes(
        root_seed=93,
        update_index=0,
        games_per_cell=1,
        policy_identity="current",
    )
    serial, _ = collect_self_play(
        {"current": model},
        plans,
        encoder_config=config,
        reward_config=RewardConfig(),
        device=torch.device("cpu"),
        active_games=8,
        max_inference_batch=64,
    )

    parallel, metrics = collect_self_play_parallel(
        {"current": model},
        plans,
        encoder_config=config,
        reward_config=RewardConfig(),
        device=torch.device("cpu"),
        workers=2,
        active_games_per_worker=4,
        max_inference_batch=64,
        max_queue_delay_ms=1.0,
    )

    assert _records(parallel) == _records(serial)
    assert metrics.games == 15
    assert metrics.inference_batch_p50 >= 1.0
    assert metrics.inference_batch_p95 >= metrics.inference_batch_p50
    assert metrics.ipc_seconds >= 0.0
    assert metrics.worker_busy_seconds > 0.0
