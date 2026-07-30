from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from garboid_pocketrocks.neural.collector import CollectorMetrics  # noqa: E402
from garboid_pocketrocks.neural.config import (  # noqa: E402
    training_encoder_config,
    training_model_config,
)
from garboid_pocketrocks.neural.model import NeuralPolicy  # noqa: E402
from garboid_pocketrocks.neural.planning import plan_mirror_episodes  # noqa: E402
from garboid_pocketrocks.neural.rollout import RolloutBatch  # noqa: E402
from garboid_pocketrocks.neural.vector_collector import (  # noqa: E402
    collect_self_play_vectorized,
)
from garboid_pocketrocks.neural.vector_pool import (  # noqa: E402
    VectorActorPool,
    VectorPoolError,
)
from garboid_pocketrocks.training.rewards import RewardConfig  # noqa: E402


def _records(
    rollout: RolloutBatch,
) -> tuple[tuple[int, int, int, float], ...]:
    return tuple(
        (
            transition.metadata.environment_seed,
            transition.metadata.learner_seat,
            transition.action,
            transition.reward,
        )
        for transition in rollout.transitions
    )


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _stable_metrics(
    metrics: CollectorMetrics,
) -> tuple[
    int,
    int,
    int,
    tuple[int, ...],
    tuple[tuple[str, int, int], ...],
    float,
    float,
]:
    return (
        metrics.games,
        metrics.decisions,
        metrics.inference_batches,
        tuple(sorted(metrics.inference_batch_sizes)),
        metrics.cell_games,
        metrics.inference_batch_p50,
        metrics.inference_batch_p95,
    )


def test_pool_reuses_actors_across_exact_policy_updates() -> None:
    torch.manual_seed(811)
    config = training_encoder_config()
    model = NeuralPolicy(config, training_model_config("small"))
    reward = RewardConfig()
    update_zero = plan_mirror_episodes(
        root_seed=811,
        update_index=0,
        games_per_cell=1,
        policy_identity="current",
    )[:4]
    update_one = plan_mirror_episodes(
        root_seed=811,
        update_index=1,
        games_per_cell=1,
        policy_identity="current",
    )[:4]

    with VectorActorPool(
        encoder_config=config,
        reward_config=reward,
        workers=2,
        engine_batch_size=1,
        max_inference_batch=16,
    ) as pool:
        worker_pids = pool.worker_pids
        first, first_metrics = pool.collect({"current": model}, update_zero)
        first_serial, first_serial_metrics = collect_self_play_vectorized(
            {"current": model},
            update_zero,
            encoder_config=config,
            reward_config=reward,
            device=torch.device("cpu"),
            engine_batch_size=1,
            max_inference_batch=16,
        )

        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(0.125)

        second, second_metrics = pool.collect({"current": model}, update_one)
        second_serial, second_serial_metrics = collect_self_play_vectorized(
            {"current": model},
            update_one,
            encoder_config=config,
            reward_config=reward,
            device=torch.device("cpu"),
            engine_batch_size=1,
            max_inference_batch=16,
        )

        assert pool.worker_pids == worker_pids
        assert _records(first) == _records(first_serial)
        assert _records(second) == _records(second_serial)
        assert _stable_metrics(first_metrics) == _stable_metrics(first_serial_metrics)
        assert _stable_metrics(second_metrics) == _stable_metrics(second_serial_metrics)

    assert pool.closed
    assert all(not _process_exists(pid) for pid in worker_pids)


def test_pool_rejects_collection_after_close() -> None:
    config = training_encoder_config()
    model = NeuralPolicy(config, training_model_config("small"))
    plans = plan_mirror_episodes(
        root_seed=17,
        update_index=0,
        games_per_cell=1,
        policy_identity="current",
    )[:1]
    pool = VectorActorPool(
        encoder_config=config,
        reward_config=RewardConfig(),
        workers=1,
    )

    pool.close()
    pool.close()

    with pytest.raises(VectorPoolError, match="closed"):
        pool.collect({"current": model}, plans)


def test_pool_propagates_worker_snapshot_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = training_encoder_config()
    model = NeuralPolicy(config, training_model_config("small"))
    plans = plan_mirror_episodes(
        root_seed=19,
        update_index=0,
        games_per_cell=1,
        policy_identity="current",
    )[:1]
    monkeypatch.setattr(model, "state_dict", lambda: {})
    pool = VectorActorPool(
        encoder_config=config,
        reward_config=RewardConfig(),
        workers=1,
    )

    with pytest.raises(VectorPoolError, match=r"vector actor 0 failed"):
        pool.collect({"current": model}, plans)

    assert pool.closed
