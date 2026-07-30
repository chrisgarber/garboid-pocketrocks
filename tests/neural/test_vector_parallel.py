from __future__ import annotations

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
from garboid_pocketrocks.neural.vector_parallel import (  # noqa: E402
    _aggregate_vector_results,
    _VectorWorkerResult,
    collect_self_play_vectorized_parallel,
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


def _worker_metrics(
    decisions: int,
    inference_batch_sizes: tuple[int, ...],
) -> CollectorMetrics:
    return CollectorMetrics(
        games=1,
        decisions=decisions,
        elapsed_seconds=1.0,
        inference_seconds=0.5,
        inference_batches=len(inference_batch_sizes),
        inference_batch_sizes=inference_batch_sizes,
        cell_games=(),
    )


def test_vector_worker_aggregation_is_canonical_by_worker_id() -> None:
    torch.manual_seed(73)
    config = training_encoder_config()
    model = NeuralPolicy(config, training_model_config("small"))
    plans = plan_mirror_episodes(
        root_seed=73,
        update_index=0,
        games_per_cell=1,
        policy_identity="current",
    )[:2]
    rollout, _ = collect_self_play_vectorized(
        {"current": model},
        plans,
        encoder_config=config,
        reward_config=RewardConfig(),
        device=torch.device("cpu"),
        engine_batch_size=1,
        max_inference_batch=16,
    )
    episode_zero, episode_one = rollout.episodes
    worker_zero = _VectorWorkerResult(
        worker_id=0,
        episodes=(episode_zero,),
        metrics=_worker_metrics(
            len(episode_zero.trajectories),
            (1, 2),
        ),
    )
    worker_one = _VectorWorkerResult(
        worker_id=1,
        episodes=(episode_one,),
        metrics=_worker_metrics(
            len(episode_one.trajectories),
            (9,),
        ),
    )
    canonical, canonical_metrics = _aggregate_vector_results(
        (worker_zero, worker_one),
        elapsed_seconds=2.0,
        queue_wait_seconds=0.2,
        ipc_seconds=0.1,
    )
    reversed_rollout, reversed_metrics = _aggregate_vector_results(
        (worker_one, worker_zero),
        elapsed_seconds=2.0,
        queue_wait_seconds=0.2,
        ipc_seconds=0.1,
    )

    assert tuple(episode.plan.episode_index for episode in canonical.episodes) == (
        0,
        1,
    )
    assert tuple(episode.plan.episode_index for episode in reversed_rollout.episodes) == (
        0,
        1,
    )
    assert reversed_metrics == canonical_metrics
    assert reversed_metrics.inference_batch_sizes == (1, 2, 9)


def test_parallel_vector_actors_match_single_process_vector_rollout() -> None:
    torch.manual_seed(81)
    config = training_encoder_config()
    model = NeuralPolicy(config, training_model_config("small"))
    plans = plan_mirror_episodes(
        root_seed=81,
        update_index=0,
        games_per_cell=1,
        policy_identity="current",
    )
    serial, _ = collect_self_play_vectorized(
        {"current": model},
        plans,
        encoder_config=config,
        reward_config=RewardConfig(),
        device=torch.device("cpu"),
        engine_batch_size=64,
        max_inference_batch=512,
    )

    parallel, metrics = collect_self_play_vectorized_parallel(
        {"current": model},
        plans,
        encoder_config=config,
        reward_config=RewardConfig(),
        workers=8,
        engine_batch_size=64,
        max_inference_batch=512,
    )

    assert _records(parallel) == _records(serial)
    assert metrics.games == 15
    assert metrics.decisions == len(parallel.transitions)
    assert metrics.ipc_seconds >= 0.0
    assert metrics.worker_busy_seconds > 0.0
