from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from garboid_pocketrocks.neural.config import (  # noqa: E402
    stage1_model_config,
    training_encoder_config,
)
from garboid_pocketrocks.neural.model import NeuralPolicy  # noqa: E402
from garboid_pocketrocks.neural.planning import plan_mirror_episodes  # noqa: E402
from garboid_pocketrocks.neural.rollout import RolloutBatch  # noqa: E402
from garboid_pocketrocks.neural.vector_collector import (  # noqa: E402
    collect_self_play_vectorized,
)
from garboid_pocketrocks.neural.vector_parallel import (  # noqa: E402
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


def test_parallel_vector_actors_match_single_process_vector_rollout() -> None:
    torch.manual_seed(81)
    config = training_encoder_config()
    model = NeuralPolicy(config, stage1_model_config())
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
