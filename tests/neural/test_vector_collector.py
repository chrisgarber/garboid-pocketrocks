from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from garboid_pocketrocks.neural.collector import collect_self_play  # noqa: E402
from garboid_pocketrocks.neural.config import (  # noqa: E402
    stage1_model_config,
    training_encoder_config,
)
from garboid_pocketrocks.neural.encoding import (  # noqa: E402
    NeuralObservationEncoder,
)
from garboid_pocketrocks.neural.model import NeuralPolicy  # noqa: E402
from garboid_pocketrocks.neural.planning import plan_mirror_episodes  # noqa: E402
from garboid_pocketrocks.neural.rollout import RolloutBatch  # noqa: E402
from garboid_pocketrocks.neural.vector_collector import (  # noqa: E402
    collect_self_play_vectorized,
    vector_plan_batches,
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


def _scores(
    rollout: RolloutBatch,
) -> tuple[tuple[int, tuple[tuple[int, int, int], ...]], ...]:
    return tuple(
        (
            episode.plan.episode_index,
            tuple(
                (score.seat, score.final_money, score.rank)
                for score in episode.result.scores
            ),
        )
        for episode in rollout.multi_seat_episodes
    )


def test_vector_plan_batches_use_homogeneous_groups_of_sixty_four() -> None:
    plans = plan_mirror_episodes(
        root_seed=42,
        update_index=0,
        games_per_cell=13,
        policy_identity="current",
    )

    batches = vector_plan_batches(plans, batch_size=64)

    assert sorted(len(batch) for batch in batches) == [1, 1, 1, 64, 64, 64]
    assert all(
        len({plan.player_count for plan in batch}) == 1
        for batch in batches
    )
    assert sorted(
        plan.episode_index for batch in batches for plan in batch
    ) == list(range(len(plans)))


def test_vector_collector_matches_scalar_sdk_self_play() -> None:
    torch.manual_seed(93)
    config = training_encoder_config()
    model = NeuralPolicy(config, stage1_model_config())
    plans = plan_mirror_episodes(
        root_seed=93,
        update_index=0,
        games_per_cell=1,
        policy_identity="current",
    )
    reward = RewardConfig(
        accounting_weight=0.7,
        win_bonus=0.5,
        placement_bonuses=(0.3, 0.2, 0.1),
        event_bonuses=(
            ("auction_resolved", 0.01),
            ("resources_awarded", 0.02),
            ("loan_acquired", 0.03),
            ("investment_acquired", 0.04),
            ("objective_claimed", 0.05),
            ("information_revealed", 0.06),
        ),
    )
    scalar, _ = collect_self_play(
        {"current": model},
        plans,
        encoder_config=config,
        reward_config=reward,
        device=torch.device("cpu"),
        active_games=64,
        max_inference_batch=512,
    )

    vector, metrics = collect_self_play_vectorized(
        {"current": model},
        plans,
        encoder_config=config,
        reward_config=reward,
        device=torch.device("cpu"),
        engine_batch_size=64,
        max_inference_batch=512,
    )

    vector_records = _records(vector)
    scalar_records = _records(scalar)
    assert tuple(record[:3] for record in vector_records) == tuple(
        record[:3] for record in scalar_records
    )
    assert tuple(record[3] for record in vector_records) == pytest.approx(
        tuple(record[3] for record in scalar_records)
    )
    assert _scores(vector) == _scores(scalar)
    assert metrics.games == 15
    assert metrics.decisions == len(vector.transitions)
    assert metrics.ipc_seconds == 0.0
    assert metrics.inference_batch_p95 >= metrics.inference_batch_p50


def test_vector_collector_skips_redundant_external_input_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_validation(*_args: object) -> None:
        raise AssertionError("trusted SDK batch state was revalidated")

    monkeypatch.setattr(
        NeuralObservationEncoder,
        "_validate",
        unexpected_validation,
    )
    config = training_encoder_config()
    model = NeuralPolicy(config, stage1_model_config())
    plans = plan_mirror_episodes(
        root_seed=7,
        update_index=0,
        games_per_cell=1,
        policy_identity="current",
    )[:1]

    rollout, metrics = collect_self_play_vectorized(
        {"current": model},
        plans,
        encoder_config=config,
        reward_config=RewardConfig(),
        device=torch.device("cpu"),
        engine_batch_size=64,
        max_inference_batch=512,
    )

    assert metrics.games == 1
    assert rollout.transitions
