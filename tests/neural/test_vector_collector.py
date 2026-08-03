from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from garboid_pocketrocks.neural.collector import collect_self_play  # noqa: E402
from garboid_pocketrocks.neural.config import (  # noqa: E402
    training_encoder_config,
    training_model_config,
)
from garboid_pocketrocks.neural.encoding import (  # noqa: E402
    NeuralObservationEncoder,
)
from garboid_pocketrocks.neural.model import NeuralPolicy  # noqa: E402
from garboid_pocketrocks.neural.planning import (  # noqa: E402
    plan_mirror_episodes,
    plan_strong_field_episodes,
)
from garboid_pocketrocks.neural.rollout import RolloutBatch  # noqa: E402
from garboid_pocketrocks.neural.vector_collector import (  # noqa: E402
    collect_self_play_vectorized,
    vector_plan_batches,
)
from garboid_pocketrocks.training.rewards import RewardConfig  # noqa: E402


def _assert_rollouts_equal(vector: RolloutBatch, scalar: RolloutBatch) -> None:
    assert tuple(episode.plan for episode in vector.episodes) == tuple(
        episode.plan for episode in scalar.episodes
    )
    for vector_episode, scalar_episode in zip(
        vector.episodes,
        scalar.episodes,
        strict=True,
    ):
        assert vector_episode.result.scores == scalar_episode.result.scores
        assert len(vector_episode.trajectories) == len(scalar_episode.trajectories)
        for vector_trajectory, scalar_trajectory in zip(
            vector_episode.trajectories,
            scalar_episode.trajectories,
            strict=True,
        ):
            assert (
                vector_trajectory.seat,
                vector_trajectory.policy_identity,
                vector_trajectory.trainable,
            ) == (
                scalar_trajectory.seat,
                scalar_trajectory.policy_identity,
                scalar_trajectory.trainable,
            )
            assert len(vector_trajectory.transitions) == len(scalar_trajectory.transitions)
            for vector_transition, scalar_transition in zip(
                vector_trajectory.transitions,
                scalar_trajectory.transitions,
                strict=True,
            ):
                assert vector_transition.action == scalar_transition.action
                assert vector_transition.reward == pytest.approx(scalar_transition.reward)
                assert vector_transition.reward_breakdown == pytest.approx(
                    scalar_transition.reward_breakdown
                )
                assert (
                    vector_transition.terminated,
                    vector_transition.truncated,
                ) == (
                    scalar_transition.terminated,
                    scalar_transition.truncated,
                )
                np.testing.assert_array_equal(
                    vector_transition.observation.history_ids,
                    scalar_transition.observation.history_ids,
                )
                np.testing.assert_array_equal(
                    vector_transition.observation.history_numeric,
                    scalar_transition.observation.history_numeric,
                )
                np.testing.assert_array_equal(
                    vector_transition.observation.history_valid,
                    scalar_transition.observation.history_valid,
                )
                np.testing.assert_array_equal(
                    vector_transition.observation.action_mask,
                    scalar_transition.observation.action_mask,
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
    assert all(len({plan.player_count for plan in batch}) == 1 for batch in batches)
    assert sorted(plan.episode_index for batch in batches for plan in batch) == list(
        range(len(plans))
    )


def test_vector_collector_matches_scalar_sdk_self_play() -> None:
    torch.manual_seed(93)
    config = training_encoder_config()
    model = NeuralPolicy(config, training_model_config("small"))
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

    _assert_rollouts_equal(vector, scalar)
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
    model = NeuralPolicy(config, training_model_config("small"))
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


def test_vector_collector_runs_fixed_and_neural_opponent_mix() -> None:
    torch.manual_seed(109)
    config = training_encoder_config()
    model = NeuralPolicy(config, training_model_config("small"))
    plans = plan_strong_field_episodes(
        root_seed=109,
        update_index=0,
        games_per_cell=1,
        policy_identity="current",
        fixed_opponent_share=Fraction(1, 2),
    )

    scalar, _ = collect_self_play(
        {"current": model},
        plans,
        encoder_config=config,
        reward_config=RewardConfig(),
        device=torch.device("cpu"),
        active_games=64,
        max_inference_batch=512,
    )
    rollout, metrics = collect_self_play_vectorized(
        {"current": model},
        plans,
        encoder_config=config,
        reward_config=RewardConfig(),
        device=torch.device("cpu"),
        engine_batch_size=64,
        max_inference_batch=512,
    )

    assert metrics.games == 15
    assert len(rollout.transitions) < metrics.decisions
    _assert_rollouts_equal(rollout, scalar)
    assert all(
        sum(trajectory.trainable for trajectory in episode.trajectories) == 1
        for episode in rollout.episodes
    )
