from __future__ import annotations

from collections import Counter

import pytest

torch = pytest.importorskip("torch")

from garboid_pocketrocks.neural.collector import (  # noqa: E402
    CollectorError,
    collect_self_play,
)
from garboid_pocketrocks.neural.config import (  # noqa: E402
    stage1_encoder_config,
    stage1_model_config,
    training_encoder_config,
)
from garboid_pocketrocks.neural.model import NeuralPolicy  # noqa: E402
from garboid_pocketrocks.neural.planning import (  # noqa: E402
    SelfPlayEpisodePlan,
    plan_mirror_episodes,
)
from garboid_pocketrocks.training.rewards import RewardConfig  # noqa: E402


def _plans() -> tuple[SelfPlayEpisodePlan, ...]:
    return plan_mirror_episodes(
        root_seed=42,
        update_index=0,
        games_per_cell=1,
        policy_identity="current",
    )


def test_collector_batches_all_live_chart_and_player_count_cells() -> None:
    torch.manual_seed(9)
    encoder_config = training_encoder_config()
    model = NeuralPolicy(encoder_config, stage1_model_config())
    model.train()
    parameters_before = tuple(
        parameter.detach().clone() for parameter in model.parameters()
    )

    rollout, metrics = collect_self_play(
        {"current": model},
        _plans(),
        encoder_config=encoder_config,
        reward_config=RewardConfig(),
        device=torch.device("cpu"),
        active_games=15,
        max_inference_batch=64,
    )

    assert metrics.games == 15
    assert metrics.decisions == len(rollout.transitions)
    assert metrics.elapsed_seconds > 0.0
    assert 0.0 <= metrics.inference_seconds <= metrics.elapsed_seconds
    assert metrics.inference_batches == len(metrics.inference_batch_sizes)
    assert max(metrics.inference_batch_sizes) > 1
    assert Counter(
        (episode.plan.ruleset_name, episode.plan.player_count)
        for episode in rollout.multi_seat_episodes
    ) == {
        (f"live-{chart}", player_count): 1
        for chart in "ABCDE"
        for player_count in (3, 4, 5)
    }
    assert metrics.cell_games == tuple(
        (f"live-{chart}", player_count, 1)
        for chart in "ABCDE"
        for player_count in (3, 4, 5)
    )
    assert all(
        len(episode.trajectories) == episode.plan.player_count
        for episode in rollout.multi_seat_episodes
    )
    assert all(
        transition.observation.action_mask[transition.action]
        for transition in rollout.transitions
    )
    assert model.training
    for before, after in zip(parameters_before, model.parameters(), strict=True):
        torch.testing.assert_close(before, after)


def test_collector_is_schedule_independent() -> None:
    torch.manual_seed(18)
    encoder_config = training_encoder_config()
    model = NeuralPolicy(encoder_config, stage1_model_config())
    plans = _plans()

    first, _ = collect_self_play(
        {"current": model},
        plans,
        encoder_config=encoder_config,
        reward_config=RewardConfig(),
        device=torch.device("cpu"),
        active_games=1,
        max_inference_batch=8,
    )
    second, _ = collect_self_play(
        {"current": model},
        tuple(reversed(plans)),
        encoder_config=encoder_config,
        reward_config=RewardConfig(),
        device=torch.device("cpu"),
        active_games=15,
        max_inference_batch=64,
    )

    first_actions = {
        (
            transition.metadata.environment_seed,
            transition.metadata.learner_seat,
            index,
        ): transition.action
        for episode in first.multi_seat_episodes
        for trajectory in episode.trajectories
        for index, transition in enumerate(trajectory.transitions)
    }
    second_actions = {
        (
            transition.metadata.environment_seed,
            transition.metadata.learner_seat,
            index,
        ): transition.action
        for episode in second.multi_seat_episodes
        for trajectory in episode.trajectories
        for index, transition in enumerate(trajectory.transitions)
    }
    assert first_actions == second_actions


@pytest.mark.parametrize(
    ("active_games", "max_inference_batch", "message"),
    [
        (0, 8, "active_games"),
        (1, 0, "max_inference_batch"),
    ],
)
def test_collector_rejects_nonpositive_batching_limits(
    active_games: int,
    max_inference_batch: int,
    message: str,
) -> None:
    encoder_config = training_encoder_config()
    model = NeuralPolicy(encoder_config, stage1_model_config())

    with pytest.raises(CollectorError, match=message):
        collect_self_play(
            {"current": model},
            _plans(),
            encoder_config=encoder_config,
            reward_config=RewardConfig(),
            device=torch.device("cpu"),
            active_games=active_games,
            max_inference_batch=max_inference_batch,
        )


def test_collector_rejects_missing_and_incompatible_policies() -> None:
    encoder_config = training_encoder_config()

    with pytest.raises(CollectorError, match="missing"):
        collect_self_play(
            {},
            _plans(),
            encoder_config=encoder_config,
            reward_config=RewardConfig(),
            device=torch.device("cpu"),
            active_games=1,
            max_inference_batch=8,
        )

    incompatible = NeuralPolicy(stage1_encoder_config(), stage1_model_config())
    with pytest.raises(CollectorError, match="encoder"):
        collect_self_play(
            {"current": incompatible},
            _plans(),
            encoder_config=encoder_config,
            reward_config=RewardConfig(),
            device=torch.device("cpu"),
            active_games=1,
            max_inference_batch=8,
        )
