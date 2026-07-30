from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from garboid_pocketrocks.neural.collector import (  # noqa: E402
    CollectorError,
    _devices_match,
    _infer_policy_requests,
    collect_self_play,
)
from garboid_pocketrocks.neural.config import (  # noqa: E402
    training_encoder_config,
    training_model_config,
)
from garboid_pocketrocks.neural.model import NeuralPolicy  # noqa: E402
from garboid_pocketrocks.neural.planning import (  # noqa: E402
    SelfPlayEpisodePlan,
    plan_mirror_episodes,
)
from garboid_pocketrocks.neural.rollout import MultiSeatEpisode  # noqa: E402
from garboid_pocketrocks.neural.self_play import (  # noqa: E402
    PendingPolicyRequest,
    PolicyResponse,
    SelfPlayGame,
)
from garboid_pocketrocks.training.rewards import RewardConfig  # noqa: E402


def _plans() -> tuple[SelfPlayEpisodePlan, ...]:
    return plan_mirror_episodes(
        root_seed=42,
        update_index=0,
        games_per_cell=1,
        policy_identity="current",
    )


def _pending_requests() -> tuple[PendingPolicyRequest, ...]:
    return SelfPlayGame.start(
        _plans()[0],
        encoder_config=training_encoder_config(),
        reward_config=RewardConfig(),
    ).pending_requests()


def test_inference_is_request_order_independent() -> None:
    torch.manual_seed(27)
    encoder_config = training_encoder_config()
    model = NeuralPolicy(encoder_config, training_model_config("small"))
    requests = _pending_requests()

    first_sizes: list[int] = []
    first, _ = _infer_policy_requests(
        {"current": model},
        requests,
        device=torch.device("cpu"),
        max_inference_batch=2,
        inference_batch_sizes=first_sizes,
    )
    reversed_sizes: list[int] = []
    reversed_responses, _ = _infer_policy_requests(
        {"current": model},
        tuple(reversed(requests)),
        device=torch.device("cpu"),
        max_inference_batch=2,
        inference_batch_sizes=reversed_sizes,
    )

    def keyed(
        responses: tuple[PolicyResponse, ...],
    ) -> dict[tuple[int, int, int], tuple[int, float, float]]:
        return {
            (response.episode_index, response.seat, response.decision_index): (
                response.action,
                response.old_log_probability,
                response.old_value,
            )
            for response in responses
        }

    expected_keys = sorted(
        (request.episode_index, request.seat, request.decision_index) for request in requests
    )
    assert [
        (response.episode_index, response.seat, response.decision_index) for response in first
    ] == expected_keys
    assert keyed(first) == keyed(reversed_responses)
    assert first_sizes == reversed_sizes == [2, 1]


@pytest.mark.parametrize("training", (False, True))
def test_inference_restores_policy_mode_after_failure(
    monkeypatch: pytest.MonkeyPatch,
    training: bool,
) -> None:
    encoder_config = training_encoder_config()
    model = NeuralPolicy(encoder_config, training_model_config("small"))
    model.train(training)

    def fail_inference(_batch: object) -> None:
        raise RuntimeError("inference failed")

    monkeypatch.setattr(model, "forward", fail_inference)

    with pytest.raises(RuntimeError, match="inference failed"):
        _infer_policy_requests(
            {"current": model},
            _pending_requests(),
            device=torch.device("cpu"),
            max_inference_batch=8,
            inference_batch_sizes=[],
        )

    assert model.training is training


def test_collector_batches_all_live_chart_and_player_count_cells() -> None:
    torch.manual_seed(9)
    encoder_config = training_encoder_config()
    model = NeuralPolicy(encoder_config, training_model_config("small"))
    model.train()
    parameters_before = tuple(parameter.detach().clone() for parameter in model.parameters())

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
    assert rollout.episodes
    assert all(isinstance(episode, MultiSeatEpisode) for episode in rollout.episodes)
    assert not hasattr(rollout, "multi_seat_episodes")
    assert Counter(
        (episode.plan.ruleset_name, episode.plan.player_count) for episode in rollout.episodes
    ) == {(f"live-{chart}", player_count): 1 for chart in "ABCDE" for player_count in (3, 4, 5)}
    assert metrics.cell_games == tuple(
        (f"live-{chart}", player_count, 1) for chart in "ABCDE" for player_count in (3, 4, 5)
    )
    assert all(
        len(episode.trajectories) == episode.plan.player_count for episode in rollout.episodes
    )
    assert rollout.transitions == tuple(
        transition
        for episode in rollout.episodes
        for trajectory in episode.trajectories
        if trajectory.trainable
        for transition in trajectory.transitions
    )
    assert all(
        transition.observation.action_mask[transition.action] for transition in rollout.transitions
    )
    assert model.training
    for before, after in zip(parameters_before, model.parameters(), strict=True):
        torch.testing.assert_close(before, after)


def test_collector_is_schedule_independent() -> None:
    torch.manual_seed(18)
    encoder_config = training_encoder_config()
    model = NeuralPolicy(encoder_config, training_model_config("small"))
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
        for episode in first.episodes
        for trajectory in episode.trajectories
        for index, transition in enumerate(trajectory.transitions)
    }
    second_actions = {
        (
            transition.metadata.environment_seed,
            transition.metadata.learner_seat,
            index,
        ): transition.action
        for episode in second.episodes
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
    model = NeuralPolicy(encoder_config, training_model_config("small"))

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

    incompatible = NeuralPolicy(
        replace(encoder_config, supported_ruleset_names=("live-A",)),
        training_model_config("small"),
    )
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


def test_unindexed_accelerator_device_matches_default_device_zero() -> None:
    assert _devices_match(torch.device("mps"), torch.device("mps:0"))
    assert _devices_match(torch.device("cuda"), torch.device("cuda:0"))
    assert not _devices_match(torch.device("cuda:1"), torch.device("cuda:0"))
