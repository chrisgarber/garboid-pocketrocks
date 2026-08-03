from __future__ import annotations

import io
import math
from typing import cast

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from garboid_pocketrocks.neural.advantages import compute_gae  # noqa: E402
from garboid_pocketrocks.neural.collector import collect_self_play  # noqa: E402
from garboid_pocketrocks.neural.config import (  # noqa: E402
    training_encoder_config,
    training_model_config,
)
from garboid_pocketrocks.neural.metrics import gameplay_metrics  # noqa: E402
from garboid_pocketrocks.neural.model import NeuralPolicy  # noqa: E402
from garboid_pocketrocks.neural.planning import (  # noqa: E402
    SelfPlayEpisodePlan,
    plan_mirror_episodes,
)
from garboid_pocketrocks.neural.ppo import (  # noqa: E402
    PPOConfig,
    PPOError,
    PPOTrainer,
    ppo_loss,
)
from garboid_pocketrocks.neural.rollout import (  # noqa: E402
    PackedRollout,
    RolloutBatch,
)
from garboid_pocketrocks.neural.seeding import (  # noqa: E402
    configure_torch_runtime,
    derive_seed,
)
from garboid_pocketrocks.training.rewards import RewardConfig  # noqa: E402


def _model(root_seed: int) -> NeuralPolicy:
    torch.manual_seed(derive_seed(root_seed, "model"))
    return NeuralPolicy(training_encoder_config(), training_model_config("small"))


def _one_game_rollout(
    model: NeuralPolicy,
    root_seed: int,
) -> tuple[tuple[SelfPlayEpisodePlan, ...], RolloutBatch]:
    plans = plan_mirror_episodes(
        root_seed=root_seed,
        update_index=0,
        games_per_cell=1,
        policy_identity="current",
    )[:1]
    rollout, _ = collect_self_play(
        {"current": model},
        plans,
        encoder_config=training_encoder_config(),
        reward_config=RewardConfig(),
        device=torch.device("cpu"),
        active_games=1,
        max_inference_batch=32,
    )
    return plans, rollout


def _serialized_torch_state(value: object) -> bytes:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def test_ppo_loss_matches_the_clipped_formula() -> None:
    config = PPOConfig()
    old_log_probability = torch.log(torch.tensor((0.50, 0.50, 0.25)))
    new_log_probability = torch.log(torch.tensor((0.75, 0.40, 0.50)))
    advantage = torch.tensor((2.0, -1.0, 0.5))
    new_value = torch.tensor((1.5, -0.5, 2.0))
    return_target = torch.tensor((1.0, 0.5, 1.5))
    entropy = torch.tensor((0.4, 0.6, 0.8))

    loss = ppo_loss(
        new_log_probability,
        new_value,
        old_log_probability,
        return_target,
        advantage,
        entropy,
        config=config,
    )

    ratio = torch.exp(new_log_probability - old_log_probability)
    expected_policy = -torch.minimum(
        ratio * advantage,
        torch.clamp(ratio, 0.8, 1.2) * advantage,
    ).mean()
    expected_value = 0.5 * torch.square(new_value - return_target).mean()
    expected_entropy = entropy.mean()
    expected_total = expected_policy + (0.5 * expected_value) - (0.01 * expected_entropy)
    torch.testing.assert_close(loss.ratio, ratio)
    torch.testing.assert_close(loss.policy, expected_policy)
    torch.testing.assert_close(loss.value, expected_value)
    torch.testing.assert_close(loss.entropy, expected_entropy)
    torch.testing.assert_close(loss.total, expected_total)

    assert config == PPOConfig(
        gamma=1.0,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_loss_coefficient=0.5,
        entropy_coefficient=0.01,
        max_gradient_norm=0.5,
        learning_rate=3e-4,
        epochs=1,
        minibatch_size=512,
    )


def test_update_does_not_change_global_torch_rng() -> None:
    configure_torch_runtime(197, deterministic_algorithms=True)
    model = _model(197)
    _, rollout = _one_game_rollout(model, 197)
    trainer = PPOTrainer(model, PPOConfig(epochs=2, minibatch_size=16))
    before = torch.get_rng_state().clone()

    trainer.update(rollout, update_seed=123)

    assert torch.equal(torch.get_rng_state(), before)


def test_policy_shift_detects_stale_rollout_without_changing_model_mode() -> None:
    configure_torch_runtime(198, deterministic_algorithms=True)
    model = _model(198)
    _, rollout = _one_game_rollout(model, 198)
    trainer = PPOTrainer(model, PPOConfig(minibatch_size=16))
    model.train()

    fresh = trainer.measure_policy_shift(rollout)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.05)
    stale = trainer.measure_policy_shift(rollout)

    assert fresh.approximate_kl == pytest.approx(0.0, abs=1e-6)
    assert fresh.clip_fraction == 0.0
    assert stale.approximate_kl > fresh.approximate_kl
    assert stale.clip_fraction > fresh.clip_fraction
    assert model.training is True


def test_update_is_exactly_repeatable_for_metrics_parameters_and_optimizer() -> None:
    configure_torch_runtime(199, deterministic_algorithms=True)
    rollout_model = _model(199)
    initial_state = {
        name: tensor.detach().clone() for name, tensor in rollout_model.state_dict().items()
    }
    _, rollout = _one_game_rollout(rollout_model, 199)
    first_model = _model(1)
    first_model.load_state_dict(initial_state)
    second_model = _model(2)
    second_model.load_state_dict(initial_state)
    config = PPOConfig(epochs=2, minibatch_size=16)
    first_trainer = PPOTrainer(first_model, config)
    second_trainer = PPOTrainer(second_model, config)

    torch.manual_seed(1)
    first = first_trainer.update(rollout, update_seed=123)
    torch.manual_seed(2)
    second = second_trainer.update(rollout, update_seed=123)

    assert first == second
    for name, tensor in first_model.state_dict().items():
        assert torch.equal(tensor, second_model.state_dict()[name])
    assert _serialized_torch_state(first_trainer.optimizer.state_dict()) == _serialized_torch_state(
        second_trainer.optimizer.state_dict()
    )


def test_minibatch_iteration_matches_one_seeded_cpu_permutation_per_epoch() -> None:
    from garboid_pocketrocks.neural.ppo import (
        _derive_local_seed,
        _iter_minibatch_indices,
    )

    actual = tuple(
        _iter_minibatch_indices(
            transition_count=7,
            epochs=2,
            minibatch_size=3,
            update_seed=123,
        )
    )
    expected: list[np.ndarray[tuple[int], np.dtype[np.int64]]] = []
    for epoch_index in range(2):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_derive_local_seed(123, "epoch", epoch_index))
        permutation = torch.randperm(7, generator=generator).numpy()
        expected.extend(permutation[start : start + 3] for start in range(0, 7, 3))

    assert all(indices.dtype == np.int64 for indices in actual)
    assert [indices.tolist() for indices in actual] == [indices.tolist() for indices in expected]


@pytest.mark.parametrize("update_seed", (-1, True, 1.5, "123"))
def test_update_seed_validation_rejects_values_that_are_not_nonnegative_integers(
    update_seed: object,
) -> None:
    from garboid_pocketrocks.neural.ppo import _validate_update_seed

    with pytest.raises(PPOError, match="update seed must be a nonnegative integer"):
        _validate_update_seed(cast(int, update_seed))


def test_update_seed_validation_accepts_large_nonnegative_integers() -> None:
    from garboid_pocketrocks.neural.ppo import _validate_update_seed

    _validate_update_seed(0)
    _validate_update_seed(2**63)


def test_update_flattens_valid_transitions_and_is_deterministic() -> None:
    configure_torch_runtime(211, deterministic_algorithms=True)
    first_model = _model(211)
    initial_state = {
        name: tensor.detach().clone() for name, tensor in first_model.state_dict().items()
    }
    plans, first_rollout = _one_game_rollout(first_model, 211)
    old_policy_quantities = tuple(
        (transition.old_log_probability, transition.old_value)
        for transition in first_rollout.transitions
    )
    first_trainer = PPOTrainer(first_model, PPOConfig())
    optimizer = first_trainer.optimizer

    first_metrics = first_trainer.update(
        first_rollout,
        update_seed=derive_seed(211, "minibatch", 0),
    )

    assert first_trainer.optimizer is optimizer
    assert first_metrics.epochs == 1
    assert first_metrics.optimizer_steps == 1
    assert first_metrics.transition_count == len(first_rollout.transitions)
    assert len(first_metrics.advantages) == first_metrics.transition_count
    assert len(first_metrics.ratios) == first_metrics.transition_count
    assert len(first_metrics.values) == first_metrics.transition_count
    assert len(first_metrics.entropies) == first_metrics.transition_count
    assert len(first_metrics.pre_clip_gradient_norms) == first_metrics.optimizer_steps
    assert len(first_metrics.post_clip_gradient_norms) == first_metrics.optimizer_steps
    diagnostic_fields = (
        "total_loss",
        "policy_loss",
        "value_loss",
        "entropy",
        "advantages",
        "ratios",
        "values",
        "entropies",
        "pre_clip_gradient_norms",
        "post_clip_gradient_norms",
        "approximate_kl",
        "clip_fraction",
    )
    assert all(
        math.isfinite(value)
        for name in diagnostic_fields
        for value in (
            getattr(first_metrics, name)
            if isinstance(getattr(first_metrics, name), tuple)
            else (getattr(first_metrics, name),)
        )
    )
    assert max(first_metrics.post_clip_gradient_norms) <= 0.5
    assert any(
        not torch.equal(tensor, initial_state[name])
        for name, tensor in first_model.state_dict().items()
    )
    assert old_policy_quantities == tuple(
        (transition.old_log_probability, transition.old_value)
        for transition in first_rollout.transitions
    )

    episode_advantages = []
    for episode in first_rollout.episodes:
        for trajectory in episode.trajectories:
            if trajectory.trainable:
                transitions = trajectory.transitions
                estimated = compute_gae(
                    torch.tensor([transition.reward for transition in transitions]),
                    torch.tensor([transition.old_value for transition in transitions]),
                    torch.tensor([transition.terminated for transition in transitions]),
                    torch.tensor([transition.truncated for transition in transitions]),
                    bootstrap_value=torch.tensor(0.0),
                    gamma=1.0,
                    gae_lambda=0.95,
                )
                episode_advantages.append(estimated.advantages)
    expected_advantages = torch.cat(episode_advantages)
    expected_advantages = (expected_advantages - expected_advantages.mean()) / (
        expected_advantages.std(unbiased=False) + 1e-8
    )
    torch.testing.assert_close(
        torch.tensor(first_metrics.advantages),
        expected_advantages,
    )

    second_model = _model(999)
    second_model.load_state_dict(initial_state)
    second_rollout, _ = collect_self_play(
        {"current": second_model},
        plans,
        encoder_config=training_encoder_config(),
        reward_config=RewardConfig(),
        device=torch.device("cpu"),
        active_games=1,
        max_inference_batch=32,
    )
    second_trainer = PPOTrainer(second_model, PPOConfig())
    second_metrics = second_trainer.update(
        second_rollout,
        update_seed=derive_seed(211, "minibatch", 0),
    )

    assert second_metrics == first_metrics
    for name, tensor in first_model.state_dict().items():
        assert torch.equal(tensor, second_model.state_dict()[name])


def test_update_uses_stored_masks_when_recomputing_policy_loss() -> None:
    configure_torch_runtime(307, deterministic_algorithms=True)
    baseline_model = _model(307)
    initial_state = {
        name: tensor.detach().clone() for name, tensor in baseline_model.state_dict().items()
    }
    _, rollout = _one_game_rollout(baseline_model, 307)
    assert all(not transition.observation.action_mask[100] for transition in rollout.transitions)

    changed_illegal_model = _model(999)
    changed_illegal_model.load_state_dict(initial_state)
    with torch.no_grad():
        changed_illegal_model.bid_head.bias[100] += 1_000.0

    seed = derive_seed(307, "minibatch", 0)
    baseline_metrics = PPOTrainer(baseline_model, PPOConfig()).update(
        rollout,
        update_seed=seed,
    )
    changed_metrics = PPOTrainer(changed_illegal_model, PPOConfig()).update(
        rollout,
        update_seed=seed,
    )

    assert changed_metrics.total_loss == baseline_metrics.total_loss
    assert changed_metrics.policy_loss == baseline_metrics.policy_loss
    assert changed_metrics.value_loss == baseline_metrics.value_loss
    assert changed_metrics.entropy == baseline_metrics.entropy
    assert changed_metrics.ratios == baseline_metrics.ratios


def test_packed_rollout_and_multi_epoch_ppo_support_all_seats() -> None:
    configure_torch_runtime(401, deterministic_algorithms=True)
    encoder_config = training_encoder_config()
    model = NeuralPolicy(encoder_config, training_model_config("small"))
    plans = plan_mirror_episodes(
        root_seed=401,
        update_index=0,
        games_per_cell=1,
        policy_identity="current",
    )[:1]
    rollout, _ = collect_self_play(
        {"current": model},
        plans,
        encoder_config=encoder_config,
        reward_config=RewardConfig(),
        device=torch.device("cpu"),
        active_games=1,
        max_inference_batch=32,
    )

    packed = PackedRollout.from_batch(rollout)
    gameplay = gameplay_metrics(rollout)
    metrics = PPOTrainer(
        model,
        PPOConfig(epochs=2, minibatch_size=16),
    ).update(rollout, update_seed=17)

    assert len(packed) == len(rollout.transitions)
    assert len(packed.trajectory_ranges) == 3
    assert packed.observation(0).action_mask[packed.actions[0]]
    assert not packed.actions.flags.writeable
    assert set(packed.phase_buckets.tolist()) == {0, 1, 2}
    assert gameplay[0].metrics.games == 3
    assert gameplay[0].metrics.decisions == len(packed)
    assert gameplay[1].key == "live-A/3"
    assert metrics.epochs == 2
    assert metrics.optimizer_steps == 2 * math.ceil(len(packed) / 16)
    assert metrics.transition_count == len(packed)
    assert math.isfinite(metrics.approximate_kl)
    assert 0.0 <= metrics.clip_fraction <= 1.0
    assert metrics.value.count == metrics.transition_count
    assert {item.dimension for item in metrics.value_slices} == {
        "all",
        "ruleset",
        "player_count",
        "phase",
    }
