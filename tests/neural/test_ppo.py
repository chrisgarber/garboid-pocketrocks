from __future__ import annotations

import math
from dataclasses import fields

import pytest

torch = pytest.importorskip("torch")

from garboid_pocketrocks.neural.advantages import compute_gae  # noqa: E402
from garboid_pocketrocks.neural.config import (  # noqa: E402
    stage1_encoder_config,
    stage1_model_config,
)
from garboid_pocketrocks.neural.model import NeuralPolicy  # noqa: E402
from garboid_pocketrocks.neural.ppo import (  # noqa: E402
    PPOConfig,
    PPOTrainer,
    ppo_loss,
)
from garboid_pocketrocks.neural.rollout import collect_rollout  # noqa: E402
from garboid_pocketrocks.neural.seeding import (  # noqa: E402
    configure_deterministic_torch,
    derive_seed,
    plan_stage1_episodes,
)


def _model(root_seed: int) -> NeuralPolicy:
    torch.manual_seed(derive_seed(root_seed, "model"))
    return NeuralPolicy(stage1_encoder_config(), stage1_model_config())


def test_ppo_loss_matches_the_clipped_stage1_formula() -> None:
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


def test_update_flattens_valid_transitions_and_is_deterministic() -> None:
    configure_deterministic_torch(211)
    first_model = _model(211)
    initial_state = {
        name: tensor.detach().clone() for name, tensor in first_model.state_dict().items()
    }
    plans = plan_stage1_episodes(root_seed=211, updates=1, games_per_update=2)
    first_rollout = collect_rollout(first_model, plans)
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
    assert first_metrics.transition_count == sum(
        len(episode.transitions) for episode in first_rollout.episodes
    )
    assert len(first_metrics.advantages) == first_metrics.transition_count
    assert len(first_metrics.ratios) == first_metrics.transition_count
    assert len(first_metrics.values) == first_metrics.transition_count
    assert len(first_metrics.entropies) == first_metrics.transition_count
    assert len(first_metrics.pre_clip_gradient_norms) == first_metrics.optimizer_steps
    assert len(first_metrics.post_clip_gradient_norms) == first_metrics.optimizer_steps
    assert all(
        math.isfinite(value)
        for field in fields(first_metrics)
        for value in (
            getattr(first_metrics, field.name)
            if isinstance(getattr(first_metrics, field.name), tuple)
            else (getattr(first_metrics, field.name),)
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
        transitions = episode.transitions
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
    second_rollout = collect_rollout(second_model, plans)
    second_trainer = PPOTrainer(second_model, PPOConfig())
    second_metrics = second_trainer.update(
        second_rollout,
        update_seed=derive_seed(211, "minibatch", 0),
    )

    assert second_metrics == first_metrics
    for name, tensor in first_model.state_dict().items():
        assert torch.equal(tensor, second_model.state_dict()[name])


def test_update_uses_stored_masks_when_recomputing_policy_loss() -> None:
    configure_deterministic_torch(307)
    plans = plan_stage1_episodes(root_seed=307, updates=1, games_per_update=2)
    baseline_model = _model(307)
    initial_state = {
        name: tensor.detach().clone() for name, tensor in baseline_model.state_dict().items()
    }
    rollout = collect_rollout(baseline_model, plans)
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
