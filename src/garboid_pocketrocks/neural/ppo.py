"""One-epoch legal-action-masked PPO for the Stage 1 training proof."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from garboid_pocketrocks.neural.advantages import compute_gae
from garboid_pocketrocks.neural.metrics import (
    ValueMetrics,
    ValueMetricSlice,
    stratified_value_metrics,
)
from garboid_pocketrocks.neural.model import NeuralPolicy
from garboid_pocketrocks.neural.policy import evaluate_masked_policy
from garboid_pocketrocks.neural.rollout import PackedRollout, RolloutBatch

_ADVANTAGE_EPSILON = 1e-8


class PPOError(ValueError):
    """Raised when inputs cannot produce a finite Stage 1 PPO update."""


@dataclass(frozen=True, slots=True)
class PPOConfig:
    """Exact Stage 1 PPO defaults."""

    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_loss_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_gradient_norm: float = 0.5
    learning_rate: float = 3e-4
    epochs: int = 1
    minibatch_size: int = 512

    def __post_init__(self) -> None:
        coefficients = (
            self.gamma,
            self.gae_lambda,
            self.clip_ratio,
            self.value_loss_coefficient,
            self.entropy_coefficient,
            self.max_gradient_norm,
            self.learning_rate,
        )
        if not all(math.isfinite(value) for value in coefficients):
            raise PPOError("PPO coefficients must be finite")
        if self.gamma != 1.0:
            raise PPOError("Stage 1 requires gamma=1.0")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise PPOError("GAE lambda must be between zero and one")
        if self.clip_ratio <= 0.0:
            raise PPOError("clip ratio must be positive")
        if self.value_loss_coefficient < 0.0 or self.entropy_coefficient < 0.0:
            raise PPOError("loss coefficients must be nonnegative")
        if self.max_gradient_norm <= 0.0:
            raise PPOError("maximum gradient norm must be positive")
        if self.learning_rate <= 0.0:
            raise PPOError("learning rate must be positive")
        if not isinstance(self.epochs, int) or isinstance(self.epochs, bool) or self.epochs <= 0:
            raise PPOError("epochs must be a positive integer")
        if (
            not isinstance(self.minibatch_size, int)
            or isinstance(self.minibatch_size, bool)
            or self.minibatch_size <= 0
        ):
            raise PPOError("minibatch size must be a positive integer")


@dataclass(frozen=True, slots=True)
class PPOLoss:
    """Differentiable clipped PPO loss and its diagnostic components."""

    total: Tensor
    policy: Tensor
    value: Tensor
    entropy: Tensor
    ratio: Tensor


@dataclass(frozen=True, slots=True)
class PPOUpdateMetrics:
    """Finite diagnostics from one complete PPO update."""

    epochs: int
    optimizer_steps: int
    transition_count: int
    total_loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    advantages: tuple[float, ...]
    ratios: tuple[float, ...]
    values: tuple[float, ...]
    entropies: tuple[float, ...]
    pre_clip_gradient_norms: tuple[float, ...]
    post_clip_gradient_norms: tuple[float, ...]
    approximate_kl: float
    clip_fraction: float
    value: ValueMetrics
    value_slices: tuple[ValueMetricSlice, ...]


@dataclass(frozen=True, slots=True)
class _TrainingTargets:
    packed: PackedRollout
    advantages: Tensor
    returns: Tensor


def ppo_loss(
    new_log_probability: Tensor,
    new_value: Tensor,
    old_log_probability: Tensor,
    return_target: Tensor,
    advantage: Tensor,
    entropy: Tensor,
    *,
    config: PPOConfig,
) -> PPOLoss:
    """Compute the exact clipped Stage 1 objective for one minibatch."""

    tensors = (
        new_log_probability,
        new_value,
        old_log_probability,
        return_target,
        advantage,
        entropy,
    )
    if any(tensor.ndim != 1 for tensor in tensors):
        raise PPOError("PPO loss inputs must be one-dimensional")
    if len({tensor.shape[0] for tensor in tensors}) != 1:
        raise PPOError("PPO loss inputs must have matching lengths")
    if tensors[0].shape[0] == 0:
        raise PPOError("PPO loss requires at least one transition")
    if any(not torch.is_floating_point(tensor) for tensor in tensors):
        raise PPOError("PPO loss inputs must be floating-point tensors")
    if any(not torch.isfinite(tensor).all().item() for tensor in tensors):
        raise PPOError("PPO loss inputs must be finite")

    ratio = torch.exp(new_log_probability - old_log_probability)
    unclipped = ratio * advantage
    clipped = (
        torch.clamp(
            ratio,
            1.0 - config.clip_ratio,
            1.0 + config.clip_ratio,
        )
        * advantage
    )
    policy = -torch.minimum(unclipped, clipped).mean()
    value = 0.5 * torch.square(new_value - return_target).mean()
    mean_entropy = entropy.mean()
    total = (
        policy
        + (config.value_loss_coefficient * value)
        - (config.entropy_coefficient * mean_entropy)
    )
    outputs = (ratio, policy, value, mean_entropy, total)
    if any(not torch.isfinite(output).all().item() for output in outputs):
        raise PPOError("PPO loss produced a nonfinite value")
    return PPOLoss(
        total=total,
        policy=policy,
        value=value,
        entropy=mean_entropy,
        ratio=ratio,
    )


class PPOTrainer:
    """Own one persistent Adam optimizer on the model's learner device."""

    def __init__(
        self,
        model: NeuralPolicy,
        config: PPOConfig = PPOConfig(),  # noqa: B008
    ) -> None:
        try:
            device = next(model.parameters()).device
        except StopIteration as error:
            raise PPOError("PPO model has no parameters") from error
        self.model = model
        self.device = device
        self.config = config
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            foreach=False,
        )

    def update(
        self,
        rollout: RolloutBatch,
        *,
        update_seed: int,
    ) -> PPOUpdateMetrics:
        """Pack a frozen rollout and apply deterministic PPO epochs."""

        if (
            not isinstance(update_seed, int)
            or isinstance(update_seed, bool)
            or not 0 <= update_seed < 2**63
        ):
            raise PPOError("update seed must be an unsigned 63-bit integer")
        _ensure_model_finite(self.model)
        targets = _training_targets(rollout, self.config)
        packed = targets.packed
        transition_count = len(packed)

        total_loss_sum = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        entropy_sum = 0.0
        ratios: list[float] = []
        values: list[float] = []
        entropies: list[float] = []
        pre_clip_norms: list[float] = []
        post_clip_norms: list[float] = []
        optimizer_steps = 0
        approximate_kl_sum = 0.0
        clip_count = 0
        self.model.train()

        for epoch_index in range(self.config.epochs):
            generator = torch.Generator(device="cpu").manual_seed(
                _derive_local_seed(update_seed, "epoch", epoch_index)
            )
            shuffled = torch.randperm(transition_count, generator=generator)
            for start in range(0, transition_count, self.config.minibatch_size):
                index_tensor = shuffled[start : start + self.config.minibatch_size]
                indices = np.asarray(index_tensor.tolist(), dtype=np.int64)
                batch = packed.batch(indices, self.device)
                output = self.model(batch)
                selection = evaluate_masked_policy(
                    output,
                    batch,
                    generator=None,
                    deterministic=True,
                )
                actions = torch.as_tensor(
                    packed.actions[indices],
                    dtype=torch.int64,
                    device=self.device,
                )
                if not batch.action_mask.gather(1, actions.unsqueeze(1)).all().item():
                    raise PPOError("rollout contains an action illegal under its stored mask")
                log_probabilities = torch.log_softmax(
                    selection.masked_logits,
                    dim=-1,
                )
                new_log_probability = log_probabilities.gather(
                    1,
                    actions.unsqueeze(1),
                ).squeeze(1)
                old_log_probability = torch.as_tensor(
                    packed.old_log_probabilities[indices],
                    dtype=output.value.dtype,
                    device=self.device,
                )
                advantage = targets.advantages[index_tensor].to(self.device)
                return_target = targets.returns[index_tensor].to(self.device)
                loss = ppo_loss(
                    new_log_probability,
                    output.value,
                    old_log_probability,
                    return_target,
                    advantage,
                    selection.entropy,
                    config=self.config,
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.total.backward()  # type: ignore[no-untyped-call]
                _ensure_gradients_finite(self.model)
                pre_clip = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_gradient_norm,
                    error_if_nonfinite=True,
                )
                post_clip = _gradient_norm(self.model)
                if not torch.isfinite(pre_clip).item() or not math.isfinite(post_clip):
                    raise PPOError("gradient clipping produced a nonfinite norm")
                self.optimizer.step()
                _ensure_model_finite(self.model)
                _ensure_optimizer_finite(self.optimizer)

                size = len(indices)
                total_loss_sum += float(loss.total.detach().item()) * size
                policy_loss_sum += float(loss.policy.detach().item()) * size
                value_loss_sum += float(loss.value.detach().item()) * size
                entropy_sum += float(loss.entropy.detach().item()) * size
                approximate_kl_sum += float(
                    (old_log_probability - new_log_probability).detach().sum().item()
                )
                clip_count += int(
                    (torch.abs(loss.ratio.detach() - 1.0) > self.config.clip_ratio).sum().item()
                )
                ratios.extend(_tensor_values(loss.ratio))
                values.extend(_tensor_values(output.value))
                entropies.extend(_tensor_values(selection.entropy))
                pre_clip_norms.append(float(pre_clip.detach().item()))
                post_clip_norms.append(post_clip)
                optimizer_steps += 1

        denominator = float(transition_count * self.config.epochs)
        phases = ("early", "middle", "late")
        value_slices = stratified_value_metrics(
            torch.from_numpy(packed.old_values.copy()),
            targets.returns,
            ruleset_names=tuple(
                f"live-{chr(ord('A') + int(index))}" for index in packed.chart_indices
            ),
            player_counts=tuple(int(count) for count in packed.player_counts),
            phases=tuple(phases[int(index)] for index in packed.phase_buckets),
        )
        return PPOUpdateMetrics(
            epochs=self.config.epochs,
            optimizer_steps=optimizer_steps,
            transition_count=transition_count,
            total_loss=total_loss_sum / denominator,
            policy_loss=policy_loss_sum / denominator,
            value_loss=value_loss_sum / denominator,
            entropy=entropy_sum / denominator,
            advantages=_tensor_values(targets.advantages),
            ratios=tuple(ratios),
            values=tuple(values),
            entropies=tuple(entropies),
            pre_clip_gradient_norms=tuple(pre_clip_norms),
            post_clip_gradient_norms=tuple(post_clip_norms),
            approximate_kl=approximate_kl_sum / denominator,
            clip_fraction=clip_count / denominator,
            value=value_slices[0].metrics,
            value_slices=value_slices,
        )


def _training_targets(
    rollout: RolloutBatch,
    config: PPOConfig,
) -> _TrainingTargets:
    try:
        packed = PackedRollout.from_batch(rollout)
    except ValueError as error:
        raise PPOError(str(error)) from error
    advantages: list[Tensor] = []
    returns: list[Tensor] = []
    for start, end in packed.trajectory_ranges:
        estimate = compute_gae(
            torch.from_numpy(packed.rewards[start:end].copy()),
            torch.from_numpy(packed.old_values[start:end].copy()),
            torch.from_numpy(packed.terminated[start:end].copy()),
            torch.from_numpy(packed.truncated[start:end].copy()),
            bootstrap_value=torch.tensor(0.0, dtype=torch.float32),
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
        advantages.append(estimate.advantages)
        returns.append(estimate.returns)

    concatenated_advantages = torch.cat(advantages)
    normalized_advantages = (concatenated_advantages - concatenated_advantages.mean()) / (
        concatenated_advantages.std(unbiased=False) + _ADVANTAGE_EPSILON
    )
    concatenated_returns = torch.cat(returns)
    if (
        not torch.isfinite(normalized_advantages).all().item()
        or not torch.isfinite(concatenated_returns).all().item()
    ):
        raise PPOError("advantage preparation produced nonfinite targets")
    return _TrainingTargets(
        packed=packed,
        advantages=normalized_advantages,
        returns=concatenated_returns,
    )


def _derive_local_seed(root_seed: int, namespace: str, index: int) -> int:
    canonical = f"{root_seed}:{namespace}:{index}".encode()
    return int.from_bytes(
        hashlib.blake2b(canonical, digest_size=8).digest(),
        "big",
    ) & ((1 << 63) - 1)


def _tensor_values(tensor: Tensor) -> tuple[float, ...]:
    return tuple(float(value) for value in tensor.detach().cpu().tolist())


def _ensure_model_finite(model: NeuralPolicy) -> None:
    if any(not torch.isfinite(parameter).all().item() for parameter in model.parameters()):
        raise PPOError("model contains a nonfinite parameter")


def _ensure_gradients_finite(model: NeuralPolicy) -> None:
    gradients = tuple(
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    )
    if not gradients:
        raise PPOError("PPO loss produced no gradients")
    if any(not torch.isfinite(gradient).all().item() for gradient in gradients):
        raise PPOError("PPO loss produced a nonfinite gradient")


def _gradient_norm(model: NeuralPolicy) -> float:
    squared_norm = sum(
        float(torch.sum(parameter.grad.detach() ** 2).item())
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    return math.sqrt(squared_norm)


def _ensure_optimizer_finite(optimizer: torch.optim.Adam) -> None:
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, Tensor) and not torch.isfinite(value).all().item():
                raise PPOError("optimizer contains nonfinite state")
