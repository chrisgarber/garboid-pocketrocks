"""Legal-action-masked PPO training."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from garboid_pocketrocks.knowledge import ruleset_name
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
    """Raised when inputs cannot produce a finite PPO update."""


@dataclass(frozen=True, slots=True)
class PPOConfig:
    """PPO optimizer and objective defaults."""

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
            raise PPOError("PPO training requires gamma=1.0")
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
class PolicyShiftMetrics:
    """Distance from stored behavior probabilities before a PPO update."""

    approximate_kl: float
    clip_fraction: float
    transition_count: int


@dataclass(frozen=True, slots=True)
class _TrainingTargets:
    packed: PackedRollout
    advantages: Tensor
    returns: Tensor


@dataclass(slots=True)
class _PPOUpdateAccumulator:
    total_loss_sum: float = 0.0
    policy_loss_sum: float = 0.0
    value_loss_sum: float = 0.0
    entropy_sum: float = 0.0
    approximate_kl_sum: float = 0.0
    clip_count: int = 0
    optimizer_steps: int = 0
    ratios: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    entropies: list[float] = field(default_factory=list)
    pre_clip_norms: list[float] = field(default_factory=list)
    post_clip_norms: list[float] = field(default_factory=list)


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
    """Compute the clipped PPO objective for one minibatch."""

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

        _validate_update_seed(update_seed)
        _ensure_model_finite(self.model)
        targets = _training_targets(rollout, self.config)
        packed = targets.packed
        transition_count = len(packed)

        accumulator = _PPOUpdateAccumulator()
        self.model.train()

        for indices in _iter_minibatch_indices(
            transition_count=transition_count,
            epochs=self.config.epochs,
            minibatch_size=self.config.minibatch_size,
            update_seed=update_seed,
        ):
            index_tensor = torch.from_numpy(indices)
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

            _record_minibatch_metrics(
                accumulator,
                loss=loss,
                new_value=output.value,
                entropy=selection.entropy,
                old_log_probability=old_log_probability,
                new_log_probability=new_log_probability,
                pre_clip=pre_clip,
                post_clip=post_clip,
                clip_ratio=self.config.clip_ratio,
            )

        return _build_update_metrics(
            accumulator,
            config=self.config,
            targets=targets,
        )

    def measure_policy_shift(self, rollout: RolloutBatch) -> PolicyShiftMetrics:
        """Measure behavior/current-policy drift without changing model state."""

        packed = PackedRollout.from_batch(rollout)
        approximate_kl_sum = 0.0
        clip_count = 0
        prior_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                for start in range(0, len(packed), self.config.minibatch_size):
                    indices = np.arange(
                        start,
                        min(start + self.config.minibatch_size, len(packed)),
                        dtype=np.int64,
                    )
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
                    new_log_probability = torch.log_softmax(
                        selection.masked_logits,
                        dim=-1,
                    ).gather(1, actions.unsqueeze(1)).squeeze(1)
                    old_log_probability = torch.as_tensor(
                        packed.old_log_probabilities[indices],
                        dtype=new_log_probability.dtype,
                        device=self.device,
                    )
                    log_ratio = new_log_probability - old_log_probability
                    ratio = torch.exp(log_ratio)
                    approximate_kl_sum += float(
                        ((ratio - 1.0) - log_ratio).sum().item()
                    )
                    clip_count += int(
                        (torch.abs(ratio - 1.0) > self.config.clip_ratio).sum().item()
                    )
        finally:
            self.model.train(prior_training)
        return PolicyShiftMetrics(
            approximate_kl=approximate_kl_sum / len(packed),
            clip_fraction=clip_count / len(packed),
            transition_count=len(packed),
        )


def _validate_update_seed(update_seed: int) -> None:
    if not isinstance(update_seed, int) or isinstance(update_seed, bool) or update_seed < 0:
        raise PPOError("update seed must be a nonnegative integer")


def _iter_minibatch_indices(
    *,
    transition_count: int,
    epochs: int,
    minibatch_size: int,
    update_seed: int,
) -> Iterator[NDArray[np.int64]]:
    for epoch_index in range(epochs):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_derive_local_seed(update_seed, "epoch", epoch_index))
        permutation = torch.randperm(
            transition_count,
            generator=generator,
        ).numpy()
        for start in range(0, transition_count, minibatch_size):
            yield permutation[start : start + minibatch_size]


def _record_minibatch_metrics(
    accumulator: _PPOUpdateAccumulator,
    *,
    loss: PPOLoss,
    new_value: Tensor,
    entropy: Tensor,
    old_log_probability: Tensor,
    new_log_probability: Tensor,
    pre_clip: Tensor,
    post_clip: float,
    clip_ratio: float,
) -> None:
    size = loss.ratio.numel()
    accumulator.total_loss_sum += float(loss.total.detach().item()) * size
    accumulator.policy_loss_sum += float(loss.policy.detach().item()) * size
    accumulator.value_loss_sum += float(loss.value.detach().item()) * size
    accumulator.entropy_sum += float(loss.entropy.detach().item()) * size
    accumulator.approximate_kl_sum += float(
        (old_log_probability - new_log_probability).detach().sum().item()
    )
    accumulator.clip_count += int((torch.abs(loss.ratio.detach() - 1.0) > clip_ratio).sum().item())
    accumulator.ratios.extend(_tensor_values(loss.ratio))
    accumulator.values.extend(_tensor_values(new_value))
    accumulator.entropies.extend(_tensor_values(entropy))
    accumulator.pre_clip_norms.append(float(pre_clip.detach().item()))
    accumulator.post_clip_norms.append(post_clip)
    accumulator.optimizer_steps += 1


def _build_update_metrics(
    accumulator: _PPOUpdateAccumulator,
    *,
    config: PPOConfig,
    targets: _TrainingTargets,
) -> PPOUpdateMetrics:
    packed = targets.packed
    transition_count = len(packed)
    denominator = float(transition_count * config.epochs)
    phases = ("early", "middle", "late")
    value_slices = stratified_value_metrics(
        torch.from_numpy(packed.old_values.copy()),
        targets.returns,
        ruleset_names=tuple(
            ruleset_name(chr(ord("A") + int(index))) for index in packed.chart_indices
        ),
        player_counts=tuple(int(count) for count in packed.player_counts),
        phases=tuple(phases[int(index)] for index in packed.phase_buckets),
    )
    return PPOUpdateMetrics(
        epochs=config.epochs,
        optimizer_steps=accumulator.optimizer_steps,
        transition_count=transition_count,
        total_loss=accumulator.total_loss_sum / denominator,
        policy_loss=accumulator.policy_loss_sum / denominator,
        value_loss=accumulator.value_loss_sum / denominator,
        entropy=accumulator.entropy_sum / denominator,
        advantages=_tensor_values(targets.advantages),
        ratios=tuple(accumulator.ratios),
        values=tuple(accumulator.values),
        entropies=tuple(accumulator.entropies),
        pre_clip_gradient_norms=tuple(accumulator.pre_clip_norms),
        post_clip_gradient_norms=tuple(accumulator.post_clip_norms),
        approximate_kl=accumulator.approximate_kl_sum / denominator,
        clip_fraction=accumulator.clip_count / denominator,
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
