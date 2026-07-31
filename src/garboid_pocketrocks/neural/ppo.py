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
from garboid_pocketrocks.neural.heuristic_auxiliary import (
    HeuristicAuxiliaryLoss,
    HeuristicAuxiliaryMetrics,
    HeuristicAuxiliaryValueConfig,
    masked_heuristic_smooth_l1_loss,
)
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
    heuristic_auxiliary_weighted_loss: float
    heuristic_auxiliary: HeuristicAuxiliaryMetrics


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
    auxiliary_unweighted_loss_sum: float = 0.0
    auxiliary_weighted_loss_sum: float = 0.0
    auxiliary_included_count: int = 0
    auxiliary_total_count: int = 0
    auxiliary_prediction_sum: float = 0.0
    auxiliary_target_sum: float = 0.0
    auxiliary_absolute_error_sum: float = 0.0


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
        heuristic_auxiliary_config: HeuristicAuxiliaryValueConfig | None = None,
    ) -> None:
        try:
            device = next(model.parameters()).device
        except StopIteration as error:
            raise PPOError("PPO model has no parameters") from error
        self.model = model
        self.device = device
        self.config = config
        self.heuristic_auxiliary_config = (
            heuristic_auxiliary_config or HeuristicAuxiliaryValueConfig()
        )
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
        targets = _training_targets(
            rollout,
            self.config,
            self.heuristic_auxiliary_config,
        )
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
            auxiliary = masked_heuristic_smooth_l1_loss(
                output.value,
                torch.as_tensor(
                    packed.heuristic_auxiliary_targets[indices],
                    dtype=output.value.dtype,
                    device=self.device,
                ),
                torch.as_tensor(
                    packed.heuristic_auxiliary_included[indices],
                    dtype=torch.bool,
                    device=self.device,
                ),
                self.heuristic_auxiliary_config,
            )
            optimization_loss = (
                loss.total
                if self.heuristic_auxiliary_config.target == "disabled"
                else loss.total + auxiliary.weighted
            )

            self.optimizer.zero_grad(set_to_none=True)
            optimization_loss.backward()  # type: ignore[no-untyped-call]
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
                optimization_loss=optimization_loss,
                auxiliary=auxiliary,
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
    optimization_loss: Tensor,
    auxiliary: HeuristicAuxiliaryLoss,
    new_value: Tensor,
    entropy: Tensor,
    old_log_probability: Tensor,
    new_log_probability: Tensor,
    pre_clip: Tensor,
    post_clip: float,
    clip_ratio: float,
) -> None:
    size = loss.ratio.numel()
    accumulator.total_loss_sum += float(optimization_loss.detach().item()) * size
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
    auxiliary_metrics = auxiliary.metrics
    auxiliary_count = auxiliary_metrics.included_count
    accumulator.auxiliary_unweighted_loss_sum += auxiliary_metrics.smooth_l1_loss * auxiliary_count
    accumulator.auxiliary_weighted_loss_sum += (
        float(auxiliary.weighted.detach().item()) * auxiliary_count
    )
    accumulator.auxiliary_included_count += auxiliary_count
    accumulator.auxiliary_total_count += auxiliary_metrics.total_count
    if auxiliary_count:
        assert auxiliary_metrics.mean_prediction is not None
        assert auxiliary_metrics.mean_target is not None
        assert auxiliary_metrics.mean_absolute_error is not None
        accumulator.auxiliary_prediction_sum += auxiliary_metrics.mean_prediction * auxiliary_count
        accumulator.auxiliary_target_sum += auxiliary_metrics.mean_target * auxiliary_count
        accumulator.auxiliary_absolute_error_sum += (
            auxiliary_metrics.mean_absolute_error * auxiliary_count
        )


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
    auxiliary_metrics = _build_auxiliary_metrics(accumulator)
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
        heuristic_auxiliary_weighted_loss=(
            accumulator.auxiliary_weighted_loss_sum / accumulator.auxiliary_included_count
            if accumulator.auxiliary_included_count
            else 0.0
        ),
        heuristic_auxiliary=auxiliary_metrics,
    )


def _training_targets(
    rollout: RolloutBatch,
    config: PPOConfig,
    heuristic_auxiliary_config: HeuristicAuxiliaryValueConfig | None = None,
) -> _TrainingTargets:
    try:
        packed = PackedRollout.from_batch(
            rollout,
            heuristic_auxiliary_config=(
                heuristic_auxiliary_config or HeuristicAuxiliaryValueConfig()
            ),
        )
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


def _build_auxiliary_metrics(
    accumulator: _PPOUpdateAccumulator,
) -> HeuristicAuxiliaryMetrics:
    count = accumulator.auxiliary_included_count
    if not count:
        return HeuristicAuxiliaryMetrics(
            included_count=0,
            total_count=accumulator.auxiliary_total_count,
            included_fraction=0.0,
            mean_prediction=None,
            mean_target=None,
            mean_absolute_error=None,
            smooth_l1_loss=0.0,
        )
    return HeuristicAuxiliaryMetrics(
        included_count=count,
        total_count=accumulator.auxiliary_total_count,
        included_fraction=count / accumulator.auxiliary_total_count,
        mean_prediction=accumulator.auxiliary_prediction_sum / count,
        mean_target=accumulator.auxiliary_target_sum / count,
        mean_absolute_error=accumulator.auxiliary_absolute_error_sum / count,
        smooth_l1_loss=accumulator.auxiliary_unweighted_loss_sum / count,
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
