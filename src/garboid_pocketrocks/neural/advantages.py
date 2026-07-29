"""Generalized advantage estimation for neural-policy training."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


class AdvantageError(ValueError):
    """Raised when a trajectory cannot produce valid advantages."""


@dataclass(frozen=True, slots=True)
class AdvantageBatch:
    """Advantages and value targets for one learner trajectory."""

    advantages: Tensor
    returns: Tensor


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    *,
    bootstrap_value: Tensor,
    gamma: float,
    gae_lambda: float,
) -> AdvantageBatch:
    """Compute Stage 1 gamma-one generalized advantage estimates."""

    _validate_inputs(
        rewards,
        values,
        terminated,
        truncated,
        bootstrap_value=bootstrap_value,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    advantages = torch.empty_like(values)
    next_value = bootstrap_value
    next_advantage = torch.zeros_like(bootstrap_value)

    for index in range(values.shape[0] - 1, -1, -1):
        nonterminal = (~terminated[index]).to(values.dtype)
        delta = rewards[index] + gamma * nonterminal * next_value - values[index]
        advantage = delta + gamma * gae_lambda * nonterminal * next_advantage
        advantages[index] = advantage
        next_value = values[index]
        next_advantage = advantage

    return AdvantageBatch(advantages=advantages, returns=advantages + values)


def _validate_inputs(
    rewards: Tensor,
    values: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    *,
    bootstrap_value: Tensor,
    gamma: float,
    gae_lambda: float,
) -> None:
    tensors = {
        "rewards": rewards,
        "values": values,
        "terminated": terminated,
        "truncated": truncated,
    }
    if any(tensor.ndim != 1 for tensor in tensors.values()):
        raise AdvantageError("trajectory inputs must be one-dimensional")

    lengths = {tensor.shape[0] for tensor in tensors.values()}
    if len(lengths) != 1:
        raise AdvantageError("trajectory inputs must have matching lengths")

    if bootstrap_value.ndim != 0:
        raise AdvantageError("bootstrap_value must be a scalar tensor")
    if terminated.dtype != torch.bool or truncated.dtype != torch.bool:
        raise AdvantageError("termination and truncation flags must be boolean")
    if torch.any(terminated & truncated).item():
        raise AdvantageError("a step cannot be both terminated and truncated")

    if not torch.is_floating_point(rewards) or not torch.is_floating_point(values):
        raise AdvantageError("rewards and values must be floating-point tensors")
    if not torch.is_floating_point(bootstrap_value):
        raise AdvantageError("bootstrap_value must be a floating-point tensor")
    if not torch.isfinite(rewards).all().item():
        raise AdvantageError("rewards must be finite")
    if not torch.isfinite(values).all().item():
        raise AdvantageError("values must be finite")
    if not torch.isfinite(bootstrap_value).item():
        raise AdvantageError("bootstrap_value must be finite")

    if gamma != 1.0:
        raise AdvantageError("Stage 1 requires gamma=1.0")
    if not math.isfinite(gae_lambda):
        raise AdvantageError("gae_lambda must be finite")
