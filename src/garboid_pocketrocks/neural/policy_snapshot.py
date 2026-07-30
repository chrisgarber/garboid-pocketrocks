"""Immutable policy snapshots for spawned local-inference actors."""

from __future__ import annotations

import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from garboid_pocketrocks.neural.config import (
    NeuralEncoderConfig,
    NeuralModelConfig,
)
from garboid_pocketrocks.neural.model import NeuralPolicy


@dataclass(frozen=True, slots=True)
class PolicySnapshot:
    """One eagerly serialized policy and its checkpointed architecture."""

    identity: str
    encoder_config: NeuralEncoderConfig
    model_config: NeuralModelConfig
    state_bytes: bytes


def snapshot_policies(
    policies: Mapping[str, NeuralPolicy],
) -> tuple[PolicySnapshot, ...]:
    """Serialize policies immediately in canonical identity order."""

    return tuple(
        PolicySnapshot(
            identity=identity,
            encoder_config=model.encoder_config,
            model_config=model.model_config,
            state_bytes=_state_bytes(model),
        )
        for identity, model in sorted(policies.items())
    )


def load_policy_snapshots(
    snapshots: Sequence[PolicySnapshot],
) -> dict[str, NeuralPolicy]:
    """Strictly rebuild policy snapshots on CPU in supplied order."""

    policies: dict[str, NeuralPolicy] = {}
    for snapshot in snapshots:
        model = NeuralPolicy(
            snapshot.encoder_config,
            snapshot.model_config,
        )
        model.load_state_dict(
            torch.load(
                io.BytesIO(snapshot.state_bytes),
                map_location="cpu",
                weights_only=True,
            ),
            strict=True,
        )
        policies[snapshot.identity] = model
    return policies


def _state_bytes(model: NeuralPolicy) -> bytes:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getvalue()
