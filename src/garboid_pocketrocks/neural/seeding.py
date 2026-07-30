"""Deterministic, namespaced seeds for the Stage 1 training proof."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

import numpy as np
import torch

_UNSIGNED_63_BIT_MASK = (1 << 63) - 1
_NUMPY_SEED_MODULUS = 1 << 32
_THREADS_CONFIGURED = False

SEED_NAMESPACES = (
    "python",
    "numpy",
    "torch",
    "model",
    "environment",
    "opponent",
    "policy",
    "minibatch",
)


@dataclass(frozen=True, slots=True)
class RuntimeSeeds:
    """The independently derived seeds installed into global runtimes."""

    python: int
    numpy: int
    torch: int


@dataclass(frozen=True, slots=True)
class EpisodePlan:
    """All deterministic choices needed to collect one learner episode."""

    update_index: int
    episode_index: int
    learner_seat: int
    environment_seed: int
    opponent_seed: int
    policy_seed: int


def derive_seed(root_seed: int, namespace: str, *indices: int) -> int:
    """Derive an unsigned 63-bit seed from a stable canonical name."""

    if not isinstance(root_seed, int) or isinstance(root_seed, bool):
        raise ValueError("root seed must be an integer")
    if namespace not in SEED_NAMESPACES:
        raise ValueError(f"unknown seed namespace {namespace!r}")
    if any(not isinstance(index, int) or isinstance(index, bool) for index in indices):
        raise ValueError("seed indices must be integers")
    canonical = ":".join((str(root_seed), namespace, *(str(index) for index in indices)))
    digest = hashlib.blake2b(canonical.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & _UNSIGNED_63_BIT_MASK


def configure_deterministic_torch(root_seed: int) -> RuntimeSeeds:
    """Seed Python, NumPy, and Torch and configure deterministic CPU execution."""

    return configure_torch_runtime(
        root_seed,
        deterministic_algorithms=True,
    )


def configure_torch_runtime(
    root_seed: int,
    *,
    deterministic_algorithms: bool,
) -> RuntimeSeeds:
    """Seed all runtimes and select strict or accelerator-compatible kernels."""

    global _THREADS_CONFIGURED

    seeds = RuntimeSeeds(
        python=derive_seed(root_seed, "python"),
        numpy=derive_seed(root_seed, "numpy"),
        torch=derive_seed(root_seed, "torch"),
    )
    random.seed(seeds.python)
    np.random.seed(seeds.numpy % _NUMPY_SEED_MODULUS)
    torch.manual_seed(seeds.torch)
    torch.use_deterministic_algorithms(deterministic_algorithms)
    if not _THREADS_CONFIGURED:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        _THREADS_CONFIGURED = True
    return seeds


def plan_stage1_episodes(
    *,
    root_seed: int,
    updates: int,
    games_per_update: int,
) -> tuple[EpisodePlan, ...]:
    """Build stable live-A/three-player plans without consuming RNG state."""

    if not isinstance(updates, int) or isinstance(updates, bool) or updates <= 0:
        raise ValueError("updates must be a positive integer")
    if (
        not isinstance(games_per_update, int)
        or isinstance(games_per_update, bool)
        or games_per_update <= 0
    ):
        raise ValueError("games per update must be a positive integer")

    return tuple(
        EpisodePlan(
            update_index=update_index,
            episode_index=episode_index,
            learner_seat=((update_index * games_per_update) + episode_index) % 3,
            environment_seed=derive_seed(
                root_seed,
                "environment",
                update_index,
                episode_index,
            ),
            opponent_seed=derive_seed(
                root_seed,
                "opponent",
                update_index,
                episode_index,
            ),
            policy_seed=derive_seed(
                root_seed,
                "policy",
                update_index,
                episode_index,
            ),
        )
        for update_index in range(updates)
        for episode_index in range(games_per_update)
    )
