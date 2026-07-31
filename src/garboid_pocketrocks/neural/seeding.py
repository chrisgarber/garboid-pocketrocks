"""Deterministic, namespaced seeds for neural training."""

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
