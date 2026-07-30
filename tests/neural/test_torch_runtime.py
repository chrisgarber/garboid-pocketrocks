from __future__ import annotations

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from garboid_pocketrocks.neural.seeding import configure_torch_runtime  # noqa: E402


def test_neural_extra_supplies_supported_cpu_torch() -> None:
    major, minor, *_ = (int(part) for part in torch.__version__.split("+")[0].split("."))

    assert (major, minor) == (2, 13)
    assert torch.tensor([1.0], device="cpu").device.type == "cpu"


def test_deterministic_setup_reseeds_all_runtimes_in_one_process() -> None:
    configure_torch_runtime(13, deterministic_algorithms=True)
    first = (
        random.random(),
        np.random.random(),
        torch.rand(4),
    )

    configure_torch_runtime(13, deterministic_algorithms=True)
    second = (
        random.random(),
        np.random.random(),
        torch.rand(4),
    )

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.get_num_threads() == 1
    assert torch.get_num_interop_threads() == 1


def test_accelerator_runtime_can_disable_strict_deterministic_algorithms() -> None:
    configure_torch_runtime(17, deterministic_algorithms=False)

    assert not torch.are_deterministic_algorithms_enabled()

    configure_torch_runtime(17, deterministic_algorithms=True)
