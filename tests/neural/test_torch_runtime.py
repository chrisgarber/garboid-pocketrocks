from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_neural_extra_supplies_supported_cpu_torch() -> None:
    major, minor, *_ = (int(part) for part in torch.__version__.split("+")[0].split("."))

    assert (major, minor) == (2, 13)
    assert torch.tensor([1.0], device="cpu").device.type == "cpu"
