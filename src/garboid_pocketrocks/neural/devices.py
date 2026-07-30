"""Explicit CPU, CUDA, and MPS device discovery."""

from __future__ import annotations

import torch


class DeviceError(ValueError):
    """Raised when a requested learner device cannot be used."""


def available_devices() -> tuple[str, ...]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return tuple(devices)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        raise DeviceError("auto device requires throughput calibration")
    if requested not in ("cpu", "cuda", "mps"):
        raise DeviceError(f"unknown device {requested!r}")
    if requested not in available_devices():
        raise DeviceError(f"requested {requested} device is unavailable")
    return torch.device(requested)


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
