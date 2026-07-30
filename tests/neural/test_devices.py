from __future__ import annotations

from collections import Counter

import pytest

torch = pytest.importorskip("torch")

from garboid_pocketrocks.neural.benchmark import (  # noqa: E402
    BenchmarkCandidate,
    BenchmarkResult,
    calibration_plans,
    choose_candidate,
)
from garboid_pocketrocks.neural.devices import DeviceError, resolve_device  # noqa: E402


def test_explicit_unavailable_device_fails_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(DeviceError, match="cuda"):
        resolve_device("cuda")


def test_auto_is_reserved_for_calibration() -> None:
    with pytest.raises(DeviceError, match="calibration"):
        resolve_device("auto")


def test_auto_selects_fastest_complete_candidate() -> None:
    def result(device: str, workers: int, rate: float) -> BenchmarkResult:
        candidate = BenchmarkCandidate(device, workers, 4, 64)
        return BenchmarkResult(
            candidate,
            games=30,
            decisions=300,
            elapsed_seconds=2.0,
            ppo_seconds=1.0,
            total_seconds=3.0,
            games_per_second=10.0,
            decisions_per_second=rate,
            inference_batch_p50=16.0,
            inference_batch_p95=32.0,
            peak_rss_bytes=None,
        )

    chosen = choose_candidate(
        (
            result("cpu", 1, 100.0),
            result("cpu", 4, 250.0),
            result("mps", 2, 220.0),
        )
    )
    assert chosen.workers == 4


def test_calibration_plans_cover_every_cell() -> None:
    plans = calibration_plans(root_seed=42, games_per_cell=2)
    counts = Counter(
        (plan.ruleset_name, plan.player_count) for plan in plans
    )

    assert len(plans) == 30
    assert len(counts) == 15
    assert set(counts.values()) == {2}
