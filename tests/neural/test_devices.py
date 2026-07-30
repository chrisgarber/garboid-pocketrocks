from __future__ import annotations

from collections import Counter

import pytest

torch = pytest.importorskip("torch")

import garboid_pocketrocks.neural.benchmark as benchmark_module  # noqa: E402
from garboid_pocketrocks.neural.benchmark import (  # noqa: E402
    BenchmarkCandidate,
    BenchmarkResult,
    calibrate,
    calibration_candidates,
    calibration_plans,
    choose_candidate,
)
from garboid_pocketrocks.neural.collector import CollectorMetrics  # noqa: E402
from garboid_pocketrocks.neural.devices import DeviceError, resolve_device  # noqa: E402
from garboid_pocketrocks.neural.ppo import PPOConfig  # noqa: E402
from garboid_pocketrocks.neural.run_config import TrainingRunConfig  # noqa: E402


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


def test_calibration_candidates_measure_vector_engine_batch_sizes() -> None:
    candidates = calibration_candidates(("cpu", "mps"))

    assert len(candidates) == 5
    assert {candidate.device for candidate in candidates} == {"cpu", "mps"}
    assert {
        candidate.workers
        for candidate in candidates
        if candidate.device == "cpu"
    } == {1, 8}
    assert {
        candidate.workers
        for candidate in candidates
        if candidate.device == "mps"
    } == {1}
    assert {
        candidate.active_games_per_worker for candidate in candidates
    } == {64, 128}
    assert {candidate.max_inference_batch for candidate in candidates} == {
        1024
    }


def test_calibration_only_times_ppo_once_per_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = (
        BenchmarkCandidate("cpu", 1, 64, 1024),
        BenchmarkCandidate("cpu", 8, 128, 1024),
    )
    metrics = CollectorMetrics(
        games=15,
        decisions=150,
        elapsed_seconds=1.0,
        inference_seconds=0.5,
        inference_batches=2,
        inference_batch_sizes=(75, 75),
        cell_games=(),
    )
    ppo_calls = 0
    ppo_epochs: list[int] = []

    class FakePPOTrainer:
        def __init__(self, _model: object, config: object) -> None:
            ppo_epochs.append(config.epochs)  # type: ignore[attr-defined]

        def update(self, *_args: object, **_kwargs: object) -> None:
            nonlocal ppo_calls
            ppo_calls += 1

    monkeypatch.setattr(
        benchmark_module,
        "calibration_candidates",
        lambda _devices: candidates,
    )
    monkeypatch.setattr(
        benchmark_module,
        "collect_self_play_vectorized",
        lambda *_args, **_kwargs: (object(), metrics),
    )
    monkeypatch.setattr(
        benchmark_module,
        "collect_self_play_vectorized_parallel",
        lambda *_args, **_kwargs: (object(), metrics),
    )
    monkeypatch.setattr(benchmark_module, "PPOTrainer", FakePPOTrainer)

    _, results = calibrate(
        TrainingRunConfig(
            device="cpu",
            ppo=PPOConfig(epochs=3),
        ),
        plans=(),
    )

    assert len(results) == 2
    assert ppo_calls == 1
    assert ppo_epochs == [3]
