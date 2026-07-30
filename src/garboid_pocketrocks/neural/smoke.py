"""Trainer-backed end-to-end neural self-play smoke."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch

from garboid_pocketrocks.neural.checkpoint import parameter_digest
from garboid_pocketrocks.neural.metrics import CalibrationBucket, ValueMetrics
from garboid_pocketrocks.neural.run_config import TrainingRunConfig
from garboid_pocketrocks.neural.trainer import resume, train
from garboid_pocketrocks.neural.training_checkpoint import load_training_checkpoint


class SmokeError(ValueError):
    """Raised when the neural smoke cannot complete safely."""


@dataclass(frozen=True, slots=True)
class SmokeResult:
    """Acceptance metrics for the full A-E, three-to-five-player smoke."""

    completed_updates: int
    completed_episodes: int
    completed_decisions: int
    cell_games: tuple[tuple[str, int, int], ...]
    games_per_second: float
    decisions_per_second: float
    value: ValueMetrics
    checkpoint_digest_verified: bool
    resume_verified: bool


def smoke_run_config() -> TrainingRunConfig:
    """Load the committed self-play smoke contract."""

    return TrainingRunConfig.from_json(Path("configs/neural/smoke.json"))


def run_smoke(
    config: TrainingRunConfig,
    output_dir: Path,
) -> SmokeResult:
    """Train once, validate the checkpoint, and execute a resume probe."""

    result = train(config, output_dir)
    loaded = load_training_checkpoint(
        result.final_checkpoint,
        device=torch.device("cpu"),
    )
    metrics = loaded.metrics
    collection = cast(dict[str, object], metrics["collection"])
    ppo = cast(dict[str, object], metrics["ppo"])
    value = _read_value_metrics(cast(dict[str, object], ppo["value"]))
    resumed_result = resume(
        result.final_checkpoint,
        result.run_dir / "resume-probe",
        max_additional_updates=1,
    )
    resumed = load_training_checkpoint(
        resumed_result.final_checkpoint,
        device=torch.device("cpu"),
    )
    smoke_result = SmokeResult(
        completed_updates=result.completed_updates,
        completed_episodes=result.completed_episodes,
        completed_decisions=result.completed_decisions,
        cell_games=loaded.manifest.progress.cell_games,
        games_per_second=_as_float(
            collection["games_per_second"],
            "games_per_second",
        ),
        decisions_per_second=_as_float(
            collection["decisions_per_second"],
            "decisions_per_second",
        ),
        value=value,
        checkpoint_digest_verified=(
            loaded.manifest.parameter_digest == parameter_digest(loaded.model.state_dict())
        ),
        resume_verified=(
            resumed_result.completed_updates == result.completed_updates + 1
            and resumed_result.completed_episodes > result.completed_episodes
            and resumed_result.completed_decisions > result.completed_decisions
            and resumed.manifest.progress.next_update_index == result.completed_updates + 1
            and resumed.manifest.lineage[-1] == str(result.final_checkpoint.resolve())
        ),
    )
    _write_json_payload(
        result.run_dir / "self-play-smoke-result.json",
        asdict(smoke_result),
    )
    return smoke_result


def _read_value_metrics(payload: dict[str, object]) -> ValueMetrics:
    calibration_payload = cast(
        list[dict[str, object]],
        payload["calibration"],
    )
    return ValueMetrics(
        count=int(cast(int, payload["count"])),
        mean_prediction=float(cast(float, payload["mean_prediction"])),
        mean_target=float(cast(float, payload["mean_target"])),
        mae=float(cast(float, payload["mae"])),
        rmse=float(cast(float, payload["rmse"])),
        bias=float(cast(float, payload["bias"])),
        explained_variance=(
            None
            if payload["explained_variance"] is None
            else float(cast(float, payload["explained_variance"]))
        ),
        correlation=(
            None if payload["correlation"] is None else float(cast(float, payload["correlation"]))
        ),
        calibration=tuple(
            CalibrationBucket(
                count=int(cast(int, bucket["count"])),
                minimum_prediction=float(cast(float, bucket["minimum_prediction"])),
                maximum_prediction=float(cast(float, bucket["maximum_prediction"])),
                mean_prediction=float(cast(float, bucket["mean_prediction"])),
                mean_target=float(cast(float, bucket["mean_target"])),
            )
            for bucket in calibration_payload
        ),
    )


def _write_json_payload(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _as_float(value: object, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise SmokeError(f"{name} must be finite")
    return float(value)
