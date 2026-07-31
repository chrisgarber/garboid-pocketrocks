"""Privacy-safe aggregate reports for heuristic bootstrap training arms."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from garboid_pocketrocks.neural.behavior_cloning import BehaviorCloningConfig
from garboid_pocketrocks.neural.heuristic_bootstrap import (
    HEURISTIC_BOOTSTRAP_ARMS,
    bootstrap_strategy,
    validate_fixed_compute_arm,
)
from garboid_pocketrocks.neural.run_config import TrainingRunConfig


class BootstrapReportingError(ValueError):
    """Raised when aggregate training evidence is incomplete or malformed."""


@dataclass(frozen=True, slots=True)
class BootstrapReportPaths:
    """Paths written after every input run passes validation."""

    learning_curves_csv: Path
    summary_json: Path
    learning_curves_digest: str
    summary_digest: str


@dataclass(frozen=True, slots=True)
class _UpdateRow:
    arm: str
    update_index: int
    games_per_cell: int
    games: int
    decisions: int
    optimizer_steps: int
    duration_seconds: float
    games_per_second: float
    decisions_per_second: float
    total_loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float
    mean_training_return: float
    return_mae: float
    return_rmse: float
    auxiliary_weighted_loss: float
    auxiliary_included_count: int
    auxiliary_total_count: int
    auxiliary_included_fraction: float
    auxiliary_smooth_l1_loss: float
    auxiliary_mean_absolute_error: float | None


@dataclass(frozen=True, slots=True)
class _RunReport:
    config: TrainingRunConfig
    arm: str
    arm_digest: str
    config_digest: str
    metrics_digest: str
    rows: tuple[_UpdateRow, ...]
    behavior_cloning: dict[str, object] | None


_CURVE_FIELDS = (
    "arm",
    "update_index",
    "completed_updates",
    "games_per_cell",
    "games",
    "decisions",
    "optimizer_steps",
    "duration_seconds",
    "cumulative_games",
    "cumulative_decisions",
    "cumulative_optimizer_steps",
    "cumulative_duration_seconds",
    "games_per_second",
    "decisions_per_second",
    "total_loss",
    "policy_loss",
    "value_loss",
    "entropy",
    "approximate_kl",
    "clip_fraction",
    "mean_training_return",
    "return_mae",
    "return_rmse",
    "auxiliary_weighted_loss",
    "auxiliary_included_count",
    "auxiliary_total_count",
    "auxiliary_included_fraction",
    "auxiliary_smooth_l1_loss",
    "auxiliary_mean_absolute_error",
)


def write_bootstrap_report(
    run_dirs: tuple[Path, ...] | list[Path],
    output_dir: Path,
) -> BootstrapReportPaths:
    """Validate development-run aggregates and write redacted reports.

    Only ``resolved-config.json``, ``metrics.jsonl``, and the optional
    ``behavior-cloning.json`` are read. Held-out, game, and decision artifacts
    are never opened or copied.
    """

    directories = tuple(run_dirs)
    if not directories:
        raise BootstrapReportingError("at least one training run is required")
    reports = tuple(_read_run(path) for path in directories)
    arms = tuple(report.arm for report in reports)
    if len(set(arms)) != len(arms):
        raise BootstrapReportingError("training runs must have distinct bootstrap arms")

    destination = output_dir.resolve()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise BootstrapReportingError("report output directory must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    curves_path = destination / "learning-curves.csv"
    summary_path = destination / "bootstrap-summary.json"
    _write_curves(curves_path, reports)
    curves_digest = _file_digest(curves_path)
    _write_summary(summary_path, reports, learning_curves_digest=curves_digest)
    return BootstrapReportPaths(
        curves_path,
        summary_path,
        curves_digest,
        _file_digest(summary_path),
    )


def _read_run(run_dir: Path) -> _RunReport:
    if not run_dir.is_dir():
        raise BootstrapReportingError("training run must be a directory")
    config_path = run_dir / "resolved-config.json"
    metrics_path = run_dir / "metrics.jsonl"
    if not config_path.is_file() or not metrics_path.is_file():
        raise BootstrapReportingError("training run is missing aggregate inputs")
    try:
        config = TrainingRunConfig.from_json(config_path)
        fixed_arm = validate_fixed_compute_arm(config)
    except (OSError, ValueError) as error:
        raise BootstrapReportingError(f"invalid resolved training config: {error}") from error
    arm = bootstrap_strategy(config)
    rows, metrics_digest = _read_metrics(metrics_path, arm=arm, config=config)
    cloning_path = run_dir / "behavior-cloning.json"
    if config.behavior_cloning is None:
        if cloning_path.exists():
            raise BootstrapReportingError("non-cloning arm contains cloning provenance")
        cloning = None
    else:
        cloning = _read_behavior_cloning(cloning_path, config.behavior_cloning)
    config_digest = _json_digest(config.to_json_dict())
    return _RunReport(
        config=config,
        arm=arm,
        arm_digest=fixed_arm.digest,
        config_digest=config_digest,
        metrics_digest=metrics_digest,
        rows=rows,
        behavior_cloning=cloning,
    )


def _read_metrics(
    path: Path,
    *,
    arm: str,
    config: TrainingRunConfig,
) -> tuple[tuple[_UpdateRow, ...], str]:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as error:
        raise BootstrapReportingError("metrics JSONL could not be read") from error
    lines = text.splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise BootstrapReportingError("metrics JSONL must contain nonblank rows")
    rows: list[_UpdateRow] = []
    for expected_index, line in enumerate(lines):
        try:
            value: object = json.loads(line, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, BootstrapReportingError) as error:
            raise BootstrapReportingError("metrics JSONL contains invalid JSON") from error
        rows.append(
            _parse_update(
                value,
                arm=arm,
                expected_index=expected_index,
                config=config,
            )
        )
    if config.max_updates is not None and len(rows) > config.max_updates:
        raise BootstrapReportingError("metrics exceed the configured update budget")
    return tuple(rows), hashlib.sha256(raw).hexdigest()


def _parse_update(
    value: object,
    *,
    arm: str,
    expected_index: int,
    config: TrainingRunConfig,
) -> _UpdateRow:
    payload = _object(value, "update metrics")
    _exact(payload, {"update_index", "games_per_cell", "duration_seconds", "collection", "ppo"})
    update_index = _integer(payload["update_index"], "update_index", minimum=0)
    if update_index != expected_index:
        raise BootstrapReportingError("metrics update indices must be contiguous from zero")
    games_per_cell = _integer(payload["games_per_cell"], "games_per_cell", minimum=1)
    if games_per_cell != config.games_per_cell:
        raise BootstrapReportingError("metrics games_per_cell contradicts run config")
    duration = _number(payload["duration_seconds"], "duration_seconds", positive=True)
    collection = _parse_collection(payload["collection"], games_per_cell)
    ppo = _parse_ppo(payload["ppo"], config)
    return _UpdateRow(
        arm=arm,
        update_index=update_index,
        games_per_cell=games_per_cell,
        games=cast(int, collection["games"]),
        decisions=cast(int, collection["decisions"]),
        optimizer_steps=cast(int, ppo["optimizer_steps"]),
        duration_seconds=duration,
        games_per_second=collection["games_per_second"],
        decisions_per_second=collection["decisions_per_second"],
        total_loss=cast(float, ppo["total_loss"]),
        policy_loss=cast(float, ppo["policy_loss"]),
        value_loss=cast(float, ppo["value_loss"]),
        entropy=cast(float, ppo["entropy"]),
        approximate_kl=cast(float, ppo["approximate_kl"]),
        clip_fraction=cast(float, ppo["clip_fraction"]),
        mean_training_return=cast(float, ppo["mean_training_return"]),
        return_mae=cast(float, ppo["return_mae"]),
        return_rmse=cast(float, ppo["return_rmse"]),
        auxiliary_weighted_loss=cast(float, ppo["auxiliary_weighted_loss"]),
        auxiliary_included_count=cast(int, ppo["auxiliary_included_count"]),
        auxiliary_total_count=cast(int, ppo["auxiliary_total_count"]),
        auxiliary_included_fraction=cast(float, ppo["auxiliary_included_fraction"]),
        auxiliary_smooth_l1_loss=cast(float, ppo["auxiliary_smooth_l1_loss"]),
        auxiliary_mean_absolute_error=cast(float | None, ppo["auxiliary_mean_absolute_error"]),
    )


def _parse_collection(value: object, games_per_cell: int) -> dict[str, int | float]:
    payload = _object(value, "collection metrics")
    expected = {
        "games",
        "decisions",
        "elapsed_seconds",
        "inference_seconds",
        "inference_batches",
        "inference_batch_sizes",
        "cell_games",
        "queue_wait_seconds",
        "ipc_seconds",
        "worker_busy_seconds",
        "inference_batch_p50",
        "inference_batch_p95",
        "games_per_second",
        "decisions_per_second",
        "mean_inference_batch_size",
    }
    _exact(payload, expected)
    games = _integer(payload["games"], "collection games", minimum=1)
    decisions = _integer(payload["decisions"], "collection decisions", minimum=1)
    if games != games_per_cell * 15:
        raise BootstrapReportingError("collection games do not cover the 15 training cells")
    inference_batches = _integer(payload["inference_batches"], "inference batches", minimum=1)
    batch_sizes = _number_list(payload["inference_batch_sizes"], "inference batch sizes")
    if len(batch_sizes) != inference_batches or sum(batch_sizes) != decisions:
        raise BootstrapReportingError("inference batch sizes do not reconcile")
    cell_games = _list(payload["cell_games"], "cell games")
    parsed_cells: set[tuple[str, int]] = set()
    cell_total = 0
    for row in cell_games:
        values = _list(row, "cell game row")
        if len(values) != 3 or not isinstance(values[0], str):
            raise BootstrapReportingError("cell game row is malformed")
        players = _integer(values[1], "cell player count", minimum=3)
        count = _integer(values[2], "cell game count", minimum=0)
        if values[0] not in {f"live-{chart}" for chart in "ABCDE"} or players not in (3, 4, 5):
            raise BootstrapReportingError("cell game row is outside the A-E/3-5 matrix")
        parsed_cells.add((values[0], players))
        cell_total += count
    if len(parsed_cells) != 15 or cell_total != games:
        raise BootstrapReportingError("cell game counts do not reconcile")
    for name in (
        "elapsed_seconds",
        "inference_seconds",
        "queue_wait_seconds",
        "ipc_seconds",
        "worker_busy_seconds",
        "inference_batch_p50",
        "inference_batch_p95",
        "mean_inference_batch_size",
    ):
        _number(payload[name], name, nonnegative=True)
    return {
        "games": games,
        "decisions": decisions,
        "games_per_second": _number(payload["games_per_second"], "games_per_second", positive=True),
        "decisions_per_second": _number(
            payload["decisions_per_second"], "decisions_per_second", positive=True
        ),
    }


def _parse_ppo(value: object, config: TrainingRunConfig) -> dict[str, int | float | None]:
    payload = _object(value, "PPO metrics")
    expected = {
        "epochs",
        "optimizer_steps",
        "transition_count",
        "total_loss",
        "policy_loss",
        "value_loss",
        "entropy",
        "advantages",
        "ratios",
        "values",
        "entropies",
        "pre_clip_gradient_norms",
        "post_clip_gradient_norms",
        "approximate_kl",
        "clip_fraction",
        "value",
        "value_slices",
        "heuristic_auxiliary_weighted_loss",
        "heuristic_auxiliary",
    }
    _exact(payload, expected)
    epochs = _integer(payload["epochs"], "PPO epochs", minimum=1)
    if epochs != config.ppo.epochs:
        raise BootstrapReportingError("PPO epochs contradict run config")
    optimizer_steps = _integer(payload["optimizer_steps"], "optimizer steps", minimum=1)
    transitions = _integer(payload["transition_count"], "transition count", minimum=1)
    _finite_list(payload["advantages"], "advantages", expected_length=transitions)
    for name in ("ratios", "values", "entropies"):
        _finite_list(payload[name], name, expected_length=transitions * epochs)
    for name in ("pre_clip_gradient_norms", "post_clip_gradient_norms"):
        _finite_list(payload[name], name, expected_length=optimizer_steps)
    value_metrics = _parse_value_metrics(payload["value"], transitions)
    if not isinstance(payload["value_slices"], list):
        raise BootstrapReportingError("value_slices must be an array")
    auxiliary = _parse_auxiliary(payload["heuristic_auxiliary"], transitions * epochs)
    result: dict[str, int | float | None] = {
        "optimizer_steps": optimizer_steps,
        "mean_training_return": value_metrics["mean_target"],
        "return_mae": value_metrics["mae"],
        "return_rmse": value_metrics["rmse"],
        "auxiliary_weighted_loss": _number(
            payload["heuristic_auxiliary_weighted_loss"],
            "auxiliary weighted loss",
            nonnegative=True,
        ),
        **auxiliary,
    }
    for name in (
        "total_loss",
        "policy_loss",
        "value_loss",
        "entropy",
        "approximate_kl",
        "clip_fraction",
    ):
        result[name] = _number(payload[name], name)
    clip_fraction = cast(float, result["clip_fraction"])
    if not 0.0 <= clip_fraction <= 1.0:
        raise BootstrapReportingError("clip_fraction must be from zero to one")
    return result


def _parse_value_metrics(value: object, expected_count: int) -> dict[str, float]:
    payload = _object(value, "value metrics")
    _exact(
        payload,
        {
            "count",
            "mean_prediction",
            "mean_target",
            "mae",
            "rmse",
            "bias",
            "explained_variance",
            "correlation",
            "calibration",
        },
    )
    if _integer(payload["count"], "value count", minimum=1) != expected_count:
        raise BootstrapReportingError("value metric count does not match transitions")
    for name in ("mean_prediction", "mean_target", "mae", "rmse", "bias"):
        _number(payload[name], name)
    for name in ("explained_variance", "correlation"):
        if payload[name] is not None:
            _number(payload[name], name)
    if not isinstance(payload["calibration"], list):
        raise BootstrapReportingError("value calibration must be an array")
    return {
        "mean_target": cast(float, payload["mean_target"]),
        "mae": cast(float, payload["mae"]),
        "rmse": cast(float, payload["rmse"]),
    }


def _parse_auxiliary(value: object, expected_total: int) -> dict[str, int | float | None]:
    payload = _object(value, "heuristic auxiliary metrics")
    _exact(
        payload,
        {
            "included_count",
            "total_count",
            "included_fraction",
            "mean_prediction",
            "mean_target",
            "mean_absolute_error",
            "smooth_l1_loss",
        },
    )
    count = _integer(payload["included_count"], "auxiliary included count", minimum=0)
    total = _integer(payload["total_count"], "auxiliary total count", minimum=1)
    if total != expected_total or count > total:
        raise BootstrapReportingError("auxiliary counts do not match PPO transitions")
    fraction = _number(payload["included_fraction"], "auxiliary fraction")
    if not math.isclose(fraction, count / total, rel_tol=0.0, abs_tol=1e-12):
        raise BootstrapReportingError("auxiliary fraction does not match counts")
    optional = ("mean_prediction", "mean_target", "mean_absolute_error")
    if count == 0 and any(payload[name] is not None for name in optional):
        raise BootstrapReportingError("masked auxiliary metrics must not report means")
    if count > 0 and any(payload[name] is None for name in optional):
        raise BootstrapReportingError("included auxiliary metrics require means")
    for name in optional:
        if payload[name] is not None:
            _number(payload[name], name)
    return {
        "auxiliary_included_count": count,
        "auxiliary_total_count": total,
        "auxiliary_included_fraction": fraction,
        "auxiliary_smooth_l1_loss": _number(
            payload["smooth_l1_loss"], "auxiliary Smooth L1 loss", nonnegative=True
        ),
        "auxiliary_mean_absolute_error": (
            None
            if payload["mean_absolute_error"] is None
            else _number(payload["mean_absolute_error"], "auxiliary MAE", nonnegative=True)
        ),
    }


def _read_behavior_cloning(
    path: Path,
    expected_config: BehaviorCloningConfig,
) -> dict[str, object]:
    if not path.is_file():
        raise BootstrapReportingError("cloning arm is missing aggregate cloning provenance")
    payload = _read_json_object(path, "behavior cloning provenance")
    return validate_behavior_cloning_payload(payload, expected_config)


def validate_behavior_cloning_payload(
    value: object,
    expected_config: BehaviorCloningConfig,
) -> dict[str, object]:
    """Validate a local cloning artifact and return its redacted public provenance."""

    payload = _object(value, "behavior cloning provenance")
    _exact(payload, {"config", "dataset", "training"})
    try:
        config = BehaviorCloningConfig.from_json_dict(payload["config"])
    except ValueError as error:
        raise BootstrapReportingError(f"invalid behavior cloning config: {error}") from error
    if config != expected_config:
        raise BootstrapReportingError("behavior cloning provenance contradicts run config")
    dataset = _object(payload["dataset"], "behavior cloning dataset")
    _exact(
        dataset,
        {
            "cell_game_counts",
            "dataset_digest",
            "example_count",
            "game_count",
            "shard_count",
            "shards",
            "teacher_identity",
            "teacher_profile_digest",
        },
    )
    training = _object(payload["training"], "behavior cloning training")
    _exact(
        training,
        {
            "config_digest",
            "dataset_digest",
            "example_count",
            "epochs",
            "elapsed_seconds",
            "optimizer_steps",
            "updates",
        },
    )
    config_digest = _string(training["config_digest"], "cloning config digest")
    dataset_digest = _string(dataset["dataset_digest"], "dataset digest")
    training_dataset_digest = _string(training["dataset_digest"], "training dataset digest")
    if config_digest != config.config_digest or training_dataset_digest != dataset_digest:
        raise BootstrapReportingError("behavior cloning digests do not reconcile")
    examples = _integer(dataset["example_count"], "demonstration examples", minimum=1)
    games = _integer(dataset["game_count"], "demonstration games", minimum=1)
    expected_games = config.rounds * config.games_per_cell * 15
    if games != expected_games:
        raise BootstrapReportingError("behavior cloning game count contradicts its config")
    cell_game_counts = _parse_cloning_cells(dataset["cell_game_counts"], config)
    shard_count = _integer(dataset["shard_count"], "cloning shard count", minimum=1)
    expected_shards = math.ceil(games / config.games_per_shard)
    if shard_count != expected_shards:
        raise BootstrapReportingError("behavior cloning shard count contradicts its config")
    shard_records = _parse_cloning_shards(dataset["shards"], config, shard_count)
    shard_digests = [cast(str, record["dataset_digest"]) for record in shard_records]
    shard_example_counts = tuple(cast(int, record["example_count"]) for record in shard_records)
    if sum(shard_example_counts) != examples:
        raise BootstrapReportingError("behavior cloning shard examples do not reconcile")
    aggregate_digest = _combined_shard_digest(shard_digests)
    if aggregate_digest != dataset_digest:
        raise BootstrapReportingError("behavior cloning aggregate dataset digest is invalid")
    epochs = _integer(training["epochs"], "cloning epochs", minimum=1)
    optimizer_steps = _integer(training["optimizer_steps"], "cloning optimizer steps", minimum=1)
    elapsed_seconds = _number(training["elapsed_seconds"], "cloning elapsed seconds", positive=True)
    training_examples = _integer(
        training["example_count"], "training demonstration examples", minimum=1
    )
    if training_examples != examples or epochs != config.epochs:
        raise BootstrapReportingError("behavior cloning compute counters do not reconcile")
    teacher_identity = _string(dataset["teacher_identity"], "teacher identity")
    teacher_profile_digest = _string(dataset["teacher_profile_digest"], "teacher profile digest")
    if (
        teacher_identity != config.teacher_identity
        or teacher_profile_digest != config.teacher_profile_digest
    ):
        raise BootstrapReportingError("behavior cloning teacher contradicts its config")
    update_rows = _list(training["updates"], "behavior cloning optimizer rows")
    if len(update_rows) != optimizer_steps:
        raise BootstrapReportingError("behavior cloning optimizer rows do not reconcile")
    _validate_cloning_updates(
        update_rows,
        shard_example_counts=shard_example_counts,
        config=config,
    )
    public_provenance: dict[str, object] = {
        "schema_version": 1,
        "method": "behavior_cloning",
        "config_digest": config_digest,
        "teacher_identity": teacher_identity,
        "teacher_profile_digest": teacher_profile_digest,
        "shard_count": shard_count,
        "demonstration_games": games,
        "demonstration_examples": examples,
        "cell_game_counts": cell_game_counts,
        "epochs": epochs,
        "optimization_order": config.optimization_order,
        "optimizer_steps": optimizer_steps,
        "elapsed_seconds": elapsed_seconds,
    }
    return {
        **public_provenance,
        "provenance_digest": _json_digest(public_provenance),
    }


def _write_curves(path: Path, reports: tuple[_RunReport, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=_CURVE_FIELDS, lineterminator="\n")
        writer.writeheader()
        for report in sorted(reports, key=lambda item: item.arm):
            cumulative_games = 0
            cumulative_decisions = 0
            cumulative_steps = 0
            cumulative_duration = 0.0
            for row in report.rows:
                cumulative_games += row.games
                cumulative_decisions += row.decisions
                cumulative_steps += row.optimizer_steps
                cumulative_duration += row.duration_seconds
                payload = {
                    **{name: getattr(row, name) for name in row.__dataclass_fields__},
                    "completed_updates": row.update_index + 1,
                    "cumulative_games": cumulative_games,
                    "cumulative_decisions": cumulative_decisions,
                    "cumulative_optimizer_steps": cumulative_steps,
                    "cumulative_duration_seconds": cumulative_duration,
                }
                writer.writerow({name: payload[name] for name in _CURVE_FIELDS})


def _write_summary(
    path: Path,
    reports: tuple[_RunReport, ...],
    *,
    learning_curves_digest: str,
) -> None:
    arms: list[dict[str, object]] = []
    for report in sorted(reports, key=lambda item: item.arm):
        rows = report.rows
        final = rows[-1]
        cloning_games = (
            cast(int, report.behavior_cloning["demonstration_games"])
            if report.behavior_cloning is not None
            else 0
        )
        cloning_steps = (
            cast(int, report.behavior_cloning["optimizer_steps"])
            if report.behavior_cloning is not None
            else 0
        )
        cloning_duration = (
            cast(float, report.behavior_cloning["elapsed_seconds"])
            if report.behavior_cloning is not None
            else 0.0
        )
        ppo_steps = sum(row.optimizer_steps for row in rows)
        ppo_duration = sum(row.duration_seconds for row in rows)
        arms.append(
            {
                "arm": report.arm,
                "arm_digest": report.arm_digest,
                "config_digest": report.config_digest,
                "metrics_digest": report.metrics_digest,
                "experiment_root_seed": report.config.root_seed,
                "model_profile": report.config.model_profile,
                "complete": len(rows) == report.config.max_updates,
                "configuration": {
                    "games_per_cell": report.config.games_per_cell,
                    "max_updates": report.config.max_updates,
                    "ppo": report.config.to_json_dict()["ppo"],
                    "reward": report.config.to_json_dict()["reward"],
                    "heuristic_auxiliary": report.config.to_json_dict()["heuristic_auxiliary"],
                    "opponent_training": report.config.opponent_training,
                },
                "compute": {
                    "completed_updates": len(rows),
                    "demonstration_games": cloning_games,
                    "ppo_games": sum(row.games for row in rows),
                    "total_training_games": cloning_games + sum(row.games for row in rows),
                    "decisions": sum(row.decisions for row in rows),
                    "optimizer_steps": {
                        "behavior_cloning": cloning_steps,
                        "ppo": ppo_steps,
                        "total": cloning_steps + ppo_steps,
                    },
                    "duration_seconds": {
                        "behavior_cloning": cloning_duration,
                        "ppo": ppo_duration,
                        "total": cloning_duration + ppo_duration,
                    },
                },
                "final_learning_metrics": {
                    "total_loss": final.total_loss,
                    "policy_loss": final.policy_loss,
                    "value_loss": final.value_loss,
                    "entropy": final.entropy,
                    "mean_training_return": final.mean_training_return,
                    "return_mae": final.return_mae,
                    "return_rmse": final.return_rmse,
                    "auxiliary_weighted_loss": final.auxiliary_weighted_loss,
                    "auxiliary_included_fraction": final.auxiliary_included_fraction,
                    "auxiliary_smooth_l1_loss": final.auxiliary_smooth_l1_loss,
                },
                "behavior_cloning": report.behavior_cloning,
            }
        )
    _write_json(
        path,
        {
            "schema_version": 1,
            "report_kind": "heuristic_bootstrap_training",
            "held_out_loaded": False,
            "learning_curves_digest": learning_curves_digest,
            "official_arm_contract": {
                "arm_count": len(HEURISTIC_BOOTSTRAP_ARMS),
                "arms": [
                    {"arm": arm.strategy, "arm_digest": arm.digest}
                    for arm in HEURISTIC_BOOTSTRAP_ARMS
                ],
            },
            "reported_arm_count": len(reports),
            "all_official_arms_present": {report.arm for report in reports}
            == {arm.strategy for arm in HEURISTIC_BOOTSTRAP_ARMS},
            "arms": arms,
        },
    )


def _parse_cloning_cells(
    value: object,
    config: BehaviorCloningConfig,
) -> list[list[object]]:
    rows = _list(value, "behavior cloning cell counts")
    parsed: list[list[object]] = []
    expected_count = config.rounds * config.games_per_cell
    for row in rows:
        values = _list(row, "behavior cloning cell row")
        if len(values) != 3 or not isinstance(values[0], str):
            raise BootstrapReportingError("behavior cloning cell row is malformed")
        players = _integer(values[1], "behavior cloning player count", minimum=3)
        games = _integer(values[2], "behavior cloning cell games", minimum=1)
        if values[0] not in {f"live-{chart}" for chart in "ABCDE"} or players not in (3, 4, 5):
            raise BootstrapReportingError("behavior cloning cell is outside A-E/3-5")
        if games != expected_count:
            raise BootstrapReportingError("behavior cloning cell budget contradicts its config")
        parsed.append([values[0], players, games])
    if len(parsed) != 15 or len({(row[0], row[1]) for row in parsed}) != 15:
        raise BootstrapReportingError("behavior cloning cells must cover A-E/3-5 exactly")
    return sorted(parsed, key=lambda row: (cast(str, row[0]), cast(int, row[1])))


def _validate_cloning_updates(
    rows: list[object],
    *,
    shard_example_counts: tuple[int, ...],
    config: BehaviorCloningConfig,
) -> None:
    coordinates: dict[tuple[int, int, int], int] = {}
    expected = {
        "shard_index",
        "epoch_index",
        "minibatch_index",
        "example_count",
        "negative_log_likelihood",
        "teacher_agreement",
        "entropy",
        "pre_clip_gradient_norm",
        "post_clip_gradient_norm",
    }
    for value in rows:
        row = _object(value, "behavior cloning optimizer row")
        _exact(row, expected)
        shard = _integer(row["shard_index"], "cloning shard index", minimum=0)
        epoch = _integer(row["epoch_index"], "cloning epoch index", minimum=0)
        examples = _integer(row["example_count"], "cloning minibatch examples", minimum=1)
        minibatch = _integer(row["minibatch_index"], "cloning minibatch index", minimum=0)
        if (
            shard >= len(shard_example_counts)
            or epoch >= config.epochs
            or examples > config.minibatch_size
        ):
            raise BootstrapReportingError("behavior cloning optimizer row exceeds its config")
        agreement = _number(row["teacher_agreement"], "teacher agreement")
        if not 0.0 <= agreement <= 1.0:
            raise BootstrapReportingError("teacher agreement must be from zero to one")
        for name in (
            "negative_log_likelihood",
            "entropy",
            "pre_clip_gradient_norm",
            "post_clip_gradient_norm",
        ):
            _number(row[name], name, nonnegative=True)
        coordinate = (shard, epoch, minibatch)
        if coordinate in coordinates:
            raise BootstrapReportingError("behavior cloning optimizer coordinates are duplicated")
        coordinates[coordinate] = examples
    expected_coordinates: dict[tuple[int, int, int], int] = {}
    for shard, shard_examples in enumerate(shard_example_counts):
        minibatches = math.ceil(shard_examples / config.minibatch_size)
        for epoch in range(config.epochs):
            for minibatch in range(minibatches):
                remaining = shard_examples - minibatch * config.minibatch_size
                expected_coordinates[(shard, epoch, minibatch)] = min(
                    config.minibatch_size, remaining
                )
    if coordinates != expected_coordinates:
        raise BootstrapReportingError(
            "behavior cloning optimizer coordinates or example counts do not reconcile"
        )


def _parse_cloning_shards(
    value: object,
    config: BehaviorCloningConfig,
    shard_count: int,
) -> list[dict[str, object]]:
    rows = _list(value, "behavior cloning shards")
    if len(rows) != shard_count:
        raise BootstrapReportingError("behavior cloning shard records do not reconcile")
    parsed: list[dict[str, object]] = []
    for expected_index, value in enumerate(rows):
        row = _object(value, "behavior cloning shard")
        _exact(row, {"shard_index", "game_count", "example_count", "dataset_digest"})
        shard_index = _integer(row["shard_index"], "cloning shard index", minimum=0)
        game_count = _integer(row["game_count"], "cloning shard games", minimum=1)
        example_count = _integer(row["example_count"], "cloning shard examples", minimum=1)
        digest = _string(row["dataset_digest"], "cloning shard dataset digest")
        if shard_index != expected_index:
            raise BootstrapReportingError("behavior cloning shard indices are not contiguous")
        expected_games = min(
            config.games_per_shard,
            config.rounds * config.games_per_cell * 15 - expected_index * config.games_per_shard,
        )
        if game_count != expected_games:
            raise BootstrapReportingError("behavior cloning shard game count is invalid")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise BootstrapReportingError("cloning shard dataset digest must be lowercase SHA-256")
        parsed.append(
            {
                "shard_index": shard_index,
                "game_count": game_count,
                "example_count": example_count,
                "dataset_digest": digest,
            }
        )
    if sum(cast(int, row["game_count"]) for row in parsed) != (
        config.rounds * config.games_per_cell * 15
    ):
        raise BootstrapReportingError("behavior cloning shard games do not reconcile")
    return parsed


def _digest_list(value: object, name: str) -> list[str]:
    values = _list(value, name)
    digests = [_string(item, name) for item in values]
    if any(
        len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest)
        for digest in digests
    ):
        raise BootstrapReportingError(f"{name} must contain lowercase SHA-256 digests")
    return digests


def _combined_shard_digest(shard_digests: list[str]) -> str:
    combined = hashlib.sha256()
    for shard_index, digest in enumerate(shard_digests):
        combined.update(shard_index.to_bytes(8, "big"))
        combined.update(bytes.fromhex(digest))
    return combined.hexdigest()


def _read_json_object(path: Path, name: str) -> dict[str, object]:
    try:
        value: object = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant
        )
    except (OSError, UnicodeError, json.JSONDecodeError, BootstrapReportingError) as error:
        raise BootstrapReportingError(f"{name} could not be read") from error
    return _object(value, name)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _json_digest(value: object) -> str:
    encoded = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reject_json_constant(value: str) -> object:
    raise BootstrapReportingError(f"nonfinite JSON constant {value!r}")


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise BootstrapReportingError(f"{name} must be an object")
    return cast(dict[str, object], value)


def _list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise BootstrapReportingError(f"{name} must be an array")
    return cast(list[object], value)


def _exact(payload: dict[str, object], expected: set[str]) -> None:
    if set(payload) != expected:
        raise BootstrapReportingError("aggregate metric keys do not match the schema")


def _integer(value: object, name: str, *, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise BootstrapReportingError(f"{name} must be an integer >= {minimum}")
    return value


def _number(
    value: object,
    name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise BootstrapReportingError(f"{name} must be finite")
    result = float(value)
    if positive and result <= 0.0:
        raise BootstrapReportingError(f"{name} must be positive")
    if nonnegative and result < 0.0:
        raise BootstrapReportingError(f"{name} must be nonnegative")
    return result


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise BootstrapReportingError(f"{name} must be a nonempty string")
    return value


def _number_list(value: object, name: str) -> list[int]:
    values = _list(value, name)
    return [_integer(item, name, minimum=1) for item in values]


def _finite_list(value: object, name: str, *, expected_length: int) -> None:
    values = _list(value, name)
    if len(values) != expected_length:
        raise BootstrapReportingError(f"{name} length does not match compute counters")
    for item in values:
        _number(item, name)
