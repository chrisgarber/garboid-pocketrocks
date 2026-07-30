"""Finite value-estimator and gameplay diagnostics for neural training."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from garboid_pocketrocks.neural.rollout import RolloutBatch, RolloutTransition


@dataclass(frozen=True, slots=True)
class CalibrationBucket:
    """One size-balanced value calibration bucket."""

    count: int
    minimum_prediction: float
    maximum_prediction: float
    mean_prediction: float
    mean_target: float


@dataclass(frozen=True, slots=True)
class ValueMetrics:
    """Scalar accuracy, association, and calibration of value predictions."""

    count: int
    mean_prediction: float
    mean_target: float
    mae: float
    rmse: float
    bias: float
    explained_variance: float | None
    correlation: float | None
    calibration: tuple[CalibrationBucket, ...]


@dataclass(frozen=True, slots=True)
class ValueMetricSlice:
    """Value diagnostics for one named population slice."""

    dimension: str
    key: str
    metrics: ValueMetrics


@dataclass(frozen=True, slots=True)
class GameplayMetrics:
    """Outcome and action aggregates for complete trajectories."""

    games: int
    decisions: int
    first_place_share: float
    mean_rank: float
    mean_final_money: float
    pass_rate: float
    mean_positive_bid: float | None
    illegal_actions: int
    faults: int


@dataclass(frozen=True, slots=True)
class GameplayMetricSlice:
    """Gameplay diagnostics for all data or one chart/player-count cell."""

    dimension: str
    key: str
    metrics: GameplayMetrics


def value_metrics(
    predictions: Tensor,
    targets: Tensor,
    *,
    buckets: int = 10,
) -> ValueMetrics:
    """Compute population statistics without emitting NaN sentinels."""

    if predictions.ndim != 1 or targets.ndim != 1:
        raise ValueError("value metric inputs must be one-dimensional")
    if predictions.shape != targets.shape:
        raise ValueError("value metric inputs must have matching shapes")
    if predictions.numel() == 0:
        raise ValueError("value metrics require at least one prediction")
    if (
        not isinstance(buckets, int)
        or isinstance(buckets, bool)
        or buckets <= 0
    ):
        raise ValueError("value metric buckets must be a positive integer")
    if (
        not torch.is_floating_point(predictions)
        or not torch.is_floating_point(targets)
        or not torch.isfinite(predictions).all().item()
        or not torch.isfinite(targets).all().item()
    ):
        raise ValueError("value metric inputs must be finite floating-point tensors")

    prediction = predictions.detach().to(dtype=torch.float64, device="cpu")
    target = targets.detach().to(dtype=torch.float64, device="cpu")
    residual = target - prediction
    prediction_variance = float(torch.var(prediction, unbiased=False).item())
    target_variance = float(torch.var(target, unbiased=False).item())
    explained_variance = (
        None
        if target_variance == 0.0
        else 1.0
        - (float(torch.var(residual, unbiased=False).item()) / target_variance)
    )
    correlation = None
    if prediction_variance > 0.0 and target_variance > 0.0:
        covariance = float(
            torch.mean(
                (prediction - prediction.mean()) * (target - target.mean())
            ).item()
        )
        correlation = covariance / math.sqrt(
            prediction_variance * target_variance
        )

    ordered = sorted(
        range(prediction.numel()),
        key=lambda index: (float(prediction[index].item()), index),
    )
    bucket_count = min(buckets, len(ordered))
    quotient, remainder = divmod(len(ordered), bucket_count)
    calibration: list[CalibrationBucket] = []
    offset = 0
    for bucket_index in range(bucket_count):
        size = quotient + int(bucket_index < remainder)
        indices = ordered[offset : offset + size]
        offset += size
        bucket_predictions = prediction[indices]
        bucket_targets = target[indices]
        calibration.append(
            CalibrationBucket(
                count=size,
                minimum_prediction=float(bucket_predictions.min().item()),
                maximum_prediction=float(bucket_predictions.max().item()),
                mean_prediction=float(bucket_predictions.mean().item()),
                mean_target=float(bucket_targets.mean().item()),
            )
        )

    return ValueMetrics(
        count=prediction.numel(),
        mean_prediction=float(prediction.mean().item()),
        mean_target=float(target.mean().item()),
        mae=float(torch.mean(torch.abs(residual)).item()),
        rmse=math.sqrt(float(torch.mean(torch.square(residual)).item())),
        bias=float(torch.mean(prediction - target).item()),
        explained_variance=explained_variance,
        correlation=correlation,
        calibration=tuple(calibration),
    )


def stratified_value_metrics(
    predictions: Tensor,
    targets: Tensor,
    *,
    ruleset_names: Sequence[str],
    player_counts: Sequence[int],
    phases: Sequence[str],
) -> tuple[ValueMetricSlice, ...]:
    """Compute global and per-chart, player-count, and phase value metrics."""

    size = predictions.numel() if predictions.ndim == 1 else -1
    if any(len(values) != size for values in (ruleset_names, player_counts, phases)):
        raise ValueError("value metric slice labels must match tensor lengths")
    slices = [
        ValueMetricSlice(
            dimension="all",
            key="all",
            metrics=value_metrics(predictions, targets),
        )
    ]
    dimensions: tuple[tuple[str, Sequence[object]], ...] = (
        ("ruleset", ruleset_names),
        ("player_count", player_counts),
        ("phase", phases),
    )
    for dimension, labels in dimensions:
        for label in sorted(set(labels), key=str):
            indices = tuple(index for index, item in enumerate(labels) if item == label)
            index_tensor = torch.tensor(indices, dtype=torch.int64, device=predictions.device)
            slices.append(
                ValueMetricSlice(
                    dimension=dimension,
                    key=str(label),
                    metrics=value_metrics(
                        predictions[index_tensor],
                        targets[index_tensor.to(targets.device)],
                    ),
                )
            )
    return tuple(slices)


def gameplay_metrics(rollout: RolloutBatch) -> tuple[GameplayMetricSlice, ...]:
    """Aggregate gameplay globally and by chart/player-count cell."""

    records: list[tuple[str, int, int, float, tuple[RolloutTransition, ...]]] = []
    for stage1_episode in rollout.episodes:
        records.append(
            (
                stage1_episode.transitions[0].metadata.ruleset_name,
                stage1_episode.transitions[0].metadata.player_count,
                stage1_episode.rank,
                float(stage1_episode.final_money),
                stage1_episode.transitions,
            )
        )
    for multi_seat_episode in rollout.multi_seat_episodes:
        scores = {
            score.seat: score for score in multi_seat_episode.result.scores
        }
        for trajectory in multi_seat_episode.trajectories:
            if trajectory.trainable:
                score = scores[trajectory.seat]
                records.append(
                    (
                        multi_seat_episode.plan.ruleset_name,
                        multi_seat_episode.plan.player_count,
                        score.rank,
                        float(score.final_money),
                        trajectory.transitions,
                    )
                )
    if not records:
        raise ValueError("gameplay metrics require at least one trajectory")
    slices = [
        GameplayMetricSlice("all", "all", _gameplay_aggregate(records))
    ]
    cells = sorted({(record[0], record[1]) for record in records})
    for ruleset_name, player_count in cells:
        selected = [
            record
            for record in records
            if record[0] == ruleset_name and record[1] == player_count
        ]
        slices.append(
            GameplayMetricSlice(
                "cell",
                f"{ruleset_name}/{player_count}",
                _gameplay_aggregate(selected),
            )
        )
    return tuple(slices)


def _gameplay_aggregate(
    records: Sequence[
        tuple[str, int, int, float, tuple[RolloutTransition, ...]]
    ],
) -> GameplayMetrics:
    transitions = tuple(
        transition for record in records for transition in record[4]
    )
    positive_bids = tuple(
        transition.action
        for transition in transitions
        if 1 <= transition.action <= 100
    )
    return GameplayMetrics(
        games=len(records),
        decisions=len(transitions),
        first_place_share=sum(record[2] == 1 for record in records) / len(records),
        mean_rank=sum(record[2] for record in records) / len(records),
        mean_final_money=sum(record[3] for record in records) / len(records),
        pass_rate=sum(transition.action == 0 for transition in transitions)
        / len(transitions),
        mean_positive_bid=(
            sum(positive_bids) / len(positive_bids) if positive_bids else None
        ),
        illegal_actions=sum(
            not bool(transition.observation.action_mask[transition.action])
            for transition in transitions
        ),
        faults=0,
    )
