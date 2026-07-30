from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from garboid_pocketrocks.neural.metrics import (  # noqa: E402
    stratified_value_metrics,
    value_metrics,
)


def test_value_metrics_match_hand_calculation() -> None:
    predictions = torch.tensor((0.0, 1.0, 2.0, 3.0))
    targets = torch.tensor((1.0, 1.0, 3.0, 3.0))

    result = value_metrics(predictions, targets, buckets=2)

    assert result.count == 4
    assert result.mean_prediction == pytest.approx(1.5)
    assert result.mean_target == pytest.approx(2.0)
    assert result.mae == pytest.approx(0.5)
    assert result.rmse == pytest.approx(math.sqrt(0.5))
    assert result.bias == pytest.approx(-0.5)
    assert result.explained_variance == pytest.approx(0.75)
    assert result.correlation == pytest.approx(math.sqrt(0.8))
    assert sum(bucket.count for bucket in result.calibration) == 4


def test_constant_targets_report_undefined_statistics_without_nan() -> None:
    result = value_metrics(torch.tensor((0.0, 1.0)), torch.ones(2), buckets=2)

    assert result.explained_variance is None
    assert result.correlation is None
    assert math.isfinite(result.rmse)


def test_value_metrics_are_sliced_by_chart_player_count_and_phase() -> None:
    slices = stratified_value_metrics(
        torch.tensor((0.0, 0.5, 1.0, 1.5)),
        torch.tensor((0.2, 0.7, 0.8, 1.2)),
        ruleset_names=("live-A", "live-A", "live-E", "live-E"),
        player_counts=(3, 5, 3, 5),
        phases=("early", "middle", "late", "late"),
    )

    assert {item.dimension for item in slices} == {
        "all",
        "ruleset",
        "player_count",
        "phase",
    }
    assert any(item.dimension == "ruleset" and item.key == "live-E" for item in slices)


@pytest.mark.parametrize(
    ("predictions", "targets", "buckets", "message"),
    [
        (torch.tensor(()), torch.tensor(()), 2, "at least one"),
        (torch.zeros((1, 1)), torch.zeros(1), 2, "one-dimensional"),
        (torch.zeros(2), torch.zeros(1), 2, "matching"),
        (torch.tensor((math.nan,)), torch.zeros(1), 2, "finite"),
        (torch.zeros(1), torch.zeros(1), 0, "buckets"),
    ],
)
def test_value_metrics_reject_invalid_inputs(
    predictions: object,
    targets: object,
    buckets: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        value_metrics(predictions, targets, buckets=buckets)  # type: ignore[arg-type]
