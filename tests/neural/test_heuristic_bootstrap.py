from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("torch")

from garboid_pocketrocks.neural.heuristic_bootstrap import (  # noqa: E402
    HEURISTIC_BOOTSTRAP_ARMS,
    REFERENCE_TRAINING_GAMES,
    bootstrap_strategy,
    validate_fixed_compute_arm,
)
from garboid_pocketrocks.neural.run_config import TrainingRunConfig  # noqa: E402


def _config(filename: str) -> TrainingRunConfig:
    return TrainingRunConfig.from_json(Path("configs/neural") / filename)


def test_every_arm_has_the_same_complete_game_budget() -> None:
    assert len({arm.digest for arm in HEURISTIC_BOOTSTRAP_ARMS}) == 5
    assert {arm.total_training_games for arm in HEURISTIC_BOOTSTRAP_ARMS} == {
        REFERENCE_TRAINING_GAMES
    }


@pytest.mark.parametrize(
    ("config", "strategy"),
    (
        (
            _config("heuristic-bootstrap-control-v1.json"),
            "fixed-compute-control-v1",
        ),
        (
            _config("heuristic-opponent-control-v1.json"),
            "focal-seat-control-v1",
        ),
        (
            _config("heuristic-behavior-cloning-v1.json"),
            "behavior-cloning-balanced-v3-v1",
        ),
        (
            _config("heuristic-auxiliary-value-v1.json"),
            "auxiliary-value-balanced-v3-v1",
        ),
        (
            _config("heuristic-opponent-curriculum-v1.json"),
            "heuristic-opponent-curriculum-v1",
        ),
    ),
)
def test_precommitted_configs_resolve_to_exact_arms(
    config: TrainingRunConfig,
    strategy: str,
) -> None:
    assert bootstrap_strategy(config) == strategy
    assert validate_fixed_compute_arm(config).strategy == strategy


def test_fixed_compute_validation_rejects_budget_drift() -> None:
    with pytest.raises(ValueError, match="does not match fixed arm"):
        validate_fixed_compute_arm(
            replace(_config("heuristic-bootstrap-control-v1.json"), max_updates=195)
        )


@pytest.mark.parametrize(
    "filename",
    (
        "heuristic-bootstrap-control-v1.json",
        "heuristic-opponent-control-v1.json",
        "heuristic-behavior-cloning-v1.json",
        "heuristic-auxiliary-value-v1.json",
        "heuristic-opponent-curriculum-v1.json",
    ),
)
def test_committed_experiment_configs_match_the_fixed_contract(filename: str) -> None:
    config = TrainingRunConfig.from_json(Path("configs/neural") / filename)

    validate_fixed_compute_arm(config)
