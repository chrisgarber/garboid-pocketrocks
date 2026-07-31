from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from garboid_pocketrocks.bots import (
    BOT_SPECS_BY_NAME,
    DEFAULT_TOURNAMENT_BOT_SPECS,
)
from garboid_pocketrocks.tournament.cli import (
    _parser,
    _resolve_bot_specs,
)


def test_parser_defaults_to_full_tournament() -> None:
    args = _parser().parse_args(())

    assert args.games == 15_000
    assert args.players == (3, 4, 5)
    assert args.charts == ("A", "B", "C", "D", "E")
    assert args.bootstrap_samples == 200
    assert args.batch_size == 64
    assert args.seed == 0


def test_bot_filters_include_then_exclude_registered_names() -> None:
    selected = _resolve_bot_specs(
        include=("random", "balanced", "passive"),
        exclude=("balanced",),
        registry=BOT_SPECS_BY_NAME,
    )

    assert tuple(spec.name for spec in selected) == ("random", "passive")


def test_bot_filters_use_curated_defaults_when_include_is_omitted() -> None:
    selected = _resolve_bot_specs(
        include=None,
        exclude=(),
        registry=BOT_SPECS_BY_NAME,
        defaults=DEFAULT_TOURNAMENT_BOT_SPECS,
    )

    assert tuple(spec.name for spec in selected) == (
        "random",
        "aggressive-v1",
        "balanced-v1",
        "passive-v1",
        "aggressive-v2",
        "balanced-v2",
        "passive-v2",
        "sdk-greedy-value-v1",
        "vector_ppo_small_v1_g1500",
        "vector_ppo_large_v1_g350k",
    )


def test_bot_filters_reject_unknown_or_empty_selection() -> None:
    with pytest.raises(ValueError, match="unknown"):
        _resolve_bot_specs(
            include=("missing",),
            exclude=(),
            registry=BOT_SPECS_BY_NAME,
        )
    with pytest.raises(ValueError, match="at least one"):
        _resolve_bot_specs(
            include=("random",),
            exclude=("random",),
            registry=BOT_SPECS_BY_NAME,
        )


def test_cli_runs_all_conditions_with_current_registry(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            "uv",
            "run",
            "garboid-tournament",
            "--games",
            "15",
            "--bootstrap-samples",
            "0",
            "--exclude-bots",
            "vector_ppo_small_v1_g1500,vector_ppo_large_v1_g350k",
            "--output-dir",
            str(tmp_path),
        ],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "ratings.csv").is_file()
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "report.html").is_file()
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    configured_names = tuple(item["name"] for item in summary["configuration"]["bots"])
    sdk_row = next(
        row for row in summary["leaderboard"] if row["bot_name"] == "sdk-greedy-value-v1"
    )

    assert "sdk-greedy-value-v1" in configured_names
    assert sdk_row["faults"] == 0
