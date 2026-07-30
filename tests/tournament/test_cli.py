from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from garboid_pocketrocks.bots import BOT_SPECS_BY_NAME
from garboid_pocketrocks.tournament.cli import (
    _parser,
    _resolve_bot_specs,
)


def test_parser_defaults_to_full_tournament() -> None:
    args = _parser().parse_args(())

    assert args.games == 10_000
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
