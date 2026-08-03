from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import garboid_pocketrocks.tournament.cli as cli_module
from garboid_pocketrocks.bots import (
    BOT_SPECS_BY_NAME,
    DEFAULT_TOURNAMENT_BOT_SPECS,
)
from garboid_pocketrocks.tournament.cli import (
    _parser,
    _resolve_bot_specs,
)
from garboid_pocketrocks.tournament.runner import TournamentRunner
from garboid_pocketrocks.tournament.schedule import TournamentConfig


def test_parser_defaults_to_full_tournament() -> None:
    args = _parser().parse_args(())

    assert args.games == 15_000
    assert args.players == (3, 4, 5)
    assert args.charts == ("A", "B", "C", "D", "E")
    assert args.bootstrap_samples == 200
    assert args.batch_size == 64
    assert args.seed is None
    assert args.decision_reports is False
    assert args.output_dir == Path("artifacts/tournaments")


def test_root_seed_resolution_preserves_ordinary_defaults_and_protects_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated_seed = (1 << 62) + 1_234_567
    requested_widths: list[int] = []

    def deterministic_randbits(width: int) -> int:
        requested_widths.append(width)
        return generated_seed

    monkeypatch.setattr(cli_module, "_secure_randbits", deterministic_randbits)

    assert cli_module._resolve_root_seed(None, decision_reports=False) == 0
    assert requested_widths == []
    assert cli_module._resolve_root_seed(None, decision_reports=True) == generated_seed
    assert requested_widths == [63]
    assert cli_module._resolve_root_seed(42, decision_reports=True) == 42
    assert requested_widths == [63]


def test_main_uses_private_entropy_for_an_omitted_diagnostic_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated_seed = (1 << 62) + 2_345_678
    captured_configs: list[object] = []

    def capture_config(config: object, **_: object) -> None:
        captured_configs.append(config)
        raise ValueError("stop after config")

    monkeypatch.setattr(cli_module, "_secure_randbits", lambda _width: generated_seed)
    monkeypatch.setattr(TournamentRunner, "run", capture_config)
    monkeypatch.setattr(sys, "argv", ["garboid-tournament", "--decision-reports"])

    with pytest.raises(SystemExit):
        cli_module.main()

    assert len(captured_configs) == 1
    captured_config = captured_configs[0]
    assert isinstance(captured_config, TournamentConfig)
    assert captured_config.root_seed == generated_seed


def test_parser_enables_decision_reports_explicitly() -> None:
    args = _parser().parse_args(("--decision-reports",))

    assert args.decision_reports is True


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
        "fixed-objective-overlay-v3",
        "fixed-objective-overlay-v2",
        "fixed-objective-overlay-v1",
        "fixed-bid-tuned-v1",
        "aggressive-v2",
        "fixed-bid-diverse-v1",
        "balanced-v2",
        "fixed-bid",
        "vector_ppo_large_v1_g350k",
        "passive-v2",
        "passive-v1",
        "aggressive-v3",
        "balanced-v3",
        "passive-v3",
        "monte-the-bookie-v1",
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
            ("vector_ppo_small_v1_g1500,vector_ppo_large_v1_g350k,vector_ppo_large_v2_g1750k"),
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
    overlay_v2_row = next(
        row for row in summary["leaderboard"] if row["bot_name"] == "fixed-objective-overlay-v2"
    )

    assert "fixed-objective-overlay-v2" in configured_names
    assert overlay_v2_row["faults"] == 0


def test_cli_writes_and_prints_decision_reports_only_when_requested(
    tmp_path: Path,
) -> None:
    ordinary_dir = tmp_path / "ordinary"
    diagnostic_dir = tmp_path / "diagnostic"
    repeated_diagnostic_dir = tmp_path / "diagnostic-repeat"
    private_seed = (1 << 62) + 7_654_321
    common = [
        "uv",
        "run",
        "garboid-tournament",
        "--games",
        "15",
        "--bootstrap-samples",
        "0",
        "--exclude-bots",
        ("vector_ppo_small_v1_g1500,vector_ppo_large_v1_g350k,vector_ppo_large_v2_g1750k"),
    ]

    ordinary = subprocess.run(
        [*common, "--output-dir", str(ordinary_dir)],
        text=True,
        capture_output=True,
    )
    diagnostic = subprocess.run(
        [
            *common,
            "--decision-reports",
            "--seed",
            str(private_seed),
            "--output-dir",
            str(diagnostic_dir),
        ],
        text=True,
        capture_output=True,
    )
    repeated_diagnostic = subprocess.run(
        [
            *common,
            "--decision-reports",
            "--seed",
            str(private_seed),
            "--output-dir",
            str(repeated_diagnostic_dir),
        ],
        text=True,
        capture_output=True,
    )

    assert ordinary.returncode == 0, ordinary.stderr
    assert diagnostic.returncode == 0, diagnostic.stderr
    assert repeated_diagnostic.returncode == 0, repeated_diagnostic.stderr
    diagnostic_names = (
        "game-summaries.jsonl",
        "game-details.jsonl",
        "decision-traces.jsonl",
        "decision-slices.csv",
    )
    assert all(not (ordinary_dir / name).exists() for name in diagnostic_names)
    ordinary_summary = json.loads((ordinary_dir / "summary.json").read_text())
    assert ordinary_summary["configuration"]["root_seed"] == 0
    assert "seed" not in diagnostic.stdout.lower()
    all_artifact_names = (
        "ratings.csv",
        "summary.json",
        "report.html",
        *diagnostic_names,
    )
    for name in all_artifact_names:
        path = diagnostic_dir / name
        assert path.is_file()
        assert path.read_bytes() == (repeated_diagnostic_dir / name).read_bytes()
        assert str(private_seed) not in path.read_text()
    for name in diagnostic_names:
        assert str(diagnostic_dir / name) in diagnostic.stdout
    assert "seed" not in (diagnostic_dir / "game-summaries.jsonl").read_text()
    assert "seed" not in (diagnostic_dir / "decision-traces.jsonl").read_text()
