from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from garboid_pocketrocks.bots import BOT_SPECS_BY_NAME
from garboid_pocketrocks.promotion import cli
from garboid_pocketrocks.promotion.analysis import (
    PromotionAnalysis,
    PromotionFailure,
    RatingDifferenceInterval,
)
from garboid_pocketrocks.promotion.reporting import (
    PromotionArtifacts,
    build_promotion_report,
)
from garboid_pocketrocks.promotion.runner import PromotionRun, PromotionRunConfig

from .test_runner import _run_inputs


def _required_args(tmp_path: Path) -> list[str]:
    return [
        "--candidate",
        "vector_ppo_large_v1_g350k",
        "--incumbent",
        "vector_ppo_small_v1_g1500",
        "--output-dir",
        str(tmp_path),
    ]


def _completed_run(tmp_path: Path, *, promoted: bool) -> PromotionRun:
    config, registry = _run_inputs(pair_count=1)
    plan_candidate = registry[config.candidate.name]
    plan_incumbent = registry[config.incumbent.name]
    failures = (
        ()
        if promoted
        else (
            PromotionFailure(
                code="interval_includes_zero",
                message="The uncertainty range includes no advantage.",
            ),
            PromotionFailure(
                code="bot_fault",
                message="A bot made an invalid decision.",
            ),
        )
    )
    analysis = PromotionAnalysis(
        requested_pairs=1,
        completed_pairs=1,
        requested_games=2,
        completed_games=2,
        rating_difference=50.0 if promoted else 0.0,
        interval=RatingDifferenceInterval(lower=10.0, upper=90.0) if promoted else None,
        bootstrap_requested=1_000,
        bootstrap_converged=1_000,
        unattributed_faults=0,
        faults_by_identity=(),
        warnings=(),
        failures=failures,
        promoted=promoted,
    )
    report = build_promotion_report(
        repository_commit="test-commit",
        candidate=plan_candidate,
        incumbent=plan_incumbent,
        opponents=(registry["opponent-a"], registry["opponent-b"]),
        development=config.development,
        held_out=config.held_out,
        bootstrap_samples=1_000,
        bootstrap_seed=0,
        workers=1,
        batch_size=64,
        analysis=analysis,
    )
    artifacts = PromotionArtifacts(
        report_json=tmp_path / "promotion-report.json",
        paired_games_jsonl=tmp_path / "paired-games.jsonl",
        corpus_snapshot_json=tmp_path / "corpus-snapshot.json",
    )
    return PromotionRun(
        config=replace(
            config,
            candidate=plan_candidate,
            incumbent=plan_incumbent,
            bootstrap_samples=1_000,
            bootstrap_seed=0,
            batch_size=64,
        ),
        plan=None,
        monte_carlo_result=None,
        report=report,
        artifacts=artifacts,
    )


def test_parser_defaults() -> None:
    args = cli._parser().parse_args(
        [
            "--candidate",
            "candidate",
            "--incumbent",
            "incumbent",
        ]
    )

    assert args.development_corpus == Path("configs/promotion/development-v1.json")
    assert args.held_out_corpus == Path("configs/promotion/held-out-v1.json")
    assert args.bootstrap_samples == 1_000
    assert args.bootstrap_seed == 0
    assert args.batch_size == 64


def test_promoted_candidate_prints_interval_report_and_exits_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = _completed_run(tmp_path, promoted=True)
    monkeypatch.setattr(
        "garboid_pocketrocks.promotion.cli.PromotionRunner.run",
        lambda *args, **kwargs: run,
    )

    exit_code = cli.main(_required_args(tmp_path))

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "passed the held-out final exam" in output
    assert "95% uncertainty interval: 10.00 to 90.00 rating points." in output
    assert f"Report: {tmp_path / 'promotion-report.json'}" in output


def test_nonpromotion_prints_every_reason_report_and_exits_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = _completed_run(tmp_path, promoted=False)
    monkeypatch.setattr(
        "garboid_pocketrocks.promotion.cli.PromotionRunner.run",
        lambda *args, **kwargs: run,
    )

    exit_code = cli.main(_required_args(tmp_path))

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "did not pass the held-out final exam" in output
    assert "- The uncertainty range includes no advantage." in output
    assert "- A bot made an invalid decision." in output
    assert f"Report: {tmp_path / 'promotion-report.json'}" in output


@pytest.mark.parametrize(
    ("extra_args", "message"),
    (
        (("--candidate", "unknown-bot"), "unknown bot"),
        (
            (
                "--candidate",
                "vector_ppo_small_v1_g1500",
                "--incumbent",
                "vector_ppo_small_v1_g1500",
            ),
            "different",
        ),
        (("--workers", "0"), "positive integer"),
        (("--bootstrap-samples", "0"), "positive integer"),
        (("--batch-size", "0"), "positive integer"),
    ),
)
def test_invalid_invocation_prints_usage_report_path_and_exits_two(
    tmp_path: Path,
    extra_args: tuple[str, ...],
    message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = _required_args(tmp_path)
    for option in extra_args[::2]:
        if option in args:
            option_index = args.index(option)
            del args[option_index : option_index + 2]
    args.extend(extra_args)

    exit_code = cli.main(args)

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "usage:" in captured.err
    assert message in captured.err
    assert str(tmp_path / "promotion-report.json") in captured.err


def test_bad_corpus_prints_direct_error_and_exits_two(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bad_corpus = tmp_path / "bad-held-out.json"
    bad_corpus.write_text("{not json", encoding="utf-8")
    args = [
        *_required_args(tmp_path / "output"),
        "--held-out-corpus",
        str(bad_corpus),
    ]

    exit_code = cli.main(args)

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "valid JSON" in captured.err
    assert str(tmp_path / "output" / "promotion-report.json") in captured.err


def test_operational_failure_prints_direct_error_and_exits_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_run(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("disk is unavailable")

    monkeypatch.setattr(
        "garboid_pocketrocks.promotion.cli.PromotionRunner.run",
        fail_run,
    )

    exit_code = cli.main(_required_args(tmp_path))

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "disk is unavailable" in captured.err
    assert str(tmp_path / "promotion-report.json") in captured.err


def test_git_commit_failure_prints_direct_error_without_a_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    git_error = subprocess.CalledProcessError(
        128,
        ("git", "rev-parse", "HEAD"),
    )

    def fail_git(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise git_error

    monkeypatch.setattr(
        "garboid_pocketrocks.promotion.runner.subprocess.run",
        fail_git,
    )

    exit_code = cli.main(_required_args(tmp_path))

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Could not determine the repository commit" in captured.err
    assert "Traceback" not in captured.err
    assert str(tmp_path / "promotion-report.json") in captured.err


@pytest.mark.parametrize(
    ("arguments", "expected"),
    (
        (("--output-dir=equals-form",), Path("equals-form")),
        (
            ("--output-dir", "first", "--output-dir", "last"),
            Path("last"),
        ),
        (
            ("--output-dir=first", "--output-dir", "last"),
            Path("last"),
        ),
        (
            ("--output-dir", "first", "--output-dir=last"),
            Path("last"),
        ),
    ),
)
def test_report_path_scan_matches_argparse_output_directory_rules(
    arguments: tuple[str, ...],
    expected: Path,
) -> None:
    assert cli._requested_output_dir(arguments) == expected


def test_parser_error_reports_last_equals_form_output_directory(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    first = tmp_path / "first"
    last = tmp_path / "last"
    args = [
        *_required_args(first),
        f"--output-dir={last}",
        "--workers",
        "0",
    ]

    exit_code = cli.main(args)

    captured = capsys.readouterr()
    assert exit_code == 2
    assert str(last / "promotion-report.json") in captured.err
    assert str(first / "promotion-report.json") not in captured.err


def test_help_explains_the_promotion_gate_in_plain_english() -> None:
    help_text = cli._parser().format_help().lower()

    for phrase in (
        "held-out",
        "bootstrap",
        "95% interval",
        "candidate",
        "incumbent",
        "development corpus",
        "output directory",
    ):
        assert phrase in help_text


def test_cli_uses_the_registered_bot_catalog() -> None:
    assert cli._BOT_REGISTRY is BOT_SPECS_BY_NAME


def test_run_config_type_is_available_to_cli_callers() -> None:
    config, _ = _run_inputs(pair_count=1)

    assert isinstance(config, PromotionRunConfig)
