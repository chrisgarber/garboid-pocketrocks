from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from garboid_pocketrocks.bots import BOT_SPECS_BY_NAME
from garboid_pocketrocks.evolution import cli
from garboid_pocketrocks.evolution.reporting import SearchArtifacts
from garboid_pocketrocks.evolution.runner import SearchFailure, SearchRun, run_search

from .test_runner import _small_inputs


@pytest.fixture(scope="module")
def complete_run() -> SearchRun:
    manifest, corpus = _small_inputs(generations=1, population=2, elites=1, cases=1)
    return run_search(
        manifest,
        corpus,
        registry=BOT_SPECS_BY_NAME,
        workers=1,
        batch_size=1,
    )


def _required_args(tmp_path: Path) -> list[str]:
    return [
        "--manifest",
        "configs/evolution/balanced-v3-search-v1.json",
        "--output-dir",
        str(tmp_path),
    ]


def _artifacts(path: Path, *, frozen: bool) -> SearchArtifacts:
    return SearchArtifacts(
        search_manifest_json=path / "search-manifest.json",
        search_report_json=path / "search-report.json",
        candidate_evaluations_jsonl=path / "candidate-evaluations.jsonl",
        selection_log_jsonl=path / "selection-log.jsonl",
        development_games_jsonl=path / "development-games.jsonl",
        development_corpus_snapshot_json=path / "development-corpus-snapshot.json",
        frozen_candidate_json=(path / "frozen-candidate.json") if frozen else None,
    )


def _stub_execution(
    monkeypatch: pytest.MonkeyPatch,
    run: SearchRun,
    output_dir: Path,
) -> None:
    monkeypatch.setattr(
        cli,
        "load_promotion_corpus",
        lambda *args, **kwargs: run.development_corpus,
    )
    monkeypatch.setattr(cli, "load_search_manifest", lambda *args, **kwargs: run.manifest)
    monkeypatch.setattr(cli, "run_search", lambda *args, **kwargs: run)
    monkeypatch.setattr(cli, "repository_commit", lambda: "test-commit")
    monkeypatch.setattr(
        cli,
        "write_search_artifacts",
        lambda *args, **kwargs: _artifacts(
            output_dir,
            frozen=run.frozen_candidate is not None,
        ),
    )


def test_parser_has_only_execution_controls() -> None:
    parser = cli._parser()
    args = parser.parse_args(["--manifest", "search.json"])
    option_strings = {option for action in parser._actions for option in action.option_strings}

    assert args.development_corpus == Path(
        "configs/promotion/historical/development-v1-17c01635.json"
    )
    assert args.batch_size == 64
    assert args.output_dir == Path("artifacts/evolution")
    assert {
        "--manifest",
        "--development-corpus",
        "--workers",
        "--batch-size",
        "--output-dir",
        "--overwrite",
    } <= option_strings
    assert (
        not {
            "--search-seed",
            "--generations",
            "--population-size",
            "--elite-count",
            "--held-out-corpus",
            "--resume",
            "--promote",
        }
        & option_strings
    )


def test_frozen_improvement_exits_zero_and_labels_development_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    complete_run: SearchRun,
) -> None:
    assert complete_run.selected_candidate is not None
    run = replace(complete_run, frozen_candidate=complete_run.selected_candidate)
    _stub_execution(monkeypatch, run, tmp_path)

    exit_code = cli.main(_required_args(tmp_path))

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "development improvement" in output
    assert "frozen" in output
    assert "held-out promotion evaluation" in output
    assert f"Report: {tmp_path / 'search-report.json'}" in output


def test_complete_no_improvement_exits_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    complete_run: SearchRun,
) -> None:
    run = replace(complete_run, frozen_candidate=None)
    _stub_execution(monkeypatch, run, tmp_path)

    exit_code = cli.main(_required_args(tmp_path))

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "without a positive development improvement" in output
    assert "no candidate was frozen" in output


def test_invalid_search_evidence_is_written_and_exits_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    complete_run: SearchRun,
) -> None:
    run = replace(
        complete_run,
        selected_candidate=None,
        frozen_candidate=None,
        failures=(SearchFailure("invalid_candidate_evidence", "Evidence was incomplete."),),
    )
    _stub_execution(monkeypatch, run, tmp_path)

    exit_code = cli.main(_required_args(tmp_path))

    output = capsys.readouterr().out
    assert exit_code == 2
    assert "invalid" in output
    assert "- Evidence was incomplete." in output
    assert f"Report: {tmp_path / 'search-report.json'}" in output


@pytest.mark.parametrize(
    ("extra_args", "message"),
    (
        (("--workers", "0"), "positive integer"),
        (("--batch-size", "0"), "positive integer"),
        (("--resume", "old-results"), "unrecognized arguments"),
    ),
)
def test_invalid_invocation_prints_usage_report_path_and_exits_two(
    tmp_path: Path,
    extra_args: tuple[str, ...],
    message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = cli.main([*_required_args(tmp_path), *extra_args])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "usage:" in captured.err
    assert message in captured.err
    assert str(tmp_path / "search-report.json") in captured.err


def test_output_preflight_happens_before_loading_or_simulation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "notes.txt").write_text("occupied")

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("expensive work started before output preflight")

    monkeypatch.setattr(cli, "load_promotion_corpus", forbidden)
    monkeypatch.setattr(cli, "run_search", forbidden)

    exit_code = cli.main(_required_args(tmp_path))

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "not empty" in captured.err


def test_operational_failure_prints_direct_error_and_exits_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    complete_run: SearchRun,
) -> None:
    _stub_execution(monkeypatch, complete_run, tmp_path)

    def fail_run(*args: object, **kwargs: object) -> None:
        raise OSError("worker unavailable")

    monkeypatch.setattr(cli, "run_search", fail_run)

    assert cli.main(_required_args(tmp_path)) == 2
    captured = capsys.readouterr()
    assert "worker unavailable" in captured.err
    assert "Traceback" not in captured.err
    assert str(tmp_path / "search-report.json") in captured.err


def test_help_explains_the_execution_only_boundary() -> None:
    help_text = cli._parser().format_help().lower()

    for phrase in (
        "evolution",
        "development",
        "manifest",
        "frozen",
        "held-out",
        "output directory",
    ):
        assert phrase in help_text
    assert "resume" not in help_text


def test_cli_uses_registered_bot_catalog() -> None:
    assert cli._BOT_REGISTRY is BOT_SPECS_BY_NAME
