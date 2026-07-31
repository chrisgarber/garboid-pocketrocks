from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import garboid_pocketrocks.evolution.diagnostics as diagnostics_module
import garboid_pocketrocks.evolution.reporting as reporting_module
from garboid_pocketrocks.bots import BOT_SPECS_BY_NAME
from garboid_pocketrocks.evolution import cli
from garboid_pocketrocks.evolution.diagnostics import WinnerDiagnosticsError
from garboid_pocketrocks.evolution.reporting import (
    SearchArtifacts,
    SearchReport,
    search_report_payload,
)
from garboid_pocketrocks.evolution.runner import SearchFailure, SearchRun, run_search

from .test_reporting import _canonical_winner_diagnostics
from .test_runner import (
    _phase_run_with_recomputed_positive_evidence,
    _run_small_phase_search,
    _small_inputs,
    _small_phase_inputs,
)


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


@pytest.fixture(scope="module")
def complete_phase_run() -> SearchRun:
    manifest, corpus = _small_phase_inputs(generations=1, cases=1)
    return _run_small_phase_search(
        manifest,
        corpus,
        workers=1,
        batch_size=1,
    )


@pytest.fixture(scope="module")
def frozen_phase_run() -> SearchRun:
    manifest, corpus = _small_phase_inputs(generations=1, cases=3)
    return _phase_run_with_recomputed_positive_evidence(
        _run_small_phase_search(
            manifest,
            corpus,
            workers=1,
            batch_size=1,
        )
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
    if run.manifest.schema_version == 2:
        monkeypatch.setattr(
            reporting_module,
            "_validate_fixed_phase_manifest_for_report",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            diagnostics_module,
            "_require_exact_winner_case_count",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            diagnostics_module,
            "_MIN_SAFE_CONTRIBUTING_GAMES",
            1,
        )
        if run.frozen_candidate is not None:
            monkeypatch.setattr(
                cli,
                "run_winner_diagnostics",
                lambda *args, **kwargs: _canonical_winner_diagnostics(run),
            )
    monkeypatch.setattr(
        cli,
        "load_promotion_corpus",
        lambda *args, **kwargs: run.development_corpus,
    )
    monkeypatch.setattr(cli, "load_search_recipe", lambda *args, **kwargs: run.manifest)
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

    assert args.development_corpus == Path("configs/promotion/development-v1.json")
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
            "--decision-reports",
            "--capture-traces",
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

    def forbidden_diagnostics(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("schema-v1 must never invoke winner diagnostics")

    monkeypatch.setattr(cli, "run_winner_diagnostics", forbidden_diagnostics)

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


def test_phase_manifest_uses_the_same_complete_no_improvement_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    complete_phase_run: SearchRun,
) -> None:
    assert complete_phase_run.manifest.schema_version == 2
    assert complete_phase_run.frozen_candidate is None
    _stub_execution(monkeypatch, complete_phase_run, tmp_path)

    exit_code = cli.main(
        [
            "--manifest",
            "configs/evolution/balanced-v4-search-v2.json",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 1
    assert "without a positive development improvement" in capsys.readouterr().out


def test_phase_manifest_frozen_improvement_exits_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    frozen_phase_run: SearchRun,
) -> None:
    assert frozen_phase_run.manifest.schema_version == 2
    assert frozen_phase_run.frozen_candidate is not None
    _stub_execution(monkeypatch, frozen_phase_run, tmp_path)
    diagnostics = _canonical_winner_diagnostics(frozen_phase_run)
    calls: list[tuple[SearchRun, object, int, int | None]] = []

    def record_diagnostics(
        run: SearchRun,
        *,
        registry: object,
        workers: int,
        batch_size: int | None,
    ) -> object:
        calls.append((run, registry, workers, batch_size))
        return diagnostics

    monkeypatch.setattr(cli, "run_winner_diagnostics", record_diagnostics)

    exit_code = cli.main(
        [
            "--manifest",
            "configs/evolution/balanced-v4-search-v2.json",
            "--workers",
            "2",
            "--batch-size",
            "1",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert calls == [(frozen_phase_run, BOT_SPECS_BY_NAME, 2, 1)]
    assert frozen_phase_run.frozen_candidate.identity in capsys.readouterr().out


@pytest.mark.parametrize(
    ("diagnostic_error", "expected_code"),
    (
        (
            WinnerDiagnosticsError(
                "diagnostic_result_mismatch",
                "PRIVATE_SENTINEL must not be retained",
            ),
            "diagnostic_result_mismatch",
        ),
        (
            RuntimeError("PRIVATE_SENTINEL must not be retained"),
            "winner_diagnostics_operational_failure",
        ),
    ),
)
def test_phase_diagnostics_failure_preserves_selection_and_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    frozen_phase_run: SearchRun,
    diagnostic_error: Exception,
    expected_code: str,
) -> None:
    _stub_execution(monkeypatch, frozen_phase_run, tmp_path)
    captured_reports: list[SearchReport] = []

    def fail_diagnostics(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise diagnostic_error

    def capture_write(
        output_dir: Path,
        *,
        report: SearchReport,
        overwrite: bool,
    ) -> SearchArtifacts:
        del overwrite
        captured_reports.append(report)
        return _artifacts(
            output_dir,
            frozen=report.run.frozen_candidate is not None,
        )

    monkeypatch.setattr(cli, "run_winner_diagnostics", fail_diagnostics)
    monkeypatch.setattr(cli, "write_search_artifacts", capture_write)

    exit_code = cli.main(
        [
            "--manifest",
            "configs/evolution/balanced-v4-search-v2.json",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 2
    assert len(captured_reports) == 1
    report = captured_reports[0]
    assert report.run.selected_candidate == frozen_phase_run.selected_candidate
    assert report.run.frozen_candidate is None
    assert report.winner_diagnostics is None
    assert report.run.failures[-1].code == expected_code
    assert "PRIVATE_SENTINEL" not in report.run.failures[-1].message
    assert "PRIVATE_SENTINEL" not in str(search_report_payload(report))
    assert "PRIVATE_SENTINEL" not in capsys.readouterr().out


def test_phase_manifest_invalid_evidence_exits_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    complete_phase_run: SearchRun,
) -> None:
    invalid = replace(
        complete_phase_run,
        selected_candidate=None,
        frozen_candidate=None,
        failures=(SearchFailure("invalid_candidate_evidence", "Phase evidence was incomplete."),),
    )
    _stub_execution(monkeypatch, invalid, tmp_path)

    exit_code = cli.main(
        [
            "--manifest",
            "configs/evolution/balanced-v4-search-v2.json",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 2
    assert "Phase evidence was incomplete." in capsys.readouterr().out


def test_phase_manifest_report_validation_failure_exits_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    complete_phase_run: SearchRun,
) -> None:
    selected = complete_phase_run.selected_candidate
    assert selected is not None
    candidate_runs = tuple(
        replace(
            candidate_run,
            evaluation=replace(candidate_run.evaluation, rating_delta=1.0),
        )
        if candidate_run.candidate == selected
        else candidate_run
        for candidate_run in complete_phase_run.candidate_runs
    )
    forged = replace(
        complete_phase_run,
        candidate_runs=candidate_runs,
        frozen_candidate=selected,
    )
    _stub_execution(monkeypatch, forged, tmp_path)

    exit_code = cli.main(
        [
            "--manifest",
            "configs/evolution/balanced-v4-search-v2.json",
            "--output-dir",
            str(tmp_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Winner diagnostics failed closed (invalid_frozen_run)." in captured.out
    assert captured.err == ""


def test_held_out_corpus_is_rejected_before_search_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    complete_run: SearchRun,
) -> None:
    held_out = replace(
        complete_run.development_corpus,
        recipe=replace(
            complete_run.development_corpus.recipe,
            purpose="held_out",
        ),
    )
    monkeypatch.setattr(cli, "load_promotion_corpus", lambda *args, **kwargs: held_out)

    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("search started with held-out input")

    monkeypatch.setattr(cli, "run_search", forbidden)

    assert cli.main(_required_args(tmp_path)) == 2
    captured = capsys.readouterr()
    assert "held-out games are the final exam" in captured.err


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
