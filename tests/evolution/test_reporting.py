from __future__ import annotations

import json
import math
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from garboid_pocketrocks.bots import BOT_SPECS_BY_NAME
from garboid_pocketrocks.evolution.reporting import (
    SearchReport,
    build_search_report,
    repository_commit,
    search_report_payload,
    validate_search_output_dir,
    write_search_artifacts,
)
from garboid_pocketrocks.evolution.runner import SearchRun, run_search
from garboid_pocketrocks.promotion.corpus import corpus_snapshot_payload

from .test_runner import _small_inputs

_REQUIRED_ARTIFACTS = (
    "search-manifest.json",
    "search-report.json",
    "candidate-evaluations.jsonl",
    "selection-log.jsonl",
    "development-games.jsonl",
    "development-corpus-snapshot.json",
)
_ALL_ARTIFACTS = (*_REQUIRED_ARTIFACTS, "frozen-candidate.json")


@pytest.fixture(scope="module")
def search_run() -> SearchRun:
    manifest, corpus = _small_inputs(generations=1, population=2, elites=1, cases=1)
    return run_search(
        manifest,
        corpus,
        registry=BOT_SPECS_BY_NAME,
        workers=1,
        batch_size=1,
    )


def _report(run: SearchRun, *, frozen: bool = False) -> SearchReport:
    selected = run.selected_candidate
    assert selected is not None
    normalized_run = replace(
        run,
        frozen_candidate=selected if frozen else None,
    )
    return build_search_report(
        normalized_run,
        repository_commit="0123456789abcdef",
        workers=1,
        batch_size=1,
    )


def test_report_payload_owns_complete_normalized_search_evidence(
    search_run: SearchRun,
) -> None:
    report = _report(search_run, frozen=True)
    assert report.run.selected_candidate is not None
    assert report.run.frozen_candidate is not None

    payload = search_report_payload(report)

    assert set(payload) == {
        "schema_version",
        "repository_commit",
        "status",
        "search",
        "development_corpus",
        "execution",
        "coverage",
        "best_result",
        "selected_candidate_identity",
        "frozen_candidate_identity",
        "failures",
        "artifacts",
    }
    assert payload["repository_commit"] == "0123456789abcdef"
    assert payload["status"] == "frozen_improvement"
    assert payload["search"] == {
        "name": report.run.manifest.name,
        "manifest_digest": report.run.manifest.digest,
        "personality": report.run.manifest.personality,
        "predecessor_name": report.run.manifest.predecessor_name,
        "generation_count": 1,
        "population_size": 2,
        "elite_count": 1,
    }
    assert payload["development_corpus"] == {
        "name": report.run.development_corpus.recipe.name,
        "digest": report.run.development_corpus.digest,
        "cases": 1,
    }
    assert payload["execution"] == {"workers": 1, "batch_size": 1}
    assert payload["coverage"] == {
        "proposed_candidates": 2,
        "evaluated_candidates": 2,
        "completed_generations": 1,
        "requested_baseline_games": 1,
        "completed_baseline_games": 1,
        "requested_candidate_games": 2,
        "completed_candidate_games": 2,
    }
    assert payload["best_result"] == {
        "candidate_identity": report.run.selected_candidate.identity,
        "generation": report.run.selected_candidate.generation,
        "slot": report.run.selected_candidate.slot,
        "worst_challenger_finish_delta": next(
            item.evaluation.worst_challenger_finish_delta
            for item in report.run.candidate_runs
            if item.candidate == report.run.selected_candidate
        ),
        "rating_delta": next(
            item.evaluation.rating_delta
            for item in report.run.candidate_runs
            if item.candidate == report.run.selected_candidate
        ),
        "normalized_finish_delta": next(
            item.evaluation.normalized_finish_delta
            for item in report.run.candidate_runs
            if item.candidate == report.run.selected_candidate
        ),
        "final_money_delta": next(
            item.evaluation.final_money_delta
            for item in report.run.candidate_runs
            if item.candidate == report.run.selected_candidate
        ),
    }
    assert payload["selected_candidate_identity"] == report.run.selected_candidate.identity
    assert payload["frozen_candidate_identity"] == report.run.frozen_candidate.identity
    assert payload["failures"] == []
    assert payload["artifacts"] == list(_ALL_ARTIFACTS)


def test_writes_byte_identical_complete_artifact_generations(
    tmp_path: Path,
    search_run: SearchRun,
) -> None:
    report = _report(search_run, frozen=True)
    assert report.run.frozen_candidate is not None

    first = write_search_artifacts(tmp_path / "first", report=report)
    second = write_search_artifacts(tmp_path / "second", report=report)

    first_paths = tuple(path for path in first.paths)
    second_paths = tuple(path for path in second.paths)
    assert tuple(path.name for path in first_paths) == _ALL_ARTIFACTS
    assert [path.read_bytes() for path in first_paths] == [
        path.read_bytes() for path in second_paths
    ]
    assert all(path.read_bytes().endswith(b"\n") for path in first_paths)

    manifest = json.loads(first.search_manifest_json.read_text())
    assert manifest["digest"] == report.run.manifest.digest
    assert json.loads(first.development_corpus_snapshot_json.read_text()) == (
        corpus_snapshot_payload(report.run.development_corpus)
    )

    evaluations = [
        json.loads(line) for line in first.candidate_evaluations_jsonl.read_text().splitlines()
    ]
    assert [item["candidate"]["identity"] for item in evaluations] == [
        item.candidate.identity for item in report.run.candidate_runs
    ]
    assert evaluations[0]["ranking_key"]["fields"] == [
        "negative_worst_challenger_finish_delta",
        "negative_rating_delta",
        "negative_normalized_finish_delta",
        "negative_final_money_delta",
        "coefficient_values",
        "candidate_identity",
    ]

    games = [json.loads(line) for line in first.development_games_jsonl.read_text().splitlines()]
    assert [item["evidence"] for item in games] == [
        "baseline",
        "candidate",
        "candidate",
    ]
    assert [item["case_index"] for item in games] == [0, 0, 0]
    assert [item["candidate_identity"] for item in games] == [
        None,
        report.run.candidate_runs[0].candidate.identity,
        report.run.candidate_runs[1].candidate.identity,
    ]

    assert first.frozen_candidate_json is not None
    frozen = json.loads(first.frozen_candidate_json.read_text())
    assert frozen["schema_version"] == 2
    assert frozen["identity"] == report.run.frozen_candidate.identity
    assert frozen["repository_commit"] == report.repository_commit
    assert frozen["source_evidence"]["search_report_sha256"]
    assert frozen["source_evidence"]["candidate_evaluations_sha256"]


def test_nonfinite_evidence_fails_before_creating_output(
    tmp_path: Path,
    search_run: SearchRun,
) -> None:
    bad_evaluation = replace(
        search_run.candidate_runs[0].evaluation,
        rating_delta=math.nan,
    )
    bad_run = replace(
        search_run,
        candidate_runs=(
            replace(search_run.candidate_runs[0], evaluation=bad_evaluation),
            *search_run.candidate_runs[1:],
        ),
        frozen_candidate=None,
    )

    with pytest.raises(ValueError, match="finite JSON"):
        write_search_artifacts(tmp_path, report=_report(bad_run))

    assert not any((tmp_path / name).exists() for name in _ALL_ARTIFACTS)


def test_output_preflight_requires_overwrite_for_nonempty_directory(
    tmp_path: Path,
) -> None:
    validate_search_output_dir(tmp_path / "missing")
    notes = tmp_path / "notes.txt"
    notes.write_text("keep me")
    with pytest.raises(FileExistsError, match="not empty"):
        validate_search_output_dir(tmp_path)
    validate_search_output_dir(tmp_path, overwrite=True)

    plain_file = tmp_path / "file"
    plain_file.write_text("plain")
    with pytest.raises(NotADirectoryError, match="not a directory"):
        validate_search_output_dir(plain_file)


def test_repository_commit_comes_from_git_rev_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def completed(command: tuple[str, ...], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert command == ("git", "rev-parse", "HEAD")
        assert kwargs == {"check": True, "capture_output": True, "text": True}
        return subprocess.CompletedProcess(command, 0, stdout="abc123\n")

    monkeypatch.setattr(
        "garboid_pocketrocks.evolution.reporting.subprocess.run",
        completed,
    )

    assert repository_commit() == "abc123"


def test_overwrite_preserves_unrelated_files_and_removes_stale_freeze(
    tmp_path: Path,
    search_run: SearchRun,
) -> None:
    write_search_artifacts(tmp_path, report=_report(search_run, frozen=True))
    unrelated = tmp_path / "notes.txt"
    unrelated.write_text("keep me")

    artifacts = write_search_artifacts(
        tmp_path,
        report=_report(search_run, frozen=False),
        overwrite=True,
    )

    assert artifacts.frozen_candidate_json is None
    assert not (tmp_path / "frozen-candidate.json").exists()
    assert unrelated.read_text() == "keep me"
    assert {path.name for path in tmp_path.iterdir()} == {
        *_REQUIRED_ARTIFACTS,
        "notes.txt",
    }


@pytest.mark.parametrize("failure_position", range(1, 8))
def test_replace_failure_rolls_back_replacement_and_stale_freeze_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    search_run: SearchRun,
    failure_position: int,
) -> None:
    write_search_artifacts(tmp_path, report=_report(search_run, frozen=True))
    previous = {name: (tmp_path / name).read_bytes() for name in _ALL_ARTIFACTS}
    unrelated = tmp_path / "notes.txt"
    unrelated.write_text("keep me")
    real_replace = os.replace
    operations = 0

    def fail_selected_operation(source: Any, destination: Any) -> None:
        nonlocal operations
        source_path = Path(source)
        destination_path = Path(destination)
        if destination_path.name in _REQUIRED_ARTIFACTS and ".backup." not in source_path.name:
            operations += 1
            if operations == failure_position:
                raise OSError(f"simulated failure {failure_position}")
        real_replace(source, destination)

    monkeypatch.setattr(
        "garboid_pocketrocks.evolution.reporting.os.replace",
        fail_selected_operation,
    )
    if failure_position == 7:
        real_unlink = Path.unlink

        def fail_stale_freeze_unlink(path: Path, *args: Any, **kwargs: Any) -> None:
            if path.name == "frozen-candidate.json" and ".backup." not in path.name:
                raise OSError("simulated failure 7")
            real_unlink(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", fail_stale_freeze_unlink)

    with pytest.raises(OSError, match=f"simulated failure {failure_position}"):
        write_search_artifacts(
            tmp_path,
            report=replace(_report(search_run), repository_commit=f"commit-{failure_position}"),
            overwrite=True,
        )

    assert {name: (tmp_path / name).read_bytes() for name in _ALL_ARTIFACTS} == previous
    assert unrelated.read_text() == "keep me"
    assert not tuple(path for path in tmp_path.iterdir() if path.name.startswith("."))
