"""Render deterministic, transactional evidence for heuristic evolution."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from garboid_pocketrocks.evolution.candidates import HeuristicCandidate
from garboid_pocketrocks.evolution.evaluation import CandidateEvaluation
from garboid_pocketrocks.evolution.manifest import search_manifest_payload
from garboid_pocketrocks.evolution.runner import CandidateRun, SearchRun
from garboid_pocketrocks.evolution.search import (
    CandidateRankingKey,
    SearchSelectionError,
    SelectionRecord,
)
from garboid_pocketrocks.promotion.corpus import corpus_snapshot_payload
from garboid_pocketrocks.simulator.monte_carlo import GameSummary
from garboid_pocketrocks.simulator.session import SessionScore

_MANIFEST_NAME = "search-manifest.json"
_REPORT_NAME = "search-report.json"
_EVALUATIONS_NAME = "candidate-evaluations.jsonl"
_SELECTION_NAME = "selection-log.jsonl"
_GAMES_NAME = "development-games.jsonl"
_CORPUS_NAME = "development-corpus-snapshot.json"
_FROZEN_NAME = "frozen-candidate.json"
_REQUIRED_ARTIFACT_NAMES = (
    _MANIFEST_NAME,
    _REPORT_NAME,
    _EVALUATIONS_NAME,
    _SELECTION_NAME,
    _GAMES_NAME,
    _CORPUS_NAME,
)
_ALL_ARTIFACT_NAMES = (*_REQUIRED_ARTIFACT_NAMES, _FROZEN_NAME)


@dataclass(frozen=True, slots=True)
class SearchReport:
    """All configuration, source evidence, and decisions from one search."""

    schema_version: int
    repository_commit: str
    workers: int
    batch_size: int
    run: SearchRun
    artifact_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SearchArtifacts:
    """Paths to one authoritative search artifact generation."""

    search_manifest_json: Path
    search_report_json: Path
    candidate_evaluations_jsonl: Path
    selection_log_jsonl: Path
    development_games_jsonl: Path
    development_corpus_snapshot_json: Path
    frozen_candidate_json: Path | None

    @property
    def paths(self) -> tuple[Path, ...]:
        """Return present artifact paths in canonical order."""

        required = (
            self.search_manifest_json,
            self.search_report_json,
            self.candidate_evaluations_jsonl,
            self.selection_log_jsonl,
            self.development_games_jsonl,
            self.development_corpus_snapshot_json,
        )
        if self.frozen_candidate_json is None:
            return required
        return (*required, self.frozen_candidate_json)


@dataclass(frozen=True, slots=True)
class _PreparedArtifact:
    target: Path
    staged: Path | None
    backup: Path | None


@dataclass(frozen=True, slots=True)
class _RollbackFailure:
    target: Path
    backup: Path | None
    error: OSError


def build_search_report(
    run: SearchRun,
    *,
    repository_commit: str,
    workers: int,
    batch_size: int,
) -> SearchReport:
    """Collect immutable search inputs and results into the report model."""

    normalized_commit = repository_commit.strip()
    if not normalized_commit:
        raise ValueError("repository commit must not be empty")
    if workers <= 0 or batch_size <= 0:
        raise ValueError("workers and batch size must be positive")
    artifact_names = (
        _ALL_ARTIFACT_NAMES if run.frozen_candidate is not None else _REQUIRED_ARTIFACT_NAMES
    )
    return SearchReport(
        schema_version=1,
        repository_commit=normalized_commit,
        workers=workers,
        batch_size=batch_size,
        run=run,
        artifact_names=artifact_names,
    )


def repository_commit() -> str:
    """Return the exact checked-out repository commit."""

    try:
        completed = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "Could not determine the repository commit with git rev-parse HEAD."
        ) from error
    commit = completed.stdout.strip()
    if not commit:
        raise RuntimeError("git did not return a repository commit")
    return commit


def search_report_payload(report: SearchReport) -> dict[str, object]:
    """Convert the authoritative report to its explicit public JSON schema."""

    run = report.run
    requested_candidate_games = sum(item.evaluation.requested_cases for item in run.candidate_runs)
    completed_candidate_games = sum(
        item.evaluation.completed_candidate_games for item in run.candidate_runs
    )
    return {
        "schema_version": report.schema_version,
        "repository_commit": report.repository_commit,
        "status": _search_status(run),
        "search": {
            "name": run.manifest.name,
            "manifest_digest": run.manifest.digest,
            "personality": run.manifest.personality,
            "predecessor_name": run.manifest.predecessor_name,
            "generation_count": run.manifest.algorithm.generation_count,
            "population_size": run.manifest.algorithm.population_size,
            "elite_count": run.manifest.algorithm.elite_count,
        },
        "development_corpus": {
            "name": run.development_corpus.recipe.name,
            "digest": run.development_corpus.digest,
            "cases": len(run.development_corpus.cases),
        },
        "execution": {
            "workers": report.workers,
            "batch_size": report.batch_size,
        },
        "coverage": {
            "proposed_candidates": len(run.candidate_runs),
            "evaluated_candidates": len(run.candidate_runs),
            "completed_generations": len(run.selections),
            "requested_baseline_games": len(run.baseline_jobs),
            "completed_baseline_games": len(run.baseline_result.game_summaries),
            "requested_candidate_games": requested_candidate_games,
            "completed_candidate_games": completed_candidate_games,
        },
        "best_result": _best_result_payload(run),
        "selected_candidate_identity": (
            None if run.selected_candidate is None else run.selected_candidate.identity
        ),
        "frozen_candidate_identity": (
            None if run.frozen_candidate is None else run.frozen_candidate.identity
        ),
        "failures": [
            {"code": failure.code, "message": failure.message} for failure in run.failures
        ],
        "artifacts": list(report.artifact_names),
    }


def write_search_artifacts(
    output_dir: Path,
    *,
    report: SearchReport,
    overwrite: bool = False,
) -> SearchArtifacts:
    """Write one rollback-protected generation of search artifacts."""

    validate_search_output_dir(output_dir, overwrite=overwrite)
    rendered = _render_artifacts(report)
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_search_output_dir(output_dir, overwrite=overwrite)
    prepared = _prepare_artifact_generation(output_dir, rendered)
    _replace_artifact_generation(prepared)
    frozen_path = output_dir / _FROZEN_NAME if report.run.frozen_candidate is not None else None
    return SearchArtifacts(
        search_manifest_json=output_dir / _MANIFEST_NAME,
        search_report_json=output_dir / _REPORT_NAME,
        candidate_evaluations_jsonl=output_dir / _EVALUATIONS_NAME,
        selection_log_jsonl=output_dir / _SELECTION_NAME,
        development_games_jsonl=output_dir / _GAMES_NAME,
        development_corpus_snapshot_json=output_dir / _CORPUS_NAME,
        frozen_candidate_json=frozen_path,
    )


def validate_search_output_dir(
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Reject unsafe search output paths before execution begins."""

    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise NotADirectoryError(f"search output path is not a directory: {output_dir}")
    if not overwrite and any(output_dir.iterdir()):
        raise FileExistsError(f"search output directory is not empty: {output_dir}")


def _search_status(run: SearchRun) -> str:
    if run.failures:
        return "failed"
    if run.frozen_candidate is not None:
        return "frozen_improvement"
    return "complete_no_improvement"


def _best_result_payload(run: SearchRun) -> dict[str, object] | None:
    candidate = run.selected_candidate
    if candidate is None:
        return None
    evaluation = _evaluation_for_identity(run, candidate.identity)
    return {
        "candidate_identity": candidate.identity,
        "generation": candidate.generation,
        "slot": candidate.slot,
        "rating_delta": evaluation.rating_delta,
        "normalized_finish_delta": evaluation.normalized_finish_delta,
        "final_money_delta": evaluation.final_money_delta,
        "worst_challenger_finish_delta": evaluation.worst_challenger_finish_delta,
    }


def _render_artifacts(report: SearchReport) -> tuple[tuple[str, str | None], ...]:
    manifest_payload = {
        **search_manifest_payload(report.run.manifest),
        "digest": report.run.manifest.digest,
    }
    manifest = _render_json_document(manifest_payload)
    search_report = _render_json_document(search_report_payload(report))
    evaluations = _render_json_lines(
        _candidate_evaluation_payload(candidate_run) for candidate_run in report.run.candidate_runs
    )
    selections = _render_json_lines(
        _selection_payload(selection) for selection in report.run.selection_records
    )
    games = _render_json_lines(_development_game_payloads(report.run))
    corpus = _render_json_document(corpus_snapshot_payload(report.run.development_corpus))
    frozen = None
    if report.run.frozen_candidate is not None:
        frozen = _render_json_document(
            _frozen_candidate_payload(
                report,
                search_report_sha256=_sha256_text(search_report),
                candidate_evaluations_sha256=_sha256_text(evaluations),
            )
        )
    return (
        (_MANIFEST_NAME, manifest),
        (_REPORT_NAME, search_report),
        (_EVALUATIONS_NAME, evaluations),
        (_SELECTION_NAME, selections),
        (_GAMES_NAME, games),
        (_CORPUS_NAME, corpus),
        (_FROZEN_NAME, frozen),
    )


def _candidate_evaluation_payload(candidate_run: CandidateRun) -> dict[str, object]:
    candidate = candidate_run.candidate
    evaluation = candidate_run.evaluation
    ranking_key: CandidateRankingKey | None = None
    try:
        ranking_key = candidate_run.evaluated_candidate.ranking_key
    except SearchSelectionError:
        pass
    return {
        "candidate": _candidate_payload(candidate),
        "coverage": {
            "requested_cases": evaluation.requested_cases,
            "completed_baseline_games": evaluation.completed_baseline_games,
            "completed_candidate_games": evaluation.completed_candidate_games,
        },
        "scores": {
            "worst_challenger_finish_delta": evaluation.worst_challenger_finish_delta,
            "challenger_finish_deltas": [
                {
                    "opponent_identity": item.opponent_identity,
                    "shared_cases": item.shared_cases,
                    "normalized_finish_delta": item.normalized_finish_delta,
                }
                for item in evaluation.challenger_finish_deltas
            ],
            "rating_delta": evaluation.rating_delta,
            "normalized_finish_delta": evaluation.normalized_finish_delta,
            "final_money_delta": evaluation.final_money_delta,
        },
        "faults": {
            "candidate": evaluation.candidate_faults,
            "incumbent": evaluation.incumbent_faults,
            "opponents": evaluation.opponent_faults,
            "unattributed": evaluation.unattributed_faults,
            "by_identity": [
                {"bot_id": identity, "count": count}
                for identity, count in evaluation.faults_by_identity
            ],
        },
        "failures": [
            {
                "code": failure.code,
                "message": failure.message,
                "invalidates_run": failure.invalidates_run,
            }
            for failure in evaluation.failures
        ],
        "valid": evaluation.valid,
        "eligible": evaluation.eligible,
        "ranking_key": (None if ranking_key is None else _ranking_key_payload(ranking_key)),
    }


def _candidate_payload(candidate: HeuristicCandidate) -> dict[str, object]:
    coefficients = candidate.genome.coefficients
    return {
        "identity": candidate.identity,
        "personality": candidate.personality,
        "generation": candidate.generation,
        "slot": candidate.slot,
        "parent_identity": candidate.parent_identity,
        "profile_digest": candidate.genome.digest,
        "coefficients": {
            "liquidity_strength": _decimal_text(coefficients.liquidity_strength),
            "future_cash_weight": _decimal_text(coefficients.future_cash_weight),
            "objective_progress_weight": _decimal_text(coefficients.objective_progress_weight),
            "bid_shading": _decimal_text(coefficients.bid_shading),
        },
    }


def _ranking_key_payload(key: CandidateRankingKey) -> dict[str, object]:
    return {
        "fields": [
            "negative_worst_challenger_finish_delta",
            "negative_rating_delta",
            "negative_normalized_finish_delta",
            "negative_final_money_delta",
            "coefficient_values",
            "candidate_identity",
        ],
        "values": [
            key.negative_worst_challenger_finish_delta,
            key.negative_rating_delta,
            key.negative_normalized_finish_delta,
            key.negative_final_money_delta,
            [_decimal_text(value) for value in key.coefficient_values],
            key.candidate_identity,
        ],
    }


def _selection_payload(record: SelectionRecord) -> dict[str, object]:
    return {
        "generation": record.generation,
        "proposal_identities": list(record.proposal_identities),
        "pool_identities": list(record.pool_identities),
        "ranked_pool_identities": list(record.ranked_pool_identities),
        "elite_identities": list(record.elite_identities),
        "ranking_keys": [
            {
                "candidate_identity": identity,
                "key": _ranking_key_payload(key),
            }
            for identity, key in record.ranking_keys
        ],
    }


def _development_game_payloads(run: SearchRun) -> Sequence[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for summary in sorted(
        run.baseline_result.game_summaries,
        key=lambda item: item.game_index,
    ):
        payloads.append(
            _development_game_payload(
                run,
                summary,
                evidence="baseline",
                candidate=None,
            )
        )
    for candidate_run in run.candidate_runs:
        for summary in sorted(
            candidate_run.result.game_summaries,
            key=lambda item: item.game_index,
        ):
            payloads.append(
                _development_game_payload(
                    run,
                    summary,
                    evidence="candidate",
                    candidate=candidate_run.candidate,
                )
            )
    return payloads


def _development_game_payload(
    run: SearchRun,
    summary: GameSummary,
    *,
    evidence: str,
    candidate: HeuristicCandidate | None,
) -> dict[str, object]:
    case = (
        run.development_corpus.cases[summary.game_index]
        if 0 <= summary.game_index < len(run.development_corpus.cases)
        else None
    )
    return {
        "evidence": evidence,
        "candidate_identity": None if candidate is None else candidate.identity,
        "candidate_generation": None if candidate is None else candidate.generation,
        "candidate_slot": None if candidate is None else candidate.slot,
        "case_index": summary.game_index,
        "case_id": None if case is None else case.case_id,
        "game": _game_summary_payload(summary),
    }


def _game_summary_payload(summary: GameSummary) -> dict[str, object]:
    return {
        "game_index": summary.game_index,
        "root_seed": summary.root_seed,
        "seed": summary.seed,
        "player_count": summary.player_count,
        "ruleset_name": summary.ruleset_name,
        "bot_names": list(summary.bot_names),
        "bot_ids": list(summary.bot_ids),
        "scores": [_session_score_payload(score) for score in summary.scores],
        "decision_counts": list(summary.decision_counts),
        "fault_counts": list(summary.fault_counts),
    }


def _session_score_payload(score: SessionScore) -> dict[str, object]:
    return {
        "seat": score.seat,
        "final_money": score.final_money,
        "rank": score.rank,
    }


def _frozen_candidate_payload(
    report: SearchReport,
    *,
    search_report_sha256: str,
    candidate_evaluations_sha256: str,
) -> dict[str, object]:
    run = report.run
    candidate = run.frozen_candidate
    assert candidate is not None
    evaluation = _evaluation_for_identity(run, candidate.identity)
    return {
        "schema_version": 2,
        **_candidate_payload(candidate),
        "predecessor_name": run.manifest.predecessor_name,
        "search": {
            "name": run.manifest.name,
            "manifest_digest": run.manifest.digest,
        },
        "development_corpus": {
            "name": run.development_corpus.recipe.name,
            "digest": run.development_corpus.digest,
        },
        "development_scores": {
            "worst_challenger_finish_delta": evaluation.worst_challenger_finish_delta,
            "rating_delta": evaluation.rating_delta,
            "normalized_finish_delta": evaluation.normalized_finish_delta,
            "final_money_delta": evaluation.final_money_delta,
        },
        "repository_commit": report.repository_commit,
        "source_evidence": {
            "search_report_sha256": search_report_sha256,
            "candidate_evaluations_sha256": candidate_evaluations_sha256,
        },
    }


def _evaluation_for_identity(
    run: SearchRun,
    identity: str,
) -> CandidateEvaluation:
    for candidate_run in run.candidate_runs:
        if candidate_run.candidate.identity == identity:
            return candidate_run.evaluation
    raise ValueError(f"frozen candidate {identity!r} has no evaluation record")


def _decimal_text(value: Decimal) -> str:
    if not value.is_finite():
        raise ValueError("Search artifacts must contain only finite JSON numbers.")
    if value.is_zero():
        return "0"
    normalized = format(value, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized or "0"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _render_json_lines(payloads: Iterable[Mapping[str, object]]) -> str:
    try:
        return "".join(
            json.dumps(
                payload,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
            for payload in payloads
        )
    except ValueError as error:
        raise ValueError("Search artifacts must contain only finite JSON numbers.") from error


def _render_json_document(payload: Mapping[str, object]) -> str:
    try:
        return json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    except ValueError as error:
        raise ValueError("Search artifacts must contain only finite JSON numbers.") from error


def _prepare_artifact_generation(
    output_dir: Path,
    rendered_artifacts: tuple[tuple[str, str | None], ...],
) -> tuple[_PreparedArtifact, ...]:
    prepared: list[_PreparedArtifact] = []
    try:
        for artifact_name, content in rendered_artifacts:
            target = output_dir / artifact_name
            staged = (
                _stage_bytes(
                    output_dir,
                    prefix=f".{artifact_name}.staged.",
                    content=content.encode("utf-8"),
                )
                if content is not None
                else None
            )
            prepared.append(_PreparedArtifact(target=target, staged=staged, backup=None))
        for index, artifact in enumerate(prepared):
            backup = (
                _stage_bytes(
                    output_dir,
                    prefix=f".{artifact.target.name}.backup.",
                    content=artifact.target.read_bytes(),
                )
                if artifact.target.exists()
                else None
            )
            prepared[index] = _PreparedArtifact(
                target=artifact.target,
                staged=artifact.staged,
                backup=backup,
            )
        return tuple(prepared)
    except Exception:
        _remove_prepared_files(prepared)
        raise


def _replace_artifact_generation(
    prepared_artifacts: tuple[_PreparedArtifact, ...],
) -> None:
    replaced: list[_PreparedArtifact] = []
    try:
        for artifact in prepared_artifacts:
            if artifact.staged is None:
                artifact.target.unlink(missing_ok=True)
            else:
                os.replace(artifact.staged, artifact.target)
            replaced.append(artifact)
    except Exception as replacement_error:
        rollback_failures = _rollback_replaced_artifacts(replaced)
        preserved_backups = tuple(
            failure.backup for failure in rollback_failures if failure.backup is not None
        )
        _remove_prepared_files(
            prepared_artifacts,
            preserve_backups=preserved_backups,
        )
        if rollback_failures:
            failure_summary = "; ".join(
                f"{failure.target}: {failure.error}" for failure in rollback_failures
            )
            recovery_summary = (
                " Recovery copies were preserved at: "
                + ", ".join(str(path) for path in preserved_backups)
                if preserved_backups
                else " No recovery copy was available for the unrestored files."
            )
            raise RuntimeError(
                "Search artifact replacement failed, and the previous artifact "
                "generation could not be fully restored. "
                f"Rollback errors: {failure_summary}.{recovery_summary}"
            ) from replacement_error
        raise
    _remove_prepared_files(prepared_artifacts)


def _rollback_replaced_artifacts(
    replaced_artifacts: Sequence[_PreparedArtifact],
) -> tuple[_RollbackFailure, ...]:
    rollback_failures: list[_RollbackFailure] = []
    for artifact in reversed(replaced_artifacts):
        try:
            if artifact.backup is None:
                artifact.target.unlink(missing_ok=True)
            else:
                os.replace(artifact.backup, artifact.target)
        except OSError as error:
            rollback_failures.append(
                _RollbackFailure(
                    target=artifact.target,
                    backup=artifact.backup,
                    error=error,
                )
            )
    return tuple(rollback_failures)


def _remove_prepared_files(
    prepared_artifacts: Sequence[_PreparedArtifact],
    *,
    preserve_backups: Collection[Path] = (),
) -> None:
    for artifact in prepared_artifacts:
        if artifact.staged is not None:
            artifact.staged.unlink(missing_ok=True)
        if artifact.backup is not None and artifact.backup not in preserve_backups:
            artifact.backup.unlink(missing_ok=True)


def _stage_bytes(
    directory: Path,
    *,
    prefix: str,
    content: bytes,
) -> Path:
    temporary_path: Path | None = None
    staged = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=directory,
            prefix=prefix,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        staged = True
        return temporary_path
    finally:
        if temporary_path is not None and not staged:
            temporary_path.unlink(missing_ok=True)
