"""Render deterministic, transactional evidence for heuristic evolution."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from decimal import Decimal
from functools import partial
from pathlib import Path

from garboid_pocketrocks.bots import BOT_SPECS_BY_NAME, BotSpec
from garboid_pocketrocks.evolution.candidates import (
    PhaseAwareHeuristicCandidate,
    SearchCandidate,
    candidate_bot_spec,
)
from garboid_pocketrocks.evolution.diagnostics import (
    WINNER_DECISION_SLICES_NAME,
    WINNER_DIAGNOSTIC_NAMES,
    WINNER_DIAGNOSTICS_JSON_NAME,
    WINNER_DIAGNOSTICS_MARKDOWN_NAME,
    WinnerDiagnostics,
    validate_winner_diagnostics_evidence,
)
from garboid_pocketrocks.evolution.evaluation import (
    CandidateEvaluation,
    evaluate_candidate,
)
from garboid_pocketrocks.evolution.manifest import (
    COEFFICIENT_NAMES,
    CoefficientValues,
    PhaseSearchManifest,
    SearchManifest,
    SearchManifestError,
    SearchRecipe,
    phase_search_manifest_payload,
    recompute_phase_search_manifest_digest,
    search_manifest_payload,
    validate_phase_search_manifest_contract,
)
from garboid_pocketrocks.evolution.planning import (
    DevelopmentPlan,
    plan_development_games,
)
from garboid_pocketrocks.evolution.runner import CandidateRun, SearchRun
from garboid_pocketrocks.evolution.search import (
    CandidateRankingKey,
    EvaluatedCandidate,
    SearchSelectionError,
    SelectionRecord,
    freeze_candidate,
    select_generation,
)
from garboid_pocketrocks.promotion.corpus import (
    corpus_snapshot_payload,
    recompute_promotion_corpus_digest,
)
from garboid_pocketrocks.simulator.monte_carlo import (
    GameJob,
    GameSummary,
    MonteCarloConfig,
)
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
_ALL_PHASE_ARTIFACT_NAMES = (*_ALL_ARTIFACT_NAMES, *WINNER_DIAGNOSTIC_NAMES)


@dataclass(frozen=True, slots=True)
class SearchReport:
    """All configuration, source evidence, and decisions from one search."""

    schema_version: int
    repository_commit: str
    workers: int
    batch_size: int
    run: SearchRun
    artifact_names: tuple[str, ...]
    winner_diagnostics: WinnerDiagnostics | None = None


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
    winner_decision_slices_csv: Path | None = None
    winner_diagnostics_json: Path | None = None
    winner_diagnostics_markdown: Path | None = None

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
        frozen = (*required, self.frozen_candidate_json)
        diagnostics = (
            self.winner_decision_slices_csv,
            self.winner_diagnostics_json,
            self.winner_diagnostics_markdown,
        )
        if all(path is not None for path in diagnostics):
            return (*frozen, *(path for path in diagnostics if path is not None))
        return frozen


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
    winner_diagnostics: WinnerDiagnostics | None = None,
) -> SearchReport:
    """Collect immutable search inputs and results into the report model."""

    normalized_commit = repository_commit.strip()
    if not normalized_commit:
        raise ValueError("repository commit must not be empty")
    if workers <= 0 or batch_size <= 0:
        raise ValueError("workers and batch size must be positive")
    if isinstance(run.manifest, PhaseSearchManifest):
        _validate_phase_report_contract(run)
        _validate_phase_candidates(run)
        _validate_phase_freeze(run)
    artifact_names: tuple[str, ...]
    if isinstance(run.manifest, PhaseSearchManifest) and run.frozen_candidate is not None:
        if winner_diagnostics is None:
            raise ValueError("frozen schema-v2 reports require winner diagnostics")
        _validate_winner_diagnostics(run, winner_diagnostics)
        artifact_names = _ALL_PHASE_ARTIFACT_NAMES
    else:
        if winner_diagnostics is not None:
            raise ValueError("winner diagnostics are allowed only for frozen schema-v2 reports")
        artifact_names = (
            _ALL_ARTIFACT_NAMES if run.frozen_candidate is not None else _REQUIRED_ARTIFACT_NAMES
        )
    return SearchReport(
        schema_version=run.manifest.schema_version,
        repository_commit=normalized_commit,
        workers=workers,
        batch_size=batch_size,
        run=run,
        artifact_names=artifact_names,
        winner_diagnostics=winner_diagnostics,
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
    payload = {
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
    if isinstance(run.manifest, PhaseSearchManifest):
        search = payload["search"]
        assert isinstance(search, dict)
        search["phase_selector"] = _phase_selector_payload(run.manifest)
        search["boundary_evidence"] = _boundary_evidence_payload(run.manifest)
    if report.winner_diagnostics is not None:
        payload["winner_diagnostics"] = {
            "winner_identity": report.winner_diagnostics.winner_identity,
            "artifacts": [
                {"name": name, "sha256": digest}
                for name, digest in report.winner_diagnostics.artifact_digests
            ],
        }
    return payload


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
    has_diagnostics = report.winner_diagnostics is not None
    return SearchArtifacts(
        search_manifest_json=output_dir / _MANIFEST_NAME,
        search_report_json=output_dir / _REPORT_NAME,
        candidate_evaluations_jsonl=output_dir / _EVALUATIONS_NAME,
        selection_log_jsonl=output_dir / _SELECTION_NAME,
        development_games_jsonl=output_dir / _GAMES_NAME,
        development_corpus_snapshot_json=output_dir / _CORPUS_NAME,
        frozen_candidate_json=frozen_path,
        winner_decision_slices_csv=(
            output_dir / WINNER_DECISION_SLICES_NAME if has_diagnostics else None
        ),
        winner_diagnostics_json=(
            output_dir / WINNER_DIAGNOSTICS_JSON_NAME if has_diagnostics else None
        ),
        winner_diagnostics_markdown=(
            output_dir / WINNER_DIAGNOSTICS_MARKDOWN_NAME if has_diagnostics else None
        ),
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
    payload: dict[str, object] = {
        "candidate_identity": candidate.identity,
        "generation": candidate.generation,
        "slot": candidate.slot,
        "rating_delta": evaluation.rating_delta,
        "normalized_finish_delta": evaluation.normalized_finish_delta,
        "final_money_delta": evaluation.final_money_delta,
    }
    if not isinstance(run.manifest, PhaseSearchManifest):
        payload["worst_challenger_finish_delta"] = evaluation.worst_challenger_finish_delta
    if isinstance(candidate, PhaseAwareHeuristicCandidate):
        assert isinstance(run.manifest, PhaseSearchManifest)
        payload["candidate"] = _candidate_payload(candidate, manifest=run.manifest)
    return payload


def _render_artifacts(report: SearchReport) -> tuple[tuple[str, str | None], ...]:
    manifest = report.run.manifest
    normalized_manifest = (
        phase_search_manifest_payload(manifest)
        if isinstance(manifest, PhaseSearchManifest)
        else search_manifest_payload(manifest)
    )
    manifest_payload = {
        **normalized_manifest,
        "digest": manifest.digest,
    }
    rendered_manifest = _render_json_document(manifest_payload)
    search_report = _render_json_document(search_report_payload(report))
    evaluations = _render_json_lines(
        _candidate_evaluation_payload(candidate_run, manifest=manifest)
        for candidate_run in report.run.candidate_runs
    )
    selections = _render_json_lines(
        _selection_payload(selection, manifest=manifest)
        for selection in report.run.selection_records
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
                selection_log_sha256=_sha256_text(selections),
                development_games_sha256=_sha256_text(games),
            )
        )
    diagnostics_by_name = (
        {}
        if report.winner_diagnostics is None
        else dict(report.winner_diagnostics.named_contents())
    )
    return (
        (_MANIFEST_NAME, rendered_manifest),
        (_REPORT_NAME, search_report),
        (_EVALUATIONS_NAME, evaluations),
        (_SELECTION_NAME, selections),
        (_GAMES_NAME, games),
        (_CORPUS_NAME, corpus),
        (_FROZEN_NAME, frozen),
        (
            WINNER_DECISION_SLICES_NAME,
            diagnostics_by_name.get(WINNER_DECISION_SLICES_NAME),
        ),
        (
            WINNER_DIAGNOSTICS_JSON_NAME,
            diagnostics_by_name.get(WINNER_DIAGNOSTICS_JSON_NAME),
        ),
        (
            WINNER_DIAGNOSTICS_MARKDOWN_NAME,
            diagnostics_by_name.get(WINNER_DIAGNOSTICS_MARKDOWN_NAME),
        ),
    )


def _candidate_evaluation_payload(
    candidate_run: CandidateRun,
    *,
    manifest: SearchRecipe,
) -> dict[str, object]:
    candidate = candidate_run.candidate
    evaluation = candidate_run.evaluation
    ranking_key: CandidateRankingKey | None = None
    try:
        ranking_key = candidate_run.evaluated_candidate.ranking_key
    except SearchSelectionError:
        pass
    scores: dict[str, object] = {
        "rating_delta": evaluation.rating_delta,
        "normalized_finish_delta": evaluation.normalized_finish_delta,
        "final_money_delta": evaluation.final_money_delta,
    }
    if not isinstance(manifest, PhaseSearchManifest):
        scores = {
            "worst_challenger_finish_delta": evaluation.worst_challenger_finish_delta,
            "challenger_finish_deltas": [
                {
                    "opponent_identity": item.opponent_identity,
                    "shared_cases": item.shared_cases,
                    "normalized_finish_delta": item.normalized_finish_delta,
                }
                for item in evaluation.challenger_finish_deltas
            ],
            **scores,
        }
    return {
        "candidate": _candidate_payload(candidate, manifest=manifest),
        "coverage": {
            "requested_cases": evaluation.requested_cases,
            "completed_baseline_games": evaluation.completed_baseline_games,
            "completed_candidate_games": evaluation.completed_candidate_games,
        },
        "scores": scores,
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
        "ranking_key": (
            None
            if ranking_key is None
            else _ranking_key_payload(
                ranking_key,
                include_challenger=not isinstance(manifest, PhaseSearchManifest),
            )
        ),
    }


def _candidate_payload(
    candidate: SearchCandidate,
    *,
    manifest: SearchRecipe,
) -> dict[str, object]:
    if isinstance(candidate, PhaseAwareHeuristicCandidate):
        if not isinstance(manifest, PhaseSearchManifest):
            raise ValueError("phase-aware candidate requires a phase-aware search manifest")
        _validate_phase_candidate(candidate, manifest)
        experts = candidate.genome.experts
        return {
            "identity": candidate.identity,
            "personality": candidate.personality,
            "generation": candidate.generation,
            "slot": candidate.slot,
            "parent_identity": candidate.parent_identity,
            "profile_digest": candidate.genome.digest,
            "phase_selector": _phase_selector_payload(manifest),
            "experts": {
                phase: _coefficient_values_payload(values)
                for phase, values in zip(
                    ("early", "middle", "late"),
                    experts.as_tuple(),
                    strict=True,
                )
            },
        }

    if not isinstance(manifest, SearchManifest):
        raise ValueError("scalar candidate requires a scalar search manifest")
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


def _ranking_key_payload(
    key: CandidateRankingKey,
    *,
    include_challenger: bool,
) -> dict[str, object]:
    fields = [
        "negative_worst_challenger_finish_delta",
        "negative_rating_delta",
        "negative_normalized_finish_delta",
        "negative_final_money_delta",
        "coefficient_values",
        "candidate_identity",
    ]
    values: list[object] = [
        key.negative_worst_challenger_finish_delta,
        key.negative_rating_delta,
        key.negative_normalized_finish_delta,
        key.negative_final_money_delta,
        [_decimal_text(value) for value in key.coefficient_values],
        key.candidate_identity,
    ]
    if not include_challenger:
        del fields[0]
        del values[0]
    return {"fields": fields, "values": values}


def _selection_payload(
    record: SelectionRecord,
    *,
    manifest: SearchRecipe,
) -> dict[str, object]:
    return {
        "generation": record.generation,
        "proposal_identities": list(record.proposal_identities),
        "pool_identities": list(record.pool_identities),
        "ranked_pool_identities": list(record.ranked_pool_identities),
        "elite_identities": list(record.elite_identities),
        "ranking_keys": [
            {
                "candidate_identity": identity,
                "key": _ranking_key_payload(
                    key,
                    include_challenger=not isinstance(manifest, PhaseSearchManifest),
                ),
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
    candidate: SearchCandidate | None,
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
    selection_log_sha256: str,
    development_games_sha256: str,
) -> dict[str, object]:
    run = report.run
    candidate = run.frozen_candidate
    assert candidate is not None
    evaluation = _evaluation_for_identity(run, candidate.identity)
    payload = {
        "schema_version": 2,
        **_candidate_payload(candidate, manifest=run.manifest),
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
    if not isinstance(run.manifest, PhaseSearchManifest):
        development_scores = payload["development_scores"]
        assert isinstance(development_scores, dict)
        payload["development_scores"] = {
            "worst_challenger_finish_delta": evaluation.worst_challenger_finish_delta,
            **development_scores,
        }
    if isinstance(run.manifest, PhaseSearchManifest):
        payload["boundary_evidence"] = _boundary_evidence_payload(run.manifest)
        assert report.winner_diagnostics is not None
        source_evidence = payload["source_evidence"]
        assert isinstance(source_evidence, dict)
        source_evidence["selection_log_sha256"] = selection_log_sha256
        source_evidence["development_games_sha256"] = development_games_sha256
        source_evidence["winner_diagnostics"] = {
            name: digest for name, digest in report.winner_diagnostics.artifact_digests
        }
    return payload


def _phase_selector_payload(manifest: PhaseSearchManifest) -> dict[str, str]:
    selector = manifest.phase_selector
    return {
        "kind": selector.kind,
        "early": selector.early,
        "middle": selector.middle,
        "late": selector.late,
    }


def _boundary_evidence_payload(manifest: PhaseSearchManifest) -> dict[str, str]:
    evidence = manifest.boundary_evidence
    return {
        "report_path": evidence.report_path,
        "report_digest": evidence.report_digest,
        "slices_path": evidence.slices_path,
        "slices_digest": evidence.slices_digest,
    }


def _coefficient_values_payload(values: CoefficientValues) -> dict[str, str]:
    coefficients = values.as_tuple()
    return {
        name: _decimal_text(value)
        for name, value in zip(COEFFICIENT_NAMES, coefficients, strict=True)
    }


def _validate_winner_diagnostics(
    run: SearchRun,
    diagnostics: WinnerDiagnostics,
) -> None:
    validate_winner_diagnostics_evidence(
        run,
        diagnostics,
        registry=BOT_SPECS_BY_NAME,
    )


def _validate_phase_report_contract(run: SearchRun) -> None:
    manifest = run.manifest
    assert isinstance(manifest, PhaseSearchManifest)
    _validate_fixed_phase_manifest_for_report(manifest)
    corpus = run.development_corpus
    if recompute_promotion_corpus_digest(corpus) != corpus.digest:
        raise ValueError("schema-v2 report development corpus digest does not match its content")
    if (
        corpus.recipe.purpose != "development"
        or corpus.recipe.name != manifest.development_corpus.name
        or corpus.digest != manifest.development_corpus.digest
    ):
        raise ValueError("schema-v2 report development corpus must match the manifest binding")
    canonical_incumbent = BOT_SPECS_BY_NAME.get(manifest.predecessor_name)
    if canonical_incumbent is None or run.incumbent is not canonical_incumbent:
        raise ValueError("schema-v2 report incumbent must be the canonical predecessor")


def _validate_fixed_phase_manifest_for_report(manifest: PhaseSearchManifest) -> None:
    if recompute_phase_search_manifest_digest(manifest) != manifest.digest:
        raise ValueError("schema-v2 report manifest digest does not match its content")
    try:
        validate_phase_search_manifest_contract(manifest)
    except SearchManifestError as error:
        raise ValueError(str(error)) from error


def _validate_phase_candidate_plan(
    candidate_run: CandidateRun,
    run: SearchRun,
) -> None:
    plan = candidate_run.plan
    candidate = candidate_run.candidate
    expected_candidate = candidate_bot_spec(candidate)
    canonical_opponents = tuple(
        BOT_SPECS_BY_NAME[name] for name in run.development_corpus.recipe.opponent_names
    )
    if plan.corpus != run.development_corpus:
        raise ValueError("schema-v2 candidate plan corpus must match the report development corpus")
    if not _bot_spec_matches(plan.candidate, expected_candidate):
        raise ValueError(
            "schema-v2 candidate plan spec must match the candidate identity and profile"
        )
    if plan.incumbent is not run.incumbent:
        raise ValueError("schema-v2 candidate plan incumbent must match the report incumbent")
    if len(plan.opponents) != len(canonical_opponents) or any(
        actual is not expected
        for actual, expected in zip(plan.opponents, canonical_opponents, strict=True)
    ):
        raise ValueError("schema-v2 candidate plan opponents must use canonical registry order")
    try:
        rebuilt = plan_development_games(
            run.development_corpus,
            candidate=expected_candidate,
            incumbent=run.incumbent,
            registry=BOT_SPECS_BY_NAME,
        )
    except ValueError as error:
        raise ValueError(
            "schema-v2 candidate plan could not be rebuilt from report-bound inputs"
        ) from error
    if not _config_matches(plan.baseline_config, run.baseline_config) or not _jobs_match(
        plan.baseline_jobs, run.baseline_jobs
    ):
        raise ValueError(
            "schema-v2 candidate plan baseline config and jobs must match the report baseline"
        )
    if not _plan_evidence_matches(plan, rebuilt):
        raise ValueError(
            "schema-v2 candidate plan config and jobs must match the rebuilt development plan"
        )


def _plan_evidence_matches(actual: DevelopmentPlan, expected: DevelopmentPlan) -> bool:
    return (
        _config_matches(actual.baseline_config, expected.baseline_config)
        and _jobs_match(actual.baseline_jobs, expected.baseline_jobs)
        and _config_matches(actual.candidate_config, expected.candidate_config)
        and _jobs_match(actual.candidate_jobs, expected.candidate_jobs)
    )


def _config_matches(actual: MonteCarloConfig, expected: MonteCarloConfig) -> bool:
    return _bot_specs_match(actual.bot_specs, expected.bot_specs) and (
        _exact_value(actual.games, expected.games)
        and _exact_value(actual.player_counts, expected.player_counts)
        and _exact_value(actual.value_charts, expected.value_charts)
        and _exact_value(actual.root_seed, expected.root_seed)
        and _exact_value(actual.objectives_enabled, expected.objectives_enabled)
        and _exact_value(actual.fault_mode, expected.fault_mode)
        and _exact_value(actual.capture_replays, expected.capture_replays)
        and _exact_value(
            actual.capture_decision_traces,
            expected.capture_decision_traces,
        )
    )


def _jobs_match(
    actual_jobs: tuple[GameJob, ...],
    expected_jobs: tuple[GameJob, ...],
) -> bool:
    if len(actual_jobs) != len(expected_jobs):
        return False
    return all(
        _bot_specs_match(actual.lineup, expected.lineup)
        and _exact_value(actual.game_index, expected.game_index)
        and _exact_value(actual.root_seed, expected.root_seed)
        and _exact_value(actual.seed, expected.seed)
        and _exact_value(actual.player_count, expected.player_count)
        and _exact_value(actual.value_chart, expected.value_chart)
        and _exact_value(actual.objectives_enabled, expected.objectives_enabled)
        and _exact_value(actual.fault_mode, expected.fault_mode)
        and _exact_value(
            actual.capture_decision_traces,
            expected.capture_decision_traces,
        )
        for actual, expected in zip(actual_jobs, expected_jobs, strict=True)
    )


def _exact_value(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, tuple):
        assert isinstance(actual, tuple)
        return len(actual) == len(expected) and all(
            _exact_value(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected, strict=True)
        )
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        return actual.keys() == expected.keys() and all(
            _exact_value(actual[key], expected[key]) for key in expected
        )
    if is_dataclass(expected):
        return all(
            _exact_value(
                getattr(actual, field.name),
                getattr(expected, field.name),
            )
            for field in fields(expected)
        )
    return actual == expected


def _bot_specs_match(
    actual_specs: tuple[BotSpec, ...],
    expected_specs: tuple[BotSpec, ...],
) -> bool:
    return len(actual_specs) == len(expected_specs) and all(
        _bot_spec_matches(actual, expected)
        for actual, expected in zip(actual_specs, expected_specs, strict=True)
    )


def _bot_spec_matches(actual: BotSpec, expected: BotSpec) -> bool:
    if actual.name != expected.name or actual.bot_id != expected.bot_id:
        return False
    actual_factory = actual.brain_factory
    expected_factory = expected.brain_factory
    if isinstance(expected_factory, partial):
        return (
            isinstance(actual_factory, partial)
            and actual_factory.func is expected_factory.func
            and _exact_value(actual_factory.args, expected_factory.args)
            and _exact_value(
                actual_factory.keywords or {},
                expected_factory.keywords or {},
            )
        )
    return actual_factory is expected_factory


def _validate_phase_freeze(run: SearchRun) -> None:
    candidate = run.frozen_candidate
    if candidate is None:
        return
    manifest = run.manifest
    assert isinstance(manifest, PhaseSearchManifest)
    if run.selected_candidate != candidate or run.failures:
        raise ValueError("phase-aware frozen candidate is inconsistent with the search result")
    expected_candidates = manifest.algorithm.generation_count * manifest.algorithm.population_size
    if len(run.candidate_runs) != expected_candidates:
        raise ValueError(
            "phase-aware frozen candidate requires complete generation and population coverage"
        )
    identities: list[str] = []
    runs_by_generation: list[list[CandidateRun]] = [
        [] for _ in range(manifest.algorithm.generation_count)
    ]
    for index, candidate_run in enumerate(run.candidate_runs):
        expected_generation, expected_slot = divmod(
            index,
            manifest.algorithm.population_size,
        )
        proposal = candidate_run.candidate
        if proposal.generation != expected_generation or proposal.slot != expected_slot:
            raise ValueError(
                "phase-aware frozen candidate requires complete ordered candidate coverage"
            )
        identities.append(proposal.identity)
        runs_by_generation[expected_generation].append(candidate_run)
    if len(set(identities)) != len(identities):
        raise ValueError("phase-aware frozen candidate requires unique candidate identities")
    if len(run.selections) != manifest.algorithm.generation_count:
        raise ValueError("phase-aware frozen candidate requires complete ordered selection history")
    recomputed_by_generation: list[list[EvaluatedCandidate]] = [
        [] for _ in range(manifest.algorithm.generation_count)
    ]
    for candidate_run in run.candidate_runs:
        recomputed_evaluation = evaluate_candidate(
            candidate_run.plan,
            run.baseline_result,
            candidate_run.result,
        )
        if recomputed_evaluation != candidate_run.evaluation:
            raise ValueError(
                "phase-aware frozen candidate requires evaluations recomputed "
                "from the recorded plans and results"
            )
        recomputed_by_generation[candidate_run.candidate.generation].append(
            EvaluatedCandidate(
                candidate=candidate_run.candidate,
                evaluation=recomputed_evaluation,
            )
        )

    recomputed_selections = []
    prior_elites: tuple[EvaluatedCandidate, ...] = ()
    for generation, proposals in enumerate(recomputed_by_generation):
        try:
            recomputed_selection = select_generation(
                generation=generation,
                proposals=tuple(proposals),
                prior_elites=prior_elites,
                elite_count=manifest.algorithm.elite_count,
            )
        except SearchSelectionError as error:
            raise ValueError(
                "phase-aware frozen candidate selection could not be recomputed "
                "from the recorded evidence"
            ) from error
        recomputed_selections.append(recomputed_selection)
        prior_elites = recomputed_selection.elites
    if tuple(recomputed_selections) != run.selections:
        raise ValueError(
            "phase-aware frozen candidate requires selections recomputed "
            "from the recorded evaluations"
        )

    recomputed_winner = prior_elites[0]
    final_winner = recomputed_winner.candidate
    if run.selected_candidate != final_winner or candidate != final_winner:
        raise ValueError(
            "phase-aware selected and frozen candidates must be the final selection winner"
        )
    matches = tuple(
        candidate_run
        for candidate_run in run.candidate_runs
        if candidate_run.candidate.identity == candidate.identity
    )
    if len(matches) != 1:
        raise ValueError("phase-aware frozen candidate must have exactly one evaluation record")
    if freeze_candidate(recomputed_winner) != candidate:
        raise ValueError(
            "phase-aware frozen candidate does not have complete fault-free positive evidence"
        )


def _validate_phase_candidates(run: SearchRun) -> None:
    manifest = run.manifest
    assert isinstance(manifest, PhaseSearchManifest)
    candidates: list[SearchCandidate] = [
        candidate_run.candidate for candidate_run in run.candidate_runs
    ]
    for candidate_run in run.candidate_runs:
        _validate_phase_candidate(candidate_run.candidate, manifest)
        _validate_phase_candidate_plan(candidate_run, run)
    for selection in run.selections:
        candidates.extend(item.candidate for item in selection.ranked_pool)
        candidates.extend(item.candidate for item in selection.elites)
    if run.selected_candidate is not None:
        candidates.append(run.selected_candidate)
    if run.frozen_candidate is not None:
        candidates.append(run.frozen_candidate)
    for candidate in candidates:
        _validate_phase_candidate(candidate, manifest)


def _validate_phase_candidate(
    candidate: SearchCandidate,
    manifest: PhaseSearchManifest,
) -> None:
    if not isinstance(candidate, PhaseAwareHeuristicCandidate):
        raise ValueError("schema-v2 reports require phase-aware candidates")
    if candidate.personality != manifest.personality:
        raise ValueError("schema-v2 candidate personality must match the manifest")
    if candidate.genome.phase_selector != manifest.phase_selector.kind:
        raise ValueError("schema-v2 candidate selector must match the manifest")


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
