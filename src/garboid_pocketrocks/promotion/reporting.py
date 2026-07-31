"""Render deterministic, machine-readable evidence for a promotion decision."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from garboid_pocketrocks.bots import BotSpec
from garboid_pocketrocks.promotion.analysis import PromotionAnalysis
from garboid_pocketrocks.promotion.corpus import (
    PromotionCorpus,
    corpus_snapshot_payload,
)
from garboid_pocketrocks.promotion.planning import (
    EffectiveOpponentPool,
    PromotionPlan,
    effective_opponent_pool_payload,
    promotion_plan_payload,
)
from garboid_pocketrocks.simulator.monte_carlo import GameSummary
from garboid_pocketrocks.simulator.runner import FaultMode
from garboid_pocketrocks.simulator.session import SessionScore

_PROMOTION_REPORT_NAME = "promotion-report.json"
_PAIRED_GAMES_NAME = "paired-games.jsonl"
_CORPUS_SNAPSHOT_NAME = "corpus-snapshot.json"
_ARTIFACT_NAMES = (
    _PROMOTION_REPORT_NAME,
    _PAIRED_GAMES_NAME,
    _CORPUS_SNAPSHOT_NAME,
)


@dataclass(frozen=True, slots=True)
class PromotionReport:
    """All configuration, evidence, and results behind one promotion decision."""

    schema_version: int
    repository_commit: str
    candidate: BotSpec
    incumbent: BotSpec
    opponents: tuple[BotSpec, ...]
    opponent_pool: EffectiveOpponentPool | None
    plan: PromotionPlan | None
    development: PromotionCorpus
    held_out: PromotionCorpus
    bootstrap_samples: int
    bootstrap_seed: int
    workers: int
    batch_size: int
    analysis: PromotionAnalysis
    artifact_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PromotionArtifacts:
    """Paths to the three authoritative promotion artifacts."""

    report_json: Path
    paired_games_jsonl: Path
    corpus_snapshot_json: Path


@dataclass(frozen=True, slots=True)
class _PreparedArtifact:
    target: Path
    staged: Path
    backup: Path | None


@dataclass(frozen=True, slots=True)
class _RollbackFailure:
    target: Path
    backup: Path | None
    error: OSError


def build_promotion_report(
    *,
    repository_commit: str,
    candidate: BotSpec,
    incumbent: BotSpec,
    opponents: tuple[BotSpec, ...],
    opponent_pool: EffectiveOpponentPool | None,
    plan: PromotionPlan | None,
    development: PromotionCorpus,
    held_out: PromotionCorpus,
    bootstrap_samples: int,
    bootstrap_seed: int,
    workers: int,
    batch_size: int,
    analysis: PromotionAnalysis,
) -> PromotionReport:
    """Collect immutable inputs and analysis into the authoritative report model."""

    return PromotionReport(
        schema_version=1,
        repository_commit=repository_commit,
        candidate=candidate,
        incumbent=incumbent,
        opponents=opponents,
        opponent_pool=opponent_pool,
        plan=plan,
        development=development,
        held_out=held_out,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        workers=workers,
        batch_size=batch_size,
        analysis=analysis,
        artifact_names=_ARTIFACT_NAMES,
    )


def write_promotion_artifacts(
    output_dir: Path,
    *,
    report: PromotionReport,
    game_summaries: Sequence[GameSummary],
    development: PromotionCorpus,
    held_out: PromotionCorpus,
    overwrite: bool = False,
) -> PromotionArtifacts:
    """Write one rollback-protected generation of promotion artifacts."""

    validate_artifact_output_dir(output_dir, overwrite=overwrite)
    _require_report_corpora(
        report,
        development=development,
        held_out=held_out,
    )
    rendered_artifacts = (
        (
            _PROMOTION_REPORT_NAME,
            _render_json_document(promotion_report_payload(report)),
        ),
        (
            _PAIRED_GAMES_NAME,
            _render_game_summaries(game_summaries),
        ),
        (
            _CORPUS_SNAPSHOT_NAME,
            _render_json_document(
                {
                    "schema_version": 1,
                    "development": corpus_snapshot_payload(report.development),
                    "held_out": corpus_snapshot_payload(report.held_out),
                }
            ),
        ),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    validate_artifact_output_dir(output_dir, overwrite=overwrite)
    prepared_artifacts = _prepare_artifact_generation(
        output_dir,
        rendered_artifacts,
    )
    _replace_artifact_generation(prepared_artifacts)

    return PromotionArtifacts(
        report_json=output_dir / _PROMOTION_REPORT_NAME,
        paired_games_jsonl=output_dir / _PAIRED_GAMES_NAME,
        corpus_snapshot_json=output_dir / _CORPUS_SNAPSHOT_NAME,
    )


def promotion_report_payload(report: PromotionReport) -> dict[str, object]:
    """Convert a report to its explicit public JSON schema."""

    analysis = report.analysis
    interval_payload: dict[str, object] | None = None
    if analysis.interval is not None:
        interval_payload = {
            "lower": analysis.interval.lower,
            "upper": analysis.interval.upper,
        }

    faults_by_identity = [
        {"bot_id": bot_id, "count": count} for bot_id, count in analysis.faults_by_identity
    ]
    total_faults = analysis.unattributed_faults + sum(
        count for _, count in analysis.faults_by_identity
    )
    return {
        "schema_version": report.schema_version,
        "repository_commit": report.repository_commit,
        "candidate": _bot_identity_payload(report.candidate),
        "incumbent": _bot_identity_payload(report.incumbent),
        "opponents": [_bot_identity_payload(opponent) for opponent in report.opponents],
        "opponent_pool": (
            None
            if report.opponent_pool is None
            else effective_opponent_pool_payload(report.opponent_pool)
        ),
        "effective_plan": (None if report.plan is None else promotion_plan_payload(report.plan)),
        "execution": {
            "bot_ids": [
                report.candidate.bot_id,
                report.incumbent.bot_id,
                *(opponent.bot_id for opponent in report.opponents),
            ],
            "games": 2 * len(report.held_out.cases),
            "player_counts": list(report.held_out.recipe.player_counts),
            "value_charts": list(report.held_out.recipe.charts),
            "root_seed": report.held_out.recipe.root_seed,
            "objectives_enabled": [True],
            "fault_mode": FaultMode.RECORD_AND_PASS.value,
            "capture_replays": False,
            "workers": report.workers,
            "batch_size": report.batch_size,
        },
        "corpora": {
            "development": _corpus_report_payload(report.development),
            "held_out": _corpus_report_payload(report.held_out),
        },
        "coverage": {
            "requested_pairs": analysis.requested_pairs,
            "completed_pairs": analysis.completed_pairs,
            "requested_games": analysis.requested_games,
            "completed_games": analysis.completed_games,
        },
        "rating_difference": analysis.rating_difference,
        "confidence_interval_95": interval_payload,
        "bootstrap": {
            "requested": report.bootstrap_samples,
            "converged": analysis.bootstrap_converged,
            "seed": report.bootstrap_seed,
        },
        "faults": {
            "total": total_faults,
            "unattributed": analysis.unattributed_faults,
            "by_identity": faults_by_identity,
        },
        "warnings": list(analysis.warnings),
        "failures": [
            {"code": failure.code, "message": failure.message} for failure in analysis.failures
        ],
        "promoted": analysis.promoted,
        "artifacts": list(report.artifact_names),
    }


def _bot_identity_payload(bot: BotSpec) -> dict[str, object]:
    return {"name": bot.name, "bot_id": bot.bot_id}


def _corpus_report_payload(corpus: PromotionCorpus) -> dict[str, object]:
    return {
        "name": corpus.recipe.name,
        "digest": corpus.digest,
        "purpose": corpus.recipe.purpose,
        "root_seed": corpus.recipe.root_seed,
        "repetitions_per_seat_cell": corpus.recipe.repetitions_per_seat_cell,
        "charts": list(corpus.recipe.charts),
        "player_counts": list(corpus.recipe.player_counts),
        "opponent_names": list(corpus.recipe.opponent_names),
        "engine_seeds": list(corpus.engine_seeds),
    }


def _require_report_corpora(
    report: PromotionReport,
    *,
    development: PromotionCorpus,
    held_out: PromotionCorpus,
) -> None:
    if development != report.development:
        raise ValueError(
            "The development corpus does not match the corpus recorded in the promotion report."
        )
    if held_out != report.held_out:
        raise ValueError(
            "The held-out corpus does not match the corpus recorded in the promotion report."
        )


def _render_game_summaries(game_summaries: Sequence[GameSummary]) -> str:
    ordered_summaries = sorted(game_summaries, key=lambda summary: summary.game_index)
    try:
        return "".join(
            json.dumps(
                _game_summary_payload(summary),
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
            for summary in ordered_summaries
        )
    except ValueError as error:
        raise ValueError("Promotion artifacts must contain only finite JSON numbers.") from error


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


def _render_json_document(payload: Mapping[str, object]) -> str:
    try:
        return json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    except ValueError as error:
        raise ValueError("Promotion artifacts must contain only finite JSON numbers.") from error


def validate_artifact_output_dir(
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Reject unsafe promotion artifact output paths before work begins."""

    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise NotADirectoryError(f"promotion output path is not a directory: {output_dir}")
    if not overwrite and any(output_dir.iterdir()):
        raise FileExistsError(f"promotion output directory is not empty: {output_dir}")


def _prepare_artifact_generation(
    output_dir: Path,
    rendered_artifacts: tuple[tuple[str, str], ...],
) -> tuple[_PreparedArtifact, ...]:
    prepared: list[_PreparedArtifact] = []
    try:
        for artifact_name, content in rendered_artifacts:
            target = output_dir / artifact_name
            staged = _stage_bytes(
                output_dir,
                prefix=f".{artifact_name}.staged.",
                content=content.encode("utf-8"),
            )
            prepared.append(
                _PreparedArtifact(
                    target=target,
                    staged=staged,
                    backup=None,
                )
            )

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
                "Promotion artifact replacement failed, and the previous artifact "
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
