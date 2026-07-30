"""Render deterministic, machine-readable evidence for a promotion decision."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from garboid_pocketrocks.bots import BotSpec
from garboid_pocketrocks.promotion.analysis import PromotionAnalysis
from garboid_pocketrocks.promotion.corpus import (
    PromotionCorpus,
    corpus_snapshot_payload,
)
from garboid_pocketrocks.simulator.monte_carlo import GameSummary
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


def build_promotion_report(
    *,
    repository_commit: str,
    candidate: BotSpec,
    incumbent: BotSpec,
    opponents: tuple[BotSpec, ...],
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
    """Write the report, sorted game summaries, and corpus snapshot atomically."""

    _validate_output_directory(output_dir, overwrite=overwrite)
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
                    "development": corpus_snapshot_payload(development),
                    "held_out": corpus_snapshot_payload(held_out),
                }
            ),
        ),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _validate_output_directory(output_dir, overwrite=overwrite)
    for artifact_name, content in rendered_artifacts:
        _atomic_write(output_dir / artifact_name, content)

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
    total_faults = sum(count for _, count in analysis.faults_by_identity)
    return {
        "schema_version": report.schema_version,
        "repository_commit": report.repository_commit,
        "candidate": _bot_identity_payload(report.candidate),
        "incumbent": _bot_identity_payload(report.incumbent),
        "opponents": [_bot_identity_payload(opponent) for opponent in report.opponents],
        "execution": {
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
        "engine_seeds": list(corpus.engine_seeds),
    }


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


def _validate_output_directory(output_dir: Path, *, overwrite: bool) -> None:
    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise NotADirectoryError(f"promotion output path is not a directory: {output_dir}")
    if not overwrite and any(output_dir.iterdir()):
        raise FileExistsError(f"promotion output directory is not empty: {output_dir}")


def _atomic_write(path: Path, content: str) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
