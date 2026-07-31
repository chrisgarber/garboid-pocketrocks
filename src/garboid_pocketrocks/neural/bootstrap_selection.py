"""Development-only selection among the five frozen neural bootstrap arms.

This module deliberately has no held-out corpus loader or held-out input.  It
freezes all completed training arms first, compares their immutable BotSpecs on
one development corpus, and records the evidence used to make that choice.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from garboid_pocketrocks.bots import BOT_SPECS_BY_NAME, BotSpec
from garboid_pocketrocks.evolution.evaluation import CandidateEvaluation, evaluate_candidate
from garboid_pocketrocks.evolution.planning import DevelopmentPlan, plan_development_games
from garboid_pocketrocks.neural.bootstrap_freeze import (
    FrozenBootstrapCandidate,
    freeze_bootstrap_candidate,
    load_frozen_bootstrap_candidate,
)
from garboid_pocketrocks.neural.heuristic_bootstrap import (
    HEURISTIC_BOOTSTRAP_ARMS,
    REFERENCE_NEURAL_IDENTITY,
    REFERENCE_PARAMETER_DIGEST,
    HeuristicBootstrapStrategy,
)
from garboid_pocketrocks.neural.tournament_bot import (
    LARGE_CHECKPOINT_PATH,
    VECTOR_PPO_LARGE_V1_G350K_BOT_SPEC,
    frozen_bootstrap_bot_spec,
)
from garboid_pocketrocks.promotion.corpus import (
    PromotionCase,
    PromotionCorpus,
    PromotionCorpusRecipe,
    corpus_snapshot_payload,
    recompute_promotion_corpus_digest,
)
from garboid_pocketrocks.simulator.monte_carlo import (
    GameSummary,
    MonteCarloResult,
    MonteCarloRunner,
)
from garboid_pocketrocks.simulator.session import SessionScore

DEVELOPMENT_CORPUS_NAME = "development-v1"
DEVELOPMENT_CORPUS_DIGEST = "17c016350dbe717641b8cd499b0908e3bc0faa811a3b4f5e574f8713a5bf2b3d"


class BootstrapSelectionError(ValueError):
    """Raised when development selection cannot produce trustworthy evidence."""


@dataclass(frozen=True, slots=True)
class BootstrapCheckpointInput:
    """One official strategy and its exact final training checkpoint."""

    strategy: HeuristicBootstrapStrategy
    training_checkpoint: Path


@dataclass(frozen=True, slots=True)
class FrozenBootstrapArm:
    """One verified immutable arm ready for development games."""

    strategy: HeuristicBootstrapStrategy
    path: Path
    candidate: FrozenBootstrapCandidate
    bot_spec: BotSpec
    freeze_manifest_digest: str


@dataclass(frozen=True, slots=True)
class BootstrapCandidateResult:
    """The planned games, raw result, and validated scores for one arm."""

    arm: FrozenBootstrapArm
    plan: DevelopmentPlan
    result: MonteCarloResult
    evaluation: CandidateEvaluation


@dataclass(frozen=True, slots=True)
class BootstrapSelectionArtifacts:
    """Transactional development-evidence files."""

    evidence_manifest_json: Path
    selection_decision_json: Path
    candidate_evaluations_jsonl: Path
    development_games_jsonl: Path
    development_corpus_snapshot_json: Path


@dataclass(frozen=True, slots=True)
class BootstrapSelectionResult:
    """Complete in-memory result of development-only bootstrap selection."""

    development_corpus: PromotionCorpus
    frozen_arms: tuple[FrozenBootstrapArm, ...]
    baseline_result: MonteCarloResult
    candidate_results: tuple[BootstrapCandidateResult, ...]
    ranked_candidate_identities: tuple[str, ...]
    selected_candidate_identity: str | None
    artifacts: BootstrapSelectionArtifacts


def run_bootstrap_development_selection(
    checkpoints: tuple[BootstrapCheckpointInput, ...] | list[BootstrapCheckpointInput],
    development_corpus: PromotionCorpus,
    *,
    bootstrap_summary_path: Path,
    frozen_arms_dir: Path,
    evidence_output_dir: Path,
    workers: int = 1,
    batch_size: int | None = None,
) -> BootstrapSelectionResult:
    """Freeze, compare, rank, and record the five official development arms.

    The only corpus accepted by this API is the already-loaded development
    corpus.  Selection uses rating delta first, normalized finish delta second,
    final-money delta third, and the immutable identity as the final tie-break.
    """

    ordered_inputs = _validate_inputs(
        checkpoints,
        development_corpus,
        bootstrap_summary_path=bootstrap_summary_path,
        frozen_arms_dir=frozen_arms_dir,
        evidence_output_dir=evidence_output_dir,
        workers=workers,
        batch_size=batch_size,
    )
    frozen_arms = _freeze_all_arms(
        ordered_inputs,
        development_corpus,
        bootstrap_summary_path=bootstrap_summary_path,
        destination=frozen_arms_dir,
    )

    baseline_result, candidate_results = _evaluate_frozen_arms(
        frozen_arms,
        development_corpus,
        workers=workers,
        batch_size=batch_size,
    )

    ranked = tuple(
        item.evaluation.candidate_identity
        for item in sorted(
            (item for item in candidate_results if _is_complete_fault_free(item.evaluation)),
            key=lambda item: _ranking_key(item.evaluation),
        )
    )
    evaluation_by_identity = {
        item.evaluation.candidate_identity: item.evaluation for item in candidate_results
    }
    selected = (
        ranked[0] if ranked and _is_strict_improvement(evaluation_by_identity[ranked[0]]) else None
    )
    artifacts = _write_evidence_artifacts(
        evidence_output_dir,
        development_corpus=development_corpus,
        frozen_arms=frozen_arms,
        baseline_result=baseline_result,
        candidate_results=candidate_results,
        ranked_identities=ranked,
        selected_identity=selected,
    )
    return BootstrapSelectionResult(
        development_corpus=development_corpus,
        frozen_arms=frozen_arms,
        baseline_result=baseline_result,
        candidate_results=candidate_results,
        ranked_candidate_identities=ranked,
        selected_candidate_identity=selected,
        artifacts=artifacts,
    )


def selected_bootstrap_bot_spec(
    evidence_output_dir: Path,
    *,
    frozen_arms_dir: Path,
) -> BotSpec:
    """Load only the exact freeze chosen by recorded development evidence.

    A later held-out runner can consume this BotSpec, but it cannot provide a
    strategy or candidate identity manually.  The recorded freeze-manifest
    digest and identity must still match the immutable directory.
    """

    evidence = _load_verified_evidence(evidence_output_dir)
    decision = evidence["decision"]
    manifest = evidence["manifest"]
    frozen_arms = _load_evidence_frozen_arms(
        decision,
        manifest=manifest,
        frozen_arms_dir=frozen_arms_dir,
    )
    corpus = _pinned_development_corpus(evidence["snapshot"])
    selected = _recompute_evidence_selection(
        evidence_output_dir,
        corpus=corpus,
        frozen_arms=frozen_arms,
        decision=decision,
    )
    selected_arm = next(
        (arm for arm in frozen_arms if arm.candidate.manifest.identity == selected),
        None,
    )
    if selected_arm is None:
        raise BootstrapSelectionError("recomputed winner has no frozen arm")
    return selected_arm.bot_spec


def _load_verified_evidence(output_dir: Path) -> dict[str, dict[str, object]]:
    """Verify the complete transactional evidence generation before selection."""

    expected_artifacts = {
        "evidence-manifest.json",
        "selection-decision.json",
        "candidate-evaluations.jsonl",
        "development-games.jsonl",
        "development-corpus-snapshot.json",
    }
    if (
        not output_dir.is_dir()
        or output_dir.is_symlink()
        or {path.name for path in output_dir.iterdir()} != expected_artifacts
        or any(path.is_symlink() for path in output_dir.iterdir())
    ):
        raise BootstrapSelectionError("development evidence artifact set is not exact")
    manifest = _read_json_object(output_dir / "evidence-manifest.json", "evidence manifest")
    expected_manifest_fields = {
        "schema_version",
        "purpose",
        "held_out_loaded",
        "development_corpus_name",
        "development_corpus_digest",
        "files",
    }
    if set(manifest) != expected_manifest_fields or (
        manifest["schema_version"] != 1
        or manifest["purpose"] != "development_bootstrap_evidence"
        or manifest["held_out_loaded"] is not False
    ):
        raise BootstrapSelectionError("development evidence manifest is not exact schema 1")
    corpus_name = manifest["development_corpus_name"]
    corpus_digest = manifest["development_corpus_digest"]
    if corpus_name != DEVELOPMENT_CORPUS_NAME or corpus_digest != DEVELOPMENT_CORPUS_DIGEST:
        raise BootstrapSelectionError("evidence does not bind committed development-v1")
    records = manifest["files"]
    if not isinstance(records, list):
        raise BootstrapSelectionError("development evidence file bindings are malformed")
    expected_bound_files = expected_artifacts - {"evidence-manifest.json"}
    bindings: dict[str, str] = {}
    for value in records:
        if not isinstance(value, dict) or set(value) != {"name", "sha256"}:
            raise BootstrapSelectionError("development evidence file binding is malformed")
        name = value["name"]
        digest = value["sha256"]
        if not isinstance(name, str) or not _is_sha256(digest) or name in bindings:
            raise BootstrapSelectionError("development evidence file binding is invalid")
        bindings[name] = digest
    if set(bindings) != expected_bound_files:
        raise BootstrapSelectionError("development evidence file bindings are not exact")
    for name, expected_digest in bindings.items():
        if _file_digest(output_dir / name) != expected_digest:
            raise BootstrapSelectionError(f"development evidence artifact digest changed: {name}")

    decision = _read_json_object(output_dir / "selection-decision.json", "selection decision")
    expected_decision_fields = {
        "schema_version",
        "purpose",
        "incumbent_identity",
        "ranking_fields",
        "ranked_candidate_identities",
        "selected_candidate_identity",
        "selection_rule",
        "frozen_arms",
    }
    if set(decision) != expected_decision_fields or (
        decision["schema_version"] != 1
        or decision["purpose"] != "development_bootstrap_selection"
        or decision["incumbent_identity"] != REFERENCE_NEURAL_IDENTITY
    ):
        raise BootstrapSelectionError("selection decision is not exact schema 1")
    snapshot = _read_json_object(
        output_dir / "development-corpus-snapshot.json",
        "development corpus snapshot",
    )
    recipe = snapshot.get("recipe")
    if (
        snapshot.get("digest") != corpus_digest
        or not isinstance(recipe, dict)
        or recipe.get("name") != corpus_name
        or recipe.get("purpose") != "development"
    ):
        raise BootstrapSelectionError("development corpus snapshot does not match manifest")
    _require_pinned_incumbent_checkpoint()
    return {"manifest": manifest, "decision": decision, "snapshot": snapshot}


def _pinned_development_corpus(snapshot: dict[str, object]) -> PromotionCorpus:
    if set(snapshot) != {"recipe", "cases", "digest"}:
        raise BootstrapSelectionError("development corpus snapshot fields are not exact")
    recipe_value = snapshot["recipe"]
    cases_value = snapshot["cases"]
    if not isinstance(recipe_value, dict) or not isinstance(cases_value, list):
        raise BootstrapSelectionError("development corpus snapshot is malformed")
    expected_recipe_fields = {
        "schema_version",
        "name",
        "purpose",
        "root_seed",
        "repetitions_per_seat_cell",
        "charts",
        "player_counts",
        "opponent_names",
    }
    if set(recipe_value) != expected_recipe_fields:
        raise BootstrapSelectionError("development recipe fields are not exact")
    if (
        recipe_value["schema_version"] != 1
        or recipe_value["name"] != DEVELOPMENT_CORPUS_NAME
        or recipe_value["purpose"] != "development"
    ):
        raise BootstrapSelectionError("development recipe identity is not pinned")
    charts = _string_array(recipe_value["charts"], "development charts")
    player_counts = _integer_array(recipe_value["player_counts"], "development players")
    opponents = _string_array(recipe_value["opponent_names"], "development opponents")
    recipe = PromotionCorpusRecipe(
        schema_version=1,
        name=DEVELOPMENT_CORPUS_NAME,
        purpose="development",
        root_seed=_json_integer(recipe_value["root_seed"], "development root seed", minimum=0),
        repetitions_per_seat_cell=_json_integer(
            recipe_value["repetitions_per_seat_cell"],
            "development repetitions",
            minimum=1,
        ),
        charts=charts,
        player_counts=player_counts,
        opponent_names=opponents,
    )
    cases: list[PromotionCase] = []
    expected_case_fields = {
        "case_id",
        "chart",
        "player_count",
        "focal_seat",
        "engine_seed",
        "opponent_names_by_seat",
    }
    for value in cases_value:
        if not isinstance(value, dict) or set(value) != expected_case_fields:
            raise BootstrapSelectionError("development case fields are not exact")
        player_count = _json_integer(value["player_count"], "case player count", minimum=3)
        lineup_value = value["opponent_names_by_seat"]
        if not isinstance(lineup_value, list) or len(lineup_value) != player_count:
            raise BootstrapSelectionError("development case lineup is incomplete")
        lineup: list[str | None] = []
        for name in lineup_value:
            if name is not None and (not isinstance(name, str) or not name):
                raise BootstrapSelectionError("development case opponent is invalid")
            lineup.append(name)
        cases.append(
            PromotionCase(
                case_id=_json_string(value["case_id"], "case id"),
                chart=_json_string(value["chart"], "case chart"),
                player_count=player_count,
                focal_seat=_json_integer(value["focal_seat"], "case focal seat", minimum=0),
                engine_seed=_json_integer(value["engine_seed"], "case engine seed", minimum=0),
                opponent_names_by_seat=tuple(lineup),
            )
        )
    corpus = PromotionCorpus(recipe=recipe, cases=tuple(cases), digest=DEVELOPMENT_CORPUS_DIGEST)
    if len(cases) != 240 or recompute_promotion_corpus_digest(corpus) != DEVELOPMENT_CORPUS_DIGEST:
        raise BootstrapSelectionError("development-v1 snapshot does not match its pinned digest")
    return corpus


def _load_evidence_frozen_arms(
    decision: dict[str, object],
    *,
    manifest: dict[str, object],
    frozen_arms_dir: Path,
) -> tuple[FrozenBootstrapArm, ...]:
    values = decision.get("frozen_arms")
    if not isinstance(values, list) or len(values) != len(HEURISTIC_BOOTSTRAP_ARMS):
        raise BootstrapSelectionError("selection decision must bind all five frozen arms")
    bindings: dict[str, dict[str, object]] = {}
    for value in values:
        if not isinstance(value, dict) or set(value) != {
            "strategy",
            "identity",
            "freeze_manifest_digest",
            "summary_digest",
        }:
            raise BootstrapSelectionError("frozen-arm binding fields are not exact")
        strategy = value["strategy"]
        if not isinstance(strategy, str) or strategy in bindings:
            raise BootstrapSelectionError("frozen-arm strategies are invalid or duplicated")
        bindings[strategy] = dict(value)
    expected_strategies = {arm.strategy for arm in HEURISTIC_BOOTSTRAP_ARMS}
    if set(bindings) != expected_strategies or tuple(
        cast(str, value["strategy"]) for value in values
    ) != tuple(arm.strategy for arm in HEURISTIC_BOOTSTRAP_ARMS):
        raise BootstrapSelectionError("selection decision does not bind the official five arms")

    frozen_root = frozen_arms_dir.resolve()
    if (
        not frozen_root.is_dir()
        or frozen_arms_dir.is_symlink()
        or {path.name for path in frozen_root.iterdir()} != expected_strategies
        or any(path.is_symlink() for path in frozen_root.iterdir())
    ):
        raise BootstrapSelectionError("frozen-arm directory set is not exact")
    arms: list[FrozenBootstrapArm] = []
    identities: set[str] = set()
    for contract in HEURISTIC_BOOTSTRAP_ARMS:
        binding = bindings[contract.strategy]
        identity = binding["identity"]
        freeze_digest = binding["freeze_manifest_digest"]
        summary_digest = binding["summary_digest"]
        if (
            not isinstance(identity, str)
            or not _is_sha256(freeze_digest)
            or not _is_sha256(summary_digest)
            or identity in identities
        ):
            raise BootstrapSelectionError("frozen-arm binding values are invalid")
        candidate_path = (frozen_root / contract.strategy).resolve()
        if candidate_path.parent != frozen_root or any(
            path.is_symlink() for path in candidate_path.rglob("*")
        ):
            raise BootstrapSelectionError("frozen arm escaped its official directory")
        if _file_digest(candidate_path / "manifest.json") != freeze_digest:
            raise BootstrapSelectionError("frozen arm digest does not match decision")
        candidate = load_frozen_bootstrap_candidate(candidate_path)
        if (
            candidate.manifest.strategy != contract.strategy
            or candidate.manifest.identity != identity
            or candidate.manifest.summary_digest != summary_digest
            or candidate.manifest.development_corpus_name != manifest["development_corpus_name"]
            or candidate.manifest.development_corpus_digest != manifest["development_corpus_digest"]
        ):
            raise BootstrapSelectionError("frozen arm provenance does not match evidence")
        identities.add(identity)
        arms.append(
            FrozenBootstrapArm(
                strategy=contract.strategy,
                path=candidate_path,
                candidate=candidate,
                bot_spec=frozen_bootstrap_bot_spec(candidate_path),
                freeze_manifest_digest=freeze_digest,
            )
        )
    if (
        len({arm.candidate.manifest.source_commit for arm in arms}) != 1
        or len({arm.candidate.manifest.summary_digest for arm in arms}) != 1
    ):
        raise BootstrapSelectionError("frozen arms do not share source and training summary")
    return tuple(arms)


def _recompute_evidence_selection(
    output_dir: Path,
    *,
    corpus: PromotionCorpus,
    frozen_arms: tuple[FrozenBootstrapArm, ...],
    decision: dict[str, object],
) -> str:
    evaluation_rows = _read_json_lines(
        output_dir / "candidate-evaluations.jsonl",
        "candidate evaluations",
    )
    if len(evaluation_rows) != len(HEURISTIC_BOOTSTRAP_ARMS):
        raise BootstrapSelectionError("candidate evidence must contain all five arms")
    stored_by_identity: dict[str, dict[str, object]] = {}
    for row in evaluation_rows:
        identity = row.get("candidate_identity")
        if not isinstance(identity, str) or identity in stored_by_identity:
            raise BootstrapSelectionError("candidate evaluation identities are invalid")
        stored_by_identity[identity] = row

    baseline_summaries: list[GameSummary] = []
    candidate_summaries: dict[str, list[GameSummary]] = {
        arm.bot_spec.name: [] for arm in frozen_arms
    }
    game_rows = _read_json_lines(output_dir / "development-games.jsonl", "development games")
    expected_games = len(corpus.cases) * (len(frozen_arms) + 1)
    if len(game_rows) != expected_games:
        raise BootstrapSelectionError("development games do not have exact five-arm coverage")
    canonical_game_payloads: list[dict[str, object]] = []
    for row_index, row in enumerate(game_rows):
        group_index, expected_game_index = divmod(row_index, len(corpus.cases))
        evidence_kind = row.get("evidence")
        candidate_identity = row.get("candidate_identity")
        summary = _game_summary_from_payload(row)
        if summary.game_index != expected_game_index:
            raise BootstrapSelectionError("development game rows are not in canonical order")
        if group_index == 0 and evidence_kind == "baseline" and candidate_identity is None:
            baseline_summaries.append(summary)
        elif group_index > 0 and (
            evidence_kind == "candidate"
            and isinstance(candidate_identity, str)
            and candidate_identity == frozen_arms[group_index - 1].bot_spec.name
        ):
            candidate_summaries[candidate_identity].append(summary)
        else:
            raise BootstrapSelectionError("development game has an unknown evidence owner")
        canonical_game_payloads.append(
            _game_payload(
                summary,
                evidence=evidence_kind,
                candidate_identity=candidate_identity,
            )
        )
    if (output_dir / "development-games.jsonl").read_text(encoding="utf-8") != _json_lines(
        canonical_game_payloads
    ):
        raise BootstrapSelectionError("development games are not canonical raw evidence")

    baseline_result = MonteCarloResult(tuple(baseline_summaries), (), ())
    recomputed: list[BootstrapCandidateResult] = []
    first_plan: DevelopmentPlan | None = None
    for arm in frozen_arms:
        plan = plan_development_games(
            corpus,
            candidate=arm.bot_spec,
            incumbent=VECTOR_PPO_LARGE_V1_G350K_BOT_SPEC,
            registry=BOT_SPECS_BY_NAME,
        )
        if first_plan is None:
            first_plan = plan
        elif (
            plan.baseline_config != first_plan.baseline_config
            or plan.baseline_jobs != first_plan.baseline_jobs
        ):
            raise BootstrapSelectionError("recomputed baselines are not identical")
        candidate_result = MonteCarloResult(
            tuple(candidate_summaries[arm.bot_spec.name]),
            (),
            (),
        )
        result = BootstrapCandidateResult(
            arm=arm,
            plan=plan,
            result=candidate_result,
            evaluation=evaluate_candidate(plan, baseline_result, candidate_result),
        )
        stored = stored_by_identity.get(arm.bot_spec.name)
        if stored is None or stored != _evaluation_payload(result):
            raise BootstrapSelectionError("stored candidate evaluation does not recompute")
        recomputed.append(result)
    if set(stored_by_identity) != {arm.bot_spec.name for arm in frozen_arms}:
        raise BootstrapSelectionError("candidate evaluations do not match frozen arms")
    canonical_evaluations = _json_lines(_evaluation_payload(item) for item in recomputed)
    if (output_dir / "candidate-evaluations.jsonl").read_text(
        encoding="utf-8"
    ) != canonical_evaluations:
        raise BootstrapSelectionError("candidate evaluations are not canonical evidence")

    ranked = tuple(
        item.evaluation.candidate_identity
        for item in sorted(
            (item for item in recomputed if _is_complete_fault_free(item.evaluation)),
            key=lambda item: _ranking_key(item.evaluation),
        )
    )
    evaluation_by_identity = {
        item.evaluation.candidate_identity: item.evaluation for item in recomputed
    }
    selected = (
        ranked[0] if ranked and _is_strict_improvement(evaluation_by_identity[ranked[0]]) else None
    )
    if selected is None:
        raise BootstrapSelectionError("recomputed evidence has no strict development winner")
    if (
        decision.get("ranking_fields")
        != [
            "rating_delta_descending",
            "normalized_finish_delta_descending",
            "final_money_delta_descending",
            "candidate_identity_ascending",
        ]
        or decision.get("selection_rule")
        != "complete fault-free winner with strictly positive rating delta"
        or decision.get("ranked_candidate_identities") != list(ranked)
        or decision.get("selected_candidate_identity") != selected
    ):
        raise BootstrapSelectionError("selection decision does not match recomputed evidence")
    if (output_dir / "selection-decision.json").read_text(encoding="utf-8") != _json_document(
        decision
    ):
        raise BootstrapSelectionError("selection decision is not canonical evidence")
    return selected


def _validate_inputs(
    checkpoints: tuple[BootstrapCheckpointInput, ...] | list[BootstrapCheckpointInput],
    development_corpus: PromotionCorpus,
    *,
    bootstrap_summary_path: Path,
    frozen_arms_dir: Path,
    evidence_output_dir: Path,
    workers: int,
    batch_size: int | None,
) -> tuple[BootstrapCheckpointInput, ...]:
    if workers < 1:
        raise BootstrapSelectionError("workers must be positive")
    if batch_size is not None and batch_size < 1:
        raise BootstrapSelectionError("batch size must be positive")
    if development_corpus.recipe.purpose != "development":
        raise BootstrapSelectionError("bootstrap selection accepts only a development corpus")
    if (
        not development_corpus.cases
        or recompute_promotion_corpus_digest(development_corpus) != development_corpus.digest
        or development_corpus.recipe.name != DEVELOPMENT_CORPUS_NAME
        or development_corpus.digest != DEVELOPMENT_CORPUS_DIGEST
    ):
        raise BootstrapSelectionError("selection requires the exact committed development-v1")
    if not bootstrap_summary_path.is_file():
        raise BootstrapSelectionError("complete bootstrap summary must be a file")
    _require_safe_paths(
        checkpoints,
        bootstrap_summary_path=bootstrap_summary_path,
        frozen_arms_dir=frozen_arms_dir,
        evidence_output_dir=evidence_output_dir,
    )
    _require_empty_directory_target(frozen_arms_dir, "frozen arms")
    _require_empty_directory_target(evidence_output_dir, "evidence output")

    canonical_incumbent = BOT_SPECS_BY_NAME.get(REFERENCE_NEURAL_IDENTITY)
    if canonical_incumbent is not VECTOR_PPO_LARGE_V1_G350K_BOT_SPEC:
        raise BootstrapSelectionError("canonical large neural incumbent is not registered")
    _require_pinned_incumbent_checkpoint()
    for opponent_name in development_corpus.recipe.opponent_names:
        opponent = BOT_SPECS_BY_NAME.get(opponent_name)
        if opponent is None or opponent.name != opponent_name:
            raise BootstrapSelectionError(
                f"development opponent {opponent_name!r} is not canonical"
            )

    required = {arm.strategy for arm in HEURISTIC_BOOTSTRAP_ARMS}
    supplied = tuple(item.strategy for item in checkpoints)
    if len(supplied) != len(required) or set(supplied) != required:
        raise BootstrapSelectionError(
            "selection requires each of the five official arms exactly once"
        )
    if any(not item.training_checkpoint.is_dir() for item in checkpoints):
        raise BootstrapSelectionError("every training checkpoint must be a directory")
    by_strategy = {item.strategy: item for item in checkpoints}
    return tuple(by_strategy[arm.strategy] for arm in HEURISTIC_BOOTSTRAP_ARMS)


def _require_pinned_incumbent_checkpoint() -> None:
    import torch

    from garboid_pocketrocks.neural.checkpoint import load_inference_checkpoint

    try:
        incumbent = load_inference_checkpoint(
            LARGE_CHECKPOINT_PATH,
            device=torch.device("cpu"),
        )
    except (OSError, ValueError) as error:
        raise BootstrapSelectionError("canonical incumbent checkpoint is invalid") from error
    if incumbent.manifest.parameter_digest != REFERENCE_PARAMETER_DIGEST:
        raise BootstrapSelectionError("canonical incumbent parameter digest has changed")


def _freeze_all_arms(
    inputs: tuple[BootstrapCheckpointInput, ...],
    development_corpus: PromotionCorpus,
    *,
    bootstrap_summary_path: Path,
    destination: Path,
) -> tuple[FrozenBootstrapArm, ...]:
    parent = destination.resolve().parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    try:
        temporary.mkdir()
        for item in inputs:
            freeze_bootstrap_candidate(
                item.training_checkpoint,
                temporary / item.strategy,
                development_corpus_name=development_corpus.recipe.name,
                development_corpus_digest=development_corpus.digest,
                bootstrap_summary_path=bootstrap_summary_path,
            )
        staged_candidates = tuple(
            load_frozen_bootstrap_candidate(temporary / item.strategy) for item in inputs
        )
        if any(
            candidate.manifest.strategy != item.strategy
            for item, candidate in zip(inputs, staged_candidates, strict=True)
        ):
            raise BootstrapSelectionError("frozen arm strategy changed during staging")
        if len({candidate.manifest.source_commit for candidate in staged_candidates}) != 1:
            raise BootstrapSelectionError("all official arms must come from one source commit")
        if len({candidate.manifest.summary_digest for candidate in staged_candidates}) != 1:
            raise BootstrapSelectionError("all official arms must bind one bootstrap summary")
        if len({candidate.manifest.identity for candidate in staged_candidates}) != len(inputs):
            raise BootstrapSelectionError("official frozen arm identities must be unique")
        _install_directory(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    arms: list[FrozenBootstrapArm] = []
    for item in inputs:
        path = (destination / item.strategy).resolve()
        candidate = load_frozen_bootstrap_candidate(path)
        arms.append(
            FrozenBootstrapArm(
                strategy=item.strategy,
                path=path,
                candidate=candidate,
                bot_spec=frozen_bootstrap_bot_spec(path),
                freeze_manifest_digest=_file_digest(path / "manifest.json"),
            )
        )
    return tuple(arms)


def _evaluate_frozen_arms(
    frozen_arms: tuple[FrozenBootstrapArm, ...],
    development_corpus: PromotionCorpus,
    *,
    workers: int,
    batch_size: int | None,
) -> tuple[MonteCarloResult, tuple[BootstrapCandidateResult, ...]]:
    """Run one reusable baseline and one matched result for every frozen arm."""

    if not frozen_arms:
        raise BootstrapSelectionError("at least one frozen arm is required")
    first_plan = plan_development_games(
        development_corpus,
        candidate=frozen_arms[0].bot_spec,
        incumbent=VECTOR_PPO_LARGE_V1_G350K_BOT_SPEC,
        registry=BOT_SPECS_BY_NAME,
    )
    baseline_result = MonteCarloRunner.run_jobs(
        first_plan.baseline_config,
        first_plan.baseline_jobs,
        workers=workers,
        batch_size=batch_size,
    )

    candidate_results: list[BootstrapCandidateResult] = []
    for arm in frozen_arms:
        plan = plan_development_games(
            development_corpus,
            candidate=arm.bot_spec,
            incumbent=VECTOR_PPO_LARGE_V1_G350K_BOT_SPEC,
            registry=BOT_SPECS_BY_NAME,
        )
        if (
            plan.baseline_config != first_plan.baseline_config
            or plan.baseline_jobs != first_plan.baseline_jobs
        ):
            raise BootstrapSelectionError(
                "all bootstrap arms must reuse the exact same development baseline"
            )
        candidate_result = MonteCarloRunner.run_jobs(
            plan.candidate_config,
            plan.candidate_jobs,
            workers=workers,
            batch_size=batch_size,
        )
        candidate_results.append(
            BootstrapCandidateResult(
                arm=arm,
                plan=plan,
                result=candidate_result,
                evaluation=evaluate_candidate(plan, baseline_result, candidate_result),
            )
        )
    return baseline_result, tuple(candidate_results)


def _ranking_key(evaluation: CandidateEvaluation) -> tuple[float, float, int, str]:
    if not _is_complete_fault_free(evaluation):
        raise BootstrapSelectionError("only complete fault-free candidates can be ranked")
    assert evaluation.rating_delta is not None
    assert evaluation.normalized_finish_delta is not None
    assert evaluation.final_money_delta is not None
    return (
        -evaluation.rating_delta,
        -evaluation.normalized_finish_delta,
        -evaluation.final_money_delta,
        evaluation.candidate_identity,
    )


def _is_complete_fault_free(evaluation: CandidateEvaluation) -> bool:
    numeric_scores = (evaluation.rating_delta, evaluation.normalized_finish_delta)
    return (
        evaluation.valid
        and evaluation.eligible
        and evaluation.requested_cases > 0
        and evaluation.completed_baseline_games == evaluation.requested_cases
        and evaluation.completed_candidate_games == evaluation.requested_cases
        and all(value is not None and math.isfinite(value) for value in numeric_scores)
        and evaluation.final_money_delta is not None
        and evaluation.candidate_faults == 0
        and evaluation.incumbent_faults == 0
        and evaluation.opponent_faults == 0
        and evaluation.unattributed_faults == 0
        and all(count == 0 for _identity, count in evaluation.faults_by_identity)
    )


def _is_strict_improvement(evaluation: CandidateEvaluation) -> bool:
    return (
        _is_complete_fault_free(evaluation)
        and evaluation.rating_delta is not None
        and evaluation.rating_delta > 0.0
    )


def _write_evidence_artifacts(
    output_dir: Path,
    *,
    development_corpus: PromotionCorpus,
    frozen_arms: tuple[FrozenBootstrapArm, ...],
    baseline_result: MonteCarloResult,
    candidate_results: tuple[BootstrapCandidateResult, ...],
    ranked_identities: tuple[str, ...],
    selected_identity: str | None,
) -> BootstrapSelectionArtifacts:
    evaluations = _json_lines(_evaluation_payload(item) for item in candidate_results)
    games = _json_lines(
        [
            *(
                _game_payload(summary, evidence="baseline", candidate_identity=None)
                for summary in baseline_result.game_summaries
            ),
            *(
                _game_payload(
                    summary,
                    evidence="candidate",
                    candidate_identity=item.evaluation.candidate_identity,
                )
                for item in candidate_results
                for summary in item.result.game_summaries
            ),
        ]
    )
    corpus = _json_document(corpus_snapshot_payload(development_corpus))
    decision = _json_document(
        {
            "schema_version": 1,
            "purpose": "development_bootstrap_selection",
            "incumbent_identity": REFERENCE_NEURAL_IDENTITY,
            "ranking_fields": [
                "rating_delta_descending",
                "normalized_finish_delta_descending",
                "final_money_delta_descending",
                "candidate_identity_ascending",
            ],
            "ranked_candidate_identities": list(ranked_identities),
            "selected_candidate_identity": selected_identity,
            "selection_rule": "complete fault-free winner with strictly positive rating delta",
            "frozen_arms": [
                {
                    "strategy": arm.strategy,
                    "identity": arm.candidate.manifest.identity,
                    "freeze_manifest_digest": arm.freeze_manifest_digest,
                    "summary_digest": arm.candidate.manifest.summary_digest,
                }
                for arm in frozen_arms
            ],
        }
    )
    rendered = {
        "candidate-evaluations.jsonl": evaluations,
        "development-games.jsonl": games,
        "development-corpus-snapshot.json": corpus,
        "selection-decision.json": decision,
    }
    manifest = _json_document(
        {
            "schema_version": 1,
            "purpose": "development_bootstrap_evidence",
            "held_out_loaded": False,
            "development_corpus_name": development_corpus.recipe.name,
            "development_corpus_digest": development_corpus.digest,
            "files": [
                {"name": name, "sha256": _text_digest(text)}
                for name, text in sorted(rendered.items())
            ],
        }
    )
    rendered["evidence-manifest.json"] = manifest
    _write_directory_transaction(output_dir, rendered)
    return BootstrapSelectionArtifacts(
        evidence_manifest_json=output_dir / "evidence-manifest.json",
        selection_decision_json=output_dir / "selection-decision.json",
        candidate_evaluations_jsonl=output_dir / "candidate-evaluations.jsonl",
        development_games_jsonl=output_dir / "development-games.jsonl",
        development_corpus_snapshot_json=output_dir / "development-corpus-snapshot.json",
    )


def _evaluation_payload(result: BootstrapCandidateResult) -> dict[str, object]:
    evaluation = result.evaluation
    return {
        "strategy": result.arm.strategy,
        "candidate_identity": evaluation.candidate_identity,
        "freeze_manifest_digest": result.arm.freeze_manifest_digest,
        "coverage": {
            "requested_cases": evaluation.requested_cases,
            "completed_baseline_games": evaluation.completed_baseline_games,
            "completed_candidate_games": evaluation.completed_candidate_games,
        },
        "scores": {
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
                {"identity": identity, "count": count}
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
        "complete_fault_free": _is_complete_fault_free(evaluation),
    }


def _game_payload(
    summary: GameSummary,
    *,
    evidence: str,
    candidate_identity: str | None,
) -> dict[str, object]:
    return {
        "evidence": evidence,
        "candidate_identity": candidate_identity,
        "game_index": summary.game_index,
        "root_seed": summary.root_seed,
        "engine_seed": summary.seed,
        "player_count": summary.player_count,
        "ruleset_name": summary.ruleset_name,
        "bot_names_by_seat": list(summary.bot_names),
        "bot_ids_by_seat": list(summary.bot_ids),
        "utilities_by_seat": [
            {
                "seat": score.seat,
                "rank": score.rank,
                "normalized_finish": (summary.player_count - score.rank)
                / (summary.player_count - 1),
                "final_money": score.final_money,
            }
            for score in summary.scores
        ],
        "decision_counts_by_seat": list(summary.decision_counts),
        "fault_counts_by_seat": list(summary.fault_counts),
    }


def _game_summary_from_payload(payload: dict[str, object]) -> GameSummary:
    expected_fields = {
        "evidence",
        "candidate_identity",
        "game_index",
        "root_seed",
        "engine_seed",
        "player_count",
        "ruleset_name",
        "bot_names_by_seat",
        "bot_ids_by_seat",
        "utilities_by_seat",
        "decision_counts_by_seat",
        "fault_counts_by_seat",
    }
    if set(payload) != expected_fields:
        raise BootstrapSelectionError("development game fields are not exact")
    game_index = _json_integer(payload["game_index"], "game index", minimum=0)
    root_seed = _json_integer(payload["root_seed"], "root seed", minimum=0)
    engine_seed = _json_integer(payload["engine_seed"], "engine seed", minimum=0)
    player_count = _json_integer(payload["player_count"], "player count", minimum=3)
    if player_count > 5:
        raise BootstrapSelectionError("development player count is invalid")
    ruleset_name = _json_string(payload["ruleset_name"], "ruleset name")
    names = _string_tuple(payload["bot_names_by_seat"], "bot names", player_count)
    identities = _string_tuple(payload["bot_ids_by_seat"], "bot identities", player_count)
    decision_counts = _integer_tuple(
        payload["decision_counts_by_seat"],
        "decision counts",
        player_count,
    )
    fault_counts = _integer_tuple(
        payload["fault_counts_by_seat"],
        "fault counts",
        player_count,
    )
    raw_utilities = payload["utilities_by_seat"]
    if not isinstance(raw_utilities, list) or len(raw_utilities) != player_count:
        raise BootstrapSelectionError("development utilities are incomplete")
    scores: list[SessionScore] = []
    for expected_seat, value in enumerate(raw_utilities):
        if not isinstance(value, dict) or set(value) != {
            "seat",
            "rank",
            "normalized_finish",
            "final_money",
        }:
            raise BootstrapSelectionError("development utility fields are not exact")
        seat = _json_integer(value["seat"], "utility seat", minimum=0)
        rank = _json_integer(value["rank"], "utility rank", minimum=1)
        money = _json_integer(value["final_money"], "utility final money")
        normalized = value["normalized_finish"]
        expected_normalized = (player_count - rank) / (player_count - 1)
        if (
            seat != expected_seat
            or rank > player_count
            or not isinstance(normalized, (int, float))
            or isinstance(normalized, bool)
            or not math.isfinite(float(normalized))
            or float(normalized) != expected_normalized
        ):
            raise BootstrapSelectionError("development utility is inconsistent")
        scores.append(SessionScore(seat=seat, final_money=money, rank=rank))
    return GameSummary(
        game_index=game_index,
        root_seed=root_seed,
        seed=engine_seed,
        player_count=player_count,
        ruleset_name=ruleset_name,
        bot_names=names,
        bot_ids=identities,
        scores=tuple(scores),
        decision_counts=decision_counts,
        fault_counts=fault_counts,
    )


def _read_json_lines(path: Path, name: str) -> tuple[dict[str, object], ...]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise BootstrapSelectionError(f"{name} contains duplicate JSON keys")
            result[key] = value
        return result

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines or any(not line.strip() for line in lines):
            raise BootstrapSelectionError(f"{name} must contain nonblank JSON lines")
        values = tuple(
            json.loads(
                line,
                object_pairs_hook=reject_duplicates,
                parse_constant=lambda value: _raise_nonfinite_json(name, value),
            )
            for line in lines
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BootstrapSelectionError(f"{name} could not be read") from error
    if any(not isinstance(value, dict) for value in values):
        raise BootstrapSelectionError(f"{name} rows must be JSON objects")
    return tuple(cast(dict[str, object], value) for value in values)


def _raise_nonfinite_json(name: str, value: str) -> object:
    raise BootstrapSelectionError(f"{name} contains non-finite JSON value {value}")


def _json_integer(value: object, name: str, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise BootstrapSelectionError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise BootstrapSelectionError(f"{name} is below its minimum")
    return value


def _json_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise BootstrapSelectionError(f"{name} must be a nonempty string")
    return value


def _string_tuple(value: object, name: str, length: int) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise BootstrapSelectionError(f"{name} are incomplete")
    return tuple(_json_string(item, name) for item in value)


def _integer_tuple(value: object, name: str, length: int) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise BootstrapSelectionError(f"{name} are incomplete")
    return tuple(_json_integer(item, name, minimum=0) for item in value)


def _string_array(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise BootstrapSelectionError(f"{name} must be a nonempty array")
    return tuple(_json_string(item, name) for item in value)


def _integer_array(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise BootstrapSelectionError(f"{name} must be a nonempty array")
    return tuple(_json_integer(item, name, minimum=0) for item in value)


def _require_empty_directory_target(path: Path, name: str) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise BootstrapSelectionError(f"{name} directory must be empty")


def _require_safe_paths(
    checkpoints: tuple[BootstrapCheckpointInput, ...] | list[BootstrapCheckpointInput],
    *,
    bootstrap_summary_path: Path,
    frozen_arms_dir: Path,
    evidence_output_dir: Path,
) -> None:
    frozen = frozen_arms_dir.resolve()
    evidence = evidence_output_dir.resolve()
    if frozen == evidence or frozen.is_relative_to(evidence) or evidence.is_relative_to(frozen):
        raise BootstrapSelectionError("frozen arms and evidence output must not overlap")
    inputs = (
        bootstrap_summary_path.resolve(),
        *(item.training_checkpoint.resolve() for item in checkpoints),
    )
    for input_path in inputs:
        if input_path.is_relative_to(frozen) or input_path.is_relative_to(evidence):
            raise BootstrapSelectionError("checkpoint and summary inputs must be outside outputs")
        if frozen.is_relative_to(input_path) or evidence.is_relative_to(input_path):
            raise BootstrapSelectionError("output directories must not be inside an input path")


def _install_directory(temporary: Path, destination: Path) -> None:
    if destination.exists():
        if not destination.is_dir() or any(destination.iterdir()):
            raise BootstrapSelectionError("frozen arms directory changed before installation")
        destination.rmdir()
    os.replace(temporary, destination)
    _sync_directory(destination.parent)


def _write_directory_transaction(output_dir: Path, rendered: dict[str, str]) -> None:
    _require_empty_directory_target(output_dir, "evidence output")
    parent = output_dir.resolve().parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = parent / f".{output_dir.name}.tmp-{uuid.uuid4().hex}"
    try:
        temporary.mkdir()
        for name, content in rendered.items():
            path = temporary / name
            path.write_text(content, encoding="utf-8")
            with path.open("rb") as file:
                os.fsync(file.fileno())
        _sync_directory(temporary)
        if output_dir.exists():
            if not output_dir.is_dir() or any(output_dir.iterdir()):
                raise BootstrapSelectionError("evidence output changed before installation")
            output_dir.rmdir()
        os.replace(temporary, output_dir)
        _sync_directory(output_dir.parent)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _json_document(value: object) -> str:
    return json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"


def _json_lines(values: Iterable[object]) -> str:
    return "".join(
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
        for value in values
    )


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_json_object(path: Path, name: str) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, nested in pairs:
            if key in result:
                raise BootstrapSelectionError(f"{name} contains duplicate JSON keys")
            result[key] = nested
        return result

    try:
        value: object = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda nested: _raise_nonfinite_json(name, nested),
        )
    except BootstrapSelectionError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BootstrapSelectionError(f"{name} could not be read") from error
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise BootstrapSelectionError(f"{name} must be a JSON object")
    return dict(value)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
