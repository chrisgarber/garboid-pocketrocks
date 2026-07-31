"""Held-out promotion gate for the recorded heuristic-bootstrap winner.

The API intentionally accepts no candidate identity, strategy, or checkpoint.
Those values come only from the verified development-selection evidence and its
bound frozen candidate.  Held-out data is not loaded until those development
bindings and the immediate incumbent have been verified.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch

from garboid_pocketrocks.bots import BOT_SPECS_BY_NAME, BotSpec
from garboid_pocketrocks.neural.bootstrap_freeze import (
    FrozenBootstrapCandidate,
    load_frozen_bootstrap_candidate,
)
from garboid_pocketrocks.neural.bootstrap_selection import selected_bootstrap_bot_spec
from garboid_pocketrocks.neural.checkpoint import load_inference_checkpoint
from garboid_pocketrocks.neural.heuristic_bootstrap import (
    HEURISTIC_BOOTSTRAP_ARMS,
    REFERENCE_NEURAL_IDENTITY,
    REFERENCE_PARAMETER_DIGEST,
)
from garboid_pocketrocks.neural.tournament_bot import (
    LARGE_CHECKPOINT_PATH,
    VECTOR_PPO_LARGE_V1_G350K_BOT_SPEC,
    FrozenBootstrapBrainFactory,
)
from garboid_pocketrocks.promotion.corpus import (
    PromotionCorpus,
    load_promotion_corpus,
    recompute_promotion_corpus_digest,
)
from garboid_pocketrocks.promotion.runner import (
    PromotionRun,
    PromotionRunConfig,
    PromotionRunner,
)

_HELD_OUT_CORPUS_NAME = "held-out-v1"
_HELD_OUT_CORPUS_DIGEST = "de686b97e9318d840554514d71158e7d30e4b1603c6692d68b73bc77947b10da"
_BOOTSTRAP_SAMPLES = 1_000
_BOOTSTRAP_SEED = 0
_BATCH_SIZE = 64
_PROMOTION_OUTPUT_SUFFIX = "held-out-promotion"
_ATTEMPT_RECEIPT_SUFFIX = "held-out-promotion-attempt.json"
_PROVENANCE_NAME = "bootstrap-promotion-provenance.json"


class BootstrapPromotionError(ValueError):
    """Raised before held-out games when bootstrap provenance is not trustworthy."""


@dataclass(frozen=True, slots=True)
class BootstrapPromotionLocations:
    """Deterministic one-shot paths derived from development evidence."""

    attempt_receipt_json: Path
    promotion_output_dir: Path
    promotion_provenance_json: Path


@dataclass(frozen=True, slots=True)
class _SelectedFreeze:
    candidate: FrozenBootstrapCandidate
    path: Path
    manifest_digest: str


def bootstrap_promotion_locations(evidence_output_dir: Path) -> BootstrapPromotionLocations:
    """Return the fixed receipt, result, and provenance paths for one selection."""

    evidence_path = evidence_output_dir.resolve()
    receipt = evidence_path.parent / f"{evidence_path.name}-{_ATTEMPT_RECEIPT_SUFFIX}"
    output = evidence_path.parent / f"{evidence_path.name}-{_PROMOTION_OUTPUT_SUFFIX}"
    return BootstrapPromotionLocations(
        attempt_receipt_json=receipt,
        promotion_output_dir=output,
        promotion_provenance_json=output / _PROVENANCE_NAME,
    )


def run_selected_bootstrap_promotion(
    *,
    evidence_output_dir: Path,
    frozen_arms_dir: Path,
    development_corpus: PromotionCorpus,
    held_out_corpus_path: Path,
    workers: int,
) -> PromotionRun:
    """Promote only the winner recorded by immutable development evidence.

    ``held_out_corpus_path`` is deliberately a path, rather than an already
    loaded corpus.  This function does not read it until selection, freeze,
    development-corpus, incumbent, and runtime-option checks have all passed.
    The ordinary promotion runner remains responsible for paired simulation,
    fault handling, and the deterministic bootstrap 95 percent interval.
    """

    candidate = selected_bootstrap_bot_spec(
        evidence_output_dir=evidence_output_dir,
        frozen_arms_dir=frozen_arms_dir,
    )
    selected_freeze = _load_selected_freeze(candidate, frozen_arms_dir=frozen_arms_dir)
    _validate_development_binding(
        candidate,
        selected_freeze.candidate,
        development_corpus=development_corpus,
    )
    incumbent = _canonical_incumbent()
    _validate_workers(workers)
    evidence_manifest_digest = _file_digest(evidence_output_dir / "evidence-manifest.json")
    locations = bootstrap_promotion_locations(evidence_output_dir)
    receipt_payload = _attempt_receipt_payload(
        candidate=candidate,
        selected_freeze=selected_freeze,
        development_corpus=development_corpus,
        evidence_manifest_digest=evidence_manifest_digest,
        workers=workers,
        output_dir_name=locations.promotion_output_dir.name,
    )
    _write_exclusive_json(locations.attempt_receipt_json, receipt_payload)

    held_out_corpus = load_promotion_corpus(
        held_out_corpus_path,
        registry=BOT_SPECS_BY_NAME,
    )
    _validate_pinned_held_out(held_out_corpus)
    _validate_canonical_opponents(held_out_corpus)
    run = PromotionRunner.run(
        PromotionRunConfig(
            candidate=candidate,
            incumbent=incumbent,
            development=development_corpus,
            held_out=held_out_corpus,
            bootstrap_samples=_BOOTSTRAP_SAMPLES,
            bootstrap_seed=_BOOTSTRAP_SEED,
            batch_size=_BATCH_SIZE,
        ),
        registry=BOT_SPECS_BY_NAME,
        workers=workers,
        output_dir=locations.promotion_output_dir,
    )
    _write_promotion_provenance(
        run,
        locations=locations,
        candidate_identity=candidate.name,
        receipt_payload=receipt_payload,
    )
    return run


def _load_selected_freeze(
    candidate: BotSpec,
    *,
    frozen_arms_dir: Path,
) -> _SelectedFreeze:
    factory = candidate.brain_factory
    if type(factory) is not FrozenBootstrapBrainFactory:
        raise BootstrapPromotionError(
            "selected candidate is not backed by a verified bootstrap freeze"
        )
    if factory.expected_identity != candidate.name or candidate.bot_id != candidate.name:
        raise BootstrapPromotionError("selected candidate identity does not match its freeze")

    frozen_root = frozen_arms_dir.resolve()
    candidate_path = factory.candidate_path.resolve()
    allowed_strategies = {arm.strategy for arm in HEURISTIC_BOOTSTRAP_ARMS}
    if candidate_path.parent != frozen_root or candidate_path.name not in allowed_strategies:
        raise BootstrapPromotionError(
            "selected candidate is outside the official frozen-arm directory"
        )
    frozen = load_frozen_bootstrap_candidate(candidate_path)
    if (
        frozen.manifest.identity != candidate.name
        or frozen.manifest.strategy != candidate_path.name
    ):
        raise BootstrapPromotionError("selected candidate changed after evidence verification")
    return _SelectedFreeze(
        candidate=frozen,
        path=candidate_path,
        manifest_digest=_file_digest(candidate_path / "manifest.json"),
    )


def _validate_development_binding(
    candidate: BotSpec,
    frozen: FrozenBootstrapCandidate,
    *,
    development_corpus: PromotionCorpus,
) -> None:
    if development_corpus.recipe.purpose != "development":
        raise BootstrapPromotionError("bootstrap promotion requires a development corpus")
    if recompute_promotion_corpus_digest(development_corpus) != development_corpus.digest:
        raise BootstrapPromotionError("development corpus digest is not immutable")
    if (
        frozen.manifest.identity != candidate.name
        or frozen.manifest.development_corpus_name != development_corpus.recipe.name
        or frozen.manifest.development_corpus_digest != development_corpus.digest
    ):
        raise BootstrapPromotionError(
            "selected freeze does not bind the supplied development corpus"
        )


def _canonical_incumbent() -> BotSpec:
    incumbent = BOT_SPECS_BY_NAME.get(REFERENCE_NEURAL_IDENTITY)
    if incumbent is not VECTOR_PPO_LARGE_V1_G350K_BOT_SPEC:
        raise BootstrapPromotionError(
            "the immediate large-v1 neural incumbent is not canonically registered"
        )
    loaded = load_inference_checkpoint(LARGE_CHECKPOINT_PATH, device=torch.device("cpu"))
    if loaded.manifest.parameter_digest != REFERENCE_PARAMETER_DIGEST:
        raise BootstrapPromotionError(
            "the immediate large-v1 neural incumbent parameter digest changed"
        )
    return incumbent


def _validate_pinned_held_out(held_out_corpus: PromotionCorpus) -> None:
    if (
        held_out_corpus.recipe.purpose != "held_out"
        or held_out_corpus.recipe.name != _HELD_OUT_CORPUS_NAME
        or held_out_corpus.digest != _HELD_OUT_CORPUS_DIGEST
        or recompute_promotion_corpus_digest(held_out_corpus) != _HELD_OUT_CORPUS_DIGEST
    ):
        raise BootstrapPromotionError(
            "held-out corpus does not match the committed held-out-v1 exam"
        )


def _validate_canonical_opponents(held_out_corpus: PromotionCorpus) -> None:
    for name in held_out_corpus.recipe.opponent_names:
        opponent = BOT_SPECS_BY_NAME.get(name)
        if opponent is None or opponent.name != name:
            raise BootstrapPromotionError(
                f"held-out opponent {name!r} is not canonically registered"
            )


def _validate_workers(workers: int) -> None:
    if type(workers) is not int or workers < 1:
        raise BootstrapPromotionError("workers must be positive")


def _attempt_receipt_payload(
    *,
    candidate: BotSpec,
    selected_freeze: _SelectedFreeze,
    development_corpus: PromotionCorpus,
    evidence_manifest_digest: str,
    workers: int,
    output_dir_name: str,
) -> dict[str, object]:
    manifest = selected_freeze.candidate.manifest
    return {
        "schema_version": 1,
        "purpose": "bootstrap_held_out_promotion_attempt",
        "evidence_manifest_digest": evidence_manifest_digest,
        "selected_candidate_identity": candidate.name,
        "selected_strategy": manifest.strategy,
        "selected_freeze_manifest_digest": selected_freeze.manifest_digest,
        "selected_summary_digest": manifest.summary_digest,
        "selected_parameter_digest": manifest.parameter_digest,
        "development_corpus_name": development_corpus.recipe.name,
        "development_corpus_digest": development_corpus.digest,
        "expected_held_out_corpus_name": _HELD_OUT_CORPUS_NAME,
        "expected_held_out_corpus_digest": _HELD_OUT_CORPUS_DIGEST,
        "incumbent_identity": REFERENCE_NEURAL_IDENTITY,
        "incumbent_parameter_digest": REFERENCE_PARAMETER_DIGEST,
        "promotion_output_directory": output_dir_name,
        "promotion_rule": {
            "bootstrap_samples": _BOOTSTRAP_SAMPLES,
            "bootstrap_seed": _BOOTSTRAP_SEED,
            "confidence_level": 0.95,
            "require_fault_free_games": True,
            "require_positive_lower_bound": True,
        },
        "execution": {
            "batch_size": _BATCH_SIZE,
            "workers": workers,
        },
    }


def _write_promotion_provenance(
    run: PromotionRun,
    *,
    locations: BootstrapPromotionLocations,
    candidate_identity: str,
    receipt_payload: dict[str, object],
) -> None:
    expected_report = locations.promotion_output_dir / "promotion-report.json"
    if run.artifacts.report_json.resolve() != expected_report.resolve():
        raise BootstrapPromotionError("promotion runner returned an unexpected report path")
    public_payload: dict[str, object] = {
        "schema_version": 1,
        "purpose": "bootstrap_held_out_promotion_provenance",
        "selected_candidate_identity": candidate_identity,
        "attempt_receipt": {
            "name": locations.attempt_receipt_json.name,
            "sha256": _file_digest(locations.attempt_receipt_json),
        },
        "promotion_report": {
            "name": expected_report.name,
            "sha256": _file_digest(expected_report),
        },
        "attempt_binding_digest": _json_digest(receipt_payload),
    }
    payload = {**public_payload, "provenance_digest": _json_digest(public_payload)}
    _write_exclusive_json(locations.promotion_provenance_json, payload)


def _write_exclusive_json(path: Path, payload: object) -> None:
    content = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o444)
    except FileExistsError as error:
        raise BootstrapPromotionError(
            f"bootstrap held-out promotion is already consumed: {path.name}"
        ) from error
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("exclusive receipt write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _file_digest(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise BootstrapPromotionError(
            f"required promotion artifact is unreadable: {path.name}"
        ) from error


def _json_digest(value: object) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
