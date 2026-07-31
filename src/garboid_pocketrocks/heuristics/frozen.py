"""Load immutable development-selected heuristic candidates."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import NoReturn, cast

from garboid_pocketrocks.bots.base import BotSpec
from garboid_pocketrocks.bots.heuristic import (
    HeuristicBotBrain,
    PhaseAwareHeuristicBotBrain,
)
from garboid_pocketrocks.heuristics.phases import PHASE_SELECTOR_NAME
from garboid_pocketrocks.heuristics.profiles import (
    HeuristicProfile,
    PhaseAwareHeuristicProfile,
)

_CATALOG_PATH = Path(__file__).with_name("frozen_candidates") / "index.json"
_INDEX_KEYS = {"schema_version", "candidates"}
_ENTRY_KEYS = {
    "candidate_evaluations_sha256",
    "development_corpus_digest",
    "development_corpus_name",
    "file",
    "identity",
    "manifest_digest",
    "personality",
    "predecessor_name",
    "profile_digest",
    "repository_commit",
    "search_name",
    "search_report_sha256",
    "sha256",
}
_PHASE_ENTRY_KEYS = {
    *_ENTRY_KEYS,
    "boundary_report_digest",
    "boundary_slices_digest",
    "development_games_sha256",
    "selection_log_sha256",
    "winner_decision_slices_sha256",
    "winner_diagnostics_json_sha256",
    "winner_diagnostics_markdown_sha256",
}
_FROZEN_V1_KEYS = {
    "coefficients",
    "development_corpus",
    "development_scores",
    "generation",
    "identity",
    "parent_identity",
    "personality",
    "predecessor_name",
    "profile_digest",
    "repository_commit",
    "schema_version",
    "search",
    "slot",
    "source_evidence",
}
_FROZEN_V2_KEYS = {
    *(_FROZEN_V1_KEYS - {"coefficients"}),
    "boundary_evidence",
    "experts",
    "phase_selector",
}
_COEFFICIENT_KEYS = {
    "bid_shading",
    "future_cash_weight",
    "liquidity_strength",
    "objective_progress_weight",
}
_COEFFICIENT_BOUNDS = {
    "liquidity_strength": (Decimal("0"), Decimal("1.5")),
    "future_cash_weight": (Decimal("0"), Decimal("2")),
    "objective_progress_weight": (Decimal("0"), Decimal("1")),
    "bid_shading": (Decimal("0"), Decimal("1")),
}
_COEFFICIENT_STEP = Decimal("0.05")
_PHASES = ("early", "middle", "late")
_PHASE_SELECTOR_KEYS = {"early", "kind", "late", "middle"}
_BOUNDARY_EVIDENCE_KEYS = {
    "report_digest",
    "report_path",
    "slices_digest",
    "slices_path",
}
_DEVELOPMENT_CORPUS_KEYS = {"digest", "name"}
_DEVELOPMENT_SCORE_KEYS_V1 = {
    "final_money_delta",
    "normalized_finish_delta",
    "rating_delta",
}
_DEVELOPMENT_SCORE_KEYS_V2 = {
    "final_money_delta",
    "normalized_finish_delta",
    "rating_delta",
    "worst_challenger_finish_delta",
}
_SEARCH_KEYS = {"manifest_digest", "name"}
_SOURCE_EVIDENCE_KEYS = {
    "candidate_evaluations_sha256",
    "search_report_sha256",
}
_PHASE_SOURCE_EVIDENCE_KEYS = {
    *_SOURCE_EVIDENCE_KEYS,
    "development_games_sha256",
    "selection_log_sha256",
    "winner_diagnostics",
}
_WINNER_DIAGNOSTIC_NAMES = (
    "winner-decision-slices.csv",
    "winner-diagnostics.json",
    "winner-diagnostics.md",
)
_WINNER_DIAGNOSTIC_KEYS = set(_WINNER_DIAGNOSTIC_NAMES)
_PERSONALITIES = ("aggressive", "balanced", "passive")
_IDENTITY_PATTERN = re.compile(
    r"(?P<personality>aggressive|balanced|passive)-v3-candidate-"
    r"g(?P<generation>[0-9]{3})-s(?P<slot>[0-9]{3})-"
    r"(?P<profile_prefix>[0-9a-f]{12})\Z"
)
_PHASE_IDENTITY_PATTERN = re.compile(
    r"(?P<personality>aggressive|balanced|passive)-v4-candidate-"
    r"g(?P<generation>[0-9]{3})-s(?P<slot>[0-9]{3})-"
    r"(?P<profile_prefix>[0-9a-f]{12})\Z"
)
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
_FIXED_PHASE_SELECTOR = {
    "kind": PHASE_SELECTOR_NAME,
    "early": "3*future>=2*total",
    "middle": "3*future>=total",
    "late": "otherwise",
}
_FIXED_BOUNDARY_EVIDENCE = {
    "report_path": "docs/benchmarks/2026-07-30-heuristic-v4-phase-boundaries.md",
    "report_digest": "9961f26f32270dcebc98df443588e96cbde2f953858cd131c66a37aeecaa9b01",
    "slices_path": (
        "docs/benchmarks/tournaments/"
        "2026-07-30-heuristic-v3-phase-boundaries-development/phase-boundary-slices.csv"
    ),
    "slices_digest": "4f8aa60edf31b28c746cb8004a4dd5468ee8ab1b26462550c914b2e3fa50d7ae",
}
_DEVELOPMENT_CORPUS_NAME = "development-v1"
_DEVELOPMENT_CORPUS_DIGEST = "17c016350dbe717641b8cd499b0908e3bc0faa811a3b4f5e574f8713a5bf2b3d"


@dataclass(frozen=True, slots=True)
class FrozenCandidate:
    """One immutable local-only candidate and its development provenance."""

    identity: str
    personality: str
    predecessor_name: str
    generation: int
    slot: int
    parent_identity: str | None
    profile: HeuristicProfile
    bot_spec: BotSpec
    search_name: str
    development_corpus_name: str
    repository_commit: str
    freeze_digest: str
    profile_digest: str
    manifest_digest: str
    search_report_digest: str
    candidate_evaluations_digest: str
    development_corpus_digest: str
    worst_challenger_finish_delta: float | None
    rating_delta: float
    normalized_finish_delta: float
    final_money_delta: int


@dataclass(frozen=True, slots=True)
class FrozenPhaseAwareCandidate:
    """One immutable local-only phase-aware candidate and its provenance."""

    identity: str
    personality: str
    predecessor_name: str
    generation: int
    slot: int
    parent_identity: str | None
    profile: PhaseAwareHeuristicProfile
    bot_spec: BotSpec
    search_name: str
    development_corpus_name: str
    repository_commit: str
    freeze_digest: str
    profile_digest: str
    manifest_digest: str
    search_report_digest: str
    candidate_evaluations_digest: str
    development_corpus_digest: str
    rating_delta: float
    normalized_finish_delta: float
    final_money_delta: int
    phase_selector_rules: tuple[tuple[str, str], ...]
    expert_digests: tuple[tuple[str, str], ...]
    boundary_report_path: str
    boundary_report_digest: str
    boundary_slices_path: str
    boundary_slices_digest: str
    selection_log_digest: str
    development_games_digest: str
    winner_diagnostics_digests: tuple[tuple[str, str], ...]


type FrozenCatalogCandidate = FrozenCandidate | FrozenPhaseAwareCandidate


class FrozenCandidateCatalogError(ValueError):
    """Explain why frozen development evidence cannot be trusted."""

    code: str

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def build_frozen_heuristic_brain(
    profile: HeuristicProfile,
    seed: int | None,
) -> HeuristicBotBrain:
    """Construct an ordinary heuristic brain from a frozen profile."""

    del seed
    return HeuristicBotBrain(profile)


def build_frozen_phase_aware_brain(
    profile: PhaseAwareHeuristicProfile,
    seed: int | None,
) -> PhaseAwareHeuristicBotBrain:
    """Construct a phase-aware brain from one frozen three-expert profile."""

    del seed
    return PhaseAwareHeuristicBotBrain(profile)


def load_frozen_candidates(
    index_path: Path = _CATALOG_PATH,
) -> tuple[FrozenCatalogCandidate, ...]:
    """Load and cross-check one strict frozen-candidate catalog."""

    index = _load_json_object(index_path, subject="frozen candidate catalog")
    _require_exact_keys(index, _INDEX_KEYS, subject="frozen candidate catalog")
    if _require_integer(index["schema_version"], field="catalog schema version") != 1:
        raise FrozenCandidateCatalogError(
            "unsupported_catalog_schema",
            "The frozen candidate catalog schema version must be 1.",
        )
    entries_value = index["candidates"]
    if not isinstance(entries_value, list) or not entries_value:
        raise FrozenCandidateCatalogError(
            "invalid_catalog_candidates",
            "The frozen candidate catalog must contain a nonempty candidate list.",
        )
    entries = tuple(
        _require_object(entry, subject=f"catalog candidate {position}")
        for position, entry in enumerate(entries_value)
    )
    for position, entry in enumerate(entries):
        untrusted_identity = entry.get("identity")
        expected_keys = (
            _PHASE_ENTRY_KEYS
            if type(untrusted_identity) is str
            and _PHASE_IDENTITY_PATTERN.fullmatch(untrusted_identity) is not None
            else _ENTRY_KEYS
        )
        _require_exact_keys(entry, expected_keys, subject=f"catalog candidate {position}")
    identities = tuple(
        _require_string(entry["identity"], field="catalog candidate identity") for entry in entries
    )
    for identity in identities:
        _candidate_schema_from_identity(identity)
    if len(set(identities)) != len(identities):
        raise FrozenCandidateCatalogError(
            "duplicate_candidate_identity",
            "The frozen candidate catalog contains a duplicate candidate identity.",
        )
    if identities != tuple(sorted(identities)):
        raise FrozenCandidateCatalogError(
            "unsorted_candidate_catalog",
            "Frozen candidate identities must appear in ascending order.",
        )

    candidates = tuple(_load_catalog_candidate(index_path.parent, entry=entry) for entry in entries)
    return candidates


def _load_catalog_candidate(
    catalog_dir: Path,
    *,
    entry: dict[str, object],
) -> FrozenCatalogCandidate:
    identity = _require_string(entry["identity"], field="catalog candidate identity")
    identity_schema = _candidate_schema_from_identity(identity)
    file_name = _require_string(entry["file"], field=f"catalog file for {identity}")
    if (
        Path(file_name).name != file_name
        or file_name != f"{identity}.json"
        or "/" in file_name
        or "\\" in file_name
    ):
        raise FrozenCandidateCatalogError(
            "invalid_candidate_file",
            f"The catalog file for {identity!r} must be exactly {identity}.json.",
        )
    freeze_digest = _require_digest(entry["sha256"], field=f"freeze digest for {identity}")
    candidate_path = catalog_dir / file_name
    try:
        candidate_bytes = candidate_path.read_bytes()
    except OSError as error:
        raise FrozenCandidateCatalogError(
            "missing_candidate_file",
            f"Could not read frozen candidate file {candidate_path}.",
        ) from error
    actual_digest = hashlib.sha256(candidate_bytes).hexdigest()
    if actual_digest != freeze_digest:
        raise FrozenCandidateCatalogError(
            "candidate_digest_mismatch",
            f"The frozen candidate file digest for {identity!r} does not match the catalog.",
        )
    payload = _load_json_object(candidate_path, subject=f"frozen candidate {identity}")
    _require_required_keys(
        payload,
        {"schema_version"},
        subject=f"frozen candidate {identity}",
    )
    candidate_schema = _require_integer(
        payload["schema_version"],
        field="frozen candidate schema version",
    )
    if candidate_schema not in (1, 2):
        raise FrozenCandidateCatalogError(
            "unsupported_candidate_schema",
            f"Frozen candidate {identity!r} must use schema version 1 or 2.",
        )
    if identity_schema is not None and candidate_schema != identity_schema:
        raise FrozenCandidateCatalogError(
            "candidate_schema_identity_mismatch",
            f"Frozen candidate {identity!r} uses schema version {candidate_schema}, "
            f"but its identity requires schema version {identity_schema}.",
        )
    if identity_schema is not None:
        _require_exact_keys(entry, _PHASE_ENTRY_KEYS, subject=f"catalog candidate {identity}")
        _require_exact_keys(payload, _FROZEN_V2_KEYS, subject=f"frozen candidate {identity}")
        return _load_phase_catalog_candidate(
            entry,
            payload=payload,
            identity=identity,
            freeze_digest=freeze_digest,
        )
    _require_exact_keys(entry, _ENTRY_KEYS, subject=f"catalog candidate {identity}")
    _require_exact_keys(payload, _FROZEN_V1_KEYS, subject=f"frozen candidate {identity}")
    identity_match = _match_identity(identity)
    payload_identity = _require_string(payload["identity"], field="frozen candidate identity")
    if payload_identity != identity:
        raise FrozenCandidateCatalogError(
            "candidate_identity_mismatch",
            f"Frozen candidate identity {payload_identity!r} does not match catalog "
            f"identity {identity!r}.",
        )
    personality = _require_string(payload["personality"], field="candidate personality")
    generation = _require_integer(payload["generation"], field="candidate generation")
    slot = _require_integer(payload["slot"], field="candidate slot")
    if (
        personality != identity_match["personality"]
        or generation != int(identity_match["generation"])
        or slot != int(identity_match["slot"])
    ):
        raise FrozenCandidateCatalogError(
            "candidate_identity_content_mismatch",
            f"Frozen candidate {identity!r} disagrees with its personality, generation, or slot.",
        )
    _require_entry_match(entry, "personality", personality, identity=identity)

    predecessor_name = _require_string(
        payload["predecessor_name"],
        field="candidate predecessor",
    )
    if predecessor_name != f"{personality}-v2":
        raise FrozenCandidateCatalogError(
            "invalid_candidate_predecessor",
            f"Frozen candidate {identity!r} must name {personality}-v2 as its predecessor.",
        )
    _require_entry_match(
        entry,
        "predecessor_name",
        predecessor_name,
        identity=identity,
    )
    parent_identity = _decode_parent_identity(
        payload["parent_identity"],
        identity=identity,
        personality=personality,
        generation=generation,
    )

    coefficients = _require_object(
        payload["coefficients"],
        subject=f"coefficients for {identity}",
    )
    _require_exact_keys(coefficients, _COEFFICIENT_KEYS, subject=f"coefficients for {identity}")
    decimal_coefficients = {
        name: _require_decimal_string(
            coefficients[name],
            field=f"{identity} coefficient {name}",
        )
        for name in _COEFFICIENT_KEYS
    }
    profile = _build_profile(identity, personality, decimal_coefficients)
    profile_digest = _require_digest(
        payload["profile_digest"],
        field=f"profile digest for {identity}",
    )
    actual_profile_digest = _profile_digest(decimal_coefficients)
    if (
        actual_profile_digest != profile_digest
        or profile_digest[:12] != identity_match["profile_prefix"]
    ):
        raise FrozenCandidateCatalogError(
            "profile_digest_mismatch",
            f"The profile digest for frozen candidate {identity!r} does not match "
            "its coefficients and identity.",
        )
    _require_entry_match(entry, "profile_digest", profile_digest, identity=identity)

    search = _require_object(payload["search"], subject=f"search provenance for {identity}")
    _require_exact_keys(search, _SEARCH_KEYS, subject=f"search provenance for {identity}")
    search_name = _require_string(search["name"], field=f"search name for {identity}")
    if search_name != f"{personality}-v3-search-v1":
        raise FrozenCandidateCatalogError(
            "search_provenance_mismatch",
            f"Frozen candidate {identity!r} has inconsistent search provenance.",
        )
    manifest_digest = _require_digest(
        search["manifest_digest"],
        field=f"manifest digest for {identity}",
    )
    _require_entry_match(entry, "search_name", search_name, identity=identity)
    _require_entry_match(entry, "manifest_digest", manifest_digest, identity=identity)

    development = _require_object(
        payload["development_corpus"],
        subject=f"development corpus provenance for {identity}",
    )
    _require_exact_keys(
        development,
        _DEVELOPMENT_CORPUS_KEYS,
        subject=f"development corpus provenance for {identity}",
    )
    corpus_name = _require_string(
        development["name"],
        field=f"development corpus name for {identity}",
    )
    if not corpus_name.startswith("development-") or "held" in corpus_name.lower():
        raise FrozenCandidateCatalogError(
            "invalid_development_corpus",
            f"Frozen candidate {identity!r} must reference a development corpus.",
        )
    corpus_digest = _require_digest(
        development["digest"],
        field=f"development corpus digest for {identity}",
    )
    _require_entry_match(
        entry,
        "development_corpus_name",
        corpus_name,
        identity=identity,
    )
    _require_entry_match(
        entry,
        "development_corpus_digest",
        corpus_digest,
        identity=identity,
    )

    source_evidence = _require_object(
        payload["source_evidence"],
        subject=f"source evidence for {identity}",
    )
    _require_exact_keys(
        source_evidence,
        _SOURCE_EVIDENCE_KEYS,
        subject=f"source evidence for {identity}",
    )
    search_report_digest = _require_digest(
        source_evidence["search_report_sha256"],
        field=f"search report digest for {identity}",
    )
    candidate_evaluations_digest = _require_digest(
        source_evidence["candidate_evaluations_sha256"],
        field=f"candidate evaluations digest for {identity}",
    )
    _require_entry_match(
        entry,
        "search_report_sha256",
        search_report_digest,
        identity=identity,
    )
    _require_entry_match(
        entry,
        "candidate_evaluations_sha256",
        candidate_evaluations_digest,
        identity=identity,
    )

    repository_commit = _require_string(
        payload["repository_commit"],
        field=f"repository commit for {identity}",
    )
    if _COMMIT_PATTERN.fullmatch(repository_commit) is None:
        raise FrozenCandidateCatalogError(
            "invalid_repository_commit",
            f"Frozen candidate {identity!r} has an invalid repository commit.",
        )
    _require_entry_match(
        entry,
        "repository_commit",
        repository_commit,
        identity=identity,
    )

    scores = _require_object(
        payload["development_scores"],
        subject=f"development scores for {identity}",
    )
    _require_exact_keys(
        scores,
        (_DEVELOPMENT_SCORE_KEYS_V1 if candidate_schema == 1 else _DEVELOPMENT_SCORE_KEYS_V2),
        subject=f"development scores for {identity}",
    )
    rating_delta = _require_finite_number(
        scores["rating_delta"],
        field=f"rating delta for {identity}",
    )
    worst_challenger_finish_delta = (
        None
        if candidate_schema == 1
        else _require_finite_number(
            scores["worst_challenger_finish_delta"],
            field=f"worst challenger finish delta for {identity}",
        )
    )
    normalized_finish_delta = _require_finite_number(
        scores["normalized_finish_delta"],
        field=f"normalized finish delta for {identity}",
    )
    final_money_delta = _require_integer(
        scores["final_money_delta"],
        field=f"final money delta for {identity}",
    )
    if rating_delta <= 0.0 or (
        worst_challenger_finish_delta is not None and worst_challenger_finish_delta <= 0.0
    ):
        raise FrozenCandidateCatalogError(
            "nonpositive_frozen_candidate",
            f"Frozen candidate {identity!r} must improve its development rating "
            "and every challenger slice.",
        )

    bot_spec = BotSpec.for_simulation(
        identity,
        partial(build_frozen_heuristic_brain, profile),
    )
    return FrozenCandidate(
        identity=identity,
        personality=personality,
        predecessor_name=predecessor_name,
        generation=generation,
        slot=slot,
        parent_identity=parent_identity,
        profile=profile,
        bot_spec=bot_spec,
        search_name=search_name,
        development_corpus_name=corpus_name,
        repository_commit=repository_commit,
        freeze_digest=freeze_digest,
        profile_digest=profile_digest,
        manifest_digest=manifest_digest,
        search_report_digest=search_report_digest,
        candidate_evaluations_digest=candidate_evaluations_digest,
        development_corpus_digest=corpus_digest,
        worst_challenger_finish_delta=worst_challenger_finish_delta,
        rating_delta=rating_delta,
        normalized_finish_delta=normalized_finish_delta,
        final_money_delta=final_money_delta,
    )


def _load_phase_catalog_candidate(
    entry: dict[str, object],
    *,
    payload: dict[str, object],
    identity: str,
    freeze_digest: str,
) -> FrozenPhaseAwareCandidate:
    identity_match = _match_phase_identity(identity)
    payload_identity = _require_string(payload["identity"], field="frozen candidate identity")
    if payload_identity != identity:
        raise FrozenCandidateCatalogError(
            "candidate_identity_mismatch",
            f"Frozen candidate identity {payload_identity!r} does not match catalog "
            f"identity {identity!r}.",
        )
    personality = _require_string(payload["personality"], field="candidate personality")
    generation = _require_integer(payload["generation"], field="candidate generation")
    slot = _require_integer(payload["slot"], field="candidate slot")
    if (
        personality != identity_match["personality"]
        or generation != int(identity_match["generation"])
        or slot != int(identity_match["slot"])
        or not 0 <= generation < 12
        or not 0 <= slot < 16
    ):
        raise FrozenCandidateCatalogError(
            "candidate_identity_content_mismatch",
            f"Frozen candidate {identity!r} has inconsistent identity personality, "
            "generation, or slot.",
        )
    _require_entry_match(entry, "personality", personality, identity=identity)

    predecessor_name = _require_string(
        payload["predecessor_name"],
        field=f"candidate predecessor for {identity}",
    )
    if predecessor_name != f"{personality}-v3":
        raise FrozenCandidateCatalogError(
            "invalid_candidate_predecessor",
            f"Frozen candidate {identity!r} must name {personality}-v3 as its predecessor.",
        )
    _require_entry_match(entry, "predecessor_name", predecessor_name, identity=identity)
    parent_identity = _decode_phase_parent_identity(
        payload["parent_identity"],
        identity=identity,
        personality=personality,
        generation=generation,
    )

    selector = _require_object(
        payload["phase_selector"],
        subject=f"phase selector for {identity}",
    )
    _require_exact_keys(selector, _PHASE_SELECTOR_KEYS, subject=f"phase selector for {identity}")
    decoded_selector = {
        key: _require_string(selector[key], field=f"phase selector {key} for {identity}")
        for key in _PHASE_SELECTOR_KEYS
    }
    if decoded_selector != _FIXED_PHASE_SELECTOR:
        raise FrozenCandidateCatalogError(
            "invalid_phase_selector",
            f"Frozen candidate {identity!r} must use the fixed public resource selector.",
        )

    experts = _require_object(payload["experts"], subject=f"phase experts for {identity}")
    _require_exact_keys(experts, set(_PHASES), subject=f"phase experts for {identity}")
    decimal_experts: dict[str, dict[str, Decimal]] = {}
    for phase in _PHASES:
        expert = _require_object(
            experts[phase],
            subject=f"{phase} phase expert for {identity}",
        )
        _require_exact_keys(
            expert,
            _COEFFICIENT_KEYS,
            subject=f"{phase} phase expert for {identity}",
        )
        decimal_experts[phase] = {
            coefficient_name: _require_grid_decimal_string(
                expert[coefficient_name],
                coefficient_name=coefficient_name,
                field=f"{identity} {phase} coefficient {coefficient_name}",
            )
            for coefficient_name in sorted(_COEFFICIENT_KEYS)
        }
    profile = _build_phase_profile(identity, personality, decimal_experts)
    expert_digests = tuple((phase, _profile_digest(decimal_experts[phase])) for phase in _PHASES)
    profile_digest = _require_digest(
        payload["profile_digest"],
        field=f"profile digest for {identity}",
    )
    actual_profile_digest = _phase_profile_digest(decimal_experts)
    if (
        actual_profile_digest != profile_digest
        or profile_digest[:12] != identity_match["profile_prefix"]
    ):
        raise FrozenCandidateCatalogError(
            "profile_digest_mismatch",
            f"The profile digest for frozen candidate {identity!r} does not match "
            "its experts, selector, and identity.",
        )
    _require_entry_match(entry, "profile_digest", profile_digest, identity=identity)

    search = _require_object(payload["search"], subject=f"search provenance for {identity}")
    _require_exact_keys(search, _SEARCH_KEYS, subject=f"search provenance for {identity}")
    search_name = _require_string(search["name"], field=f"search name for {identity}")
    if search_name != f"{personality}-v4-search-v2":
        raise FrozenCandidateCatalogError(
            "search_provenance_mismatch",
            f"Frozen candidate {identity!r} has inconsistent search provenance.",
        )
    manifest_digest = _require_digest(
        search["manifest_digest"],
        field=f"manifest digest for {identity}",
    )
    _require_entry_match(entry, "search_name", search_name, identity=identity)
    _require_entry_match(entry, "manifest_digest", manifest_digest, identity=identity)

    development = _require_object(
        payload["development_corpus"],
        subject=f"development corpus provenance for {identity}",
    )
    _require_exact_keys(
        development,
        _DEVELOPMENT_CORPUS_KEYS,
        subject=f"development corpus provenance for {identity}",
    )
    corpus_name = _require_string(
        development["name"],
        field=f"development corpus name for {identity}",
    )
    corpus_digest = _require_digest(
        development["digest"],
        field=f"development corpus digest for {identity}",
    )
    if corpus_name != _DEVELOPMENT_CORPUS_NAME or corpus_digest != _DEVELOPMENT_CORPUS_DIGEST:
        raise FrozenCandidateCatalogError(
            "invalid_development_corpus",
            f"Frozen candidate {identity!r} must use the fixed development-v1 corpus.",
        )
    _require_entry_match(entry, "development_corpus_name", corpus_name, identity=identity)
    _require_entry_match(entry, "development_corpus_digest", corpus_digest, identity=identity)

    boundary = _require_object(
        payload["boundary_evidence"],
        subject=f"boundary evidence for {identity}",
    )
    _require_exact_keys(
        boundary,
        _BOUNDARY_EVIDENCE_KEYS,
        subject=f"boundary evidence for {identity}",
    )
    decoded_boundary = {
        "report_path": _require_string(
            boundary["report_path"],
            field=f"boundary report path for {identity}",
        ),
        "report_digest": _require_digest(
            boundary["report_digest"],
            field=f"boundary report digest for {identity}",
        ),
        "slices_path": _require_string(
            boundary["slices_path"],
            field=f"boundary slices path for {identity}",
        ),
        "slices_digest": _require_digest(
            boundary["slices_digest"],
            field=f"boundary slices digest for {identity}",
        ),
    }
    if decoded_boundary != _FIXED_BOUNDARY_EVIDENCE:
        raise FrozenCandidateCatalogError(
            "boundary_provenance_mismatch",
            f"Frozen candidate {identity!r} has inconsistent fixed boundary evidence.",
        )
    _require_entry_match(
        entry,
        "boundary_report_digest",
        decoded_boundary["report_digest"],
        identity=identity,
    )
    _require_entry_match(
        entry,
        "boundary_slices_digest",
        decoded_boundary["slices_digest"],
        identity=identity,
    )

    source = _require_object(
        payload["source_evidence"],
        subject=f"source evidence for {identity}",
    )
    _require_exact_keys(
        source,
        _PHASE_SOURCE_EVIDENCE_KEYS,
        subject=f"source evidence for {identity}",
    )
    search_report_digest = _require_digest(
        source["search_report_sha256"],
        field=f"search report digest for {identity}",
    )
    candidate_evaluations_digest = _require_digest(
        source["candidate_evaluations_sha256"],
        field=f"candidate evaluations digest for {identity}",
    )
    selection_log_digest = _require_digest(
        source["selection_log_sha256"],
        field=f"selection log digest for {identity}",
    )
    development_games_digest = _require_digest(
        source["development_games_sha256"],
        field=f"development games digest for {identity}",
    )
    winner_diagnostics = _require_object(
        source["winner_diagnostics"],
        subject=f"winner diagnostics for {identity}",
    )
    _require_exact_keys(
        winner_diagnostics,
        _WINNER_DIAGNOSTIC_KEYS,
        subject=f"winner diagnostics for {identity}",
    )
    diagnostic_digests = tuple(
        (
            name,
            _require_digest(
                winner_diagnostics[name],
                field=f"winner diagnostics {name} digest for {identity}",
            ),
        )
        for name in _WINNER_DIAGNOSTIC_NAMES
    )
    source_entry_fields = (
        ("search_report_sha256", search_report_digest),
        ("candidate_evaluations_sha256", candidate_evaluations_digest),
        ("selection_log_sha256", selection_log_digest),
        ("development_games_sha256", development_games_digest),
        ("winner_decision_slices_sha256", diagnostic_digests[0][1]),
        ("winner_diagnostics_json_sha256", diagnostic_digests[1][1]),
        ("winner_diagnostics_markdown_sha256", diagnostic_digests[2][1]),
    )
    for field, digest in source_entry_fields:
        _require_entry_match(entry, field, digest, identity=identity)

    repository_commit = _require_string(
        payload["repository_commit"],
        field=f"repository commit for {identity}",
    )
    if _COMMIT_PATTERN.fullmatch(repository_commit) is None:
        raise FrozenCandidateCatalogError(
            "invalid_repository_commit",
            f"Frozen candidate {identity!r} has an invalid repository commit.",
        )
    _require_entry_match(entry, "repository_commit", repository_commit, identity=identity)

    scores = _require_object(
        payload["development_scores"],
        subject=f"development scores for {identity}",
    )
    _require_exact_keys(
        scores,
        _DEVELOPMENT_SCORE_KEYS_V1,
        subject=f"development scores for {identity}",
    )
    rating_delta = _require_finite_number(
        scores["rating_delta"],
        field=f"rating delta for {identity}",
    )
    normalized_finish_delta = _require_finite_number(
        scores["normalized_finish_delta"],
        field=f"normalized finish delta for {identity}",
    )
    final_money_delta = _require_integer(
        scores["final_money_delta"],
        field=f"final money delta for {identity}",
    )
    if rating_delta <= 0.0:
        raise FrozenCandidateCatalogError(
            "nonpositive_frozen_candidate",
            f"Frozen candidate {identity!r} must have a positive development rating.",
        )

    bot_spec = BotSpec.for_simulation(
        identity,
        partial(build_frozen_phase_aware_brain, profile),
    )
    return FrozenPhaseAwareCandidate(
        identity=identity,
        personality=personality,
        predecessor_name=predecessor_name,
        generation=generation,
        slot=slot,
        parent_identity=parent_identity,
        profile=profile,
        bot_spec=bot_spec,
        search_name=search_name,
        development_corpus_name=corpus_name,
        repository_commit=repository_commit,
        freeze_digest=freeze_digest,
        profile_digest=profile_digest,
        manifest_digest=manifest_digest,
        search_report_digest=search_report_digest,
        candidate_evaluations_digest=candidate_evaluations_digest,
        development_corpus_digest=corpus_digest,
        rating_delta=rating_delta,
        normalized_finish_delta=normalized_finish_delta,
        final_money_delta=final_money_delta,
        phase_selector_rules=tuple(
            (key, decoded_selector[key]) for key in ("kind", "early", "middle", "late")
        ),
        expert_digests=expert_digests,
        boundary_report_path=decoded_boundary["report_path"],
        boundary_report_digest=decoded_boundary["report_digest"],
        boundary_slices_path=decoded_boundary["slices_path"],
        boundary_slices_digest=decoded_boundary["slices_digest"],
        selection_log_digest=selection_log_digest,
        development_games_digest=development_games_digest,
        winner_diagnostics_digests=diagnostic_digests,
    )


def _build_phase_profile(
    identity: str,
    personality: str,
    experts: Mapping[str, Mapping[str, Decimal]],
) -> PhaseAwareHeuristicProfile:
    try:
        named_experts = {
            phase: _build_profile(identity, personality, experts[phase]) for phase in _PHASES
        }
        return PhaseAwareHeuristicProfile(
            name=personality,
            early=named_experts["early"],
            middle=named_experts["middle"],
            late=named_experts["late"],
            phase_selector=PHASE_SELECTOR_NAME,
        )
    except (KeyError, ValueError) as error:
        raise FrozenCandidateCatalogError(
            "invalid_candidate_coefficients",
            f"Frozen candidate {identity!r} has invalid phase-aware coefficients: {error}",
        ) from error


def _phase_profile_digest(
    experts: Mapping[str, Mapping[str, Decimal]],
) -> str:
    payload = {
        "experts": {
            phase: {name: _decimal_text(experts[phase][name]) for name in sorted(_COEFFICIENT_KEYS)}
            for phase in _PHASES
        },
        "phase_selector": PHASE_SELECTOR_NAME,
    }
    encoded = (
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_profile(
    identity: str,
    personality: str,
    coefficients: Mapping[str, Decimal],
) -> HeuristicProfile:
    try:
        return HeuristicProfile(
            name=personality,
            liquidity_strength=float(coefficients["liquidity_strength"]),
            future_cash_weight=float(coefficients["future_cash_weight"]),
            objective_progress_weight=float(coefficients["objective_progress_weight"]),
            bid_shading=float(coefficients["bid_shading"]),
        )
    except ValueError as error:
        raise FrozenCandidateCatalogError(
            "invalid_candidate_coefficients",
            f"Frozen candidate {identity!r} has invalid heuristic coefficients: {error}",
        ) from error


def _profile_digest(coefficients: Mapping[str, Decimal]) -> str:
    payload = {name: _decimal_text(coefficients[name]) for name in sorted(_COEFFICIENT_KEYS)}
    encoded = (
        json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _decode_parent_identity(
    value: object,
    *,
    identity: str,
    personality: str,
    generation: int,
) -> str | None:
    if value is None and generation == 0:
        return None
    parent = _require_string(value, field=f"parent identity for {identity}")
    parent_match = _match_identity(parent)
    if parent_match["personality"] != personality or int(parent_match["generation"]) >= generation:
        raise FrozenCandidateCatalogError(
            "invalid_parent_identity",
            f"Frozen candidate {identity!r} has an inconsistent parent identity.",
        )
    return parent


def _decode_phase_parent_identity(
    value: object,
    *,
    identity: str,
    personality: str,
    generation: int,
) -> str | None:
    if value is None:
        if generation == 0:
            return None
        raise FrozenCandidateCatalogError(
            "invalid_parent_identity",
            f"Frozen candidate {identity!r} must name an earlier v4 parent.",
        )
    parent = _require_string(value, field=f"parent identity for {identity}")
    matched_parent = _PHASE_IDENTITY_PATTERN.fullmatch(parent)
    if matched_parent is None:
        raise FrozenCandidateCatalogError(
            "invalid_parent_identity",
            f"Frozen candidate {identity!r} must name a canonical v4 parent identity.",
        )
    parent_match = matched_parent.groupdict()
    if (
        parent_match["personality"] != personality
        or int(parent_match["generation"]) >= generation
        or not 0 <= int(parent_match["slot"]) < 16
    ):
        raise FrozenCandidateCatalogError(
            "invalid_parent_identity",
            f"Frozen candidate {identity!r} has an inconsistent v4 parent identity.",
        )
    return parent


def _match_identity(identity: str) -> dict[str, str]:
    matched = _IDENTITY_PATTERN.fullmatch(identity)
    if matched is None:
        raise FrozenCandidateCatalogError(
            "invalid_candidate_identity",
            f"Frozen candidate identity {identity!r} is not canonical.",
        )
    return matched.groupdict()


def _match_phase_identity(identity: str) -> dict[str, str]:
    matched = _PHASE_IDENTITY_PATTERN.fullmatch(identity)
    if matched is None:
        raise FrozenCandidateCatalogError(
            "invalid_candidate_identity",
            f"Phase-aware frozen candidate identity {identity!r} is not canonical v4.",
        )
    return matched.groupdict()


def _candidate_schema_from_identity(identity: str) -> int | None:
    if _PHASE_IDENTITY_PATTERN.fullmatch(identity) is not None:
        return 2
    _match_identity(identity)
    return None


def _require_entry_match(
    entry: dict[str, object],
    field: str,
    payload_value: str,
    *,
    identity: str,
) -> None:
    entry_value = _require_string(
        entry[field],
        field=f"catalog {field} for {identity}",
    )
    if entry_value != payload_value:
        subject = field.replace("_", " ")
        raise FrozenCandidateCatalogError(
            "catalog_provenance_mismatch",
            f"The catalog {subject} for {identity!r} does not match its frozen payload.",
        )


def _require_digest(value: object, *, field: str) -> str:
    digest = _require_string(value, field=field)
    if _DIGEST_PATTERN.fullmatch(digest) is None:
        raise FrozenCandidateCatalogError(
            "invalid_provenance_digest",
            f"The {field} must be a lowercase SHA-256 digest.",
        )
    return digest


def _require_decimal_string(value: object, *, field: str) -> Decimal:
    if not isinstance(value, str) or value.strip() != value or not value:
        raise FrozenCandidateCatalogError(
            "invalid_candidate_coefficients",
            f"The {field} must be a canonical finite decimal string.",
        )
    try:
        decimal = Decimal(value)
    except InvalidOperation as error:
        raise FrozenCandidateCatalogError(
            "invalid_candidate_coefficients",
            f"The {field} must be a canonical finite decimal string.",
        ) from error
    if not decimal.is_finite() or _decimal_text(decimal) != value:
        raise FrozenCandidateCatalogError(
            "invalid_candidate_coefficients",
            f"The {field} must be a canonical finite decimal string.",
        )
    return decimal


def _require_grid_decimal_string(
    value: object,
    *,
    coefficient_name: str,
    field: str,
) -> Decimal:
    decimal = _require_decimal_string(value, field=field)
    minimum, maximum = _COEFFICIENT_BOUNDS[coefficient_name]
    if not minimum <= decimal <= maximum or (decimal - minimum) % _COEFFICIENT_STEP != 0:
        raise FrozenCandidateCatalogError(
            "invalid_candidate_coefficients",
            f"The {field} must be on the fixed 0.05 coefficient grid.",
        )
    return decimal


def _decimal_text(value: Decimal) -> str:
    if value.is_zero():
        return "0"
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _require_finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FrozenCandidateCatalogError(
            "invalid_development_score",
            f"The {field} must be a finite number.",
        )
    result = float(value)
    if not math.isfinite(result):
        raise FrozenCandidateCatalogError(
            "invalid_development_score",
            f"The {field} must be a finite number.",
        )
    return result


def _require_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FrozenCandidateCatalogError(
            "invalid_integer",
            f"The {field} must be an integer.",
        )
    return value


def _require_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise FrozenCandidateCatalogError(
            "invalid_string",
            f"The {field} must be a nonempty string.",
        )
    return value


def _require_object(value: object, *, subject: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise FrozenCandidateCatalogError(
            "invalid_object",
            f"The {subject} must be a JSON object.",
        )
    return cast(dict[str, object], value)


def _require_exact_keys(
    payload: dict[str, object],
    expected: set[str],
    *,
    subject: str,
) -> None:
    actual = set(payload)
    if actual == expected:
        return
    missing = sorted(key.replace("_", " ") for key in expected - actual)
    unknown = sorted(key.replace("_", " ") for key in actual - expected)
    raise FrozenCandidateCatalogError(
        "invalid_object_keys",
        f"The {subject} has invalid keys; missing {missing}, unknown {unknown}.",
    )


def _require_required_keys(
    payload: dict[str, object],
    required: set[str],
    *,
    subject: str,
) -> None:
    missing = sorted(key.replace("_", " ") for key in required - set(payload))
    if missing:
        raise FrozenCandidateCatalogError(
            "invalid_object_keys",
            f"The {subject} has invalid keys; missing {missing}.",
        )


class _NonFiniteJsonNumber(ValueError):
    """Internal signal for JSON NaN and infinity."""


def _load_json_object(path: Path, *, subject: str) -> dict[str, object]:
    def reject_nonfinite(value: str) -> NoReturn:
        raise _NonFiniteJsonNumber(value)

    def reject_duplicate_keys(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        decoded: dict[str, object] = {}
        for key, value in pairs:
            if key in decoded:
                raise FrozenCandidateCatalogError(
                    "duplicate_json_key",
                    f"The {subject} contains duplicate JSON key {key!r}.",
                )
            decoded[key] = value
        return decoded

    try:
        decoded = cast(
            object,
            json.loads(
                path.read_text(encoding="utf-8"),
                parse_constant=reject_nonfinite,
                object_pairs_hook=reject_duplicate_keys,
            ),
        )
    except (OSError, json.JSONDecodeError, UnicodeError, _NonFiniteJsonNumber) as error:
        raise FrozenCandidateCatalogError(
            "invalid_json",
            f"The {subject} at {path} must be readable finite JSON.",
        ) from error
    return _require_object(decoded, subject=subject)


FROZEN_CANDIDATES = load_frozen_candidates()
FROZEN_CANDIDATES_BY_NAME: Mapping[str, FrozenCatalogCandidate] = MappingProxyType(
    {candidate.identity: candidate for candidate in FROZEN_CANDIDATES}
)
