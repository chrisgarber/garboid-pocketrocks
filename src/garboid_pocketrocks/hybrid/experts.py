"""Fail-closed catalog of experts that already passed promotion."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast, final

from garboid_pocketrocks.bots.base import BotSpec
from garboid_pocketrocks.bots.heuristic import (
    AGGRESSIVE_HEURISTIC_V3_BOT_SPEC,
    BALANCED_HEURISTIC_V3_BOT_SPEC,
    PASSIVE_HEURISTIC_V3_BOT_SPEC,
)
from garboid_pocketrocks.bots.registry import BOT_SPECS_BY_NAME
from garboid_pocketrocks.heuristics.frozen import FrozenCandidate, load_frozen_candidates
from garboid_pocketrocks.heuristics.profiles import HEURISTIC_V3, HeuristicProfile
from garboid_pocketrocks.neural.tournament_bot import (
    LARGE_CHECKPOINT_PATH,
    VECTOR_PPO_LARGE_V1_G350K_BOT_SPEC,
)

type ExecutableKind = Literal["heuristic_profile", "neural_checkpoint"]

_CATALOG_PATH = Path(__file__).with_name("promoted_experts-v1.json")
# This is the trust anchor for the compact receipts. Changing promotion history
# requires an explicit code review, not an unnoticed data-file edit.
_CATALOG_SHA256 = "ad6904d24776f6572fc290b2a4c6259e12a7567bbff0e6d296240c5ebd592243"
_EXPECTED_EXPERT_NAMES = (
    "aggressive-v3",
    "balanced-v3",
    "passive-v3",
    "vector_ppo_large_v1_g350k",
)
_EXPECTED_INCUMBENTS = {
    "aggressive-v3": "aggressive-v2",
    "balanced-v3": "balanced-v2",
    "passive-v3": "passive-v2",
    "vector_ppo_large_v1_g350k": "vector_ppo_small_v1_g1500",
}
_DEVELOPMENT_CORPUS = (
    "development-v1",
    "17c016350dbe717641b8cd499b0908e3bc0faa811a3b4f5e574f8713a5bf2b3d",
)
_HELD_OUT_CORPUS = (
    "held-out-v1",
    "de686b97e9318d840554514d71158e7d30e4b1603c6692d68b73bc77947b10da",
)
_EXPECTED_RULE = {
    "charts": ["A", "B", "C", "D", "E"],
    "fault_mode": "record_and_pass",
    "objectives_enabled": [True],
    "player_counts": [3, 4, 5],
}
_EXPECTED_HEURISTIC_EXECUTABLES: Mapping[str, tuple[BotSpec, HeuristicProfile]] = MappingProxyType(
    {
        "aggressive-v3": (AGGRESSIVE_HEURISTIC_V3_BOT_SPEC, HEURISTIC_V3.aggressive),
        "balanced-v3": (BALANCED_HEURISTIC_V3_BOT_SPEC, HEURISTIC_V3.balanced),
        "passive-v3": (PASSIVE_HEURISTIC_V3_BOT_SPEC, HEURISTIC_V3.passive),
    }
)
_CATALOG_KEYS = {"experts", "schema_version"}
_ENTRY_KEYS = {
    "executable",
    "expert_name",
    "incumbent_name",
    "promoted_candidate_identity",
    "promotion",
    "promotion_evidence_digest",
}
_EXECUTABLE_KEYS = {"artifact_digest", "artifact_name", "digest", "kind"}
_PROMOTION_KEYS = {
    "bootstrap",
    "confidence_interval_95",
    "corpora",
    "coverage",
    "failures",
    "faults",
    "promoted",
    "rating_difference",
    "repository_commit",
    "rule",
    "source_artifact",
    "source_sha256",
    "warnings",
}
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
_VERIFIED_CATALOG_TOKEN = object()


class PromotedExpertCatalogError(ValueError):
    """Explain why an expert cannot be trusted as promoted."""

    code: str

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class PromotedExpert:
    """One immutable expert with verified promotion and executable evidence."""

    name: str
    bot_spec: BotSpec
    promoted_candidate_identity: str
    incumbent_name: str
    promotion_evidence_digest: str
    executable_kind: ExecutableKind
    executable_digest: str
    executable_artifact_digest: str
    development_corpus_name: str
    development_corpus_digest: str
    held_out_corpus_name: str
    held_out_corpus_digest: str
    promotion_repository_commit: str
    rating_difference: float
    confidence_interval_lower: float
    confidence_interval_upper: float


@final
class VerifiedPromotedExpertCatalog:
    """Opaque proof that the exact pinned catalog passed every verifier."""

    __slots__ = ("__catalog_digest", "__experts", "__token")
    __catalog_digest: str
    __experts: tuple[PromotedExpert, ...]
    __token: object

    def __init__(
        self,
        experts: tuple[PromotedExpert, ...],
        *,
        catalog_digest: str,
        token: object,
    ) -> None:
        if token is not _VERIFIED_CATALOG_TOKEN or catalog_digest != _CATALOG_SHA256:
            raise TypeError(
                "verified promoted expert catalogs can only be loaded from package data"
            )
        object.__setattr__(self, "_VerifiedPromotedExpertCatalog__experts", experts)
        object.__setattr__(self, "_VerifiedPromotedExpertCatalog__catalog_digest", catalog_digest)
        object.__setattr__(self, "_VerifiedPromotedExpertCatalog__token", token)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("verified promoted expert catalogs are immutable")

    def __iter__(self) -> Iterator[PromotedExpert]:
        return iter(self.__experts)

    def __len__(self) -> int:
        return len(self.__experts)

    def __getitem__(self, index: int) -> PromotedExpert:
        return self.__experts[index]

    def _verified_experts(self) -> tuple[PromotedExpert, ...]:
        if self.__token is not _VERIFIED_CATALOG_TOKEN or self.__catalog_digest != _CATALOG_SHA256:
            raise TypeError("promoted expert catalog is not a valid pinned catalog")
        return self.__experts


@dataclass(frozen=True, slots=True)
class ExpertAvailability:
    """A runtime probe result that does not rewrite promotion eligibility."""

    expert_name: str
    available: bool
    reason: Literal[
        "available",
        "availability_not_reported",
        "runtime_dependency_missing",
        "construction_failed",
    ]
    detail: str | None = None


def load_promoted_experts() -> VerifiedPromotedExpertCatalog:
    """Load the one pinned catalog and reject any missing or edited evidence."""

    try:
        catalog_bytes = _CATALOG_PATH.read_bytes()
    except OSError as error:
        raise PromotedExpertCatalogError(
            "missing_catalog",
            "The promoted expert receipt catalog is missing.",
        ) from error
    if hashlib.sha256(catalog_bytes).hexdigest() != _CATALOG_SHA256:
        raise PromotedExpertCatalogError(
            "catalog_digest_mismatch",
            "The promoted expert receipt catalog was edited without updating its trust anchor.",
        )
    payload = _decode_json_object(catalog_bytes, subject="promoted expert catalog")
    catalog_digest = hashlib.sha256(catalog_bytes).hexdigest()
    experts = _validate_catalog_payload(payload, bot_specs=BOT_SPECS_BY_NAME)
    _verify_retained_sources(experts, payload)
    return VerifiedPromotedExpertCatalog(
        experts,
        catalog_digest=catalog_digest,
        token=_VERIFIED_CATALOG_TOKEN,
    )


def check_expert_availability(expert: PromotedExpert) -> ExpertAvailability:
    """Construct one eligible expert and return a stable, plain-English diagnostic."""

    try:
        expert.bot_spec.make_brain(seed=0)
    except (ImportError, ModuleNotFoundError) as error:
        return ExpertAvailability(
            expert_name=expert.name,
            available=False,
            reason="runtime_dependency_missing",
            detail=f"{type(error).__name__}: {error}",
        )
    except Exception as error:  # noqa: BLE001 - availability must become a diagnostic
        return ExpertAvailability(
            expert_name=expert.name,
            available=False,
            reason="construction_failed",
            detail=f"{type(error).__name__}: {error}",
        )
    return ExpertAvailability(expert_name=expert.name, available=True, reason="available")


def _validate_catalog_payload(
    payload: dict[str, object],
    *,
    bot_specs: Mapping[str, BotSpec],
) -> tuple[PromotedExpert, ...]:
    _require_exact_keys(payload, _CATALOG_KEYS, subject="promoted expert catalog")
    if _require_int(payload["schema_version"], field="catalog schema version") != 1:
        raise PromotedExpertCatalogError(
            "unsupported_catalog_schema",
            "The promoted expert catalog schema version must be 1.",
        )
    raw_entries = payload["experts"]
    if not isinstance(raw_entries, list):
        raise PromotedExpertCatalogError(
            "invalid_expert_entries",
            "The promoted expert catalog experts field must be a list.",
        )
    entries = tuple(
        _require_object(value, subject=f"expert entry {index}")
        for index, value in enumerate(raw_entries)
    )
    names = tuple(
        _require_string(entry.get("expert_name"), field="expert name") for entry in entries
    )
    if names != _EXPECTED_EXPERT_NAMES:
        raise PromotedExpertCatalogError(
            "unexpected_expert_roster",
            "The initial promoted expert roster must be exactly aggressive-v3, balanced-v3, "
            "passive-v3, and vector_ppo_large_v1_g350k in that order.",
        )
    if len(set(names)) != len(names):
        raise PromotedExpertCatalogError(
            "duplicate_expert",
            "The promoted expert catalog contains a duplicate expert.",
        )
    experts = tuple(_validate_entry(entry, bot_specs=bot_specs) for entry in entries)
    candidate_identities = tuple(expert.promoted_candidate_identity for expert in experts)
    evidence_digests = tuple(expert.promotion_evidence_digest for expert in experts)
    if len(set(candidate_identities)) != len(candidate_identities):
        raise PromotedExpertCatalogError(
            "duplicate_promoted_candidate",
            "Two expert entries bind the same promoted candidate identity.",
        )
    if len(set(evidence_digests)) != len(evidence_digests):
        raise PromotedExpertCatalogError(
            "duplicate_promotion_evidence",
            "Two expert entries bind the same promotion receipt.",
        )
    _verify_executable_evidence(experts)
    return experts


def _validate_entry(
    entry: dict[str, object],
    *,
    bot_specs: Mapping[str, BotSpec],
) -> PromotedExpert:
    _require_exact_keys(entry, _ENTRY_KEYS, subject="promoted expert entry")
    name = _require_string(entry["expert_name"], field="expert name")
    if name not in bot_specs:
        raise PromotedExpertCatalogError(
            "unregistered_expert",
            f"Promoted expert {name!r} is not a runnable registered BotSpec.",
        )
    spec = bot_specs[name]
    if spec.name != name or spec.bot_id != name:
        raise PromotedExpertCatalogError(
            "non_versioned_expert_identity",
            f"Promoted expert {name!r} must use its explicit local simulation identity.",
        )
    candidate_identity = _require_string(
        entry["promoted_candidate_identity"],
        field=f"promoted candidate identity for {name}",
    )
    incumbent_name = _require_string(
        entry["incumbent_name"],
        field=f"incumbent identity for {name}",
    )
    if incumbent_name != _EXPECTED_INCUMBENTS[name]:
        raise PromotedExpertCatalogError(
            "promotion_incumbent_mismatch",
            f"Promotion evidence for {name!r} does not name its exact predecessor.",
        )
    executable = _require_object(entry["executable"], subject=f"executable evidence for {name}")
    _require_exact_keys(executable, _EXECUTABLE_KEYS, subject=f"executable evidence for {name}")
    kind_value = _require_string(executable["kind"], field=f"executable kind for {name}")
    if kind_value not in ("heuristic_profile", "neural_checkpoint"):
        raise PromotedExpertCatalogError(
            "unknown_executable_kind",
            f"Promoted expert {name!r} has an unknown executable kind.",
        )
    kind = cast(ExecutableKind, kind_value)
    artifact_name = _require_string(
        executable["artifact_name"], field=f"executable artifact name for {name}"
    )
    if artifact_name != candidate_identity:
        raise PromotedExpertCatalogError(
            "executable_identity_mismatch",
            f"Executable evidence for {name!r} does not name its promoted candidate.",
        )
    executable_digest = _require_digest(executable["digest"], field=f"executable digest for {name}")
    artifact_digest = _require_digest(
        executable["artifact_digest"], field=f"executable artifact digest for {name}"
    )

    promotion = _require_object(entry["promotion"], subject=f"promotion evidence for {name}")
    _require_exact_keys(promotion, _PROMOTION_KEYS, subject=f"promotion evidence for {name}")
    expected_evidence_digest = _canonical_digest(
        {
            "candidate_identity": candidate_identity,
            "incumbent_name": incumbent_name,
            "promotion": promotion,
        }
    )
    evidence_digest = _require_digest(
        entry["promotion_evidence_digest"], field=f"promotion evidence digest for {name}"
    )
    if evidence_digest != expected_evidence_digest:
        raise PromotedExpertCatalogError(
            "promotion_evidence_digest_mismatch",
            f"Promotion evidence for {name!r} does not match its digest.",
        )
    _validate_successful_promotion(name, promotion)
    corpora = _require_object(promotion["corpora"], subject=f"promotion corpora for {name}")
    development = _require_named_corpus(corpora, "development", expected=_DEVELOPMENT_CORPUS)
    held_out = _require_named_corpus(corpora, "held_out", expected=_HELD_OUT_CORPUS)
    interval = _require_object(
        promotion["confidence_interval_95"], subject=f"promotion interval for {name}"
    )
    lower = _require_finite_number(interval.get("lower"), field=f"lower interval for {name}")
    upper = _require_finite_number(interval.get("upper"), field=f"upper interval for {name}")
    rating_difference = _require_finite_number(
        promotion["rating_difference"], field=f"rating difference for {name}"
    )
    repository_commit = _require_string(
        promotion["repository_commit"], field=f"promotion repository commit for {name}"
    )
    if _COMMIT_PATTERN.fullmatch(repository_commit) is None:
        raise PromotedExpertCatalogError(
            "invalid_promotion_commit",
            f"Promotion evidence for {name!r} must bind an exact source commit.",
        )
    return PromotedExpert(
        name=name,
        bot_spec=spec,
        promoted_candidate_identity=candidate_identity,
        incumbent_name=incumbent_name,
        promotion_evidence_digest=evidence_digest,
        executable_kind=kind,
        executable_digest=executable_digest,
        executable_artifact_digest=artifact_digest,
        development_corpus_name=development[0],
        development_corpus_digest=development[1],
        held_out_corpus_name=held_out[0],
        held_out_corpus_digest=held_out[1],
        promotion_repository_commit=repository_commit,
        rating_difference=rating_difference,
        confidence_interval_lower=lower,
        confidence_interval_upper=upper,
    )


def _validate_successful_promotion(name: str, promotion: dict[str, object]) -> None:
    if promotion["promoted"] is not True:
        raise PromotedExpertCatalogError(
            "promotion_failed",
            f"Expert {name!r} does not have a successful promotion decision.",
        )
    rating = _require_finite_number(
        promotion["rating_difference"], field=f"rating difference for {name}"
    )
    interval = _require_object(
        promotion["confidence_interval_95"], subject=f"promotion interval for {name}"
    )
    _require_exact_keys(interval, {"lower", "upper"}, subject=f"promotion interval for {name}")
    lower = _require_finite_number(interval["lower"], field=f"lower interval for {name}")
    upper = _require_finite_number(interval["upper"], field=f"upper interval for {name}")
    if not 0.0 < lower <= rating <= upper:
        raise PromotedExpertCatalogError(
            "nonpositive_promotion_interval",
            f"Expert {name!r} must have a positive, ordered 95% promotion interval.",
        )
    bootstrap = _require_object(promotion["bootstrap"], subject=f"bootstrap evidence for {name}")
    _require_exact_keys(
        bootstrap, {"converged", "requested", "seed"}, subject=f"bootstrap evidence for {name}"
    )
    requested = _require_int(bootstrap["requested"], field=f"requested bootstrap fits for {name}")
    converged = _require_int(bootstrap["converged"], field=f"converged bootstrap fits for {name}")
    seed = _require_int(bootstrap["seed"], field=f"bootstrap seed for {name}")
    if (requested, converged, seed) != (1000, 1000, 0):
        raise PromotedExpertCatalogError(
            "incomplete_bootstrap",
            f"Expert {name!r} must bind all 1,000 converged fits from bootstrap seed 0.",
        )
    coverage = _require_object(promotion["coverage"], subject=f"coverage evidence for {name}")
    _require_exact_keys(
        coverage,
        {"completed_games", "completed_pairs", "requested_games", "requested_pairs"},
        subject=f"coverage evidence for {name}",
    )
    coverage_values = tuple(
        _require_int(coverage[field], field=f"{field} for {name}")
        for field in ("completed_pairs", "requested_pairs", "completed_games", "requested_games")
    )
    if coverage_values != (480, 480, 960, 960):
        raise PromotedExpertCatalogError(
            "incomplete_promotion_coverage",
            f"Expert {name!r} must bind all 480 matched pairs and 960 games.",
        )
    if _require_int(promotion["faults"], field=f"fault count for {name}") != 0:
        raise PromotedExpertCatalogError(
            "promotion_faults",
            f"Expert {name!r} has promotion faults.",
        )
    for field in ("failures", "warnings"):
        value = promotion[field]
        if value != []:
            raise PromotedExpertCatalogError(
                f"promotion_{field}",
                f"Expert {name!r} has recorded promotion {field}.",
            )
    if promotion["rule"] != _EXPECTED_RULE:
        raise PromotedExpertCatalogError(
            "promotion_rule_mismatch",
            f"Expert {name!r} was not evaluated under the exact common promotion rule.",
        )
    _require_digest(promotion["source_sha256"], field=f"promotion source digest for {name}")
    source = _require_string(promotion["source_artifact"], field=f"promotion source for {name}")
    if Path(source).is_absolute() or ".." in Path(source).parts:
        raise PromotedExpertCatalogError(
            "invalid_promotion_source",
            f"Expert {name!r} has an unsafe promotion evidence path.",
        )


def _verify_executable_evidence(experts: tuple[PromotedExpert, ...]) -> None:
    frozen_by_identity = {candidate.identity: candidate for candidate in load_frozen_candidates()}
    for expert in experts:
        if expert.executable_kind == "heuristic_profile":
            _verify_heuristic_expert(expert, frozen_by_identity)
        else:
            _verify_neural_expert(expert)


def _verify_heuristic_expert(
    expert: PromotedExpert,
    frozen_by_identity: Mapping[str, FrozenCandidate],
) -> None:
    try:
        frozen = frozen_by_identity[expert.promoted_candidate_identity]
    except KeyError as error:
        raise PromotedExpertCatalogError(
            "missing_promoted_profile",
            f"Promoted expert {expert.name!r} has no packaged frozen profile.",
        ) from error
    if (
        frozen.freeze_digest != expert.executable_artifact_digest
        or frozen.profile_digest != expert.executable_digest
    ):
        raise PromotedExpertCatalogError(
            "promoted_profile_digest_mismatch",
            f"Packaged profile evidence for {expert.name!r} was edited.",
        )
    try:
        expected_spec, released_profile = _EXPECTED_HEURISTIC_EXECUTABLES[expert.name]
    except KeyError as error:
        raise PromotedExpertCatalogError(
            "unexpected_heuristic_identity",
            f"Promoted heuristic identity {expert.name!r} is not in the initial roster.",
        ) from error
    if expert.bot_spec != expected_spec:
        raise PromotedExpertCatalogError(
            "released_botspec_mismatch",
            f"Released identity {expert.name!r} does not bind its versioned brain factory.",
        )
    if released_profile != frozen.profile:
        raise PromotedExpertCatalogError(
            "released_profile_mismatch",
            f"Released identity {expert.name!r} does not run its promoted frozen profile.",
        )


def _verify_neural_expert(expert: PromotedExpert) -> None:
    if expert.promoted_candidate_identity != "vector_ppo_large_v1_g350k":
        raise PromotedExpertCatalogError(
            "unexpected_neural_identity",
            "The initial neural expert must be vector_ppo_large_v1_g350k.",
        )
    if expert.bot_spec != VECTOR_PPO_LARGE_V1_G350K_BOT_SPEC:
        raise PromotedExpertCatalogError(
            "released_botspec_mismatch",
            "The neural expert does not bind the promoted large checkpoint factory.",
        )
    manifest_path = LARGE_CHECKPOINT_PATH / "manifest.json"
    model_path = LARGE_CHECKPOINT_PATH / "model.pt"
    try:
        manifest = _decode_json_object(manifest_path.read_bytes(), subject="neural manifest")
        model_digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    except OSError as error:
        raise PromotedExpertCatalogError(
            "missing_promoted_checkpoint",
            "The promoted neural checkpoint bundle is incomplete.",
        ) from error
    if (
        manifest.get("parameter_digest") != expert.executable_digest
        or manifest.get("model_sha256") != expert.executable_artifact_digest
        or model_digest != expert.executable_artifact_digest
    ):
        raise PromotedExpertCatalogError(
            "promoted_checkpoint_digest_mismatch",
            "The promoted neural checkpoint does not match its eligibility receipt.",
        )


def _verify_retained_sources(
    experts: tuple[PromotedExpert, ...],
    payload: dict[str, object],
) -> None:
    """Verify source receipts in a checkout while keeping installed wheels usable."""

    repository_root = _repository_root()
    if not (repository_root / "pyproject.toml").is_file():
        return
    raw_entries = cast(list[object], payload["experts"])
    for expert, raw_entry in zip(experts, raw_entries, strict=True):
        entry = cast(dict[str, object], raw_entry)
        promotion = cast(dict[str, object], entry["promotion"])
        source_path = repository_root / cast(str, promotion["source_artifact"])
        try:
            actual_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        except OSError as error:
            raise PromotedExpertCatalogError(
                "missing_promotion_source",
                f"The retained promotion source for {expert.name!r} is missing.",
            ) from error
        if actual_digest != promotion["source_sha256"]:
            raise PromotedExpertCatalogError(
                "promotion_source_digest_mismatch",
                f"The retained promotion source for {expert.name!r} was edited.",
            )


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _require_named_corpus(
    corpora: dict[str, object],
    purpose: str,
    *,
    expected: tuple[str, str],
) -> tuple[str, str]:
    _require_exact_keys(corpora, {"development", "held_out"}, subject="promotion corpora")
    corpus = _require_object(corpora[purpose], subject=f"{purpose} promotion corpus")
    _require_exact_keys(corpus, {"digest", "name"}, subject=f"{purpose} promotion corpus")
    actual = (
        _require_string(corpus["name"], field=f"{purpose} corpus name"),
        _require_digest(corpus["digest"], field=f"{purpose} corpus digest"),
    )
    if actual != expected:
        raise PromotedExpertCatalogError(
            "promotion_corpus_mismatch",
            f"The {purpose} corpus does not match the pinned common promotion corpus.",
        )
    return actual


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _decode_json_object(raw: bytes, *, subject: str) -> dict[str, object]:
    try:
        value: object = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PromotedExpertCatalogError(
            "invalid_json",
            f"The {subject} is not valid JSON.",
        ) from error
    return _require_object(value, subject=subject)


def _require_object(value: object, *, subject: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise PromotedExpertCatalogError("invalid_object", f"The {subject} must be an object.")
    return cast(dict[str, object], value)


def _require_exact_keys(value: Mapping[str, object], expected: set[str], *, subject: str) -> None:
    if set(value) != expected:
        raise PromotedExpertCatalogError(
            "unexpected_fields",
            f"The {subject} must contain exactly {sorted(expected)!r}.",
        )


def _require_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise PromotedExpertCatalogError("invalid_string", f"The {field} must be a string.")
    return value


def _require_digest(value: object, *, field: str) -> str:
    digest = _require_string(value, field=field)
    if _DIGEST_PATTERN.fullmatch(digest) is None:
        raise PromotedExpertCatalogError("invalid_digest", f"The {field} must be a SHA-256 digest.")
    return digest


def _require_int(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise PromotedExpertCatalogError("invalid_integer", f"The {field} must be an integer.")
    return value


def _require_finite_number(value: object, *, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise PromotedExpertCatalogError("invalid_number", f"The {field} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise PromotedExpertCatalogError("nonfinite_number", f"The {field} must be finite.")
    return result


def promoted_experts_by_name(
    catalog: VerifiedPromotedExpertCatalog | None = None,
) -> Mapping[str, PromotedExpert]:
    """Return an immutable lookup derived only from the exact pinned catalog."""

    verified = load_promoted_experts() if catalog is None else catalog
    roster = _require_verified_catalog(verified)
    return MappingProxyType({expert.name: expert for expert in roster})


def _require_verified_catalog(
    catalog: VerifiedPromotedExpertCatalog,
) -> tuple[PromotedExpert, ...]:
    if type(catalog) is not VerifiedPromotedExpertCatalog:
        raise TypeError("expert selection requires the exact pinned verified catalog")
    return catalog._verified_experts()
