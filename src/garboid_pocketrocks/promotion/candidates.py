"""Resolve promotion candidates without widening the released bot registry."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields
from math import isfinite
from typing import Protocol

from garboid_pocketrocks.bots import BOT_SPECS_BY_NAME, BotSpec
from garboid_pocketrocks.heuristics.phases import HeuristicPhase
from garboid_pocketrocks.promotion.corpus import (
    PromotionCorpus,
    recompute_promotion_corpus_digest,
)

_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
_PHASE_NAMES: tuple[HeuristicPhase, ...] = ("early", "middle", "late")
_COEFFICIENT_NAMES = (
    "liquidity_strength",
    "future_cash_weight",
    "objective_progress_weight",
    "bid_shading",
)
_PHASE_SELECTOR_NAMES = ("kind", "early", "middle", "late")
_WINNER_DIAGNOSTIC_NAMES = (
    "winner-decision-slices.csv",
    "winner-diagnostics.json",
    "winner-diagnostics.md",
)


class FrozenCandidateRecord(Protocol):
    """The promotion-facing fields supplied by the frozen candidate catalog."""

    @property
    def identity(self) -> str: ...

    @property
    def bot_spec(self) -> BotSpec: ...

    @property
    def predecessor_name(self) -> str: ...

    @property
    def development_corpus_name(self) -> str: ...

    @property
    def development_corpus_digest(self) -> str: ...

    @property
    def search_name(self) -> str: ...

    @property
    def repository_commit(self) -> str: ...

    @property
    def freeze_digest(self) -> str: ...

    @property
    def profile_digest(self) -> str: ...

    @property
    def manifest_digest(self) -> str: ...

    @property
    def search_report_digest(self) -> str: ...

    @property
    def candidate_evaluations_digest(self) -> str: ...


@dataclass(frozen=True, slots=True)
class FrozenCandidateProvenance:
    """Content digests and bindings carried from search into promotion."""

    candidate_name: str
    candidate_bot_id: str
    predecessor_name: str
    development_corpus_name: str
    development_corpus_digest: str
    search_name: str
    repository_commit: str
    freeze_digest: str
    profile_digest: str
    manifest_digest: str
    search_report_digest: str
    candidate_evaluations_digest: str


@dataclass(frozen=True, slots=True)
class FrozenPhaseAwareCandidateProvenance(FrozenCandidateProvenance):
    """Complete frozen search evidence for one phase-aware candidate."""

    freeze_schema_version: int
    personality: str
    phase_selector_rules: tuple[tuple[str, str], ...]
    expert_profiles: tuple[tuple[str, tuple[tuple[str, float], ...]], ...]
    expert_digests: tuple[tuple[str, str], ...]
    boundary_report_path: str
    boundary_report_digest: str
    boundary_slices_path: str
    boundary_slices_digest: str
    selection_log_digest: str
    development_games_digest: str
    winner_diagnostics_digests: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class ResolvedPromotionCandidate:
    """A runnable candidate and optional evidence from deterministic search."""

    bot_spec: BotSpec
    frozen_provenance: FrozenCandidateProvenance | None
    _source_frozen_candidate: FrozenCandidateRecord | None = field(
        default=None,
        compare=False,
        repr=False,
    )


class PromotionCandidateError(ValueError):
    """Explain why a requested promotion candidate cannot be trusted."""


def load_frozen_candidate_catalog() -> Mapping[str, FrozenCandidateRecord]:
    """Load the trusted catalog inside the caller's operational error boundary."""

    from garboid_pocketrocks.heuristics.frozen import FROZEN_CANDIDATES_BY_NAME

    return FROZEN_CANDIDATES_BY_NAME


def resolve_promotion_candidate(
    name: str,
    *,
    registry: Mapping[str, BotSpec],
    frozen_candidates: Mapping[str, FrozenCandidateRecord],
) -> ResolvedPromotionCandidate:
    """Resolve an exact released or frozen candidate name."""

    registered = registry.get(name)
    if registered is not None:
        return ResolvedPromotionCandidate(
            bot_spec=registered,
            frozen_provenance=None,
        )

    frozen = frozen_candidates.get(name)
    if frozen is None:
        raise PromotionCandidateError(f"unknown bot name: {name}")
    frozen_identity = getattr(frozen, "identity", None)
    if (
        type(frozen_identity) is not str
        or frozen_identity != name
        or frozen.bot_spec.name != name
        or frozen.bot_spec.bot_id != name
    ):
        raise PromotionCandidateError(
            "frozen candidate catalog keys, identities, names, and simulation bot IDs "
            "must match exactly"
        )

    provenance = _provenance_from_frozen_candidate(frozen)
    _validate_frozen_provenance(provenance)
    return ResolvedPromotionCandidate(
        bot_spec=frozen.bot_spec,
        frozen_provenance=provenance,
        _source_frozen_candidate=frozen,
    )


def validate_promotion_candidate(
    candidate: ResolvedPromotionCandidate,
    *,
    incumbent: BotSpec,
    development: PromotionCorpus,
    registry: Mapping[str, BotSpec] | None,
) -> None:
    """Bind a compared candidate to one exact trusted registry or catalog record."""

    del registry  # The caller's scheduling registry is never an identity authority.
    provenance = candidate.frozen_provenance
    canonical_released = BOT_SPECS_BY_NAME.get(candidate.bot_spec.name)
    if provenance is None and canonical_released is not None:
        if canonical_released is not candidate.bot_spec:
            raise PromotionCandidateError(
                "the registered candidate must be the canonical released registry spec"
            )
        return

    trusted_candidates = load_frozen_candidate_catalog()
    if provenance is None:
        if candidate.bot_spec.name not in trusted_candidates:
            return
        raise PromotionCandidateError(
            "a frozen candidate must carry provenance from the trusted catalog"
        )
    _validate_frozen_provenance(provenance)
    if (
        provenance.candidate_name != candidate.bot_spec.name
        or provenance.candidate_bot_id != candidate.bot_spec.bot_id
    ):
        raise PromotionCandidateError("the frozen candidate identity does not match its provenance")

    trusted = trusted_candidates.get(provenance.candidate_name)
    if trusted is None:
        raise PromotionCandidateError(
            "the candidate is absent from the trusted frozen candidate catalog"
        )
    if (
        candidate._source_frozen_candidate is not None
        and trusted is not candidate._source_frozen_candidate
    ):
        raise PromotionCandidateError(
            "the resolved source does not match the exact trusted frozen candidate record"
        )
    if trusted.bot_spec is not candidate.bot_spec:
        raise PromotionCandidateError(
            "the candidate BotSpec does not match the trusted frozen candidate record"
        )
    if provenance != _provenance_from_frozen_candidate(trusted):
        raise PromotionCandidateError(
            "the provenance does not match the trusted frozen candidate record"
        )

    canonical_predecessor = BOT_SPECS_BY_NAME.get(provenance.predecessor_name)
    if incumbent is not canonical_predecessor:
        raise PromotionCandidateError(
            "the frozen candidate requires its exact canonical predecessor"
        )
    if development.recipe.purpose != "development":
        raise PromotionCandidateError("the frozen candidate requires a development corpus")
    recomputed_digest = recompute_promotion_corpus_digest(development)
    if recomputed_digest != development.digest:
        raise PromotionCandidateError(
            "the development corpus stored digest does not match its recipe and cases"
        )
    if (
        provenance.development_corpus_name != development.recipe.name
        or provenance.development_corpus_digest != development.digest
    ):
        raise PromotionCandidateError(
            "the frozen candidate development corpus does not match the invoked corpus"
        )


def validate_frozen_promotion_opponents(
    provenance: FrozenCandidateProvenance | None,
    opponents: Iterable[BotSpec | None],
    *,
    required_names: Iterable[str],
) -> None:
    """Require exact released opponent objects for a frozen candidate."""

    if provenance is None:
        return
    resolved_opponents = tuple(opponents)
    reported_names = tuple(opponent.name for opponent in resolved_opponents if opponent is not None)
    required_name_tuple = tuple(required_names)
    if (
        len(reported_names) != len(set(reported_names))
        or len(required_name_tuple) != len(set(required_name_tuple))
        or set(reported_names) != set(required_name_tuple)
    ):
        raise PromotionCandidateError(
            "frozen candidate opponents must exactly cover the held-out opponent names"
        )
    for opponent in resolved_opponents:
        if opponent is None or BOT_SPECS_BY_NAME.get(opponent.name) is not opponent:
            raise PromotionCandidateError(
                "a frozen candidate requires every exact canonical released opponent"
            )


def validate_frozen_candidate_provenance(
    provenance: FrozenCandidateProvenance,
) -> None:
    """Validate the exact public shape of frozen promotion provenance."""

    _validate_frozen_provenance(provenance)


def _provenance_from_frozen_candidate(
    frozen: FrozenCandidateRecord,
) -> FrozenCandidateProvenance:
    common_fields = {
        "candidate_name": frozen.bot_spec.name,
        "candidate_bot_id": frozen.bot_spec.bot_id,
        "predecessor_name": frozen.predecessor_name,
        "development_corpus_name": frozen.development_corpus_name,
        "development_corpus_digest": frozen.development_corpus_digest,
        "search_name": frozen.search_name,
        "repository_commit": frozen.repository_commit,
        "freeze_digest": frozen.freeze_digest,
        "profile_digest": frozen.profile_digest,
        "manifest_digest": frozen.manifest_digest,
        "search_report_digest": frozen.search_report_digest,
        "candidate_evaluations_digest": frozen.candidate_evaluations_digest,
    }

    # Import lazily so malformed catalog data remains inside the CLI's
    # operational error boundary.
    from garboid_pocketrocks.heuristics.frozen import FrozenPhaseAwareCandidate

    if type(frozen) is FrozenPhaseAwareCandidate:
        profile = frozen.profile
        expert_profiles = tuple(
            (
                phase,
                tuple(
                    (coefficient, getattr(profile.profile_for_phase(phase), coefficient))
                    for coefficient in _COEFFICIENT_NAMES
                ),
            )
            for phase in _PHASE_NAMES
        )
        return FrozenPhaseAwareCandidateProvenance(
            **common_fields,
            freeze_schema_version=2,
            personality=frozen.personality,
            phase_selector_rules=frozen.phase_selector_rules,
            expert_profiles=expert_profiles,
            expert_digests=frozen.expert_digests,
            boundary_report_path=frozen.boundary_report_path,
            boundary_report_digest=frozen.boundary_report_digest,
            boundary_slices_path=frozen.boundary_slices_path,
            boundary_slices_digest=frozen.boundary_slices_digest,
            selection_log_digest=frozen.selection_log_digest,
            development_games_digest=frozen.development_games_digest,
            winner_diagnostics_digests=frozen.winner_diagnostics_digests,
        )

    return FrozenCandidateProvenance(
        **common_fields,
    )


def _validate_frozen_provenance(provenance: FrozenCandidateProvenance) -> None:
    if type(provenance) not in (
        FrozenCandidateProvenance,
        FrozenPhaseAwareCandidateProvenance,
    ):
        raise PromotionCandidateError(
            "frozen candidate provenance must use the exact provenance type"
        )
    non_builtin_strings = tuple(
        field.name
        for field in fields(FrozenCandidateProvenance)
        if type(getattr(provenance, field.name)) is not str
    )
    if non_builtin_strings:
        raise PromotionCandidateError(
            "frozen candidate provenance fields must be built-in strings: "
            + ", ".join(non_builtin_strings)
        )
    named_fields = {
        "candidate": provenance.candidate_name,
        "candidate bot ID": provenance.candidate_bot_id,
        "predecessor": provenance.predecessor_name,
        "development corpus": provenance.development_corpus_name,
        "search": provenance.search_name,
    }
    invalid_names = tuple(name for name, value in named_fields.items() if not value)
    if invalid_names:
        raise PromotionCandidateError(
            "frozen candidate provenance has empty names for " + ", ".join(invalid_names)
        )
    digests = {
        "development corpus": provenance.development_corpus_digest,
        "freeze": provenance.freeze_digest,
        "profile": provenance.profile_digest,
        "manifest": provenance.manifest_digest,
        "search report": provenance.search_report_digest,
        "candidate evaluations": provenance.candidate_evaluations_digest,
    }
    invalid_digests = tuple(
        name
        for name, digest in digests.items()
        if not isinstance(digest, str) or _DIGEST_PATTERN.fullmatch(digest) is None
    )
    if invalid_digests:
        raise PromotionCandidateError(
            "frozen candidate provenance has invalid SHA-256 digests for "
            + ", ".join(invalid_digests)
        )
    if _COMMIT_PATTERN.fullmatch(provenance.repository_commit) is None:
        raise PromotionCandidateError(
            "frozen candidate provenance has an invalid repository commit"
        )
    if type(provenance) is FrozenPhaseAwareCandidateProvenance:
        _validate_phase_aware_provenance(provenance)


def _validate_phase_aware_provenance(
    provenance: FrozenPhaseAwareCandidateProvenance,
) -> None:
    """Validate the exact primitive shape carried across promotion boundaries."""

    if type(provenance.freeze_schema_version) is not int or provenance.freeze_schema_version != 2:
        raise PromotionCandidateError(
            "phase-aware candidate provenance must use freeze schema version 2"
        )
    if type(provenance.personality) is not str or provenance.personality not in (
        "aggressive",
        "balanced",
        "passive",
    ):
        raise PromotionCandidateError("phase-aware candidate provenance has an invalid personality")
    _validate_named_string_pairs(
        provenance.phase_selector_rules,
        expected_names=_PHASE_SELECTOR_NAMES,
        subject="phase selector rules",
        digest_values=False,
    )
    _validate_named_string_pairs(
        provenance.expert_digests,
        expected_names=_PHASE_NAMES,
        subject="expert digests",
        digest_values=True,
    )
    _validate_expert_profiles(provenance.expert_profiles)

    path_fields = {
        "boundary report path": provenance.boundary_report_path,
        "boundary slices path": provenance.boundary_slices_path,
    }
    if any(type(value) is not str or not value for value in path_fields.values()):
        raise PromotionCandidateError(
            "phase-aware candidate provenance has invalid boundary evidence paths"
        )
    phase_digests = {
        "boundary report": provenance.boundary_report_digest,
        "boundary slices": provenance.boundary_slices_digest,
        "selection log": provenance.selection_log_digest,
        "development games": provenance.development_games_digest,
    }
    if any(
        type(digest) is not str or _DIGEST_PATTERN.fullmatch(digest) is None
        for digest in phase_digests.values()
    ):
        raise PromotionCandidateError(
            "phase-aware candidate provenance has invalid evidence digests"
        )
    _validate_named_string_pairs(
        provenance.winner_diagnostics_digests,
        expected_names=_WINNER_DIAGNOSTIC_NAMES,
        subject="winner diagnostics digests",
        digest_values=True,
    )


def _validate_named_string_pairs(
    value: object,
    *,
    expected_names: tuple[str, ...],
    subject: str,
    digest_values: bool,
) -> None:
    if (
        type(value) is not tuple
        or tuple(
            item[0]
            for item in value
            if type(item) is tuple and len(item) == 2 and type(item[0]) is str
        )
        != expected_names
        or len(value) != len(expected_names)
        or any(
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not str
            or not item[1]
            for item in value
        )
    ):
        raise PromotionCandidateError(f"phase-aware candidate provenance has invalid {subject}")
    if digest_values and any(_DIGEST_PATTERN.fullmatch(item[1]) is None for item in value):
        raise PromotionCandidateError(f"phase-aware candidate provenance has invalid {subject}")


def _validate_expert_profiles(value: object) -> None:
    if type(value) is not tuple or len(value) != len(_PHASE_NAMES):
        raise PromotionCandidateError(
            "phase-aware candidate provenance has invalid expert profiles"
        )
    for phase_index, phase_entry in enumerate(value):
        if (
            type(phase_entry) is not tuple
            or len(phase_entry) != 2
            or phase_entry[0] != _PHASE_NAMES[phase_index]
            or type(phase_entry[0]) is not str
            or type(phase_entry[1]) is not tuple
            or len(phase_entry[1]) != len(_COEFFICIENT_NAMES)
        ):
            raise PromotionCandidateError(
                "phase-aware candidate provenance has invalid expert profiles"
            )
        for coefficient_index, coefficient_entry in enumerate(phase_entry[1]):
            if (
                type(coefficient_entry) is not tuple
                or len(coefficient_entry) != 2
                or coefficient_entry[0] != _COEFFICIENT_NAMES[coefficient_index]
                or type(coefficient_entry[0]) is not str
                or type(coefficient_entry[1]) is not float
                or not isfinite(coefficient_entry[1])
            ):
                raise PromotionCandidateError(
                    "phase-aware candidate provenance has invalid expert profiles"
                )
