"""Resolve promotion candidates without widening the released bot registry."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from typing import Protocol

from garboid_pocketrocks.bots import BOT_SPECS_BY_NAME, BotSpec
from garboid_pocketrocks.promotion.corpus import (
    PromotionCorpus,
    recompute_promotion_corpus_digest,
)

_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")


class FrozenCandidateRecord(Protocol):
    """The promotion-facing fields supplied by the frozen candidate catalog."""

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
class ResolvedPromotionCandidate:
    """A runnable candidate and optional evidence from deterministic search."""

    bot_spec: BotSpec
    frozen_provenance: FrozenCandidateProvenance | None


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
    if frozen.bot_spec.name != name or frozen.bot_spec.bot_id != name:
        raise PromotionCandidateError(
            "frozen candidate catalog keys, names, and simulation bot IDs must match exactly"
        )

    provenance = _provenance_from_frozen_candidate(frozen)
    _validate_frozen_provenance(provenance)
    return ResolvedPromotionCandidate(
        bot_spec=frozen.bot_spec,
        frozen_provenance=provenance,
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


def _provenance_from_frozen_candidate(
    frozen: FrozenCandidateRecord,
) -> FrozenCandidateProvenance:
    return FrozenCandidateProvenance(
        candidate_name=frozen.bot_spec.name,
        candidate_bot_id=frozen.bot_spec.bot_id,
        predecessor_name=frozen.predecessor_name,
        development_corpus_name=frozen.development_corpus_name,
        development_corpus_digest=frozen.development_corpus_digest,
        search_name=frozen.search_name,
        repository_commit=frozen.repository_commit,
        freeze_digest=frozen.freeze_digest,
        profile_digest=frozen.profile_digest,
        manifest_digest=frozen.manifest_digest,
        search_report_digest=frozen.search_report_digest,
        candidate_evaluations_digest=frozen.candidate_evaluations_digest,
    )


def _validate_frozen_provenance(provenance: FrozenCandidateProvenance) -> None:
    if type(provenance) is not FrozenCandidateProvenance:
        raise PromotionCandidateError(
            "frozen candidate provenance must use the exact provenance type"
        )
    non_builtin_strings = tuple(
        field.name
        for field in fields(provenance)
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
