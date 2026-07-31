"""Rank deterministic candidate pools and record every selection decision."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal

from garboid_pocketrocks.evolution.candidates import (
    HeuristicCandidate,
    PhaseAwareHeuristicCandidate,
    SearchCandidate,
)
from garboid_pocketrocks.evolution.evaluation import CandidateEvaluation
from garboid_pocketrocks.heuristics.phases import PHASE_SELECTOR_NAME


@dataclass(frozen=True, slots=True, order=True)
class CandidateRankingKey:
    """The exact ascending key implementing the documented fitness order."""

    negative_rating_delta: float
    negative_normalized_finish_delta: float
    negative_final_money_delta: int
    coefficient_values: tuple[Decimal, ...]
    candidate_identity: str

    def as_tuple(
        self,
    ) -> tuple[
        float,
        float,
        int,
        tuple[Decimal, ...],
        str,
    ]:
        """Return the canonical ranking fields in comparison order."""

        return (
            self.negative_rating_delta,
            self.negative_normalized_finish_delta,
            self.negative_final_money_delta,
            self.coefficient_values,
            self.candidate_identity,
        )


@dataclass(frozen=True, slots=True)
class EvaluatedCandidate:
    """One proposal paired with its complete development evidence."""

    candidate: SearchCandidate
    evaluation: CandidateEvaluation

    @property
    def ranking_key(self) -> CandidateRankingKey:
        """Return the exact deterministic selection key."""

        return candidate_ranking_key(self)


@dataclass(frozen=True, slots=True)
class SelectionRecord:
    """The complete inputs, ranking, and elite decision for one generation."""

    generation: int
    proposal_identities: tuple[str, ...]
    pool_identities: tuple[str, ...]
    ranked_pool_identities: tuple[str, ...]
    elite_identities: tuple[str, ...]
    ranking_keys: tuple[tuple[str, CandidateRankingKey], ...]


@dataclass(frozen=True, slots=True)
class GenerationSelection:
    """The ranked pool, selected elites, and immutable decision record."""

    ranked_pool: tuple[EvaluatedCandidate, ...]
    elites: tuple[EvaluatedCandidate, ...]
    record: SelectionRecord


class SearchSelectionError(ValueError):
    """Explain why a generation cannot make a trustworthy selection."""

    code: str

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def candidate_ranking_key(item: EvaluatedCandidate) -> CandidateRankingKey:
    """Build the exact ascending key for one valid scored candidate."""

    _require_candidate_invariants(item.candidate)
    _require_matching_identity(item)
    evaluation = item.evaluation
    if (
        not evaluation.valid
        or evaluation.rating_delta is None
        or evaluation.normalized_finish_delta is None
        or evaluation.final_money_delta is None
        or not math.isfinite(evaluation.rating_delta)
        or not math.isfinite(evaluation.normalized_finish_delta)
    ):
        raise SearchSelectionError(
            "invalid_candidate_evidence",
            f"Candidate {item.candidate.identity!r} does not have complete finite evidence.",
        )
    return CandidateRankingKey(
        negative_rating_delta=-evaluation.rating_delta,
        negative_normalized_finish_delta=-evaluation.normalized_finish_delta,
        negative_final_money_delta=-evaluation.final_money_delta,
        coefficient_values=_candidate_coefficient_values(item.candidate),
        candidate_identity=item.candidate.identity,
    )


def rank_candidate_pool(
    pool: tuple[EvaluatedCandidate, ...],
) -> tuple[EvaluatedCandidate, ...]:
    """Rank every separate proposal without deduplicating repeated genomes."""

    if not pool:
        raise SearchSelectionError(
            "empty_candidate_pool",
            "A generation candidate pool must not be empty.",
        )
    _require_compatible_candidate_pool(pool)
    identities = tuple(item.candidate.identity for item in pool)
    if len(set(identities)) != len(identities):
        raise SearchSelectionError(
            "duplicate_candidate_identity",
            "A generation candidate pool contains duplicate candidate identities.",
        )
    return tuple(sorted(pool, key=candidate_ranking_key))


def select_generation(
    *,
    generation: int,
    proposals: tuple[EvaluatedCandidate, ...],
    prior_elites: tuple[EvaluatedCandidate, ...],
    elite_count: int,
) -> GenerationSelection:
    """Rank generation zero or one deterministic mu-plus-lambda pool."""

    if generation < 0:
        raise ValueError("generation must be nonnegative")
    if elite_count <= 0:
        raise ValueError("elite count must be positive")
    if not proposals:
        raise SearchSelectionError(
            "empty_generation",
            f"Generation {generation} has no evaluated proposals.",
        )
    if generation == 0 and prior_elites:
        raise SearchSelectionError(
            "unexpected_prior_elites",
            "Generation zero cannot include prior elites.",
        )
    if generation > 0 and not prior_elites:
        raise SearchSelectionError(
            "missing_prior_elites",
            f"Generation {generation} requires the prior ranked elites.",
        )
    if any(item.candidate.generation != generation for item in proposals):
        raise SearchSelectionError(
            "proposal_generation_mismatch",
            f"Generation {generation} contains a proposal from another generation.",
        )
    if any(item.candidate.generation >= generation for item in prior_elites):
        raise SearchSelectionError(
            "prior_elite_generation_mismatch",
            f"Generation {generation} contains an elite that is not from an earlier generation.",
        )

    pool = (*prior_elites, *proposals)
    ranked = rank_candidate_pool(pool)
    eligible = tuple(item for item in ranked if item.evaluation.eligible)
    if len(eligible) < elite_count:
        raise SearchSelectionError(
            "insufficient_eligible_candidates",
            f"Generation {generation} has {len(eligible)} eligible candidate(s); "
            f"{elite_count} are required.",
        )
    elites = eligible[:elite_count]
    record = SelectionRecord(
        generation=generation,
        proposal_identities=tuple(item.candidate.identity for item in proposals),
        pool_identities=tuple(item.candidate.identity for item in pool),
        ranked_pool_identities=tuple(item.candidate.identity for item in ranked),
        elite_identities=tuple(item.candidate.identity for item in elites),
        ranking_keys=tuple((item.candidate.identity, item.ranking_key) for item in ranked),
    )
    return GenerationSelection(
        ranked_pool=ranked,
        elites=elites,
        record=record,
    )


def freeze_candidate(item: EvaluatedCandidate) -> SearchCandidate | None:
    """Return the winner only when development evidence permits a freeze."""

    _require_candidate_invariants(item.candidate)
    _require_matching_identity(item)
    evaluation = item.evaluation
    if (
        not evaluation.valid
        or not evaluation.eligible
        or evaluation.rating_delta is None
        or evaluation.rating_delta <= 0.0
        or not math.isfinite(evaluation.rating_delta)
        or evaluation.normalized_finish_delta is None
        or not math.isfinite(evaluation.normalized_finish_delta)
        or evaluation.final_money_delta is None
        or evaluation.requested_cases <= 0
        or evaluation.completed_baseline_games != evaluation.requested_cases
        or evaluation.completed_candidate_games != evaluation.requested_cases
        or evaluation.candidate_faults != 0
        or evaluation.incumbent_faults != 0
        or evaluation.opponent_faults != 0
        or evaluation.unattributed_faults != 0
        or evaluation.faults_by_identity
    ):
        return None
    return item.candidate


def _require_matching_identity(item: EvaluatedCandidate) -> None:
    if item.evaluation.candidate_identity != item.candidate.identity:
        raise SearchSelectionError(
            "candidate_identity_mismatch",
            f"Evaluation identity {item.evaluation.candidate_identity!r} does not "
            f"match proposal {item.candidate.identity!r}.",
        )


def _require_compatible_candidate_pool(
    pool: tuple[EvaluatedCandidate, ...],
) -> None:
    candidates = tuple(item.candidate for item in pool)
    candidate_families = {type(candidate) for candidate in candidates}
    if len(candidate_families) != 1:
        raise SearchSelectionError(
            "mixed_candidate_family",
            "A candidate pool cannot mix scalar and phase-aware policies.",
        )

    personalities = {candidate.personality for candidate in candidates}
    if len(personalities) != 1:
        raise SearchSelectionError(
            "mixed_candidate_personality",
            "Every candidate in one pool must use the same personality.",
        )

    if isinstance(candidates[0], PhaseAwareHeuristicCandidate):
        selectors = {
            candidate.genome.phase_selector
            for candidate in candidates
            if isinstance(candidate, PhaseAwareHeuristicCandidate)
        }
        if len(selectors) != 1:
            raise SearchSelectionError(
                "mixed_candidate_phase_selector",
                "Every phase-aware candidate in one pool must use the same selector.",
            )

    for candidate in candidates:
        _require_candidate_invariants(candidate)


def _require_candidate_invariants(candidate: SearchCandidate) -> None:
    if candidate.personality not in ("aggressive", "balanced", "passive"):
        raise SearchSelectionError(
            "invalid_candidate_personality",
            f"Candidate {candidate.identity!r} has an unknown personality.",
        )
    if (
        isinstance(candidate, PhaseAwareHeuristicCandidate)
        and candidate.genome.phase_selector != PHASE_SELECTOR_NAME
    ):
        raise SearchSelectionError(
            "invalid_candidate_phase_selector",
            f"Candidate {candidate.identity!r} does not use the fixed phase selector.",
        )


def _candidate_coefficient_values(candidate: SearchCandidate) -> tuple[Decimal, ...]:
    """Return every policy value in the candidate's canonical genome order."""

    if isinstance(candidate, PhaseAwareHeuristicCandidate):
        return candidate.genome.experts.as_loci()
    if isinstance(candidate, HeuristicCandidate):
        return candidate.genome.coefficients.as_tuple()
    raise TypeError(f"unsupported candidate type {type(candidate).__name__}")
