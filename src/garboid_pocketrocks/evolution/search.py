"""Rank deterministic candidate pools and record every selection decision."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal

from garboid_pocketrocks.evolution.candidates import HeuristicCandidate
from garboid_pocketrocks.evolution.evaluation import CandidateEvaluation


@dataclass(frozen=True, slots=True, order=True)
class CandidateRankingKey:
    """The exact ascending key implementing the documented fitness order."""

    negative_worst_challenger_finish_delta: float
    negative_rating_delta: float
    negative_normalized_finish_delta: float
    negative_final_money_delta: int
    coefficient_values: tuple[Decimal, Decimal, Decimal, Decimal]
    candidate_identity: str

    def as_tuple(
        self,
    ) -> tuple[
        float,
        float,
        float,
        int,
        tuple[Decimal, Decimal, Decimal, Decimal],
        str,
    ]:
        """Return the canonical ranking fields in comparison order."""

        return (
            self.negative_worst_challenger_finish_delta,
            self.negative_rating_delta,
            self.negative_normalized_finish_delta,
            self.negative_final_money_delta,
            self.coefficient_values,
            self.candidate_identity,
        )


@dataclass(frozen=True, slots=True)
class EvaluatedCandidate:
    """One proposal paired with its complete development evidence."""

    candidate: HeuristicCandidate
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

    _require_matching_identity(item)
    evaluation = item.evaluation
    if (
        not evaluation.valid
        or evaluation.worst_challenger_finish_delta is None
        or evaluation.rating_delta is None
        or evaluation.normalized_finish_delta is None
        or evaluation.final_money_delta is None
        or not math.isfinite(evaluation.worst_challenger_finish_delta)
        or not math.isfinite(evaluation.rating_delta)
        or not math.isfinite(evaluation.normalized_finish_delta)
    ):
        raise SearchSelectionError(
            "invalid_candidate_evidence",
            f"Candidate {item.candidate.identity!r} does not have complete finite evidence.",
        )
    return CandidateRankingKey(
        negative_worst_challenger_finish_delta=-evaluation.worst_challenger_finish_delta,
        negative_rating_delta=-evaluation.rating_delta,
        negative_normalized_finish_delta=-evaluation.normalized_finish_delta,
        negative_final_money_delta=-evaluation.final_money_delta,
        coefficient_values=item.candidate.genome.coefficients.as_tuple(),
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


def freeze_candidate(item: EvaluatedCandidate) -> HeuristicCandidate | None:
    """Return the winner only when development evidence permits a freeze."""

    _require_matching_identity(item)
    evaluation = item.evaluation
    if (
        not evaluation.valid
        or not evaluation.eligible
        or evaluation.worst_challenger_finish_delta is None
        or evaluation.worst_challenger_finish_delta <= 0.0
        or not math.isfinite(evaluation.worst_challenger_finish_delta)
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
