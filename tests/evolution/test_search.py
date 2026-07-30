from __future__ import annotations

from decimal import Decimal

import pytest

from garboid_pocketrocks.evolution.candidates import (
    CoefficientGenome,
    HeuristicCandidate,
)
from garboid_pocketrocks.evolution.evaluation import (
    CandidateEvaluation,
    EvaluationFailure,
)
from garboid_pocketrocks.evolution.manifest import CoefficientValues
from garboid_pocketrocks.evolution.search import (
    EvaluatedCandidate,
    SearchSelectionError,
    freeze_candidate,
    rank_candidate_pool,
    select_generation,
)


def test_ranks_by_exact_fitness_then_coefficients_and_identity() -> None:
    rating_winner = _evaluated(0, 0, "0.40", rating=11.0, finish=0.0, money=0)
    finish_winner = _evaluated(0, 1, "0.45", rating=10.0, finish=2.0, money=0)
    money_winner = _evaluated(0, 2, "0.50", rating=10.0, finish=1.0, money=20)
    coefficient_winner = _evaluated(0, 3, "0.10", rating=10.0, finish=1.0, money=10)
    identity_winner = _evaluated(0, 4, "0.20", rating=10.0, finish=1.0, money=10)
    identity_loser = _evaluated(1, 0, "0.20", rating=10.0, finish=1.0, money=10)
    ineligible = _evaluated(
        0,
        5,
        "0.00",
        rating=999.0,
        finish=999.0,
        money=999,
        eligible=False,
    )

    ranked = rank_candidate_pool(
        (
            ineligible,
            identity_loser,
            money_winner,
            identity_winner,
            rating_winner,
            coefficient_winner,
            finish_winner,
        )
    )

    assert tuple(item.candidate.identity for item in ranked) == (
        ineligible.candidate.identity,
        rating_winner.candidate.identity,
        finish_winner.candidate.identity,
        money_winner.candidate.identity,
        coefficient_winner.candidate.identity,
        identity_winner.candidate.identity,
        identity_loser.candidate.identity,
    )
    assert rating_winner.ranking_key.as_tuple() == (
        -11.0,
        -0.0,
        0,
        rating_winner.candidate.genome.coefficients.as_tuple(),
        rating_winner.candidate.identity,
    )


def test_mu_plus_lambda_selection_records_the_complete_pool_and_elites() -> None:
    initial = tuple(
        _evaluated(0, slot, f"0.{slot}", rating=float(slot), finish=0.0, money=0)
        for slot in range(4)
    )

    first = select_generation(
        generation=0,
        proposals=initial,
        prior_elites=(),
        elite_count=2,
    )

    assert tuple(item.candidate.slot for item in first.elites) == (3, 2)
    assert first.record.generation == 0
    assert first.record.proposal_identities == tuple(item.candidate.identity for item in initial)
    assert first.record.pool_identities == first.record.proposal_identities
    assert first.record.ranked_pool_identities == tuple(
        item.candidate.identity for item in first.ranked_pool
    )
    assert first.record.elite_identities == tuple(item.candidate.identity for item in first.elites)
    assert tuple(identity for identity, _key in first.record.ranking_keys) == (
        first.record.ranked_pool_identities
    )

    children = tuple(
        _evaluated(1, slot, f"0.{slot + 4}", rating=float(slot + 1), finish=0.0, money=0)
        for slot in range(4)
    )
    second = select_generation(
        generation=1,
        proposals=children,
        prior_elites=first.elites,
        elite_count=2,
    )

    assert second.record.proposal_identities == tuple(item.candidate.identity for item in children)
    assert second.record.pool_identities == (
        *(item.candidate.identity for item in first.elites),
        *(item.candidate.identity for item in children),
    )
    assert len(second.ranked_pool) == 6
    assert tuple(item.evaluation.rating_delta for item in second.elites) == (4.0, 3.0)


def test_selection_fails_closed_for_invalid_or_insufficient_evidence() -> None:
    invalid = _evaluated(0, 0, "0.10", rating=None, valid=False, eligible=False)
    ineligible = _evaluated(0, 1, "0.20", rating=20.0, eligible=False)
    eligible = _evaluated(0, 2, "0.30", rating=10.0)

    with pytest.raises(SearchSelectionError) as invalid_error:
        select_generation(
            generation=0,
            proposals=(invalid, eligible),
            prior_elites=(),
            elite_count=1,
        )

    assert invalid_error.value.code == "invalid_candidate_evidence"

    with pytest.raises(SearchSelectionError) as insufficient_error:
        select_generation(
            generation=0,
            proposals=(ineligible, eligible),
            prior_elites=(),
            elite_count=2,
        )

    assert insufficient_error.value.code == "insufficient_eligible_candidates"


@pytest.mark.parametrize(
    ("changes", "expected"),
    (
        ({}, True),
        ({"rating": 0.0}, False),
        ({"rating": -0.01}, False),
        ({"eligible": False}, False),
        ({"valid": False, "eligible": False, "rating": None}, False),
        ({"completed": 2}, False),
        ({"candidate_faults": 1, "eligible": False}, False),
    ),
)
def test_freezes_only_a_complete_fault_free_positive_valid_winner(
    changes: dict[str, object],
    expected: bool,
) -> None:
    winner = _evaluated(
        0,
        0,
        "0.40",
        **{"rating": 1.0, **changes},  # type: ignore[arg-type]
    )

    assert (freeze_candidate(winner) is winner.candidate) is expected


def _evaluated(
    generation: int,
    slot: int,
    liquidity: str,
    *,
    rating: float | None,
    finish: float = 0.0,
    money: int = 0,
    valid: bool = True,
    eligible: bool = True,
    completed: int = 3,
    candidate_faults: int = 0,
) -> EvaluatedCandidate:
    candidate = HeuristicCandidate(
        personality="balanced",
        generation=generation,
        slot=slot,
        genome=CoefficientGenome(
            CoefficientValues(
                liquidity_strength=Decimal(liquidity),
                future_cash_weight=Decimal("0.75"),
                objective_progress_weight=Decimal("0.20"),
                bid_shading=Decimal("0.25"),
            )
        ),
        parent_identity=None if generation == 0 else "balanced-parent",
    )
    failures = (
        ()
        if valid and eligible
        else (
            EvaluationFailure(
                "fixture_failure",
                "Synthetic invalid or ineligible evidence.",
                not valid,
            ),
        )
    )
    evaluation = CandidateEvaluation(
        candidate_identity=candidate.identity,
        incumbent_identity="balanced-v2",
        requested_cases=3,
        completed_baseline_games=completed,
        completed_candidate_games=completed,
        rating_delta=rating,
        normalized_finish_delta=None if rating is None else finish,
        final_money_delta=None if rating is None else money,
        candidate_faults=candidate_faults,
        incumbent_faults=0,
        opponent_faults=0,
        unattributed_faults=0,
        faults_by_identity=(),
        failures=failures,
        valid=valid,
        eligible=eligible,
    )
    return EvaluatedCandidate(candidate=candidate, evaluation=evaluation)
