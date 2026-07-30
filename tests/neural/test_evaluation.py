from __future__ import annotations

from collections import Counter

from garboid_pocketrocks.neural.evaluation import (
    EvaluationReport,
    plan_paired_evaluation,
    promotion_decision,
)
from garboid_pocketrocks.neural.league import League, plan_league_episodes
from garboid_pocketrocks.neural.planning import SeatPolicy


def test_paired_evaluation_rotates_every_candidate_seat_and_cell() -> None:
    plans = plan_paired_evaluation(
        root_seed=99,
        candidate_identity="candidate",
        incumbent_identity="incumbent",
        games_per_seat_cell=2,
    )
    counts = Counter(
        (
            plan.ruleset_name,
            plan.player_count,
            plan.seat_policies.index(SeatPolicy("candidate", False)),
        )
        for plan in plans
    )
    assert set(counts.values()) == {2}
    assert len(counts) == 5 * sum((3, 4, 5))


def _report(low: float) -> EvaluationReport:
    return EvaluationReport(
        candidate_identity="candidate",
        incumbent_identity="champion",
        games=10,
        utility_delta=0.1,
        confidence_low=low,
        confidence_high=0.2,
        illegal_actions=0,
        faults=0,
    )


def test_promotion_requires_positive_lower_bound_and_clean_play() -> None:
    assert not promotion_decision(_report(-0.01))
    assert promotion_decision(_report(0.01))
    assert League("champion", ("champion",)).promote(_report(0.01)).champion == (
        "candidate"
    )


def test_league_games_train_only_current_policy_seats() -> None:
    plans = plan_league_episodes(
        root_seed=42,
        update_index=10,
        games_per_cell=10,
        current_identity="current",
        historical_identities=("champion", "older"),
        league_fraction=0.2,
    )

    assert any(
        any(not seat.trainable for seat in plan.seat_policies)
        for plan in plans
    )
    assert all(
        seat.trainable == (seat.identity == "current")
        for plan in plans
        for seat in plan.seat_policies
    )
