from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import FrozenInstanceError
from fractions import Fraction

import pytest

pytest.importorskip("torch")

from garboid_pocketrocks.neural.heuristic_curriculum import (  # noqa: E402
    FOCAL_SEAT_CONTROL_V1,
    HEURISTIC_OPPONENT_CURRICULUM_V1,
    RELEASED_HEURISTIC_V3_IDENTITIES,
    HeuristicCurriculumStage,
    HeuristicOpponentCurriculum,
    plan_heuristic_curriculum_episodes,
)


def test_v1_schedule_is_the_frozen_196_update_bootstrap_curriculum() -> None:
    schedule = HEURISTIC_OPPONENT_CURRICULUM_V1

    assert schedule.identity == "heuristic-opponent-curriculum-v1"
    assert schedule.total_updates == 196
    assert schedule.heuristic_identities == (
        "aggressive-v3",
        "balanced-v3",
        "passive-v3",
    )
    assert tuple(
        (stage.name, stage.first_update, stage.stop_update, stage.heuristic_share)
        for stage in schedule.stages
    ) == (
        ("teacher-only", 0, 49, Fraction(1, 1)),
        ("teacher-mostly", 49, 98, Fraction(3, 4)),
        ("teacher-self-play-mix", 98, 147, Fraction(1, 2)),
        ("self-play-mostly", 147, 196, Fraction(1, 4)),
    )
    with pytest.raises(FrozenInstanceError):
        schedule.total_updates = 197  # type: ignore[misc]


def test_focal_control_uses_the_same_planner_without_teacher_seats() -> None:
    resolved = plan_heuristic_curriculum_episodes(
        root_seed=42,
        update_index=98,
        games_per_cell=7,
        learner_policy_identity="current",
        curriculum=FOCAL_SEAT_CONTROL_V1,
    )

    assert resolved.heuristic_opponent_count == 0
    assert all(
        sum(policy.trainable for policy in plan.seat_policies) == 1
        and {policy.identity for policy in plan.seat_policies} == {"current"}
        for plan in resolved.plans
    )
    assert len(FOCAL_SEAT_CONTROL_V1.digest) == 64
    assert FOCAL_SEAT_CONTROL_V1.digest != HEURISTIC_OPPONENT_CURRICULUM_V1.digest


@pytest.mark.parametrize(
    ("update_index", "expected_name", "expected_share"),
    [
        (0, "teacher-only", Fraction(1, 1)),
        (48, "teacher-only", Fraction(1, 1)),
        (49, "teacher-mostly", Fraction(3, 4)),
        (97, "teacher-mostly", Fraction(3, 4)),
        (98, "teacher-self-play-mix", Fraction(1, 2)),
        (146, "teacher-self-play-mix", Fraction(1, 2)),
        (147, "self-play-mostly", Fraction(1, 4)),
        (195, "self-play-mostly", Fraction(1, 4)),
    ],
)
def test_stage_boundaries_are_half_open_and_unambiguous(
    update_index: int,
    expected_name: str,
    expected_share: Fraction,
) -> None:
    stage = HEURISTIC_OPPONENT_CURRICULUM_V1.stage_for_update(update_index)

    assert stage.name == expected_name
    assert stage.heuristic_share == expected_share


@pytest.mark.parametrize("update_index", [-1, 196, True])
def test_schedule_rejects_updates_outside_the_experiment(update_index: int) -> None:
    with pytest.raises(ValueError, match="outside the curriculum"):
        HEURISTIC_OPPONENT_CURRICULUM_V1.stage_for_update(update_index)


def test_curriculum_config_rejects_aliases_and_stage_gaps() -> None:
    stages = (
        HeuristicCurriculumStage("first", 0, 1, Fraction(1, 1)),
        HeuristicCurriculumStage("second", 2, 3, Fraction(0, 1)),
    )
    with pytest.raises(ValueError, match="released v3"):
        HeuristicOpponentCurriculum(
            identity="bad-alias",
            total_updates=1,
            heuristic_identities=("aggressive", "balanced", "passive"),
            stages=(HeuristicCurriculumStage("all", 0, 1, Fraction(1, 1)),),
        )
    with pytest.raises(ValueError, match="contiguous"):
        HeuristicOpponentCurriculum(
            identity="gapped",
            total_updates=3,
            heuristic_identities=RELEASED_HEURISTIC_V3_IDENTITIES,
            stages=stages,
        )


@pytest.mark.parametrize(
    ("update_index", "share"),
    [(0, Fraction(1, 1)), (49, Fraction(3, 4)), (98, Fraction(1, 2)), (147, Fraction(1, 4))],
)
def test_plan_resolves_exact_teacher_share_and_balanced_teacher_identities(
    update_index: int,
    share: Fraction,
) -> None:
    resolved = plan_heuristic_curriculum_episodes(
        root_seed=42,
        update_index=update_index,
        games_per_cell=7,
        learner_policy_identity="current",
    )
    opponent_count = sum(plan.player_count - 1 for plan in resolved.plans)
    expected_teacher_count = opponent_count * share.numerator // share.denominator
    observed = Counter(
        policy.identity
        for plan in resolved.plans
        for policy in plan.seat_policies
        if not policy.trainable
    )

    assert resolved.heuristic_opponent_count == expected_teacher_count
    assert resolved.self_play_opponent_count == opponent_count - expected_teacher_count
    assert sum(observed[name] for name in RELEASED_HEURISTIC_V3_IDENTITIES) == (
        expected_teacher_count
    )
    assert observed["current"] == opponent_count - expected_teacher_count
    teacher_counts = [observed[name] for name in RELEASED_HEURISTIC_V3_IDENTITIES]
    assert max(teacher_counts) - min(teacher_counts) <= 1


def test_plan_has_one_focal_learner_and_balances_seats_in_every_cell() -> None:
    resolved = plan_heuristic_curriculum_episodes(
        root_seed=42,
        update_index=98,
        games_per_cell=119,
        learner_policy_identity="current",
    )
    cell_counts = Counter((plan.ruleset_name, plan.player_count) for plan in resolved.plans)
    focal_counts: dict[tuple[str, int], Counter[int]] = defaultdict(Counter)
    for plan in resolved.plans:
        focal_seats = [seat for seat, policy in enumerate(plan.seat_policies) if policy.trainable]
        assert len(focal_seats) == 1
        assert plan.seat_policies[focal_seats[0]].identity == "current"
        focal_counts[(plan.ruleset_name, plan.player_count)][focal_seats[0]] += 1

    assert len(resolved.plans) == 119 * 15
    assert set(cell_counts.values()) == {119}
    for counts in focal_counts.values():
        assert max(counts.values()) - min(counts.values()) <= 1


def test_plans_are_repeatable_and_root_seed_changes_curriculum_assignments() -> None:
    first = plan_heuristic_curriculum_episodes(
        root_seed=42,
        update_index=98,
        games_per_cell=3,
        learner_policy_identity="current",
    )
    repeated = plan_heuristic_curriculum_episodes(
        root_seed=42,
        update_index=98,
        games_per_cell=3,
        learner_policy_identity="current",
    )
    another_seed = plan_heuristic_curriculum_episodes(
        root_seed=43,
        update_index=98,
        games_per_cell=3,
        learner_policy_identity="current",
    )

    assert first == repeated
    assert first.plans != another_seed.plans
    assert all(len(plan.seat_sampling_seeds) == plan.player_count for plan in first.plans)
