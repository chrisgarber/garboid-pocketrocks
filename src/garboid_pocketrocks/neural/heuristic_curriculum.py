"""Deterministic focal-seat plans for the heuristic-opponent ablation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from fractions import Fraction

from garboid_pocketrocks.neural.heuristic_teachers import (
    RELEASED_HEURISTIC_V3_IDENTITIES as RELEASED_HEURISTIC_V3_IDENTITIES,
)
from garboid_pocketrocks.neural.planning import (
    SeatPolicy,
    SelfPlayEpisodePlan,
    plan_mirror_episodes,
)

_UNSIGNED_63_BIT_MASK = (1 << 63) - 1


@dataclass(frozen=True, slots=True)
class HeuristicCurriculumStage:
    """One half-open range of updates with a fixed teacher-opponent share."""

    name: str
    first_update: int
    stop_update: int
    heuristic_share: Fraction

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("curriculum stage name must be nonempty")
        if (
            not isinstance(self.first_update, int)
            or isinstance(self.first_update, bool)
            or self.first_update < 0
        ):
            raise ValueError("first_update must be a nonnegative integer")
        if (
            not isinstance(self.stop_update, int)
            or isinstance(self.stop_update, bool)
            or self.stop_update <= self.first_update
        ):
            raise ValueError("stop_update must be greater than first_update")
        if not isinstance(self.heuristic_share, Fraction):
            raise ValueError("heuristic_share must be an exact Fraction")
        if not 0 <= self.heuristic_share <= 1:
            raise ValueError("heuristic_share must be from zero to one")

    def contains(self, update_index: int) -> bool:
        """Return whether this stage owns the update index."""

        return self.first_update <= update_index < self.stop_update


@dataclass(frozen=True, slots=True)
class HeuristicOpponentCurriculum:
    """Immutable teacher identities and the complete update-stage schedule."""

    identity: str
    total_updates: int
    heuristic_identities: tuple[str, ...]
    stages: tuple[HeuristicCurriculumStage, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "heuristic_identities", tuple(self.heuristic_identities))
        object.__setattr__(self, "stages", tuple(self.stages))
        if not self.identity:
            raise ValueError("curriculum identity must be nonempty")
        if (
            not isinstance(self.total_updates, int)
            or isinstance(self.total_updates, bool)
            or self.total_updates <= 0
        ):
            raise ValueError("total_updates must be a positive integer")
        if self.heuristic_identities != RELEASED_HEURISTIC_V3_IDENTITIES:
            raise ValueError("curriculum must pin the released v3 heuristic identities")
        if not self.stages:
            raise ValueError("curriculum must contain at least one stage")
        expected_first_update = 0
        names: set[str] = set()
        for stage in self.stages:
            if stage.name in names:
                raise ValueError("curriculum stage names must be unique")
            names.add(stage.name)
            if stage.first_update != expected_first_update:
                raise ValueError("curriculum stages must be contiguous from update zero")
            expected_first_update = stage.stop_update
        if expected_first_update != self.total_updates:
            raise ValueError("curriculum stages must cover every configured update")

    def stage_for_update(self, update_index: int) -> HeuristicCurriculumStage:
        """Resolve exactly one stage, rejecting updates outside the experiment."""

        if (
            not isinstance(update_index, int)
            or isinstance(update_index, bool)
            or not 0 <= update_index < self.total_updates
        ):
            raise ValueError("update_index is outside the curriculum")
        return next(stage for stage in self.stages if stage.contains(update_index))


HEURISTIC_OPPONENT_CURRICULUM_V1 = HeuristicOpponentCurriculum(
    identity="heuristic-opponent-curriculum-v1",
    total_updates=196,
    heuristic_identities=RELEASED_HEURISTIC_V3_IDENTITIES,
    stages=(
        HeuristicCurriculumStage("teacher-only", 0, 49, Fraction(1, 1)),
        HeuristicCurriculumStage("teacher-mostly", 49, 98, Fraction(3, 4)),
        HeuristicCurriculumStage("teacher-self-play-mix", 98, 147, Fraction(1, 2)),
        HeuristicCurriculumStage("self-play-mostly", 147, 196, Fraction(1, 4)),
    ),
)


@dataclass(frozen=True, slots=True)
class ResolvedHeuristicCurriculumUpdate:
    """A fully resolved update plan and its auditable opponent counts."""

    stage: HeuristicCurriculumStage
    plans: tuple[SelfPlayEpisodePlan, ...]
    heuristic_opponent_count: int
    self_play_opponent_count: int


def plan_heuristic_curriculum_episodes(
    *,
    root_seed: int,
    update_index: int,
    games_per_cell: int,
    learner_policy_identity: str,
    curriculum: HeuristicOpponentCurriculum = HEURISTIC_OPPONENT_CURRICULUM_V1,
) -> ResolvedHeuristicCurriculumUpdate:
    """Plan one balanced update with one learning seat in every game.

    The engine and neural action seeds retain the existing self-play namespaces.
    Separate curriculum namespaces choose focal seats, teacher slots, and teacher
    identities, so changing one selection rule cannot perturb the others.
    """

    if not isinstance(curriculum, HeuristicOpponentCurriculum):
        raise ValueError("curriculum must be a HeuristicOpponentCurriculum")
    stage = curriculum.stage_for_update(update_index)
    base_plans = plan_mirror_episodes(
        root_seed=root_seed,
        update_index=update_index,
        games_per_cell=games_per_cell,
        policy_identity=learner_policy_identity,
    )

    focal_seats = tuple(
        _focal_seat(root_seed, plan, games_per_cell=games_per_cell) for plan in base_plans
    )
    opponent_slots = tuple(
        (plan_index, seat)
        for plan_index, plan in enumerate(base_plans)
        for seat in range(plan.player_count)
        if seat != focal_seats[plan_index]
    )
    ordered_for_kind = sorted(
        opponent_slots,
        key=lambda slot: (
            _curriculum_seed(
                root_seed,
                "heuristic-opponent-kind",
                update_index,
                base_plans[slot[0]].episode_index,
                slot[1],
            ),
            slot,
        ),
    )
    heuristic_count = (
        len(opponent_slots) * stage.heuristic_share.numerator // stage.heuristic_share.denominator
    )
    heuristic_slots = frozenset(ordered_for_kind[:heuristic_count])
    ordered_for_identity = sorted(
        heuristic_slots,
        key=lambda slot: (
            _curriculum_seed(
                root_seed,
                "heuristic-opponent-identity",
                update_index,
                base_plans[slot[0]].episode_index,
                slot[1],
            ),
            slot,
        ),
    )
    identity_by_slot = {
        slot: curriculum.heuristic_identities[index % len(curriculum.heuristic_identities)]
        for index, slot in enumerate(ordered_for_identity)
    }

    resolved_plans: list[SelfPlayEpisodePlan] = []
    for plan_index, plan in enumerate(base_plans):
        policies = tuple(
            SeatPolicy(learner_policy_identity, trainable=True)
            if seat == focal_seats[plan_index]
            else SeatPolicy(
                identity_by_slot.get((plan_index, seat), learner_policy_identity),
                trainable=False,
            )
            for seat in range(plan.player_count)
        )
        resolved_plans.append(replace(plan, seat_policies=policies))

    return ResolvedHeuristicCurriculumUpdate(
        stage=stage,
        plans=tuple(resolved_plans),
        heuristic_opponent_count=heuristic_count,
        self_play_opponent_count=len(opponent_slots) - heuristic_count,
    )


def _focal_seat(
    root_seed: int,
    plan: SelfPlayEpisodePlan,
    *,
    games_per_cell: int,
) -> int:
    repetition = plan.episode_index // 15
    if not 0 <= repetition < games_per_cell:
        raise ValueError("episode index is incompatible with games_per_cell")
    chart_index = (plan.episode_index % 15) // 3
    offset = (
        _curriculum_seed(
            root_seed,
            "focal-seat",
            plan.update_index,
            chart_index,
            plan.player_count,
        )
        % plan.player_count
    )
    return (repetition + offset) % plan.player_count


def _curriculum_seed(root_seed: int, namespace: str, *indices: int) -> int:
    if not isinstance(root_seed, int) or isinstance(root_seed, bool):
        raise ValueError("root_seed must be an integer")
    if not namespace:
        raise ValueError("seed namespace must be nonempty")
    if any(not isinstance(index, int) or isinstance(index, bool) for index in indices):
        raise ValueError("seed indices must be integers")
    canonical = ":".join((str(root_seed), namespace, *(str(index) for index in indices)))
    digest = hashlib.blake2b(canonical.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & _UNSIGNED_63_BIT_MASK
