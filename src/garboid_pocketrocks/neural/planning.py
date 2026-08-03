"""Deterministic episode and per-decision plans for neural self-play."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from fractions import Fraction

from garboid_pocketrocks.knowledge import ruleset_name, value_chart_from_ruleset_name
from garboid_pocketrocks.neural.opponent_pool import STRONG_FIELD_POOL_V1

_UNSIGNED_63_BIT_MASK = (1 << 63) - 1


@dataclass(frozen=True, slots=True)
class SeatPolicy:
    """One immutable policy assignment for a game seat."""

    identity: str
    trainable: bool

    def __post_init__(self) -> None:
        if not self.identity:
            raise ValueError("policy identity must be nonempty")


@dataclass(frozen=True, slots=True)
class SelfPlayEpisodePlan:
    """Every deterministic input required to play one self-play game."""

    update_index: int
    episode_index: int
    ruleset_name: str
    player_count: int
    engine_seed: int
    seat_sampling_seeds: tuple[int, ...]
    seat_policies: tuple[SeatPolicy, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "seat_sampling_seeds", tuple(self.seat_sampling_seeds))
        object.__setattr__(self, "seat_policies", tuple(self.seat_policies))
        if (
            not isinstance(self.update_index, int)
            or isinstance(self.update_index, bool)
            or self.update_index < 0
        ):
            raise ValueError("update_index must be a nonnegative integer")
        if (
            not isinstance(self.episode_index, int)
            or isinstance(self.episode_index, bool)
            or self.episode_index < 0
        ):
            raise ValueError("episode_index must be a nonnegative integer")
        if self.ruleset_name not in {ruleset_name(chart) for chart in "ABCDE"}:
            raise ValueError("ruleset_name must identify a supported live chart")
        if self.player_count not in (3, 4, 5):
            raise ValueError("player_count must be three, four, or five")
        if not _is_seed(self.engine_seed):
            raise ValueError("engine_seed must be an unsigned 63-bit integer")
        if len(self.seat_sampling_seeds) != self.player_count or any(
            not _is_seed(seed) for seed in self.seat_sampling_seeds
        ):
            raise ValueError("seat sampling seeds must match player_count")
        if len(self.seat_policies) != self.player_count:
            raise ValueError("seat policies must match player_count")


def plan_mirror_episodes(
    *,
    root_seed: int,
    update_index: int,
    games_per_cell: int,
    policy_identity: str,
) -> tuple[SelfPlayEpisodePlan, ...]:
    """Plan balanced all-seat mirror games across all 15 training cells."""

    _require_int("root_seed", root_seed)
    _require_nonnegative_int("update_index", update_index)
    if (
        not isinstance(games_per_cell, int)
        or isinstance(games_per_cell, bool)
        or games_per_cell <= 0
    ):
        raise ValueError("games_per_cell must be a positive integer")
    if not isinstance(policy_identity, str) or not policy_identity:
        raise ValueError("policy_identity must be a nonempty string")

    plans: list[SelfPlayEpisodePlan] = []
    for _repetition in range(games_per_cell):
        for chart in "ABCDE":
            for player_count in (3, 4, 5):
                episode_index = len(plans)
                plans.append(
                    SelfPlayEpisodePlan(
                        update_index=update_index,
                        episode_index=episode_index,
                        ruleset_name=ruleset_name(chart),
                        player_count=player_count,
                        engine_seed=_derive_seed(
                            root_seed,
                            "engine",
                            update_index,
                            episode_index,
                        ),
                        seat_sampling_seeds=tuple(
                            _derive_seed(
                                root_seed,
                                "policy",
                                update_index,
                                episode_index,
                                seat,
                            )
                            for seat in range(player_count)
                        ),
                        seat_policies=(SeatPolicy(identity=policy_identity, trainable=True),)
                        * player_count,
                    )
                )
    return tuple(plans)


def plan_strong_field_episodes(
    *,
    root_seed: int,
    update_index: int,
    games_per_cell: int,
    policy_identity: str,
    fixed_opponent_share: Fraction,
) -> tuple[SelfPlayEpisodePlan, ...]:
    """Plan focal-seat learning against a fixed/self-play opponent mix."""

    if not isinstance(fixed_opponent_share, Fraction) or not 0 <= fixed_opponent_share <= 1:
        raise ValueError("fixed_opponent_share must be an exact Fraction from zero to one")
    base = plan_mirror_episodes(
        root_seed=root_seed,
        update_index=update_index,
        games_per_cell=games_per_cell,
        policy_identity=policy_identity,
    )
    focal_seats = tuple(
        (plan.episode_index // 15 + _derive_seed(
            root_seed,
            "focal-seat",
            update_index,
            ord(value_chart_from_ruleset_name(plan.ruleset_name)),
            plan.player_count,
        ))
        % plan.player_count
        for plan in base
    )
    opponent_slots = tuple(
        (plan_index, seat)
        for plan_index, plan in enumerate(base)
        for seat in range(plan.player_count)
        if seat != focal_seats[plan_index]
    )
    ordered_slots = sorted(
        opponent_slots,
        key=lambda slot: (
            _derive_seed(
                root_seed,
                "fixed-opponent-kind",
                update_index,
                base[slot[0]].episode_index,
                slot[1],
            ),
            slot,
        ),
    )
    fixed_count = (
        len(opponent_slots)
        * fixed_opponent_share.numerator
        // fixed_opponent_share.denominator
    )
    fixed_slots = frozenset(ordered_slots[:fixed_count])
    ordered_fixed = sorted(
        fixed_slots,
        key=lambda slot: (
            _derive_seed(
                root_seed,
                "fixed-opponent-identity",
                update_index,
                base[slot[0]].episode_index,
                slot[1],
            ),
            slot,
        ),
    )
    identity_by_slot = {
        slot: STRONG_FIELD_POOL_V1[index % len(STRONG_FIELD_POOL_V1)].name
        for index, slot in enumerate(ordered_fixed)
    }
    return tuple(
        SelfPlayEpisodePlan(
            update_index=plan.update_index,
            episode_index=plan.episode_index,
            ruleset_name=plan.ruleset_name,
            player_count=plan.player_count,
            engine_seed=plan.engine_seed,
            seat_sampling_seeds=plan.seat_sampling_seeds,
            seat_policies=tuple(
                SeatPolicy(policy_identity, trainable=True)
                if seat == focal_seats[plan_index]
                else SeatPolicy(
                    identity_by_slot.get((plan_index, seat), policy_identity),
                    trainable=False,
                )
                for seat in range(plan.player_count)
            ),
        )
        for plan_index, plan in enumerate(base)
    )


def decision_seed(
    plan: SelfPlayEpisodePlan,
    seat: int,
    decision_index: int,
) -> int:
    """Derive a schedule-independent seed for one seat decision."""

    if not isinstance(seat, int) or isinstance(seat, bool) or not 0 <= seat < plan.player_count:
        raise ValueError("seat is outside the episode player count")
    _require_nonnegative_int("decision_index", decision_index)
    return _derive_seed(
        plan.seat_sampling_seeds[seat],
        "decision",
        plan.update_index,
        plan.episode_index,
        seat,
        decision_index,
    )


def _derive_seed(root_seed: int, namespace: str, *indices: int) -> int:
    _require_int("root_seed", root_seed)
    if not namespace:
        raise ValueError("seed namespace must be nonempty")
    if any(not isinstance(index, int) or isinstance(index, bool) for index in indices):
        raise ValueError("seed indices must be integers")
    canonical = ":".join((str(root_seed), namespace, *(str(index) for index in indices)))
    digest = hashlib.blake2b(canonical.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & _UNSIGNED_63_BIT_MASK


def _require_int(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")


def _require_nonnegative_int(name: str, value: object) -> None:
    _require_int(name, value)
    assert isinstance(value, int)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _is_seed(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value < 2**63
