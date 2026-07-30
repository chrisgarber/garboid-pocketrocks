"""Deterministic episode and per-decision plans for neural self-play."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

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
        if self.ruleset_name not in {
            "live-A",
            "live-B",
            "live-C",
            "live-D",
            "live-E",
        }:
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
                        ruleset_name=f"live-{chart}",
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
