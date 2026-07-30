"""Held-out paired seat-rotated policy evaluation plans."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from garboid_pocketrocks.neural.planning import (
    SeatPolicy,
    SelfPlayEpisodePlan,
)


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    candidate_identity: str
    incumbent_identity: str
    games: int
    utility_delta: float
    confidence_low: float
    confidence_high: float
    illegal_actions: int
    faults: int


def plan_paired_evaluation(
    *,
    root_seed: int,
    candidate_identity: str,
    incumbent_identity: str,
    games_per_seat_cell: int,
) -> tuple[SelfPlayEpisodePlan, ...]:
    if games_per_seat_cell <= 0:
        raise ValueError("games_per_seat_cell must be positive")
    plans: list[SelfPlayEpisodePlan] = []
    for repetition in range(games_per_seat_cell):
        for chart in "ABCDE":
            for player_count in (3, 4, 5):
                for candidate_seat in range(player_count):
                    index = len(plans)
                    policies = tuple(
                        SeatPolicy(
                            candidate_identity if seat == candidate_seat else incumbent_identity,
                            False,
                        )
                        for seat in range(player_count)
                    )
                    plans.append(
                        SelfPlayEpisodePlan(
                            update_index=0,
                            episode_index=index,
                            ruleset_name=f"live-{chart}",
                            player_count=player_count,
                            engine_seed=_seed(
                                root_seed,
                                repetition,
                                ord(chart),
                                player_count,
                                candidate_seat,
                            ),
                            seat_sampling_seeds=tuple(
                                _seed(root_seed, index, seat, 1) for seat in range(player_count)
                            ),
                            seat_policies=policies,
                        )
                    )
    return tuple(plans)


def promotion_decision(report: EvaluationReport) -> bool:
    return report.confidence_low > 0.0 and report.illegal_actions == 0 and report.faults == 0


def _seed(root_seed: int, *parts: int) -> int:
    encoded = ":".join(("evaluation", str(root_seed), *(str(part) for part in parts))).encode()
    return int.from_bytes(
        hashlib.blake2b(encoded, digest_size=8).digest(),
        "big",
    ) & ((1 << 63) - 1)
