"""Immutable checkpoint league and deterministic league-game mixing."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import floor

from garboid_pocketrocks.neural.evaluation import (
    EvaluationReport,
    promotion_decision,
)
from garboid_pocketrocks.neural.planning import (
    SeatPolicy,
    SelfPlayEpisodePlan,
    plan_mirror_episodes,
)


@dataclass(frozen=True, slots=True)
class League:
    champion: str
    promoted: tuple[str, ...]

    def promote(self, report: EvaluationReport) -> League:
        if report.incumbent_identity != self.champion:
            raise ValueError("evaluation incumbent is not the league champion")
        if report.candidate_identity in self.promoted:
            raise ValueError("candidate is already in the league")
        if not promotion_decision(report):
            return self
        return League(
            champion=report.candidate_identity,
            promoted=(*self.promoted, report.candidate_identity),
        )


def plan_league_episodes(
    *,
    root_seed: int,
    update_index: int,
    games_per_cell: int,
    current_identity: str,
    historical_identities: tuple[str, ...],
    league_fraction: float,
) -> tuple[SelfPlayEpisodePlan, ...]:
    if not 0.0 <= league_fraction < 1.0:
        raise ValueError("league_fraction must be from zero to one")
    if league_fraction > 0.0 and not historical_identities:
        raise ValueError("league games require historical identities")
    plans = plan_mirror_episodes(
        root_seed=root_seed,
        update_index=update_index,
        games_per_cell=games_per_cell,
        policy_identity=current_identity,
    )
    league_games = floor(games_per_cell * league_fraction)
    mixed: list[SelfPlayEpisodePlan] = []
    for plan in plans:
        repetition = plan.episode_index // 15
        if repetition >= league_games:
            mixed.append(plan)
            continue
        current_seat = repetition % plan.player_count
        policies = tuple(
            SeatPolicy(current_identity, True)
            if seat == current_seat
            else SeatPolicy(
                historical_identities[
                    (plan.episode_index + seat) % len(historical_identities)
                ],
                False,
            )
            for seat in range(plan.player_count)
        )
        mixed.append(replace(plan, seat_policies=policies))
    return tuple(mixed)
