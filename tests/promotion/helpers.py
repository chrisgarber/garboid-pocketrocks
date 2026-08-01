from __future__ import annotations

from dataclasses import dataclass, replace

from garboid_pocketrocks.bots import BotBrain, BotSpec, RandomBot
from garboid_pocketrocks.promotion.candidates import FrozenCandidateProvenance
from garboid_pocketrocks.promotion.corpus import (
    PromotionCase,
    PromotionCorpus,
    PromotionCorpusRecipe,
)
from garboid_pocketrocks.promotion.planning import PromotionPlan, plan_paired_games
from garboid_pocketrocks.simulator.monte_carlo import (
    GameJob,
    GameSummary,
    MonteCarloResult,
)
from garboid_pocketrocks.simulator.session import SessionScore


@dataclass(frozen=True, slots=True)
class FrozenCandidateFixture:
    identity: str
    bot_spec: BotSpec
    predecessor_name: str
    development_corpus_name: str
    development_corpus_digest: str
    search_name: str
    repository_commit: str
    freeze_digest: str
    profile_digest: str
    manifest_digest: str
    search_report_digest: str
    candidate_evaluations_digest: str


class EvilFactory:
    """Build different behavior while claiming equality with every factory."""

    def __call__(self, seed: int | None) -> BotBrain:
        return RandomBot.build_brain(seed)

    def __eq__(self, other: object) -> bool:
        del other
        return True


class EvilString(str):
    """Carry different text while claiming equality with every string."""

    __hash__ = str.__hash__

    def __eq__(self, other: object) -> bool:
        del other
        return True

    def __ne__(self, other: object) -> bool:
        del other
        return False


class EvilProvenance(FrozenCandidateProvenance):
    """Carry forged fields while claiming equality with every provenance."""

    def __eq__(self, other: object) -> bool:
        del other
        return True

    def __ne__(self, other: object) -> bool:
        del other
        return False


def evil_provenance(
    trusted: FrozenCandidateProvenance,
) -> EvilProvenance:
    return EvilProvenance(
        candidate_name=trusted.candidate_name,
        candidate_bot_id=trusted.candidate_bot_id,
        predecessor_name=trusted.predecessor_name,
        development_corpus_name=trusted.development_corpus_name,
        development_corpus_digest=trusted.development_corpus_digest,
        search_name="forged-search-name",
        repository_commit=trusted.repository_commit,
        freeze_digest=trusted.freeze_digest,
        profile_digest=trusted.profile_digest,
        manifest_digest=trusted.manifest_digest,
        search_report_digest=trusted.search_report_digest,
        candidate_evaluations_digest=trusted.candidate_evaluations_digest,
    )


def frozen_candidate_fixture(
    *,
    bot_spec: BotSpec,
    predecessor_name: str,
    development: PromotionCorpus,
) -> FrozenCandidateFixture:
    identity = bot_spec.name
    return FrozenCandidateFixture(
        identity=identity,
        bot_spec=bot_spec,
        predecessor_name=predecessor_name,
        development_corpus_name=development.recipe.name,
        development_corpus_digest=development.digest,
        search_name="fixture-v3-search-v1",
        repository_commit="1" * 40,
        freeze_digest="a" * 64,
        profile_digest="b" * 64,
        manifest_digest="c" * 64,
        search_report_digest="d" * 64,
        candidate_evaluations_digest="e" * 64,
    )


def promotion_plan(*, pair_count: int = 3) -> PromotionPlan:
    """Build a small real promotion plan with connected comparison games."""

    candidate = _bot_spec("candidate")
    incumbent = _bot_spec("incumbent")
    opponents = (_bot_spec("opponent-a"), _bot_spec("opponent-b"))
    cases = tuple(
        PromotionCase(
            case_id=f"fixture-held-out-v1:{chr(65 + index)}:3:seat-{index % 3}:repeat-0",
            chart=chr(65 + index),
            player_count=3,
            focal_seat=index % 3,
            engine_seed=12_345 + index,
            opponent_names_by_seat=_opponents_by_seat(index % 3),
        )
        for index in range(pair_count)
    )
    corpus = PromotionCorpus(
        recipe=PromotionCorpusRecipe(
            schema_version=1,
            name="fixture-held-out-v1",
            purpose="held_out",
            root_seed=90_001,
            repetitions_per_seat_cell=1,
            charts=tuple(case.chart for case in cases),
            player_counts=(3,),
            opponent_names=tuple(spec.name for spec in opponents),
        ),
        cases=cases,
        digest="0" * 64,
    )
    return plan_paired_games(
        corpus,
        candidate=candidate,
        incumbent=incumbent,
        registry={spec.name: spec for spec in opponents},
    )


def summary_for_job(
    job: GameJob,
    *,
    final_money: tuple[int, ...],
    ranks: tuple[int, ...],
) -> GameSummary:
    """Create an exact summary for a planned job with explicit results."""

    return GameSummary(
        game_index=job.game_index,
        root_seed=job.root_seed,
        seed=job.seed,
        player_count=job.player_count,
        ruleset_name=f"live-{job.value_chart}",
        bot_names=tuple(spec.name for spec in job.lineup),
        bot_ids=tuple(spec.bot_id for spec in job.lineup),
        scores=tuple(
            SessionScore(seat=seat, final_money=money, rank=rank)
            for seat, (money, rank) in enumerate(zip(final_money, ranks, strict=True))
        ),
        decision_counts=(10,) * job.player_count,
        fault_counts=(0,) * job.player_count,
    )


def result_for_plan(
    plan: PromotionPlan,
    *,
    candidate_wins: bool = True,
) -> MonteCarloResult:
    """Create exact twins where the selected compared bot wins or loses."""

    summaries: list[GameSummary] = []
    for pair in plan.pairs:
        focal_seat = pair.case.focal_seat
        winning_money = tuple(100 if seat == focal_seat else 30 - seat for seat in range(3))
        losing_money = tuple(0 if seat == focal_seat else 100 - seat for seat in range(3))
        other_seats = tuple(seat for seat in range(3) if seat != focal_seat)
        winning_rank_by_seat = {focal_seat: 1, other_seats[0]: 2, other_seats[1]: 3}
        losing_rank_by_seat = {other_seats[0]: 1, other_seats[1]: 2, focal_seat: 3}
        winning_ranks = tuple(winning_rank_by_seat[seat] for seat in range(3))
        losing_ranks = tuple(losing_rank_by_seat[seat] for seat in range(3))
        candidate_money, candidate_ranks = (
            (winning_money, winning_ranks) if candidate_wins else (losing_money, losing_ranks)
        )
        incumbent_money, incumbent_ranks = (
            (losing_money, losing_ranks) if candidate_wins else (winning_money, winning_ranks)
        )
        summaries.extend(
            (
                summary_for_job(
                    pair.candidate_game,
                    final_money=candidate_money,
                    ranks=candidate_ranks,
                ),
                summary_for_job(
                    pair.incumbent_game,
                    final_money=incumbent_money,
                    ranks=incumbent_ranks,
                ),
            )
        )
    return MonteCarloResult(game_summaries=tuple(summaries), bot_statistics=(), replays=())


def replace_summary(
    result: MonteCarloResult,
    game_index: int,
    **changes: object,
) -> MonteCarloResult:
    """Replace one summary while preserving the rest of the real result."""

    summaries = tuple(
        replace(summary, **changes)  # type: ignore[arg-type]
        if summary.game_index == game_index
        else summary
        for summary in result.game_summaries
    )
    return replace(result, game_summaries=summaries)


def _bot_spec(name: str) -> BotSpec:
    return BotSpec.for_simulation(name, RandomBot.build_brain)


def _opponents_by_seat(focal_seat: int) -> tuple[str | None, ...]:
    opponents = iter(("opponent-a", "opponent-b"))
    return tuple(None if seat == focal_seat else next(opponents) for seat in range(3))
