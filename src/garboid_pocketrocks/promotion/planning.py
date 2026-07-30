"""Build matched candidate and incumbent games for the promotion gate."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from garboid_pocketrocks.bots import BotSpec
from garboid_pocketrocks.promotion.corpus import PromotionCase, PromotionCorpus
from garboid_pocketrocks.simulator.monte_carlo import GameJob, MonteCarloConfig
from garboid_pocketrocks.simulator.runner import FaultMode


@dataclass(frozen=True, slots=True)
class PairedGamePlan:
    """Two copies of one case with the compared bot occupying the same seat."""

    pair_index: int
    case: PromotionCase
    candidate_game: GameJob
    incumbent_game: GameJob


@dataclass(frozen=True, slots=True)
class PromotionPlan:
    """The complete, immutable set of matched games for one comparison."""

    candidate: BotSpec
    incumbent: BotSpec
    opponents: tuple[BotSpec, ...]
    pairs: tuple[PairedGamePlan, ...]
    monte_carlo_config: MonteCarloConfig

    @property
    def jobs(self) -> tuple[GameJob, ...]:
        """Return each candidate game immediately before its incumbent twin."""

        return tuple(
            game for pair in self.pairs for game in (pair.candidate_game, pair.incumbent_game)
        )


class PromotionPlanningError(ValueError):
    """Explain why a fair paired promotion plan cannot be built."""

    code: str

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def plan_paired_games(
    held_out: PromotionCorpus,
    *,
    candidate: BotSpec,
    incumbent: BotSpec,
    registry: Mapping[str, BotSpec],
) -> PromotionPlan:
    """Place candidate and incumbent into exact twin copies of every held-out case."""

    if held_out.recipe.purpose != "held_out":
        raise PromotionPlanningError(
            "held_out_corpus_required",
            "Promotion games must come from a held-out corpus that was not used for tuning.",
        )
    if _identities_overlap(candidate, incumbent):
        raise PromotionPlanningError(
            "candidate_incumbent_identity_collision",
            "Candidate and incumbent must have different names and different bot IDs.",
        )

    opponent_specs_by_name = _resolve_recipe_opponents(held_out, registry=registry)
    _require_no_compared_bot_opponent_collisions(
        candidate=candidate,
        incumbent=incumbent,
        opponents=tuple(opponent_specs_by_name.values()),
    )

    pairs: list[PairedGamePlan] = []
    opponents_in_first_seen_order: list[BotSpec] = []
    seen_opponent_ids: set[str] = set()
    for pair_index, case in enumerate(held_out.cases):
        candidate_lineup, incumbent_lineup = _build_twin_lineups(
            case,
            candidate=candidate,
            incumbent=incumbent,
            opponent_specs_by_name=opponent_specs_by_name,
        )
        for seat, spec in enumerate(candidate_lineup):
            if seat == case.focal_seat or spec.bot_id in seen_opponent_ids:
                continue
            seen_opponent_ids.add(spec.bot_id)
            opponents_in_first_seen_order.append(spec)

        pairs.append(
            PairedGamePlan(
                pair_index=pair_index,
                case=case,
                candidate_game=_game_job(
                    held_out=held_out,
                    case=case,
                    pair_index=pair_index,
                    variant_offset=0,
                    lineup=candidate_lineup,
                ),
                incumbent_game=_game_job(
                    held_out=held_out,
                    case=case,
                    pair_index=pair_index,
                    variant_offset=1,
                    lineup=incumbent_lineup,
                ),
            )
        )

    opponents = tuple(opponents_in_first_seen_order)
    monte_carlo_config = MonteCarloConfig(
        bot_specs=(candidate, incumbent, *opponents),
        games=2 * len(held_out.cases),
        player_counts=held_out.recipe.player_counts,
        value_charts=held_out.recipe.charts,
        root_seed=held_out.recipe.root_seed,
        objectives_enabled=(True,),
        fault_mode=FaultMode.RECORD_AND_PASS,
    )
    return PromotionPlan(
        candidate=candidate,
        incumbent=incumbent,
        opponents=opponents,
        pairs=tuple(pairs),
        monte_carlo_config=monte_carlo_config,
    )


def _resolve_recipe_opponents(
    held_out: PromotionCorpus,
    *,
    registry: Mapping[str, BotSpec],
) -> dict[str, BotSpec]:
    resolved: dict[str, BotSpec] = {}
    for opponent_name in held_out.recipe.opponent_names:
        spec = registry.get(opponent_name)
        if spec is None or spec.name != opponent_name:
            raise PromotionPlanningError(
                "opponent_identity_mismatch",
                f"Corpus opponent {opponent_name!r} does not match its registered identity.",
            )
        resolved[opponent_name] = spec
    return resolved


def _require_no_compared_bot_opponent_collisions(
    *,
    candidate: BotSpec,
    incumbent: BotSpec,
    opponents: tuple[BotSpec, ...],
) -> None:
    for opponent in opponents:
        if _identities_overlap(candidate, opponent):
            raise PromotionPlanningError(
                "candidate_opponent_identity_collision",
                "The candidate must have a different name and bot ID from every opponent.",
            )
        if _identities_overlap(incumbent, opponent):
            raise PromotionPlanningError(
                "incumbent_opponent_identity_collision",
                "The incumbent must have a different name and bot ID from every opponent.",
            )


def _identities_overlap(first: BotSpec, second: BotSpec) -> bool:
    return first.name == second.name or first.bot_id == second.bot_id


def _build_twin_lineups(
    case: PromotionCase,
    *,
    candidate: BotSpec,
    incumbent: BotSpec,
    opponent_specs_by_name: Mapping[str, BotSpec],
) -> tuple[tuple[BotSpec, ...], tuple[BotSpec, ...]]:
    if (
        len(case.opponent_names_by_seat) != case.player_count
        or case.focal_seat not in range(case.player_count)
        or case.opponent_names_by_seat[case.focal_seat] is not None
        or sum(name is None for name in case.opponent_names_by_seat) != 1
    ):
        raise PromotionPlanningError(
            "opponent_identity_mismatch",
            f"Promotion case {case.case_id!r} does not describe one opponent per non-focal seat.",
        )

    candidate_lineup: list[BotSpec] = []
    incumbent_lineup: list[BotSpec] = []
    for seat, opponent_name in enumerate(case.opponent_names_by_seat):
        if seat == case.focal_seat:
            candidate_lineup.append(candidate)
            incumbent_lineup.append(incumbent)
            continue
        if opponent_name is None:
            raise PromotionPlanningError(
                "opponent_identity_mismatch",
                f"Promotion case {case.case_id!r} is missing an opponent at seat {seat}.",
            )
        opponent = opponent_specs_by_name.get(opponent_name)
        if opponent is None:
            raise PromotionPlanningError(
                "opponent_identity_mismatch",
                f"Promotion case {case.case_id!r} uses undeclared opponent {opponent_name!r}.",
            )
        candidate_lineup.append(opponent)
        incumbent_lineup.append(opponent)
    return tuple(candidate_lineup), tuple(incumbent_lineup)


def _game_job(
    *,
    held_out: PromotionCorpus,
    case: PromotionCase,
    pair_index: int,
    variant_offset: int,
    lineup: tuple[BotSpec, ...],
) -> GameJob:
    return GameJob(
        game_index=2 * pair_index + variant_offset,
        root_seed=held_out.recipe.root_seed,
        seed=case.engine_seed,
        player_count=case.player_count,
        value_chart=case.chart,
        objectives_enabled=True,
        lineup=lineup,
        fault_mode=FaultMode.RECORD_AND_PASS,
    )
