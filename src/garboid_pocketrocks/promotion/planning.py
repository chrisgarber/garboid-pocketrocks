"""Build matched candidate and incumbent games for the promotion gate."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal

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
class OpponentExclusion:
    """One configured opponent omitted because it is a compared identity."""

    opponent: BotSpec
    reason: Literal["candidate", "incumbent"]


@dataclass(frozen=True, slots=True)
class EffectiveOpponentPool:
    """The configured, excluded, and remaining opponents for one comparison."""

    configured: tuple[BotSpec, ...]
    exclusions: tuple[OpponentExclusion, ...]
    remaining: tuple[BotSpec, ...]


@dataclass(frozen=True, slots=True)
class PromotionPlan:
    """The complete, immutable set of matched games for one comparison."""

    candidate: BotSpec
    incumbent: BotSpec
    source_corpus_name: str
    source_corpus_digest: str
    opponent_pool: EffectiveOpponentPool
    opponents: tuple[BotSpec, ...]
    pairs: tuple[PairedGamePlan, ...]
    monte_carlo_config: MonteCarloConfig
    digest: str

    @property
    def jobs(self) -> tuple[GameJob, ...]:
        """Return each candidate game immediately before its incumbent twin."""

        return tuple(
            game for pair in self.pairs for game in (pair.candidate_game, pair.incumbent_game)
        )


class PromotionPlanningError(ValueError):
    """Explain why a fair paired promotion plan cannot be built."""

    code: str
    opponent_pool: EffectiveOpponentPool | None

    def __init__(
        self,
        code: str,
        message: str,
        *,
        opponent_pool: EffectiveOpponentPool | None = None,
    ) -> None:
        self.code = code
        self.opponent_pool = opponent_pool
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

    configured_opponents = _resolve_recipe_opponents(held_out, registry=registry)
    for source_case in held_out.cases:
        _build_twin_lineups(
            source_case,
            candidate=candidate,
            incumbent=incumbent,
            opponent_specs_by_name=configured_opponents,
        )
    opponent_pool = _effective_opponent_pool(
        candidate=candidate,
        incumbent=incumbent,
        configured=tuple(configured_opponents.values()),
        required=_required_ordinary_opponents(held_out),
    )
    opponent_specs_by_name = {spec.name: spec for spec in opponent_pool.remaining}
    effective_cases = tuple(
        _effective_case(
            case,
            held_out=held_out,
            opponent_pool=opponent_pool,
        )
        for case in held_out.cases
    )

    pairs: list[PairedGamePlan] = []
    opponents_in_first_seen_order: list[BotSpec] = []
    seen_opponent_ids: set[str] = set()
    for pair_index, case in enumerate(effective_cases):
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
    plan = PromotionPlan(
        candidate=candidate,
        incumbent=incumbent,
        source_corpus_name=held_out.recipe.name,
        source_corpus_digest=held_out.digest,
        opponent_pool=opponent_pool,
        opponents=opponents,
        pairs=tuple(pairs),
        monte_carlo_config=monte_carlo_config,
        digest="",
    )
    return replace(plan, digest=_promotion_plan_digest(plan))


def _required_ordinary_opponents(held_out: PromotionCorpus) -> int:
    """Return the largest number of ordinary opponents required by one case."""

    return max(held_out.recipe.player_counts) - 1


def _resolve_recipe_opponents(
    held_out: PromotionCorpus,
    *,
    registry: Mapping[str, BotSpec],
) -> dict[str, BotSpec]:
    resolved: dict[str, BotSpec] = {}
    seen_names: set[str] = set()
    seen_bot_ids: set[str] = set()
    for opponent_name in held_out.recipe.opponent_names:
        spec = registry.get(opponent_name)
        if spec is None or spec.name != opponent_name:
            raise PromotionPlanningError(
                "opponent_identity_mismatch",
                f"Corpus opponent {opponent_name!r} does not match its registered identity.",
            )
        if spec.name in seen_names or spec.bot_id in seen_bot_ids:
            raise PromotionPlanningError(
                "opponent_identity_mismatch",
                "Corpus opponents must have different names and different bot IDs.",
            )
        seen_names.add(spec.name)
        seen_bot_ids.add(spec.bot_id)
        resolved[opponent_name] = spec
    return resolved


def _effective_opponent_pool(
    *,
    candidate: BotSpec,
    incumbent: BotSpec,
    configured: tuple[BotSpec, ...],
    required: int,
) -> EffectiveOpponentPool:
    exclusions: list[OpponentExclusion] = []
    remaining: list[BotSpec] = []
    for opponent in configured:
        candidate_name_matches = candidate.name == opponent.name
        candidate_id_matches = candidate.bot_id == opponent.bot_id
        if candidate_name_matches and candidate_id_matches:
            exclusions.append(OpponentExclusion(opponent, "candidate"))
            continue
        if candidate_name_matches or candidate_id_matches:
            raise PromotionPlanningError(
                "candidate_opponent_identity_collision",
                "A configured opponent partially matches the candidate identity; "
                "name and bot ID must either both match or both differ.",
            )

        incumbent_name_matches = incumbent.name == opponent.name
        incumbent_id_matches = incumbent.bot_id == opponent.bot_id
        if incumbent_name_matches and incumbent_id_matches:
            exclusions.append(OpponentExclusion(opponent, "incumbent"))
            continue
        if incumbent_name_matches or incumbent_id_matches:
            raise PromotionPlanningError(
                "incumbent_opponent_identity_collision",
                "A configured opponent partially matches the incumbent identity; "
                "name and bot ID must either both match or both differ.",
            )
        remaining.append(opponent)

    pool = EffectiveOpponentPool(
        configured=configured,
        exclusions=tuple(exclusions),
        remaining=tuple(remaining),
    )
    if len(pool.remaining) < required:
        raise PromotionPlanningError(
            "insufficient_eligible_opponents",
            f"Promotion requires {required} distinct ordinary opponents after "
            f"excluding compared identities, but only {len(pool.remaining)} remain.",
            opponent_pool=pool,
        )
    return pool


def _identities_overlap(first: BotSpec, second: BotSpec) -> bool:
    return first.name == second.name or first.bot_id == second.bot_id


def _effective_case(
    case: PromotionCase,
    *,
    held_out: PromotionCorpus,
    opponent_pool: EffectiveOpponentPool,
) -> PromotionCase:
    repetition = _case_repetition(case)
    try:
        chart_index = held_out.recipe.charts.index(case.chart)
    except ValueError as error:
        raise PromotionPlanningError(
            "opponent_identity_mismatch",
            f"Promotion case {case.case_id!r} uses chart {case.chart!r} outside its corpus recipe.",
            opponent_pool=opponent_pool,
        ) from error

    rotation = (repetition + chart_index + case.player_count + case.focal_seat) % len(
        opponent_pool.remaining
    )
    rotated = opponent_pool.remaining[rotation:] + opponent_pool.remaining[:rotation]
    selected = iter(rotated[: case.player_count - 1])
    return replace(
        case,
        opponent_names_by_seat=tuple(
            None if seat == case.focal_seat else next(selected).name
            for seat in range(case.player_count)
        ),
    )


def _case_repetition(case: PromotionCase) -> int:
    marker = ":repeat-"
    _, separator, raw_repetition = case.case_id.rpartition(marker)
    try:
        repetition = int(raw_repetition)
    except ValueError as error:
        raise PromotionPlanningError(
            "opponent_identity_mismatch",
            f"Promotion case {case.case_id!r} does not end with a valid repetition index.",
        ) from error
    if not separator or repetition < 0:
        raise PromotionPlanningError(
            "opponent_identity_mismatch",
            f"Promotion case {case.case_id!r} does not end with a valid repetition index.",
        )
    return repetition


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
    seen_opponent_names: set[str] = set()
    seen_opponent_ids: set[str] = set()
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
        if opponent.name in seen_opponent_names or opponent.bot_id in seen_opponent_ids:
            raise PromotionPlanningError(
                "opponent_identity_mismatch",
                f"Promotion case {case.case_id!r} places one opponent identity in "
                "more than one non-focal seat.",
            )
        seen_opponent_names.add(opponent.name)
        seen_opponent_ids.add(opponent.bot_id)
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


def effective_opponent_pool_payload(pool: EffectiveOpponentPool) -> dict[str, object]:
    """Render the exact configured, excluded, and remaining opponent identities."""

    return {
        "configured": [_bot_identity_payload(spec) for spec in pool.configured],
        "exclusions": [
            {
                "opponent": _bot_identity_payload(exclusion.opponent),
                "reason": exclusion.reason,
            }
            for exclusion in pool.exclusions
        ],
        "remaining": [_bot_identity_payload(spec) for spec in pool.remaining],
    }


def promotion_plan_payload(plan: PromotionPlan) -> dict[str, object]:
    """Render the canonical effective plan and its digest."""

    return {
        **_promotion_plan_digest_payload(plan),
        "digest": plan.digest,
    }


def _promotion_plan_digest(plan: PromotionPlan) -> str:
    payload = _promotion_plan_digest_payload(plan)
    encoded = (
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _promotion_plan_digest_payload(plan: PromotionPlan) -> dict[str, object]:
    return {
        "schema_version": 1,
        "source_corpus": {
            "name": plan.source_corpus_name,
            "digest": plan.source_corpus_digest,
        },
        "candidate": _bot_identity_payload(plan.candidate),
        "incumbent": _bot_identity_payload(plan.incumbent),
        "opponent_pool": effective_opponent_pool_payload(plan.opponent_pool),
        "pairs": [
            {
                "pair_index": pair.pair_index,
                "case": {
                    "case_id": pair.case.case_id,
                    "chart": pair.case.chart,
                    "player_count": pair.case.player_count,
                    "focal_seat": pair.case.focal_seat,
                    "engine_seed": pair.case.engine_seed,
                    "opponent_names_by_seat": list(pair.case.opponent_names_by_seat),
                },
                "candidate_lineup": [
                    _bot_identity_payload(spec) for spec in pair.candidate_game.lineup
                ],
                "incumbent_lineup": [
                    _bot_identity_payload(spec) for spec in pair.incumbent_game.lineup
                ],
            }
            for pair in plan.pairs
        ],
    }


def _bot_identity_payload(spec: BotSpec) -> dict[str, str]:
    return {"name": spec.name, "bot_id": spec.bot_id}
