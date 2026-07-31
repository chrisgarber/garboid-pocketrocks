"""Build matched, development-only jobs for one heuristic candidate."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from garboid_pocketrocks.bots import BotSpec
from garboid_pocketrocks.promotion.corpus import PromotionCase, PromotionCorpus
from garboid_pocketrocks.simulator.monte_carlo import GameJob, MonteCarloConfig
from garboid_pocketrocks.simulator.runner import FaultMode


@dataclass(frozen=True, slots=True)
class DevelopmentPlan:
    """Reusable incumbent evidence and matched jobs for one candidate."""

    corpus: PromotionCorpus
    candidate: BotSpec
    incumbent: BotSpec
    opponents: tuple[BotSpec, ...]
    baseline_config: MonteCarloConfig
    baseline_jobs: tuple[GameJob, ...]
    candidate_config: MonteCarloConfig
    candidate_jobs: tuple[GameJob, ...]


class DevelopmentPlanningError(ValueError):
    """Explain why matched development games cannot be planned safely."""

    code: str

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def plan_development_games(
    corpus: PromotionCorpus,
    *,
    candidate: BotSpec,
    incumbent: BotSpec,
    registry: Mapping[str, BotSpec],
) -> DevelopmentPlan:
    """Build exact development twins whose baseline can be reused."""

    if corpus.recipe.purpose != "development":
        raise DevelopmentPlanningError(
            "development_corpus_required",
            "Evolution may plan only a development corpus.",
        )
    if not corpus.cases:
        raise DevelopmentPlanningError(
            "empty_development_corpus",
            "Evolution requires at least one development case.",
        )
    engine_seeds = tuple(case.engine_seed for case in corpus.cases)
    if (
        not _is_nonnegative_integer(corpus.recipe.root_seed)
        or any(not _is_nonnegative_integer(seed) for seed in engine_seeds)
        or len(set(engine_seeds)) != len(engine_seeds)
    ):
        raise DevelopmentPlanningError(
            "invalid_development_seed",
            "Development root and engine seeds must be nonnegative integers, "
            "and every engine seed must be unique.",
        )
    if candidate.name != candidate.bot_id:
        raise DevelopmentPlanningError(
            "candidate_identity_mismatch",
            "Evolution candidates must use one explicit local name and bot ID.",
        )
    if _identities_overlap(candidate, incumbent):
        raise DevelopmentPlanningError(
            "candidate_incumbent_identity_collision",
            "Candidate and incumbent must have different names and bot IDs.",
        )

    opponents_by_name = _resolve_opponents(corpus, registry=registry)
    opponents = tuple(opponents_by_name.values())
    _require_no_compared_identity_collisions(
        candidate=candidate,
        incumbent=incumbent,
        opponents=opponents,
    )

    baseline_jobs: list[GameJob] = []
    candidate_jobs: list[GameJob] = []
    for case_index, case in enumerate(corpus.cases):
        _validate_case(case, opponents_by_name=opponents_by_name)
        baseline_jobs.append(
            _job(
                case_index,
                corpus=corpus,
                case=case,
                focal_bot=incumbent,
                opponents_by_name=opponents_by_name,
            )
        )
        candidate_jobs.append(
            _job(
                case_index,
                corpus=corpus,
                case=case,
                focal_bot=candidate,
                opponents_by_name=opponents_by_name,
            )
        )

    player_counts = tuple(dict.fromkeys(case.player_count for case in corpus.cases))
    charts = tuple(dict.fromkeys(case.chart for case in corpus.cases))
    return DevelopmentPlan(
        corpus=corpus,
        candidate=candidate,
        incumbent=incumbent,
        opponents=opponents,
        baseline_config=MonteCarloConfig(
            bot_specs=(incumbent, *opponents),
            games=len(corpus.cases),
            player_counts=player_counts,
            value_charts=charts,
            root_seed=corpus.recipe.root_seed,
            objectives_enabled=(True,),
            fault_mode=FaultMode.RECORD_AND_PASS,
        ),
        baseline_jobs=tuple(baseline_jobs),
        candidate_config=MonteCarloConfig(
            bot_specs=(candidate, *opponents),
            games=len(corpus.cases),
            player_counts=player_counts,
            value_charts=charts,
            root_seed=corpus.recipe.root_seed,
            objectives_enabled=(True,),
            fault_mode=FaultMode.RECORD_AND_PASS,
        ),
        candidate_jobs=tuple(candidate_jobs),
    )


def _resolve_opponents(
    corpus: PromotionCorpus,
    *,
    registry: Mapping[str, BotSpec],
) -> dict[str, BotSpec]:
    resolved: dict[str, BotSpec] = {}
    seen_ids: set[str] = set()
    for opponent_name in corpus.recipe.opponent_names:
        spec = registry.get(opponent_name)
        if (
            spec is None
            or spec.name != opponent_name
            or spec.bot_id in seen_ids
            or opponent_name in resolved
        ):
            raise DevelopmentPlanningError(
                "opponent_identity_mismatch",
                f"Development opponent {opponent_name!r} does not have "
                "one unique matching registered identity.",
            )
        resolved[opponent_name] = spec
        seen_ids.add(spec.bot_id)
    return resolved


def _require_no_compared_identity_collisions(
    *,
    candidate: BotSpec,
    incumbent: BotSpec,
    opponents: tuple[BotSpec, ...],
) -> None:
    for opponent in opponents:
        if _identities_overlap(candidate, opponent):
            raise DevelopmentPlanningError(
                "candidate_opponent_identity_collision",
                "Candidate identity must differ from every development opponent.",
            )
        if _identities_overlap(incumbent, opponent):
            raise DevelopmentPlanningError(
                "incumbent_opponent_identity_collision",
                "Incumbent identity must differ from every development opponent.",
            )


def _validate_case(
    case: PromotionCase,
    *,
    opponents_by_name: Mapping[str, BotSpec],
) -> None:
    if not 0 <= case.focal_seat < case.player_count:
        raise DevelopmentPlanningError(
            "invalid_development_case",
            f"Development case {case.case_id!r} has an invalid focal seat.",
        )
    if len(case.opponent_names_by_seat) != case.player_count:
        raise DevelopmentPlanningError(
            "invalid_development_case",
            f"Development case {case.case_id!r} has an incomplete lineup.",
        )
    for seat, opponent_name in enumerate(case.opponent_names_by_seat):
        if seat == case.focal_seat:
            if opponent_name is not None:
                raise DevelopmentPlanningError(
                    "invalid_development_case",
                    f"Development case {case.case_id!r} does not reserve its focal seat.",
                )
        elif opponent_name is None or opponent_name not in opponents_by_name:
            raise DevelopmentPlanningError(
                "invalid_development_case",
                f"Development case {case.case_id!r} has an unknown opponent lineup.",
            )
    opponent_names = tuple(name for name in case.opponent_names_by_seat if name is not None)
    opponent_ids = tuple(opponents_by_name[name].bot_id for name in opponent_names)
    if len(set(opponent_names)) != len(opponent_names) or len(set(opponent_ids)) != len(
        opponent_ids
    ):
        raise DevelopmentPlanningError(
            "invalid_development_case",
            f"Development case {case.case_id!r} must use distinct opponent "
            "names and bot IDs at every non-focal seat.",
        )


def _job(
    game_index: int,
    *,
    corpus: PromotionCorpus,
    case: PromotionCase,
    focal_bot: BotSpec,
    opponents_by_name: Mapping[str, BotSpec],
) -> GameJob:
    lineup = tuple(
        focal_bot if seat == case.focal_seat else opponents_by_name[_required_name(name)]
        for seat, name in enumerate(case.opponent_names_by_seat)
    )
    return GameJob(
        game_index=game_index,
        root_seed=corpus.recipe.root_seed,
        seed=case.engine_seed,
        player_count=case.player_count,
        value_chart=case.chart,
        objectives_enabled=True,
        lineup=lineup,
        fault_mode=FaultMode.RECORD_AND_PASS,
    )


def _required_name(name: str | None) -> str:
    assert name is not None
    return name


def _identities_overlap(first: BotSpec, second: BotSpec) -> bool:
    return first.name == second.name or first.bot_id == second.bot_id


def _is_nonnegative_integer(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0
