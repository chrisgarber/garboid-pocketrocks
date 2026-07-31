"""Execute deterministic heuristic generations on development games only."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from garboid_pocketrocks.bots import BOT_SPECS_BY_NAME, BotSpec
from garboid_pocketrocks.evolution.candidates import (
    HeuristicCandidate,
    build_initial_population,
    build_mutation_population,
    candidate_bot_spec,
)
from garboid_pocketrocks.evolution.evaluation import (
    CandidateEvaluation,
    evaluate_candidate,
)
from garboid_pocketrocks.evolution.manifest import (
    SearchManifest,
    recompute_search_manifest_digest,
)
from garboid_pocketrocks.evolution.planning import (
    DevelopmentPlan,
    plan_development_games,
)
from garboid_pocketrocks.evolution.search import (
    EvaluatedCandidate,
    GenerationSelection,
    SearchSelectionError,
    SelectionRecord,
    freeze_candidate,
    select_generation,
)
from garboid_pocketrocks.promotion.corpus import (
    PromotionCorpus,
    recompute_promotion_corpus_digest,
)
from garboid_pocketrocks.simulator.monte_carlo import (
    GameJob,
    MonteCarloConfig,
    MonteCarloResult,
    MonteCarloRunner,
)


@dataclass(frozen=True, slots=True)
class SearchFailure:
    """One stable reason a deterministic search could not complete."""

    code: str
    message: str


@dataclass(frozen=True, slots=True)
class CandidateRun:
    """One proposal, its exact plan and games, and its evaluation."""

    candidate: HeuristicCandidate
    plan: DevelopmentPlan
    result: MonteCarloResult
    evaluation: CandidateEvaluation

    @property
    def evaluated_candidate(self) -> EvaluatedCandidate:
        """Return the proposal/evaluation pair consumed by selection."""

        return EvaluatedCandidate(
            candidate=self.candidate,
            evaluation=self.evaluation,
        )


@dataclass(frozen=True, slots=True)
class SearchRun:
    """All in-memory source evidence and decisions from one search."""

    manifest: SearchManifest
    development_corpus: PromotionCorpus
    incumbent: BotSpec
    baseline_config: MonteCarloConfig
    baseline_jobs: tuple[GameJob, ...]
    baseline_result: MonteCarloResult
    candidate_runs: tuple[CandidateRun, ...]
    selections: tuple[GenerationSelection, ...]
    selected_candidate: HeuristicCandidate | None
    frozen_candidate: HeuristicCandidate | None
    failures: tuple[SearchFailure, ...]

    @property
    def selection_records(self) -> tuple[SelectionRecord, ...]:
        """Return selection evidence in generation order."""

        return tuple(selection.record for selection in self.selections)


class SearchRunError(ValueError):
    """Explain why a search invocation cannot begin safely."""

    code: str

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def run_search(
    manifest: SearchManifest,
    development_corpus: PromotionCorpus,
    *,
    registry: Mapping[str, BotSpec],
    workers: int,
    batch_size: int | None = None,
) -> SearchRun:
    """Run every deterministic proposal with one cached incumbent baseline."""

    _validate_invocation(
        manifest,
        development_corpus,
        registry=registry,
        workers=workers,
        batch_size=batch_size,
    )
    incumbent = registry[manifest.predecessor_name]
    population = build_initial_population(manifest)
    first_plan = plan_development_games(
        development_corpus,
        candidate=candidate_bot_spec(population[0]),
        incumbent=incumbent,
        registry=registry,
    )
    baseline_result = MonteCarloRunner.run_jobs(
        first_plan.baseline_config,
        first_plan.baseline_jobs,
        workers=workers,
        batch_size=batch_size,
    )

    candidate_runs: list[CandidateRun] = []
    selections: list[GenerationSelection] = []
    prior_elites: tuple[EvaluatedCandidate, ...] = ()
    for generation in range(manifest.algorithm.generation_count):
        if generation > 0:
            population = build_mutation_population(
                manifest,
                generation=generation,
                ranked_elites=tuple(item.candidate for item in prior_elites),
            )
        generation_runs = tuple(
            _run_candidate(
                candidate,
                development_corpus=development_corpus,
                incumbent=incumbent,
                registry=registry,
                baseline_config=first_plan.baseline_config,
                baseline_jobs=first_plan.baseline_jobs,
                baseline_result=baseline_result,
                workers=workers,
                batch_size=batch_size,
                prepared_plan=first_plan if generation == 0 and candidate.slot == 0 else None,
            )
            for candidate in population
        )
        candidate_runs.extend(generation_runs)
        evaluated = tuple(item.evaluated_candidate for item in generation_runs)
        try:
            selection = select_generation(
                generation=generation,
                proposals=evaluated,
                prior_elites=prior_elites,
                elite_count=manifest.algorithm.elite_count,
            )
        except SearchSelectionError as error:
            return SearchRun(
                manifest=manifest,
                development_corpus=development_corpus,
                incumbent=incumbent,
                baseline_config=first_plan.baseline_config,
                baseline_jobs=first_plan.baseline_jobs,
                baseline_result=baseline_result,
                candidate_runs=tuple(candidate_runs),
                selections=tuple(selections),
                selected_candidate=None,
                frozen_candidate=None,
                failures=(SearchFailure(error.code, str(error)),),
            )
        selections.append(selection)
        prior_elites = selection.elites

    winner = prior_elites[0]
    return SearchRun(
        manifest=manifest,
        development_corpus=development_corpus,
        incumbent=incumbent,
        baseline_config=first_plan.baseline_config,
        baseline_jobs=first_plan.baseline_jobs,
        baseline_result=baseline_result,
        candidate_runs=tuple(candidate_runs),
        selections=tuple(selections),
        selected_candidate=winner.candidate,
        frozen_candidate=freeze_candidate(winner),
        failures=(),
    )


def _run_candidate(
    candidate: HeuristicCandidate,
    *,
    development_corpus: PromotionCorpus,
    incumbent: BotSpec,
    registry: Mapping[str, BotSpec],
    baseline_config: MonteCarloConfig,
    baseline_jobs: tuple[GameJob, ...],
    baseline_result: MonteCarloResult,
    workers: int,
    batch_size: int | None,
    prepared_plan: DevelopmentPlan | None,
) -> CandidateRun:
    plan = (
        prepared_plan
        if prepared_plan is not None
        else plan_development_games(
            development_corpus,
            candidate=candidate_bot_spec(candidate),
            incumbent=incumbent,
            registry=registry,
        )
    )
    if plan.baseline_config != baseline_config or plan.baseline_jobs != baseline_jobs:
        raise SearchRunError(
            "baseline_plan_mismatch",
            f"Candidate {candidate.identity!r} did not reproduce the cached baseline plan.",
        )
    result = MonteCarloRunner.run_jobs(
        plan.candidate_config,
        plan.candidate_jobs,
        workers=workers,
        batch_size=batch_size,
    )
    evaluation = evaluate_candidate(plan, baseline_result, result)
    return CandidateRun(
        candidate=candidate,
        plan=plan,
        result=result,
        evaluation=evaluation,
    )


def _validate_invocation(
    manifest: SearchManifest,
    development_corpus: PromotionCorpus,
    *,
    registry: Mapping[str, BotSpec],
    workers: int,
    batch_size: int | None,
) -> None:
    if workers <= 0:
        raise SearchRunError("invalid_workers", "Search workers must be positive.")
    if batch_size is not None and batch_size <= 0:
        raise SearchRunError("invalid_batch_size", "Search batch size must be positive.")
    if recompute_search_manifest_digest(manifest) != manifest.digest:
        raise SearchRunError(
            "stale_manifest_digest",
            "The search manifest digest does not match its normalized content.",
        )
    if recompute_promotion_corpus_digest(development_corpus) != development_corpus.digest:
        raise SearchRunError(
            "stale_development_corpus_digest",
            "The development corpus digest does not match its normalized recipe and cases.",
        )
    if (
        development_corpus.recipe.purpose != "development"
        or development_corpus.recipe.name != manifest.development_corpus.name
        or development_corpus.digest != manifest.development_corpus.digest
    ):
        raise SearchRunError(
            "development_corpus_mismatch",
            "The search corpus must match the manifest's exact development binding.",
        )
    incumbent = registry.get(manifest.predecessor_name)
    if incumbent is None or incumbent.name != manifest.predecessor_name:
        raise SearchRunError(
            "unknown_predecessor",
            f"Manifest predecessor {manifest.predecessor_name!r} is not registered.",
        )
    canonical_incumbent = BOT_SPECS_BY_NAME.get(manifest.predecessor_name)
    if canonical_incumbent is None or incumbent is not canonical_incumbent:
        raise SearchRunError(
            "noncanonical_predecessor",
            f"Manifest predecessor {manifest.predecessor_name!r} must be the "
            "exact canonical released registry spec.",
        )
    for opponent_name in development_corpus.recipe.opponent_names:
        canonical_opponent = BOT_SPECS_BY_NAME.get(opponent_name)
        if canonical_opponent is None or registry.get(opponent_name) is not canonical_opponent:
            raise SearchRunError(
                "noncanonical_opponent",
                f"Development opponent {opponent_name!r} must be the "
                "exact canonical released registry spec.",
            )
    algorithm = manifest.algorithm
    if (
        algorithm.generation_count <= 0
        or algorithm.population_size <= 0
        or algorithm.elite_count <= 0
        or algorithm.elite_count > algorithm.population_size
    ):
        raise SearchRunError(
            "invalid_algorithm",
            "Search generations, population, and elites must form a positive valid algorithm.",
        )
