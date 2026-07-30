"""Run the complete held-out promotion gate and preserve its evidence."""

from __future__ import annotations

import math
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from garboid_pocketrocks.bots import BotSpec
from garboid_pocketrocks.promotion.analysis import (
    PromotionAnalysis,
    PromotionFailure,
    analyze_promotion,
)
from garboid_pocketrocks.promotion.corpus import (
    PromotionCorpus,
    PromotionCorpusError,
    validate_corpus_separation,
)
from garboid_pocketrocks.promotion.planning import (
    PromotionPlan,
    PromotionPlanningError,
    plan_paired_games,
)
from garboid_pocketrocks.promotion.reporting import (
    PromotionArtifacts,
    PromotionReport,
    build_promotion_report,
    validate_artifact_output_dir,
    write_promotion_artifacts,
)
from garboid_pocketrocks.simulator.errors import SimulationError
from garboid_pocketrocks.simulator.monte_carlo import (
    MonteCarloResult,
    MonteCarloRunner,
)
from garboid_pocketrocks.tournament.rating import TournamentRatingError


@dataclass(frozen=True, slots=True)
class PromotionRunConfig:
    """Every strategy-neutral input needed for one promotion comparison."""

    candidate: BotSpec
    incumbent: BotSpec
    development: PromotionCorpus
    held_out: PromotionCorpus
    bootstrap_samples: int = 1_000
    bootstrap_seed: int = 0
    batch_size: int = 64


@dataclass(frozen=True, slots=True)
class PromotionRun:
    """The plan, simulation result, decision, and files from one gate run."""

    config: PromotionRunConfig
    plan: PromotionPlan | None
    monte_carlo_result: MonteCarloResult | None
    report: PromotionReport
    artifacts: PromotionArtifacts


class PromotionRunner:
    """Coordinate validation, paired simulation, analysis, and reporting."""

    @staticmethod
    def run(
        config: PromotionRunConfig,
        *,
        registry: Mapping[str, BotSpec],
        workers: int,
        output_dir: Path,
        overwrite: bool = False,
        repository_commit: str | None = None,
    ) -> PromotionRun:
        """Run one comparison, writing a fail-closed report for domain failures."""

        validate_artifact_output_dir(output_dir, overwrite=overwrite)
        resolved_repository_commit = (
            repository_commit if repository_commit is not None else _repository_commit()
        )
        plan: PromotionPlan | None = None
        result: MonteCarloResult | None = None
        try:
            validate_corpus_separation(config.development, config.held_out)
            plan = plan_paired_games(
                config.held_out,
                candidate=config.candidate,
                incumbent=config.incumbent,
                registry=registry,
            )
            result = MonteCarloRunner.run_jobs(
                plan.monte_carlo_config,
                plan.jobs,
                workers=workers,
                batch_size=config.batch_size,
            )
            analysis = analyze_promotion(
                plan,
                result,
                bootstrap_samples=config.bootstrap_samples,
                bootstrap_seed=config.bootstrap_seed,
                workers=workers,
            )
            analysis = _replace_nonfinite_analysis(analysis)
        except PromotionCorpusError as error:
            analysis = _failure_analysis(
                config,
                plan=plan,
                result=result,
                code=error.code,
                message=str(error),
            )
        except PromotionPlanningError as error:
            analysis = _failure_analysis(
                config,
                plan=plan,
                result=result,
                code=error.code,
                message=str(error),
            )
        except SimulationError as error:
            analysis = _failure_analysis(
                config,
                plan=plan,
                result=result,
                code="simulation_failed",
                message=f"The promotion games could not be completed: {error}",
            )
        except TournamentRatingError as error:
            analysis = _failure_analysis(
                config,
                plan=plan,
                result=result,
                code="rating_fit_failed",
                message=f"The rating model could not analyze the promotion games: {error}",
            )

        opponents = (
            plan.opponents
            if plan is not None
            else _resolved_report_opponents(config.held_out, registry=registry)
        )
        report = build_promotion_report(
            repository_commit=resolved_repository_commit,
            candidate=config.candidate,
            incumbent=config.incumbent,
            opponents=opponents,
            development=config.development,
            held_out=config.held_out,
            bootstrap_samples=config.bootstrap_samples,
            bootstrap_seed=config.bootstrap_seed,
            workers=workers,
            batch_size=config.batch_size,
            analysis=analysis,
        )
        artifacts = write_promotion_artifacts(
            output_dir,
            report=report,
            game_summaries=() if result is None else result.game_summaries,
            development=config.development,
            held_out=config.held_out,
            overwrite=overwrite,
        )
        return PromotionRun(
            config=config,
            plan=plan,
            monte_carlo_result=result,
            report=report,
            artifacts=artifacts,
        )


def _failure_analysis(
    config: PromotionRunConfig,
    *,
    plan: PromotionPlan | None,
    result: MonteCarloResult | None,
    code: str,
    message: str,
) -> PromotionAnalysis:
    requested_pairs = len(plan.pairs) if plan is not None else len(config.held_out.cases)
    requested_games = len(plan.jobs) if plan is not None else requested_pairs * 2
    completed_games = len(result.game_summaries) if result is not None else 0
    return PromotionAnalysis(
        requested_pairs=requested_pairs,
        completed_pairs=0,
        requested_games=requested_games,
        completed_games=completed_games,
        rating_difference=None,
        interval=None,
        bootstrap_requested=config.bootstrap_samples,
        bootstrap_converged=0,
        faults_by_identity=(),
        warnings=(),
        failures=(PromotionFailure(code=code, message=message),),
        promoted=False,
    )


def _replace_nonfinite_analysis(analysis: PromotionAnalysis) -> PromotionAnalysis:
    numeric_results = (
        () if analysis.rating_difference is None else (analysis.rating_difference,)
    ) + (() if analysis.interval is None else (analysis.interval.lower, analysis.interval.upper))
    if all(math.isfinite(value) for value in numeric_results):
        return analysis

    failure = PromotionFailure(
        code="nonfinite_analysis",
        message="The promotion analysis returned a number that was not finite.",
    )
    failures = tuple(
        sorted(
            {*analysis.failures, failure},
            key=lambda item: (item.code, item.message),
        )
    )
    return replace(
        analysis,
        rating_difference=None,
        interval=None,
        failures=failures,
        promoted=False,
    )


def _resolved_report_opponents(
    held_out: PromotionCorpus,
    *,
    registry: Mapping[str, BotSpec],
) -> tuple[BotSpec, ...]:
    opponents: list[BotSpec] = []
    seen_bot_ids: set[str] = set()
    for name in held_out.recipe.opponent_names:
        spec = registry.get(name)
        if spec is None or spec.name != name or spec.bot_id in seen_bot_ids:
            continue
        seen_bot_ids.add(spec.bot_id)
        opponents.append(spec)
    return tuple(opponents)


def _repository_commit() -> str:
    try:
        completed = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "Could not determine the repository commit with git rev-parse HEAD."
        ) from error
    commit = completed.stdout.strip()
    if not commit:
        raise RuntimeError("git did not return a repository commit")
    return commit
