from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from garboid_pocketrocks.simulator.monte_carlo import (
    MonteCarloResult,
    MonteCarloRunner,
)
from garboid_pocketrocks.tournament.analysis import (
    BootstrapSummary,
    TournamentAnalysis,
    analyze_tournament,
    bootstrap_rating_intervals,
)
from garboid_pocketrocks.tournament.rating import (
    PlackettLuceFit,
    fit_plackett_luce,
    observations_from_games,
)
from garboid_pocketrocks.tournament.reporting import (
    TournamentArtifacts,
    validate_artifact_output_dir,
    write_tournament_artifacts,
)
from garboid_pocketrocks.tournament.schedule import (
    TournamentConfig,
    TournamentPlan,
    TournamentPlanner,
)


@dataclass(frozen=True, slots=True)
class TournamentRun:
    config: TournamentConfig
    plan: TournamentPlan
    monte_carlo_result: MonteCarloResult
    fit: PlackettLuceFit
    analysis: TournamentAnalysis
    bootstrap: BootstrapSummary
    artifacts: TournamentArtifacts


class TournamentRunner:
    @staticmethod
    def run(
        config: TournamentConfig,
        *,
        workers: int = 1,
        output_dir: Path,
        overwrite: bool = False,
    ) -> TournamentRun:
        validate_artifact_output_dir(output_dir, overwrite=overwrite)
        plan = TournamentPlanner.plan(config)
        monte_carlo_result = MonteCarloRunner.run_jobs(
            plan.monte_carlo_config,
            plan.jobs,
            workers=workers,
            batch_size=config.batch_size,
        )
        observations = observations_from_games(monte_carlo_result.game_summaries)
        bot_ids = tuple(spec.bot_id for spec in config.bot_specs)
        fit = fit_plackett_luce(observations, bot_ids)
        analysis = analyze_tournament(monte_carlo_result, fit)
        try:
            bootstrap = bootstrap_rating_intervals(
                monte_carlo_result.game_summaries,
                bot_ids,
                samples=config.bootstrap_samples,
                root_seed=config.root_seed,
                workers=workers,
            )
        except Exception as error:
            bootstrap = BootstrapSummary(
                requested=config.bootstrap_samples,
                converged=0,
                intervals=(),
                warnings=(
                    f"bootstrap failed with {type(error).__name__}; "
                    "confidence intervals are unavailable",
                ),
            )
        if config.decision_reports:
            from garboid_pocketrocks.diagnostics.analysis import build_decision_report

            decision_report = build_decision_report(
                monte_carlo_result.decision_traces,
                game_summaries=monte_carlo_result.game_summaries,
                game_details=monte_carlo_result.game_details,
                bot_statistics=monte_carlo_result.bot_statistics,
                tournament_analysis=analysis,
            )
        else:
            decision_report = None
        artifacts = write_tournament_artifacts(
            output_dir=output_dir,
            overwrite=overwrite,
            config=config,
            plan=plan,
            fit=fit,
            analysis=analysis,
            bootstrap=bootstrap,
            decision_report=decision_report,
        )
        return TournamentRun(
            config=config,
            plan=plan,
            monte_carlo_result=monte_carlo_result,
            fit=fit,
            analysis=analysis,
            bootstrap=bootstrap,
            artifacts=artifacts,
        )
