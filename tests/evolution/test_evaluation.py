from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import garboid_pocketrocks.evolution.evaluation as evaluation_module
from garboid_pocketrocks.bots import (
    BOT_SPECS_BY_NAME,
    BotSpec,
    RandomBot,
)
from garboid_pocketrocks.evolution.candidates import (
    build_initial_population,
    candidate_bot_spec,
)
from garboid_pocketrocks.evolution.evaluation import evaluate_candidate
from garboid_pocketrocks.evolution.manifest import load_search_recipe
from garboid_pocketrocks.evolution.planning import (
    DevelopmentPlan,
    plan_development_games,
)
from garboid_pocketrocks.knowledge import ruleset_name
from garboid_pocketrocks.promotion.corpus import (
    PromotionCase,
    PromotionCorpus,
    PromotionCorpusRecipe,
    load_promotion_corpus,
)
from garboid_pocketrocks.simulator.monte_carlo import (
    GameJob,
    GameSummary,
    MonteCarloResult,
    MonteCarloRunner,
)
from garboid_pocketrocks.simulator.session import SessionScore

REPOSITORY_ROOT = Path(__file__).parents[2]


def test_computes_rating_and_exact_paired_tie_break_deltas() -> None:
    plan = _plan()
    baseline, candidate = _winning_candidate_results(plan)

    evaluation = evaluate_candidate(plan, baseline, candidate)

    assert evaluation.candidate_identity == plan.candidate.name
    assert evaluation.requested_cases == 3
    assert evaluation.completed_baseline_games == 3
    assert evaluation.completed_candidate_games == 3
    assert evaluation.rating_delta is not None
    assert evaluation.rating_delta > 0
    assert evaluation.normalized_finish_delta == 3.0
    assert evaluation.final_money_delta == 300
    assert evaluation.candidate_faults == 0
    assert evaluation.incumbent_faults == 0
    assert evaluation.opponent_faults == 0
    assert evaluation.unattributed_faults == 0
    assert evaluation.failures == ()
    assert evaluation.valid is True
    assert evaluation.eligible is True


def test_missing_evidence_invalidates_the_evaluation_without_partial_scores() -> None:
    plan = _plan()
    baseline, candidate = _winning_candidate_results(plan)
    baseline = replace(baseline, game_summaries=baseline.game_summaries[:-1])

    evaluation = evaluate_candidate(plan, baseline, candidate)

    assert _failure_codes(evaluation) == ("missing_baseline_game",)
    assert evaluation.completed_baseline_games == 2
    assert evaluation.completed_candidate_games == 3
    assert evaluation.rating_delta is None
    assert evaluation.normalized_finish_delta is None
    assert evaluation.final_money_delta is None
    assert evaluation.valid is False
    assert evaluation.eligible is False


def test_candidate_fault_is_recorded_as_ineligible_but_does_not_corrupt_baseline() -> None:
    plan = _plan()
    baseline, candidate = _winning_candidate_results(plan)
    summary = candidate.game_summaries[0]
    focal_seat = plan.corpus.cases[0].focal_seat
    faults = tuple(2 if seat == focal_seat else 0 for seat in range(summary.player_count))
    candidate = _replace_summary(candidate, 0, fault_counts=faults)

    evaluation = evaluate_candidate(plan, baseline, candidate)

    assert _failure_codes(evaluation) == ("candidate_fault",)
    assert evaluation.failures[0].invalidates_run is False
    assert evaluation.candidate_faults == 2
    assert evaluation.incumbent_faults == 0
    assert evaluation.opponent_faults == 0
    assert evaluation.rating_delta is not None
    assert evaluation.valid is True
    assert evaluation.eligible is False


@pytest.mark.parametrize("evidence", ("baseline", "candidate"))
def test_incumbent_or_opponent_fault_invalidates_reusable_evidence(evidence: str) -> None:
    plan = _plan()
    baseline, candidate = _winning_candidate_results(plan)
    result = baseline if evidence == "baseline" else candidate
    summary = result.game_summaries[0]
    focal_seat = plan.corpus.cases[0].focal_seat
    fault_seat = focal_seat if evidence == "baseline" else (focal_seat + 1) % 3
    faults = tuple(1 if seat == fault_seat else 0 for seat in range(summary.player_count))
    result = _replace_summary(result, 0, fault_counts=faults)
    if evidence == "baseline":
        baseline = result
    else:
        candidate = result

    evaluation = evaluate_candidate(plan, baseline, candidate)

    expected_code = "incumbent_fault" if evidence == "baseline" else "opponent_fault"
    assert _failure_codes(evaluation) == (expected_code,)
    assert evaluation.failures[0].invalidates_run is True
    assert evaluation.valid is False
    assert evaluation.eligible is False
    assert evaluation.rating_delta is None


@pytest.mark.parametrize(
    ("changes", "expected_code"),
    (
        ({"root_seed": -1}, "seed_mismatch"),
        ({"seed": -1}, "seed_mismatch"),
        ({"ruleset_name": "live-E"}, "ruleset_mismatch"),
        ({"player_count": 4}, "player_count_mismatch"),
        ({"bot_names": ("wrong", "opponent-a", "opponent-b")}, "identity_mismatch"),
        ({"bot_ids": ("wrong", "opponent-a", "opponent-b")}, "identity_mismatch"),
        (
            {
                "scores": (
                    SessionScore(seat=0, final_money=100, rank=1),
                    SessionScore(seat=0, final_money=20, rank=2),
                    SessionScore(seat=2, final_money=10, rank=3),
                )
            },
            "player_count_mismatch",
        ),
        ({"decision_counts": (10, 10)}, "player_count_mismatch"),
        ({"fault_counts": (0, 0)}, "player_count_mismatch"),
        (
            {
                "scores": (
                    SessionScore(seat=0, final_money=100, rank=0),
                    SessionScore(seat=1, final_money=20, rank=2),
                    SessionScore(seat=2, final_money=10, rank=3),
                )
            },
            "invalid_game_evidence",
        ),
    ),
)
def test_one_mutated_candidate_field_invalidates_all_scores(
    changes: dict[str, object],
    expected_code: str,
) -> None:
    plan = _plan()
    baseline, candidate = _winning_candidate_results(plan)
    candidate = _replace_summary(candidate, 0, **changes)

    evaluation = evaluate_candidate(plan, baseline, candidate)

    assert expected_code in _failure_codes(evaluation)
    assert evaluation.valid is False
    assert evaluation.eligible is False
    assert evaluation.rating_delta is None
    assert evaluation.normalized_finish_delta is None
    assert evaluation.final_money_delta is None


def test_duplicate_and_unexpected_evidence_cannot_satisfy_case_coverage() -> None:
    plan = _plan()
    baseline, candidate = _winning_candidate_results(plan)
    duplicate = baseline.game_summaries[0]
    unexpected = replace(candidate.game_summaries[0], game_index=99)
    baseline = replace(
        baseline,
        game_summaries=(duplicate, duplicate, *baseline.game_summaries[1:]),
    )
    candidate = replace(candidate, game_summaries=(unexpected, *candidate.game_summaries[1:]))

    evaluation = evaluate_candidate(plan, baseline, candidate)

    assert _failure_codes(evaluation) == (
        "duplicate_baseline_game",
        "missing_candidate_game",
        "unexpected_candidate_game",
    )
    assert evaluation.completed_baseline_games == 2
    assert evaluation.completed_candidate_games == 2
    assert evaluation.valid is False


def test_faults_with_untrusted_identity_are_not_misattributed_to_the_candidate() -> None:
    plan = _plan()
    baseline, candidate = _winning_candidate_results(plan)
    candidate = _replace_summary(
        candidate,
        0,
        bot_ids=("untrusted", "opponent-a", "opponent-b"),
        fault_counts=(4, 0, 0),
    )

    evaluation = evaluate_candidate(plan, baseline, candidate)

    assert _failure_codes(evaluation) == ("identity_mismatch", "unattributed_fault")
    assert evaluation.candidate_faults == 0
    assert evaluation.unattributed_faults == 4
    assert evaluation.faults_by_identity == ()
    assert evaluation.valid is False


def test_rank_must_be_derived_from_final_money_before_scoring() -> None:
    plan = _plan()
    baseline, candidate = _winning_candidate_results(plan)
    summary = candidate.game_summaries[0]
    candidate = _replace_summary(
        candidate,
        0,
        scores=tuple(
            replace(score, rank=2 if score.seat == 0 else 1 if score.seat == 1 else 3)
            for score in summary.scores
        ),
    )

    evaluation = evaluate_candidate(plan, baseline, candidate)

    assert _failure_codes(evaluation) == ("invalid_game_evidence",)
    assert evaluation.valid is False
    assert evaluation.rating_delta is None


def test_equal_final_money_requires_equal_competition_ranks() -> None:
    plan = _plan()
    baseline, candidate = _winning_candidate_results(plan)
    candidate = _replace_summary(
        candidate,
        0,
        scores=(
            SessionScore(seat=0, final_money=100, rank=1),
            SessionScore(seat=1, final_money=100, rank=1),
            SessionScore(seat=2, final_money=25, rank=3),
        ),
    )

    evaluation = evaluate_candidate(plan, baseline, candidate)

    assert "invalid_game_evidence" not in _failure_codes(evaluation)
    assert evaluation.valid is True


def test_malformed_rating_observation_is_invalid_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    baseline, candidate = _winning_candidate_results(plan)

    def malformed_observation(*args: object) -> object:
        raise ValueError("duplicate bot identity in ranking")

    monkeypatch.setattr(
        evaluation_module,
        "observations_from_games",
        malformed_observation,
    )

    evaluation = evaluate_candidate(plan, baseline, candidate)

    assert _failure_codes(evaluation) == ("rating_observation_failed",)
    assert evaluation.valid is False
    assert evaluation.rating_delta is None


@pytest.mark.parametrize(
    "manifest_name",
    ("balanced-v3-search-v1.json", "balanced-v4-search-v2.json"),
)
def test_real_development_jobs_match_across_serial_batch_and_workers(
    manifest_name: str,
) -> None:
    corpus = load_promotion_corpus(
        REPOSITORY_ROOT / "configs/promotion/development-v1.json",
        registry=BOT_SPECS_BY_NAME,
    )
    manifest = load_search_recipe(
        REPOSITORY_ROOT / "configs/evolution" / manifest_name,
        development_corpus=corpus,
    )
    candidate = candidate_bot_spec(build_initial_population(manifest)[1])
    small_corpus = replace(corpus, cases=corpus.cases[:2], digest="small-real-corpus")
    plan = plan_development_games(
        small_corpus,
        candidate=candidate,
        incumbent=BOT_SPECS_BY_NAME[manifest.predecessor_name],
        registry=BOT_SPECS_BY_NAME,
    )

    serial_baseline = MonteCarloRunner.run_jobs(
        plan.baseline_config,
        plan.baseline_jobs,
        workers=1,
    )
    serial_candidate = MonteCarloRunner.run_jobs(
        plan.candidate_config,
        plan.candidate_jobs,
        workers=1,
    )
    batched_baseline = MonteCarloRunner.run_jobs(
        plan.baseline_config,
        plan.baseline_jobs,
        workers=1,
        batch_size=1,
    )
    batched_candidate = MonteCarloRunner.run_jobs(
        plan.candidate_config,
        plan.candidate_jobs,
        workers=1,
        batch_size=1,
    )
    worker_baseline = MonteCarloRunner.run_jobs(
        plan.baseline_config,
        plan.baseline_jobs,
        workers=2,
        batch_size=1,
    )
    worker_candidate = MonteCarloRunner.run_jobs(
        plan.candidate_config,
        plan.candidate_jobs,
        workers=2,
        batch_size=1,
    )

    assert batched_baseline == serial_baseline == worker_baseline
    assert batched_candidate == serial_candidate == worker_candidate
    expected = evaluate_candidate(plan, serial_baseline, serial_candidate)
    assert evaluate_candidate(plan, batched_baseline, batched_candidate) == expected
    assert evaluate_candidate(plan, worker_baseline, worker_candidate) == expected
    assert expected.valid is True
    assert expected.eligible is True


def _failure_codes(evaluation: object) -> tuple[str, ...]:
    return tuple(failure.code for failure in evaluation.failures)  # type: ignore[attr-defined]


def _plan() -> DevelopmentPlan:
    candidate = _spec("balanced-v3-candidate-test")
    incumbent = _spec("balanced-v2")
    opponents = (_spec("opponent-a"), _spec("opponent-b"))
    cases = tuple(
        PromotionCase(
            case_id=f"fixture-development-v1:{chr(65 + index)}:3:seat-{index}:repeat-0",
            chart=chr(65 + index),
            player_count=3,
            focal_seat=index,
            engine_seed=10_001 + index,
            opponent_names_by_seat=_opponents_by_seat(index),
        )
        for index in range(3)
    )
    corpus = PromotionCorpus(
        recipe=PromotionCorpusRecipe(
            schema_version=1,
            name="fixture-development-v1",
            purpose="development",
            root_seed=9_001,
            repetitions_per_seat_cell=1,
            charts=("A", "B", "C"),
            player_counts=(3,),
            opponent_names=tuple(spec.name for spec in opponents),
        ),
        cases=cases,
        digest="d" * 64,
    )
    return plan_development_games(
        corpus,
        candidate=candidate,
        incumbent=incumbent,
        registry={spec.name: spec for spec in opponents},
    )


def _winning_candidate_results(
    plan: DevelopmentPlan,
) -> tuple[MonteCarloResult, MonteCarloResult]:
    baseline = tuple(
        _summary_for_job(
            job,
            focal_seat=case.focal_seat,
            focal_money=0,
            focal_rank=3,
        )
        for case, job in zip(plan.corpus.cases, plan.baseline_jobs, strict=True)
    )
    candidate = tuple(
        _summary_for_job(
            job,
            focal_seat=case.focal_seat,
            focal_money=100,
            focal_rank=1,
        )
        for case, job in zip(plan.corpus.cases, plan.candidate_jobs, strict=True)
    )
    return _result(baseline), _result(candidate)


def _summary_for_job(
    job: GameJob,
    *,
    focal_seat: int,
    focal_money: int,
    focal_rank: int,
) -> GameSummary:
    other_seats = tuple(seat for seat in range(job.player_count) if seat != focal_seat)
    other_ranks = (1, 2) if focal_rank == 3 else (2, 3)
    ranks = {
        focal_seat: focal_rank,
        **dict(zip(other_seats, other_ranks, strict=True)),
    }
    money = {
        focal_seat: focal_money,
        **dict(zip(other_seats, (50, 25), strict=True)),
    }
    return GameSummary(
        game_index=job.game_index,
        root_seed=job.root_seed,
        seed=job.seed,
        player_count=job.player_count,
        ruleset_name=ruleset_name(job.value_chart, job.objectives_enabled),
        bot_names=tuple(spec.name for spec in job.lineup),
        bot_ids=tuple(spec.bot_id for spec in job.lineup),
        scores=tuple(
            SessionScore(
                seat=seat,
                final_money=money[seat],
                rank=ranks[seat],
            )
            for seat in range(job.player_count)
        ),
        decision_counts=(10,) * job.player_count,
        fault_counts=(0,) * job.player_count,
    )


def _result(summaries: tuple[GameSummary, ...]) -> MonteCarloResult:
    return MonteCarloResult(
        game_summaries=summaries,
        bot_statistics=(),
        replays=(),
    )


def _replace_summary(
    result: MonteCarloResult,
    game_index: int,
    **changes: object,
) -> MonteCarloResult:
    return replace(
        result,
        game_summaries=tuple(
            replace(summary, **changes)  # type: ignore[arg-type]
            if summary.game_index == game_index
            else summary
            for summary in result.game_summaries
        ),
    )


def _spec(name: str) -> BotSpec:
    return BotSpec.for_simulation(name, RandomBot.build_brain)


def _opponents_by_seat(focal_seat: int) -> tuple[str | None, ...]:
    opponents = iter(("opponent-a", "opponent-b"))
    return tuple(None if seat == focal_seat else next(opponents) for seat in range(3))
