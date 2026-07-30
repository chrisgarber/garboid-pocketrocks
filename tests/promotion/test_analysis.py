from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

import garboid_pocketrocks.promotion.analysis as analysis_module
from garboid_pocketrocks.promotion.analysis import (
    PromotionFailure,
    analyze_promotion,
    bootstrap_paired_rating_differences,
)
from garboid_pocketrocks.simulator.session import SessionScore
from garboid_pocketrocks.tournament.rating import (
    TournamentRatingError,
    observations_from_games,
)

from .helpers import (
    promotion_plan,
    replace_summary,
    result_for_plan,
    summary_for_job,
)


def _failure_codes(analysis: Any) -> tuple[str, ...]:
    return tuple(failure.code for failure in analysis.failures)


@pytest.mark.parametrize(
    ("change", "expected_code"),
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
        ({"fault_counts": (0, 1, 0)}, "bot_fault"),
    ),
)
def test_validation_fails_closed_for_one_mutated_field(
    change: dict[str, object],
    expected_code: str,
) -> None:
    plan = promotion_plan(pair_count=1)
    result = replace_summary(result_for_plan(plan), 0, **change)

    analysis = analyze_promotion(
        plan,
        result,
        bootstrap_samples=10,
        bootstrap_seed=42,
    )

    assert expected_code in _failure_codes(analysis)
    assert analysis.completed_pairs == int(expected_code == "bot_fault")
    assert analysis.promoted is False


def test_reports_missing_and_unexpected_games_in_stable_order() -> None:
    plan = promotion_plan(pair_count=1)
    result = result_for_plan(plan)
    unexpected = replace(result.game_summaries[0], game_index=99)
    result = replace(result, game_summaries=(unexpected,))

    analysis = analyze_promotion(
        plan,
        result,
        bootstrap_samples=10,
        bootstrap_seed=42,
    )

    assert analysis.failures == tuple(
        sorted(
            set(analysis.failures),
            key=lambda item: (item.code, item.message),
        )
    )
    assert _failure_codes(analysis) == (
        "missing_paired_game",
        "missing_paired_game",
        "unexpected_game",
    )
    assert analysis.completed_games == 0
    assert analysis.completed_pairs == 0


def test_duplicate_game_index_is_unexpected_and_cannot_complete_a_pair() -> None:
    plan = promotion_plan(pair_count=1)
    result = result_for_plan(plan)
    result = replace(
        result,
        game_summaries=(
            result.game_summaries[0],
            result.game_summaries[0],
            result.game_summaries[1],
        ),
    )

    analysis = analyze_promotion(
        plan,
        result,
        bootstrap_samples=10,
        bootstrap_seed=42,
    )

    assert "unexpected_game" in _failure_codes(analysis)
    assert analysis.completed_games == 1
    assert analysis.completed_pairs == 0


def test_any_nonzero_fault_is_counted_by_identity_and_fails() -> None:
    plan = promotion_plan(pair_count=1)
    result = result_for_plan(plan)
    faulty = replace(result.game_summaries[1], fault_counts=(2, 0, 3))
    result = replace(result, game_summaries=(result.game_summaries[0], faulty))

    analysis = analyze_promotion(
        plan,
        result,
        bootstrap_samples=10,
        bootstrap_seed=42,
    )

    assert _failure_codes(analysis) == ("bot_fault",)
    expected_faults = tuple(
        sorted(
            (
                (faulty.bot_ids[0], 2),
                (faulty.bot_ids[2], 3),
            )
        )
    )
    assert analysis.faults_by_identity == expected_faults


def test_faults_remain_attributed_when_the_same_summary_has_a_seed_mismatch() -> None:
    plan = promotion_plan(pair_count=1)
    result = result_for_plan(plan)
    faulty = replace(
        result.game_summaries[0],
        seed=-1,
        fault_counts=(2, 0, 3),
    )
    result = replace(result, game_summaries=(faulty, result.game_summaries[1]))

    analysis = analyze_promotion(
        plan,
        result,
        bootstrap_samples=10,
        bootstrap_seed=42,
    )

    assert _failure_codes(analysis) == ("bot_fault", "seed_mismatch")
    assert analysis.faults_by_identity == tuple(
        sorted(
            (
                (faulty.bot_ids[0], 2),
                (faulty.bot_ids[2], 3),
            )
        )
    )
    assert analysis.promoted is False


@pytest.mark.parametrize(
    "changes",
    (
        {
            "bot_names": ("untrusted-name", "opponent-a", "opponent-b"),
            "fault_counts": (4, 0, 0),
        },
        {
            "bot_ids": ("untrusted-id", "opponent-a", "opponent-b"),
            "fault_counts": (4, 0, 0),
        },
        {
            "bot_ids": ("candidate", "opponent-a"),
            "fault_counts": (4, 0, 0),
        },
        {
            "bot_ids": ("candidate", "opponent-a", "opponent-b"),
            "fault_counts": (4, 0),
        },
    ),
)
def test_faults_are_not_attributed_through_malformed_or_mismatched_identities(
    changes: dict[str, object],
) -> None:
    plan = promotion_plan(pair_count=1)
    result = replace_summary(result_for_plan(plan), 0, **changes)

    analysis = analyze_promotion(
        plan,
        result,
        bootstrap_samples=10,
        bootstrap_seed=42,
    )

    assert "bot_fault" in _failure_codes(analysis)
    assert analysis.faults_by_identity == ()
    assert analysis.promoted is False


def test_fits_rating_difference_and_deterministic_interval() -> None:
    plan = promotion_plan()
    result = result_for_plan(plan)

    first = analyze_promotion(
        plan,
        result,
        bootstrap_samples=100,
        bootstrap_seed=42,
        workers=1,
    )
    parallel = analyze_promotion(
        plan,
        result,
        bootstrap_samples=100,
        bootstrap_seed=42,
        workers=2,
    )

    assert first.rating_difference is not None
    assert first.rating_difference > 0
    assert first.interval is not None
    assert first.bootstrap_converged >= 90
    assert parallel == first


def test_bootstrap_resamples_complete_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = promotion_plan()
    candidate_rankings = (
        {"candidate": 1, "opponent-a": 2, "opponent-b": 3},
        {"candidate": 1, "opponent-a": 3, "opponent-b": 2},
        {"candidate": 1, "opponent-a": 2, "opponent-b": 2},
    )
    incumbent_rankings = (
        {"incumbent": 3, "opponent-a": 1, "opponent-b": 2},
        {"incumbent": 3, "opponent-a": 2, "opponent-b": 1},
        {"incumbent": 2, "opponent-a": 1, "opponent-b": 1},
    )
    pairs = tuple(
        (
            summary_for_job(
                pair.candidate_game,
                final_money=(100, 50, 0),
                ranks=tuple(
                    candidate_rankings[pair.pair_index][spec.bot_id]
                    for spec in pair.candidate_game.lineup
                ),
            ),
            summary_for_job(
                pair.incumbent_game,
                final_money=(100, 50, 0),
                ranks=tuple(
                    incumbent_rankings[pair.pair_index][spec.bot_id]
                    for spec in pair.incumbent_game.lineup
                ),
            ),
        )
        for pair in plan.pairs
    )
    pair_signatures = tuple(
        (
            observations_from_games((candidate_game,))[0].rank_groups,
            observations_from_games((incumbent_game,))[0].rank_groups,
        )
        for candidate_game, incumbent_game in pairs
    )
    assert len({signature for pair in pair_signatures for signature in pair}) == 6
    recorded_groups: list[tuple[tuple[tuple[str, ...], ...], ...]] = []
    real_fit = analysis_module.fit_plackett_luce

    def record_fit(observations: Any, bot_ids: tuple[str, ...]) -> Any:
        recorded_groups.append(tuple(observation.rank_groups for observation in observations))
        return real_fit(observations, bot_ids)

    monkeypatch.setattr(analysis_module, "fit_plackett_luce", record_fit)

    bootstrap_paired_rating_differences(
        pairs,
        tuple(spec.bot_id for spec in plan.monte_carlo_config.bot_specs),
        candidate_id=plan.candidate.bot_id,
        incumbent_id=plan.incumbent.bot_id,
        samples=10,
        root_seed=42,
    )

    for replicate in recorded_groups:
        for candidate_signature, incumbent_signature in pair_signatures:
            assert replicate.count(candidate_signature) == replicate.count(incumbent_signature)


def test_parallel_bootstrap_failure_retries_all_samples_serially(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = promotion_plan()
    result = result_for_plan(plan)
    pairs = tuple(
        (result.game_summaries[index], result.game_summaries[index + 1])
        for index in range(0, len(result.game_summaries), 2)
    )
    bot_ids = tuple(spec.bot_id for spec in plan.monte_carlo_config.bot_specs)
    expected = bootstrap_paired_rating_differences(
        pairs,
        bot_ids,
        candidate_id=plan.candidate.bot_id,
        incumbent_id=plan.incumbent.bot_id,
        samples=10,
        root_seed=42,
    )

    def unavailable_process_pool(*args: object, **kwargs: object) -> None:
        raise PermissionError("workers unavailable")

    monkeypatch.setattr(
        analysis_module,
        "ProcessPoolExecutor",
        unavailable_process_pool,
    )
    retried = bootstrap_paired_rating_differences(
        pairs,
        bot_ids,
        candidate_id=plan.candidate.bot_id,
        incumbent_id=plan.incumbent.bot_id,
        samples=10,
        root_seed=42,
        workers=2,
    )

    assert retried[:2] == expected[:2]
    assert retried[2] == (
        "Parallel bootstrap failed with PermissionError; all samples were retried serially.",
    )


def test_bootstrap_counts_rating_fit_failures_as_not_converged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = promotion_plan(pair_count=1)
    result = result_for_plan(plan)
    pairs = ((result.game_summaries[0], result.game_summaries[1]),)

    def fail_fit(*args: object, **kwargs: object) -> None:
        raise TournamentRatingError("synthetic optimizer failure")

    monkeypatch.setattr(analysis_module, "fit_plackett_luce", fail_fit)
    values, converged, warnings = bootstrap_paired_rating_differences(
        pairs,
        tuple(spec.bot_id for spec in plan.monte_carlo_config.bot_specs),
        candidate_id=plan.candidate.bot_id,
        incumbent_id=plan.incumbent.bot_id,
        samples=10,
        root_seed=42,
    )

    assert values == ()
    assert converged == 0
    assert warnings == ("10 of 10 bootstrap fits did not converge and were excluded.",)


def test_interval_including_zero_fails_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = promotion_plan()
    result = result_for_plan(plan)
    monkeypatch.setattr(
        analysis_module,
        "bootstrap_paired_rating_differences",
        lambda *args, **kwargs: ((-1.0, 1.0), 2, ()),
    )

    analysis = analyze_promotion(
        plan,
        result,
        bootstrap_samples=2,
        bootstrap_seed=42,
    )

    assert _failure_codes(analysis) == ("interval_includes_zero",)
    assert analysis.promoted is False


def test_passing_interval_promotes(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = promotion_plan()
    result = result_for_plan(plan)
    monkeypatch.setattr(
        analysis_module,
        "bootstrap_paired_rating_differences",
        lambda *args, **kwargs: ((1.0, 2.0), 2, ()),
    )

    analysis = analyze_promotion(
        plan,
        result,
        bootstrap_samples=2,
        bootstrap_seed=42,
    )

    assert analysis.failures == ()
    assert analysis.interval is not None and analysis.interval.lower > 0
    assert analysis.promoted is True


def test_insufficient_bootstrap_convergence_fails_without_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = promotion_plan()
    result = result_for_plan(plan)
    monkeypatch.setattr(
        analysis_module,
        "bootstrap_paired_rating_differences",
        lambda *args, **kwargs: ((1.0,) * 8, 8, ()),
    )

    analysis = analyze_promotion(
        plan,
        result,
        bootstrap_samples=10,
        bootstrap_seed=42,
    )

    assert _failure_codes(analysis) == ("bootstrap_incomplete",)
    assert analysis.interval is None
    assert analysis.promoted is False


@pytest.mark.parametrize(
    "bootstrap_values",
    (
        (math.nan,) * 10,
        (math.inf,) * 10,
    ),
)
def test_nonfinite_bootstrap_interval_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    bootstrap_values: tuple[float, ...],
) -> None:
    plan = promotion_plan()
    result = result_for_plan(plan)
    monkeypatch.setattr(
        analysis_module,
        "bootstrap_paired_rating_differences",
        lambda *args, **kwargs: (bootstrap_values, len(bootstrap_values), ()),
    )

    analysis = analyze_promotion(
        plan,
        result,
        bootstrap_samples=10,
        bootstrap_seed=42,
    )

    assert _failure_codes(analysis) == ("nonfinite_analysis",)
    assert analysis.interval is None
    assert analysis.promoted is False


def test_nonfinite_point_estimate_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = promotion_plan()
    result = result_for_plan(plan)
    ratings = {
        spec.bot_id: SimpleNamespace(rating=math.nan) for spec in plan.monte_carlo_config.bot_specs
    }
    monkeypatch.setattr(
        analysis_module,
        "fit_plackett_luce",
        lambda *args, **kwargs: SimpleNamespace(ratings_by_id=ratings),
    )

    analysis = analyze_promotion(
        plan,
        result,
        bootstrap_samples=10,
        bootstrap_seed=42,
    )

    assert _failure_codes(analysis) == ("nonfinite_analysis",)
    assert analysis.rating_difference is None
    assert analysis.interval is None


def test_rating_fit_failure_is_a_stable_domain_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = promotion_plan()
    result = result_for_plan(plan)

    def fail_fit(*args: object, **kwargs: object) -> None:
        raise TournamentRatingError("synthetic optimizer failure")

    monkeypatch.setattr(analysis_module, "fit_plackett_luce", fail_fit)

    analysis = analyze_promotion(
        plan,
        result,
        bootstrap_samples=10,
        bootstrap_seed=42,
    )

    assert _failure_codes(analysis) == ("rating_fit_failed",)
    assert analysis.rating_difference is None
    assert analysis.interval is None


@pytest.mark.parametrize(
    ("bootstrap_samples", "workers", "message"),
    (
        (0, 1, "bootstrap samples must be positive"),
        (-1, 1, "bootstrap samples must be positive"),
        (1, 0, "bootstrap workers must be positive"),
        (1, -1, "bootstrap workers must be positive"),
    ),
)
def test_rejects_invalid_bootstrap_configuration(
    bootstrap_samples: int,
    workers: int,
    message: str,
) -> None:
    plan = promotion_plan(pair_count=1)

    with pytest.raises(ValueError, match=message):
        analyze_promotion(
            plan,
            result_for_plan(plan),
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=42,
            workers=workers,
        )


def test_promotion_failure_has_plain_english_diagnostic() -> None:
    failure = PromotionFailure("example", "This explains what went wrong.")

    assert failure.message == "This explains what went wrong."
