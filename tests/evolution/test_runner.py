from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import garboid_pocketrocks.evolution.runner as runner_module
from garboid_pocketrocks.bots import BOT_SPECS_BY_NAME, BotSpec, RandomBot
from garboid_pocketrocks.evolution.candidates import (
    build_initial_population,
    build_mutation_population,
)
from garboid_pocketrocks.evolution.evaluation import (
    CandidateEvaluation,
    ChallengerFinishDelta,
)
from garboid_pocketrocks.evolution.manifest import (
    SearchManifest,
    load_search_manifest,
    recompute_search_manifest_digest,
)
from garboid_pocketrocks.evolution.runner import SearchRunError, run_search
from garboid_pocketrocks.promotion.corpus import (
    PromotionCorpus,
    load_promotion_corpus,
    recompute_promotion_corpus_digest,
)
from garboid_pocketrocks.simulator.monte_carlo import MonteCarloRunner

REPOSITORY_ROOT = Path(__file__).parents[2]


def test_runs_one_cached_baseline_and_complete_mu_plus_lambda_generations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, corpus = _small_inputs(generations=2, population=4, elites=2, cases=2)
    real_run_jobs = MonteCarloRunner.run_jobs
    focal_identities: list[str] = []

    def record_run_jobs(*args: object, **kwargs: object) -> object:
        config = args[0]
        focal_identities.append(config.bot_specs[0].name)  # type: ignore[attr-defined]
        return real_run_jobs(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(MonteCarloRunner, "run_jobs", record_run_jobs)

    run = run_search(
        manifest,
        corpus,
        registry=BOT_SPECS_BY_NAME,
        workers=1,
        batch_size=1,
    )

    assert focal_identities.count(manifest.predecessor_name) == 1
    assert len(focal_identities) == 1 + (
        manifest.algorithm.generation_count * manifest.algorithm.population_size
    )
    assert len(run.candidate_runs) == 8
    assert len(run.selection_records) == 2
    assert len(run.selection_records[0].pool_identities) == 4
    assert len(run.selection_records[1].pool_identities) == 6
    assert run.failures == ()
    assert run.selected_candidate is not None

    initial = build_initial_population(manifest)
    assert tuple(item.candidate for item in run.candidate_runs[:4]) == initial
    expected_children = build_mutation_population(
        manifest,
        generation=1,
        ranked_elites=tuple(item.candidate for item in run.selections[0].elites),
    )
    assert tuple(item.candidate for item in run.candidate_runs[4:]) == expected_children
    assert tuple(
        candidate_run.evaluation.candidate_identity for candidate_run in run.candidate_runs
    ) == tuple(candidate_run.candidate.identity for candidate_run in run.candidate_runs)


def test_repeated_and_worker_runs_have_identical_sequences_results_and_winner() -> None:
    manifest, corpus = _small_inputs(generations=2, population=4, elites=2, cases=1)

    first = run_search(
        manifest,
        corpus,
        registry=BOT_SPECS_BY_NAME,
        workers=1,
        batch_size=1,
    )
    repeated = run_search(
        manifest,
        corpus,
        registry=BOT_SPECS_BY_NAME,
        workers=1,
        batch_size=1,
    )
    worker = run_search(
        manifest,
        corpus,
        registry=BOT_SPECS_BY_NAME,
        workers=2,
        batch_size=1,
    )

    expected_sequence = tuple(item.candidate for item in first.candidate_runs)
    expected_evaluations = tuple(item.evaluation for item in first.candidate_runs)
    expected_games = tuple(item.result.game_summaries for item in first.candidate_runs)
    expected_winner = (
        None if first.selected_candidate is None else first.selected_candidate.identity
    )
    expected_freeze = None if first.frozen_candidate is None else first.frozen_candidate.identity
    for run in (repeated, worker):
        assert tuple(item.candidate for item in run.candidate_runs) == expected_sequence
        assert tuple(item.evaluation for item in run.candidate_runs) == expected_evaluations
        assert tuple(item.result.game_summaries for item in run.candidate_runs) == expected_games
        assert run.baseline_result == first.baseline_result
        assert run.selection_records == first.selection_records
        assert (
            None if run.selected_candidate is None else run.selected_candidate.identity
        ) == expected_winner
        assert (
            None if run.frozen_candidate is None else run.frozen_candidate.identity
        ) == expected_freeze


def test_invalid_candidate_evidence_stops_after_the_complete_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, corpus = _small_inputs(generations=2, population=4, elites=2, cases=1)
    real_run_jobs = MonteCarloRunner.run_jobs
    calls = 0

    def omit_first_candidate_game(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        result = real_run_jobs(*args, **kwargs)  # type: ignore[arg-type]
        if calls == 2:
            return replace(result, game_summaries=())
        return result

    monkeypatch.setattr(MonteCarloRunner, "run_jobs", omit_first_candidate_game)

    run = run_search(
        manifest,
        corpus,
        registry=BOT_SPECS_BY_NAME,
        workers=1,
        batch_size=1,
    )

    assert calls == 1 + manifest.algorithm.population_size
    assert len(run.candidate_runs) == manifest.algorithm.population_size
    assert run.selection_records == ()
    assert tuple(failure.code for failure in run.failures) == ("invalid_candidate_evidence",)
    assert run.selected_candidate is None
    assert run.frozen_candidate is None


def test_complete_nonpositive_search_selects_but_does_not_freeze(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, corpus = _small_inputs(generations=1, population=2, elites=1, cases=1)

    def tied_evaluation(plan: object, *args: object) -> CandidateEvaluation:
        return CandidateEvaluation(
            candidate_identity=plan.candidate.name,  # type: ignore[attr-defined]
            incumbent_identity=plan.incumbent.name,  # type: ignore[attr-defined]
            requested_cases=1,
            completed_baseline_games=1,
            completed_candidate_games=1,
            worst_challenger_finish_delta=0.0,
            challenger_finish_deltas=(
                ChallengerFinishDelta(
                    opponent_identity="challenger",
                    shared_cases=1,
                    normalized_finish_delta=0.0,
                ),
            ),
            rating_delta=0.0,
            normalized_finish_delta=0.0,
            final_money_delta=0,
            candidate_faults=0,
            incumbent_faults=0,
            opponent_faults=0,
            unattributed_faults=0,
            faults_by_identity=(),
            failures=(),
            valid=True,
            eligible=True,
        )

    monkeypatch.setattr(runner_module, "evaluate_candidate", tied_evaluation)

    run = run_search(
        manifest,
        corpus,
        registry=BOT_SPECS_BY_NAME,
        workers=1,
        batch_size=1,
    )

    assert run.failures == ()
    assert run.selected_candidate is not None
    assert run.frozen_candidate is None


def test_rejects_a_corpus_that_does_not_match_the_manifest_binding() -> None:
    manifest, corpus = _small_inputs(generations=1, population=2, elites=1, cases=1)
    mismatched = replace(corpus, recipe=replace(corpus.recipe, name="other-development"))
    mismatched = replace(
        mismatched,
        digest=recompute_promotion_corpus_digest(mismatched),
    )

    with pytest.raises(SearchRunError) as raised:
        run_search(
            manifest,
            mismatched,
            registry=BOT_SPECS_BY_NAME,
            workers=1,
        )

    assert raised.value.code == "development_corpus_mismatch"


def test_rejects_a_forged_predecessor_before_simulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, corpus = _small_inputs(generations=1, population=2, elites=1, cases=1)
    canonical = BOT_SPECS_BY_NAME[manifest.predecessor_name]
    forged = BotSpec(
        name=canonical.name,
        bot_id=canonical.bot_id,
        brain_factory=RandomBot.build_brain,
    )
    registry = {**BOT_SPECS_BY_NAME, forged.name: forged}
    assert forged is not canonical
    assert (forged.name, forged.bot_id) == (canonical.name, canonical.bot_id)

    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("simulation started with a forged predecessor")

    monkeypatch.setattr(MonteCarloRunner, "run_jobs", forbidden)

    with pytest.raises(SearchRunError) as raised:
        run_search(
            manifest,
            corpus,
            registry=registry,
            workers=1,
            batch_size=1,
        )

    assert raised.value.code == "noncanonical_predecessor"


def test_rejects_each_forged_development_opponent_before_simulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, corpus = _small_inputs(generations=1, population=2, elites=1, cases=1)

    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("simulation started with a forged development opponent")

    monkeypatch.setattr(MonteCarloRunner, "run_jobs", forbidden)

    for opponent_name in corpus.recipe.opponent_names:
        canonical = BOT_SPECS_BY_NAME[opponent_name]
        forged = BotSpec(
            name=canonical.name,
            bot_id=canonical.bot_id,
            brain_factory=RandomBot.build_brain,
        )
        registry = {**BOT_SPECS_BY_NAME, forged.name: forged}
        assert forged is not canonical
        assert (forged.name, forged.bot_id) == (canonical.name, canonical.bot_id)

        with pytest.raises(SearchRunError) as raised:
            run_search(
                manifest,
                corpus,
                registry=registry,
                workers=1,
                batch_size=1,
            )

        assert raised.value.code == "noncanonical_opponent"


@pytest.mark.parametrize("stale_source", ("manifest", "corpus"))
def test_rejects_stale_source_digest_before_simulation(
    monkeypatch: pytest.MonkeyPatch,
    stale_source: str,
) -> None:
    manifest, corpus = _small_inputs(generations=1, population=2, elites=1, cases=1)
    if stale_source == "manifest":
        manifest = replace(manifest, search_seed=manifest.search_seed + 1)
        expected_code = "stale_manifest_digest"
    else:
        corpus = replace(corpus, cases=())
        expected_code = "stale_development_corpus_digest"

    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("simulation started with stale source evidence")

    monkeypatch.setattr(MonteCarloRunner, "run_jobs", forbidden)

    with pytest.raises(SearchRunError) as raised:
        run_search(
            manifest,
            corpus,
            registry=BOT_SPECS_BY_NAME,
            workers=1,
            batch_size=1,
        )

    assert raised.value.code == expected_code


def _small_inputs(
    *,
    generations: int,
    population: int,
    elites: int,
    cases: int,
) -> tuple[SearchManifest, PromotionCorpus]:
    corpus = load_promotion_corpus(
        REPOSITORY_ROOT
        / "configs/promotion/development-balanced-v3-broad-v1.json",
        registry=BOT_SPECS_BY_NAME,
    )
    manifest = load_search_manifest(
        REPOSITORY_ROOT / "configs/evolution/balanced-v3-search-v1.json",
        development_corpus=corpus,
    )
    corpus = replace(corpus, cases=corpus.cases[:cases])
    corpus = replace(corpus, digest=recompute_promotion_corpus_digest(corpus))
    manifest = replace(
        manifest,
        development_corpus=replace(
            manifest.development_corpus,
            digest=corpus.digest,
        ),
        algorithm=replace(
            manifest.algorithm,
            generation_count=generations,
            population_size=population,
            elite_count=elites,
        ),
    )
    manifest = replace(manifest, digest=recompute_search_manifest_digest(manifest))
    return manifest, corpus
