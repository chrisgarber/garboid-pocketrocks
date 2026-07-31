from __future__ import annotations

from dataclasses import replace

import pytest

from garboid_pocketrocks.bots import BotSpec, RandomBot
from garboid_pocketrocks.evolution.planning import (
    DevelopmentPlanningError,
    plan_development_games,
)
from garboid_pocketrocks.promotion.corpus import (
    PromotionCase,
    PromotionCorpus,
    PromotionCorpusRecipe,
)
from garboid_pocketrocks.simulator.runner import FaultMode


def _spec(name: str, *, bot_id: str | None = None) -> BotSpec:
    return BotSpec(
        name=name,
        bot_id=name if bot_id is None else bot_id,
        brain_factory=RandomBot.build_brain,
    )


def _registry() -> dict[str, BotSpec]:
    return {name: _spec(name) for name in ("opponent-a", "opponent-b", "opponent-c", "opponent-d")}


def test_plans_exact_candidate_and_reusable_baseline_twins() -> None:
    corpus = _corpus()
    candidate = _spec("balanced-v3-candidate-test")
    incumbent = _spec("balanced-v2")
    opponents = tuple(_spec(f"opponent-{letter}") for letter in "abcd")
    registry = {spec.name: spec for spec in opponents}

    plan = plan_development_games(
        corpus,
        candidate=candidate,
        incumbent=incumbent,
        registry=registry,
    )

    assert plan.corpus is corpus
    assert plan.candidate is candidate
    assert plan.incumbent is incumbent
    assert plan.opponents == opponents
    assert len(plan.baseline_jobs) == len(corpus.cases)
    assert len(plan.candidate_jobs) == len(corpus.cases)
    assert plan.baseline_config.games == len(corpus.cases)
    assert plan.candidate_config.games == len(corpus.cases)
    assert plan.baseline_config.bot_specs == (incumbent, *opponents)
    assert plan.candidate_config.bot_specs == (candidate, *opponents)
    assert plan.baseline_config.root_seed == corpus.recipe.root_seed
    assert plan.candidate_config.root_seed == corpus.recipe.root_seed
    assert plan.baseline_config.player_counts == (3, 5)
    assert plan.candidate_config.player_counts == (3, 5)
    assert plan.baseline_config.value_charts == ("A", "E")
    assert plan.candidate_config.value_charts == ("A", "E")
    assert plan.baseline_config.fault_mode is FaultMode.RECORD_AND_PASS
    assert plan.candidate_config.fault_mode is FaultMode.RECORD_AND_PASS

    for case_index, (case, baseline, proposed) in enumerate(
        zip(corpus.cases, plan.baseline_jobs, plan.candidate_jobs, strict=True)
    ):
        assert baseline.game_index == proposed.game_index == case_index
        assert baseline.root_seed == proposed.root_seed == corpus.recipe.root_seed
        assert baseline.seed == proposed.seed == case.engine_seed
        assert baseline.value_chart == proposed.value_chart == case.chart
        assert baseline.player_count == proposed.player_count == case.player_count
        assert baseline.objectives_enabled is proposed.objectives_enabled is True
        assert baseline.fault_mode is proposed.fault_mode is FaultMode.RECORD_AND_PASS
        assert baseline.capture_decision_traces is proposed.capture_decision_traces is False
        assert baseline.lineup[case.focal_seat] is incumbent
        assert proposed.lineup[case.focal_seat] is candidate
        assert (
            tuple(
                spec.name if seat != case.focal_seat else None
                for seat, spec in enumerate(baseline.lineup)
            )
            == case.opponent_names_by_seat
        )
        assert (
            tuple(
                spec.name if seat != case.focal_seat else None
                for seat, spec in enumerate(proposed.lineup)
            )
            == case.opponent_names_by_seat
        )


def test_rejects_a_held_out_corpus_before_planning_any_jobs() -> None:
    corpus = replace(
        _corpus(),
        recipe=replace(_corpus().recipe, purpose="held_out"),
    )

    with pytest.raises(DevelopmentPlanningError) as raised:
        plan_development_games(
            corpus,
            candidate=_spec("candidate"),
            incumbent=_spec("incumbent"),
            registry=_registry(),
        )

    assert raised.value.code == "development_corpus_required"
    assert "development" in str(raised.value)


def test_baseline_jobs_are_reusable_across_distinct_candidate_plans() -> None:
    corpus = _corpus()
    incumbent = _spec("balanced-v2")
    registry = _registry()

    first = plan_development_games(
        corpus,
        candidate=_spec("balanced-v3-candidate-first"),
        incumbent=incumbent,
        registry=registry,
    )
    second = plan_development_games(
        corpus,
        candidate=_spec("balanced-v3-candidate-second"),
        incumbent=incumbent,
        registry=registry,
    )

    assert first.baseline_config == second.baseline_config
    assert first.baseline_jobs == second.baseline_jobs
    assert first.candidate_config != second.candidate_config
    assert first.candidate_jobs != second.candidate_jobs


def test_rejects_repeated_opponent_identity_within_one_case() -> None:
    corpus = _corpus()
    repeated = replace(
        corpus.cases[0],
        opponent_names_by_seat=("opponent-a", None, "opponent-a"),
    )

    with pytest.raises(DevelopmentPlanningError) as raised:
        plan_development_games(
            replace(corpus, cases=(repeated, *corpus.cases[1:])),
            candidate=_spec("candidate"),
            incumbent=_spec("incumbent"),
            registry=_registry(),
        )

    assert raised.value.code == "invalid_development_case"
    assert "distinct" in str(raised.value)


def test_requires_candidate_name_and_bot_id_to_be_the_same_local_identity() -> None:
    with pytest.raises(DevelopmentPlanningError) as raised:
        plan_development_games(
            _corpus(),
            candidate=_spec("candidate", bot_id="remote-or-moving-id"),
            incumbent=_spec("incumbent"),
            registry=_registry(),
        )

    assert raised.value.code == "candidate_identity_mismatch"
    assert "local" in str(raised.value)


@pytest.mark.parametrize("root_seed", (True, -1))
def test_rejects_invalid_development_root_seed(root_seed: int) -> None:
    corpus = _corpus()

    with pytest.raises(DevelopmentPlanningError) as raised:
        plan_development_games(
            replace(corpus, recipe=replace(corpus.recipe, root_seed=root_seed)),
            candidate=_spec("candidate"),
            incumbent=_spec("incumbent"),
            registry=_registry(),
        )

    assert raised.value.code == "invalid_development_seed"


@pytest.mark.parametrize("engine_seed", (False, -1))
def test_rejects_invalid_development_engine_seed(engine_seed: int) -> None:
    corpus = _corpus()
    invalid_case = replace(corpus.cases[0], engine_seed=engine_seed)

    with pytest.raises(DevelopmentPlanningError) as raised:
        plan_development_games(
            replace(corpus, cases=(invalid_case, *corpus.cases[1:])),
            candidate=_spec("candidate"),
            incumbent=_spec("incumbent"),
            registry=_registry(),
        )

    assert raised.value.code == "invalid_development_seed"


def test_rejects_duplicate_development_engine_seeds() -> None:
    corpus = _corpus()
    duplicate = replace(corpus.cases[1], engine_seed=corpus.cases[0].engine_seed)

    with pytest.raises(DevelopmentPlanningError) as raised:
        plan_development_games(
            replace(corpus, cases=(corpus.cases[0], duplicate)),
            candidate=_spec("candidate"),
            incumbent=_spec("incumbent"),
            registry=_registry(),
        )

    assert raised.value.code == "invalid_development_seed"
    assert "unique" in str(raised.value)


@pytest.mark.parametrize(
    ("candidate", "incumbent", "registry", "expected_code"),
    (
        (
            _spec("same"),
            _spec("same"),
            _registry(),
            "candidate_incumbent_identity_collision",
        ),
        (
            _spec("candidate"),
            _spec("incumbent", bot_id="candidate"),
            _registry(),
            "candidate_incumbent_identity_collision",
        ),
        (
            _spec("opponent-a"),
            _spec("incumbent"),
            _registry(),
            "candidate_opponent_identity_collision",
        ),
        (
            _spec("candidate"),
            _spec("opponent-a"),
            _registry(),
            "incumbent_opponent_identity_collision",
        ),
        (
            _spec("candidate"),
            _spec("incumbent"),
            {
                **_registry(),
                "opponent-a": _spec("wrong-name"),
            },
            "opponent_identity_mismatch",
        ),
    ),
)
def test_rejects_ambiguous_or_colliding_bot_identities(
    candidate: BotSpec,
    incumbent: BotSpec,
    registry: dict[str, BotSpec],
    expected_code: str,
) -> None:
    with pytest.raises(DevelopmentPlanningError) as raised:
        plan_development_games(
            _corpus(),
            candidate=candidate,
            incumbent=incumbent,
            registry=registry,
        )

    assert raised.value.code == expected_code


def _corpus() -> PromotionCorpus:
    opponent_names = tuple(f"opponent-{letter}" for letter in "abcd")
    cases = (
        PromotionCase(
            case_id="fixture-development-v1:A:3:seat-1:repeat-0",
            chart="A",
            player_count=3,
            focal_seat=1,
            engine_seed=10_001,
            opponent_names_by_seat=("opponent-a", None, "opponent-b"),
        ),
        PromotionCase(
            case_id="fixture-development-v1:E:5:seat-4:repeat-0",
            chart="E",
            player_count=5,
            focal_seat=4,
            engine_seed=10_002,
            opponent_names_by_seat=(
                "opponent-a",
                "opponent-b",
                "opponent-c",
                "opponent-d",
                None,
            ),
        ),
    )
    return PromotionCorpus(
        recipe=PromotionCorpusRecipe(
            schema_version=1,
            name="fixture-development-v1",
            purpose="development",
            root_seed=9_001,
            repetitions_per_seat_cell=1,
            charts=("A", "E"),
            player_counts=(3, 5),
            opponent_names=opponent_names,
        ),
        cases=cases,
        digest="d" * 64,
    )
