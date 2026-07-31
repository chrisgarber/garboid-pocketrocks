from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from garboid_pocketrocks.bots import BOT_SPECS_BY_NAME, BotSpec, RandomBot
from garboid_pocketrocks.promotion import planning as promotion_planning
from garboid_pocketrocks.promotion.corpus import (
    PromotionCase,
    PromotionCorpus,
    PromotionCorpusRecipe,
    load_promotion_corpus,
)
from garboid_pocketrocks.promotion.planning import (
    PromotionPlanningError,
    plan_paired_games,
)
from garboid_pocketrocks.simulator.runner import FaultMode


def _bot_spec(name: str, *, bot_id: str | None = None) -> BotSpec:
    if bot_id is None:
        return BotSpec.for_simulation(name, RandomBot.build_brain)
    return BotSpec(
        name=name,
        bot_id=bot_id,
        brain_factory=RandomBot.build_brain,
    )


def _held_out_corpus(
    *,
    opponent_names_by_seat: tuple[str | None, ...] = (
        None,
        "opponent-b",
        "opponent-a",
    ),
    opponent_names: tuple[str, ...] = ("opponent-a", "opponent-b"),
) -> PromotionCorpus:
    player_count = len(opponent_names_by_seat)
    case = PromotionCase(
        case_id="fixture-held-out-v1:A:3:seat-0:repeat-0",
        chart="A",
        player_count=player_count,
        focal_seat=0,
        engine_seed=12345,
        opponent_names_by_seat=opponent_names_by_seat,
    )
    recipe = PromotionCorpusRecipe(
        schema_version=1,
        name="fixture-held-out-v1",
        purpose="held_out",
        root_seed=90001,
        repetitions_per_seat_cell=1,
        charts=("A",),
        player_counts=(player_count,),
        opponent_names=opponent_names,
    )
    return PromotionCorpus(recipe=recipe, cases=(case,), digest="0" * 64)


def _identities() -> tuple[BotSpec, BotSpec, BotSpec, BotSpec]:
    return (
        _bot_spec("candidate"),
        _bot_spec("incumbent"),
        _bot_spec("opponent-a"),
        _bot_spec("opponent-b"),
    )


def test_plans_candidate_and_incumbent_as_exact_twin_games() -> None:
    candidate, incumbent, opponent_a, opponent_b = _identities()
    corpus = _held_out_corpus()

    plan = plan_paired_games(
        corpus,
        candidate=candidate,
        incumbent=incumbent,
        registry={
            opponent_a.name: opponent_a,
            opponent_b.name: opponent_b,
        },
    )

    assert plan.candidate == candidate
    assert plan.incumbent == incumbent
    assert plan.opponents == (opponent_b, opponent_a)
    assert len(plan.pairs) == 1

    pair = plan.pairs[0]
    assert pair.pair_index == 0
    assert pair.case == corpus.cases[0]
    assert pair.candidate_game.game_index == 0
    assert pair.incumbent_game.game_index == 1
    assert pair.candidate_game.seed == pair.incumbent_game.seed == pair.case.engine_seed
    assert pair.candidate_game.root_seed == pair.incumbent_game.root_seed == 90001
    assert pair.candidate_game.player_count == pair.incumbent_game.player_count == 3
    assert pair.candidate_game.value_chart == pair.incumbent_game.value_chart == "A"
    assert pair.candidate_game.objectives_enabled is True
    assert pair.incumbent_game.objectives_enabled is True
    assert pair.candidate_game.fault_mode is FaultMode.RECORD_AND_PASS
    assert pair.incumbent_game.fault_mode is FaultMode.RECORD_AND_PASS
    assert pair.candidate_game.lineup[pair.case.focal_seat] == candidate
    assert pair.incumbent_game.lineup[pair.case.focal_seat] == incumbent

    for seat, opponent_name in enumerate(pair.case.opponent_names_by_seat):
        if seat == pair.case.focal_seat:
            continue
        candidate_opponent = pair.candidate_game.lineup[seat]
        incumbent_opponent = pair.incumbent_game.lineup[seat]
        assert candidate_opponent == incumbent_opponent
        assert candidate_opponent.name == opponent_name
        assert candidate_opponent.bot_id == incumbent_opponent.bot_id


def test_excludes_exact_candidate_and_incumbent_identities_from_eligible_pool() -> None:
    candidate = _bot_spec("candidate")
    incumbent = _bot_spec("incumbent")
    opponents = tuple(_bot_spec(f"opponent-{letter}") for letter in "abcd")
    configured = (candidate, incumbent, *opponents)
    corpus = _held_out_corpus(
        opponent_names_by_seat=(None, "candidate", "incumbent"),
        opponent_names=tuple(spec.name for spec in configured),
    )

    plan = plan_paired_games(
        corpus,
        candidate=candidate,
        incumbent=incumbent,
        registry={spec.name: spec for spec in configured},
    )

    assert plan.opponent_pool.configured == configured
    assert tuple(
        (exclusion.opponent, exclusion.reason) for exclusion in plan.opponent_pool.exclusions
    ) == ((candidate, "candidate"), (incumbent, "incumbent"))
    assert plan.opponent_pool.remaining == opponents
    assert plan.opponents == (opponents[3], opponents[0])
    assert plan.pairs[0].case.opponent_names_by_seat == (
        None,
        opponents[3].name,
        opponents[0].name,
    )
    assert len(plan.digest) == 64
    int(plan.digest, 16)
    assert plan.digest == "ac8034bc307397eeac50f4e727736cf901a68f32bd9e6c4707da9f89857ab84a"


def test_fails_closed_when_compared_identity_filter_leaves_too_few_opponents() -> None:
    candidate = _bot_spec("candidate")
    incumbent = _bot_spec("incumbent")
    opponents = tuple(_bot_spec(f"opponent-{letter}") for letter in "abc")
    configured = (candidate, incumbent, *opponents)
    corpus = _held_out_corpus(
        opponent_names_by_seat=(
            None,
            "candidate",
            "incumbent",
            "opponent-a",
            "opponent-b",
        ),
        opponent_names=tuple(spec.name for spec in configured),
    )

    with pytest.raises(PromotionPlanningError) as captured:
        plan_paired_games(
            corpus,
            candidate=candidate,
            incumbent=incumbent,
            registry={spec.name: spec for spec in configured},
        )

    assert captured.value.code == "insufficient_eligible_opponents"
    assert captured.value.opponent_pool is not None
    assert captured.value.opponent_pool.remaining == opponents


def test_plan_digest_covers_every_executable_job_field() -> None:
    candidate, incumbent, opponent_a, opponent_b = _identities()
    plan = plan_paired_games(
        _held_out_corpus(),
        candidate=candidate,
        incumbent=incumbent,
        registry={
            opponent_a.name: opponent_a,
            opponent_b.name: opponent_b,
        },
    )
    first_pair = plan.pairs[0]
    changed_jobs = (
        replace(first_pair.candidate_game, game_index=99),
        replace(first_pair.candidate_game, root_seed=77),
        replace(first_pair.candidate_game, seed=88),
        replace(first_pair.candidate_game, objectives_enabled=False),
    )

    for changed_job in changed_jobs:
        changed_pair = replace(first_pair, candidate_game=changed_job)
        changed_plan = replace(plan, pairs=(changed_pair,))
        assert promotion_planning._promotion_plan_digest(changed_plan) != plan.digest


def test_plan_digest_covers_monte_carlo_execution_configuration() -> None:
    candidate, incumbent, opponent_a, opponent_b = _identities()
    plan = plan_paired_games(
        _held_out_corpus(),
        candidate=candidate,
        incumbent=incumbent,
        registry={
            opponent_a.name: opponent_a,
            opponent_b.name: opponent_b,
        },
    )
    changed_plan = replace(
        plan,
        monte_carlo_config=replace(
            plan.monte_carlo_config,
            capture_replays=True,
        ),
    )

    assert promotion_planning._promotion_plan_digest(changed_plan) != plan.digest


def test_committed_plan_pins_filtered_lineups_exposures_and_digest() -> None:
    corpus = load_promotion_corpus(
        Path("configs/promotion/held-out-v1.json"),
        registry=BOT_SPECS_BY_NAME,
    )

    plan = plan_paired_games(
        corpus,
        candidate=BOT_SPECS_BY_NAME["vector_ppo_large_v1_g350k"],
        incumbent=BOT_SPECS_BY_NAME["vector_ppo_small_v1_g1500"],
        registry=BOT_SPECS_BY_NAME,
    )

    assert tuple(
        (exclusion.opponent.name, exclusion.reason) for exclusion in plan.opponent_pool.exclusions
    ) == (("vector_ppo_large_v1_g350k", "candidate"),)
    assert plan.pairs[0].case.opponent_names_by_seat == (
        None,
        "aggressive-v2",
        "balanced-v2",
    )
    assert plan.pairs[-1].case.opponent_names_by_seat == (
        "passive-v1",
        "aggressive-v2",
        "balanced-v2",
        "passive-v2",
        None,
    )
    exposures = Counter(
        opponent_name
        for pair in plan.pairs
        for opponent_name in pair.case.opponent_names_by_seat
        if opponent_name is not None
    )
    assert exposures == {
        "aggressive-v1": 253,
        "balanced-v1": 254,
        "passive-v1": 254,
        "aggressive-v2": 254,
        "balanced-v2": 253,
        "passive-v2": 252,
    }
    assert plan.digest == "b131e59b7c3a59ff90d54f7b63fb80e09c0956d221508b2806c4e3ebd0bdcba1"


def test_flattens_pairs_in_contiguous_candidate_then_incumbent_order() -> None:
    candidate, incumbent, opponent_a, opponent_b = _identities()
    corpus = _held_out_corpus()
    second_case = replace(
        corpus.cases[0],
        case_id="fixture-held-out-v1:B:3:seat-1:repeat-0",
        chart="B",
        focal_seat=1,
        engine_seed=67890,
        opponent_names_by_seat=("opponent-a", None, "opponent-b"),
    )
    corpus = replace(
        corpus,
        recipe=replace(corpus.recipe, charts=("A", "B")),
        cases=(*corpus.cases, second_case),
    )

    plan = plan_paired_games(
        corpus,
        candidate=candidate,
        incumbent=incumbent,
        registry={
            opponent_a.name: opponent_a,
            opponent_b.name: opponent_b,
        },
    )

    assert plan.jobs == tuple(
        game for pair in plan.pairs for game in (pair.candidate_game, pair.incumbent_game)
    )
    assert [job.game_index for job in plan.jobs] == [0, 1, 2, 3]


def test_builds_exact_simulator_configuration_in_first_seen_identity_order() -> None:
    candidate, incumbent, opponent_a, opponent_b = _identities()
    corpus = _held_out_corpus()

    plan = plan_paired_games(
        corpus,
        candidate=candidate,
        incumbent=incumbent,
        registry={
            opponent_a.name: opponent_a,
            opponent_b.name: opponent_b,
        },
    )

    config = plan.monte_carlo_config
    assert config.bot_specs == (candidate, incumbent, opponent_b, opponent_a)
    assert len({spec.bot_id for spec in config.bot_specs}) == len(config.bot_specs)
    assert config.games == 2
    assert config.player_counts == corpus.recipe.player_counts
    assert config.value_charts == corpus.recipe.charts
    assert config.root_seed == corpus.recipe.root_seed
    assert config.objectives_enabled == (True,)
    assert config.fault_mode is FaultMode.RECORD_AND_PASS


@pytest.mark.parametrize(
    ("candidate", "incumbent"),
    (
        (_bot_spec("same-name", bot_id="candidate-id"), _bot_spec("same-name", bot_id="other-id")),
        (_bot_spec("candidate", bot_id="same-id"), _bot_spec("incumbent", bot_id="same-id")),
    ),
)
def test_rejects_candidate_incumbent_name_or_id_collisions(
    candidate: BotSpec,
    incumbent: BotSpec,
) -> None:
    _, _, opponent_a, opponent_b = _identities()

    with pytest.raises(PromotionPlanningError) as captured:
        plan_paired_games(
            _held_out_corpus(),
            candidate=candidate,
            incumbent=incumbent,
            registry={
                opponent_a.name: opponent_a,
                opponent_b.name: opponent_b,
            },
        )

    assert captured.value.code == "candidate_incumbent_identity_collision"
    assert str(captured.value)


@pytest.mark.parametrize(
    "candidate",
    (
        _bot_spec("opponent-a", bot_id="candidate-id"),
        _bot_spec("candidate", bot_id="opponent-a"),
    ),
)
def test_rejects_candidate_opponent_name_or_id_collisions(candidate: BotSpec) -> None:
    _, incumbent, opponent_a, opponent_b = _identities()

    with pytest.raises(PromotionPlanningError) as captured:
        plan_paired_games(
            _held_out_corpus(),
            candidate=candidate,
            incumbent=incumbent,
            registry={
                opponent_a.name: opponent_a,
                opponent_b.name: opponent_b,
            },
        )

    assert captured.value.code == "candidate_opponent_identity_collision"
    assert str(captured.value)


@pytest.mark.parametrize(
    "incumbent",
    (
        _bot_spec("opponent-b", bot_id="incumbent-id"),
        _bot_spec("incumbent", bot_id="opponent-b"),
    ),
)
def test_rejects_incumbent_opponent_name_or_id_collisions(incumbent: BotSpec) -> None:
    candidate, _, opponent_a, opponent_b = _identities()

    with pytest.raises(PromotionPlanningError) as captured:
        plan_paired_games(
            _held_out_corpus(),
            candidate=candidate,
            incumbent=incumbent,
            registry={
                opponent_a.name: opponent_a,
                opponent_b.name: opponent_b,
            },
        )

    assert captured.value.code == "incumbent_opponent_identity_collision"
    assert str(captured.value)


@pytest.mark.parametrize(
    "registry",
    (
        {"opponent-a": _bot_spec("opponent-a")},
        {
            "opponent-a": _bot_spec("renamed-opponent"),
            "opponent-b": _bot_spec("opponent-b"),
        },
    ),
)
def test_rejects_registry_entries_that_do_not_match_corpus_opponents(
    registry: dict[str, BotSpec],
) -> None:
    candidate, incumbent, _, _ = _identities()

    with pytest.raises(PromotionPlanningError) as captured:
        plan_paired_games(
            _held_out_corpus(),
            candidate=candidate,
            incumbent=incumbent,
            registry=registry,
        )

    assert captured.value.code == "opponent_identity_mismatch"
    assert str(captured.value)


def test_rejects_distinct_opponent_names_that_share_one_bot_id() -> None:
    candidate, incumbent, _, _ = _identities()

    with pytest.raises(PromotionPlanningError) as captured:
        plan_paired_games(
            _held_out_corpus(),
            candidate=candidate,
            incumbent=incumbent,
            registry={
                "opponent-a": _bot_spec("opponent-a", bot_id="shared-opponent-id"),
                "opponent-b": _bot_spec("opponent-b", bot_id="shared-opponent-id"),
            },
        )

    assert captured.value.code == "opponent_identity_mismatch"
    assert "different names and different bot IDs" in str(captured.value)


def test_rejects_a_case_that_seats_one_opponent_identity_more_than_once() -> None:
    candidate, incumbent, opponent_a, opponent_b = _identities()
    corpus = _held_out_corpus(
        opponent_names_by_seat=(None, "opponent-a", "opponent-a"),
    )

    with pytest.raises(PromotionPlanningError) as captured:
        plan_paired_games(
            corpus,
            candidate=candidate,
            incumbent=incumbent,
            registry={
                opponent_a.name: opponent_a,
                opponent_b.name: opponent_b,
            },
        )

    assert captured.value.code == "opponent_identity_mismatch"
    assert "more than one non-focal seat" in str(captured.value)


def test_requires_every_case_opponent_to_be_declared_by_the_recipe() -> None:
    candidate, incumbent, opponent_a, opponent_b = _identities()
    undeclared = _bot_spec("undeclared-opponent")
    corpus = _held_out_corpus(
        opponent_names_by_seat=(None, "opponent-a", "undeclared-opponent"),
    )

    with pytest.raises(PromotionPlanningError) as captured:
        plan_paired_games(
            corpus,
            candidate=candidate,
            incumbent=incumbent,
            registry={
                opponent_a.name: opponent_a,
                opponent_b.name: opponent_b,
                undeclared.name: undeclared,
            },
        )

    assert captured.value.code == "opponent_identity_mismatch"
    assert str(captured.value)


def test_requires_a_held_out_corpus() -> None:
    candidate, incumbent, opponent_a, opponent_b = _identities()
    corpus = _held_out_corpus()
    development_corpus = replace(
        corpus,
        recipe=replace(corpus.recipe, purpose="development"),
    )

    with pytest.raises(PromotionPlanningError) as captured:
        plan_paired_games(
            development_corpus,
            candidate=candidate,
            incumbent=incumbent,
            registry={
                opponent_a.name: opponent_a,
                opponent_b.name: opponent_b,
            },
        )

    assert captured.value.code == "held_out_corpus_required"
    assert str(captured.value)
