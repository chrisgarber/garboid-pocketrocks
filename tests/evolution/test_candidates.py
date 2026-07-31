from __future__ import annotations

import json
import pickle
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal, localcontext
from functools import partial
from pathlib import Path

import pytest
from pocketrocks import OBJECTIVES, ActionId, DecisionContext, Suit

from garboid_pocketrocks.bots import BOT_SPECS_BY_NAME
from garboid_pocketrocks.bots.heuristic import (
    BalancedHeuristicV2Brain,
    HeuristicBotBrain,
)
from garboid_pocketrocks.evolution.candidates import (
    CoefficientGenome,
    HeuristicCandidate,
    build_initial_population,
    build_mutation_population,
    candidate_bot_spec,
    candidate_profile,
)
from garboid_pocketrocks.evolution.manifest import (
    COEFFICIENT_NAMES,
    CoefficientGrid,
    CoefficientGrids,
    CoefficientValues,
    SearchManifest,
    load_search_manifest,
)
from garboid_pocketrocks.knowledge import RulesetKnowledge
from garboid_pocketrocks.promotion.corpus import load_promotion_corpus

REPOSITORY_ROOT = Path(__file__).parents[2]


@pytest.fixture(scope="module")
def balanced_manifest() -> SearchManifest:
    corpus = load_promotion_corpus(
        REPOSITORY_ROOT / "configs/promotion/development-balanced-v3-broad-v1.json",
        registry=BOT_SPECS_BY_NAME,
    )
    return load_search_manifest(
        REPOSITORY_ROOT / "configs/evolution/balanced-v3-search-v1.json",
        development_corpus=corpus,
    )


def _decimal_tuple(*values: str) -> tuple[Decimal, ...]:
    return tuple(Decimal(value) for value in values)


def _load_large_grid_manifest(tmp_path: Path) -> SearchManifest:
    corpus = load_promotion_corpus(
        REPOSITORY_ROOT / "configs/promotion/development-balanced-v3-broad-v1.json",
        registry=BOT_SPECS_BY_NAME,
    )
    payload = json.loads(
        (REPOSITORY_ROOT / "configs/evolution/balanced-v3-search-v1.json").read_text(
            encoding="utf-8"
        )
    )
    payload["coefficient_grids"]["liquidity_strength"] = {
        "minimum": "0.4",
        "maximum": "70000000000000000000000000000.4",
        "step": "10000000000000000000000000000",
    }
    manifest_path = tmp_path / "balanced-v3-search-v1.json"
    manifest_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with localcontext() as context:
        context.prec = 50
        return load_search_manifest(manifest_path, development_corpus=corpus)


def _knowledge(*, private_cards: int = 0) -> RulesetKnowledge:
    return RulesetKnowledge(
        name="candidate-test",
        player_count=3,
        starting_cash=30,
        private_cards_per_player=private_cards,
        resource_counts=(3, 3, 3, 3, 3),
        action_counts=(12, 8, 3, 2, 3, 2),
        value_chart=(0, 4, 8, 12, 16, 20),
        objective_pool=tuple(sorted(OBJECTIVES)),
        active_objective_count=0,
        objectives_enabled=False,
    )


def _context(
    *,
    decision_kind: str = "submitBid",
    hand: tuple[int, ...] = (),
    legal_max: int | None = 9,
) -> DecisionContext:
    return DecisionContext(
        request_id="candidate-test",
        deadline_at=2**63 - 1,
        received_at=0,
        decision_kind=decision_kind,  # type: ignore[arg-type]
        player_count=3,
        starting_cash=30,
        value_chart=(0, 4, 8, 12, 16, 20),
        objective_ids=(),
        current_action_id=int(ActionId.AUCTION2),
        current_resource_ids=(int(Suit.BRICK), int(Suit.WOOD)),
        cash_by_seat=(30, 30, 30),
        tiebreak_seat=2,
        won_resource_counts_by_seat=((0, 0, 0, 0, 0),) * 3,
        revealed_info_counts_by_seat=((0, 0, 0, 0, 0),) * 3,
        owned_objective_ids_by_seat=((), (), ()),
        bot_seat=0,
        current_hand_suit_ids=hand,
        legal_max_amount=legal_max,
        revealable_count=len(hand),
    )


def test_genome_is_immutable_and_has_one_canonical_content_digest() -> None:
    first = CoefficientGenome(
        CoefficientValues(
            liquidity_strength=Decimal("0.400"),
            future_cash_weight=Decimal("0.750"),
            objective_progress_weight=Decimal("0.200"),
            bid_shading=Decimal("0.250"),
        )
    )
    equivalent = CoefficientGenome(
        CoefficientValues(
            liquidity_strength=Decimal("0.4"),
            future_cash_weight=Decimal("0.75"),
            objective_progress_weight=Decimal("0.2"),
            bid_shading=Decimal("0.25"),
        )
    )

    assert first.digest == equivalent.digest
    assert first.digest == "429d9a62e92febf80a9f73c41a39994db9f80bcb2cefa3fff70e8668ff8caea8"
    with pytest.raises(FrozenInstanceError):
        first.digest = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        first.coefficients.bid_shading = Decimal("0.30")  # type: ignore[misc]


def test_generation_zero_is_the_incumbent_followed_by_golden_grid_samples(
    balanced_manifest: SearchManifest,
) -> None:
    population = build_initial_population(balanced_manifest)

    assert tuple(candidate.genome.coefficients.as_tuple() for candidate in population) == (
        _decimal_tuple("0.4", "0.75", "0.2", "0.25"),
        _decimal_tuple("0.5", "0.95", "0", "0.25"),
        _decimal_tuple("1.35", "0.5", "0.9", "0.7"),
        _decimal_tuple("1.15", "1.25", "0.75", "0.7"),
        _decimal_tuple("1", "1.6", "0.35", "0.7"),
        _decimal_tuple("1.05", "0.85", "0.8", "0.65"),
        _decimal_tuple("1.25", "1.8", "0.1", "0.05"),
        _decimal_tuple("1.25", "1.55", "0.15", "0.4"),
        _decimal_tuple("0.75", "1.55", "0", "0.25"),
        _decimal_tuple("0.35", "0.5", "0.8", "0.15"),
        _decimal_tuple("0.6", "2", "0.1", "0.35"),
        _decimal_tuple("0.5", "0.3", "0.55", "0.35"),
    )
    assert population[0].genome.coefficients == balanced_manifest.initial_coefficients
    assert tuple(candidate.generation for candidate in population) == (0,) * 12
    assert tuple(candidate.slot for candidate in population) == tuple(range(12))
    assert all(candidate.parent_identity is None for candidate in population)
    assert population[0].identity == "balanced-v3-candidate-g000-s000-429d9a62e92f"
    assert population[-1].identity == "balanced-v3-candidate-g000-s011-6918592759cd"
    assert build_initial_population(balanced_manifest) == population


def test_initial_sampling_is_independent_of_decimal_context_with_large_endpoints(
    tmp_path: Path,
) -> None:
    manifest = _load_large_grid_manifest(tmp_path)

    assert manifest.coefficient_grids.liquidity_strength.maximum == Decimal(
        "70000000000000000000000000000.4"
    )
    populations = []
    for precision in (8, 28, 50):
        with localcontext() as context:
            context.prec = precision
            populations.append(build_initial_population(manifest))

    assert populations[0] == populations[1] == populations[2]
    assert any(
        candidate.genome.coefficients.liquidity_strength > Decimal("1e20")
        for candidate in populations[0]
    )


def test_repeated_genomes_remain_separate_slot_proposals(
    balanced_manifest: SearchManifest,
) -> None:
    coefficients = balanced_manifest.initial_coefficients
    constant_grids = CoefficientGrids(
        liquidity_strength=CoefficientGrid(
            coefficients.liquidity_strength,
            coefficients.liquidity_strength,
            Decimal("0.05"),
        ),
        future_cash_weight=CoefficientGrid(
            coefficients.future_cash_weight,
            coefficients.future_cash_weight,
            Decimal("0.05"),
        ),
        objective_progress_weight=CoefficientGrid(
            coefficients.objective_progress_weight,
            coefficients.objective_progress_weight,
            Decimal("0.05"),
        ),
        bid_shading=CoefficientGrid(
            coefficients.bid_shading,
            coefficients.bid_shading,
            Decimal("0.05"),
        ),
    )
    duplicate_manifest = replace(
        balanced_manifest,
        algorithm=replace(balanced_manifest.algorithm, population_size=3),
        coefficient_grids=constant_grids,
    )

    population = build_initial_population(duplicate_manifest)

    assert tuple(candidate.genome for candidate in population) == (population[0].genome,) * 3
    assert len({candidate.identity for candidate in population}) == 3
    assert tuple(candidate.slot for candidate in population) == (0, 1, 2)


def test_later_generation_has_golden_one_field_mutations_and_cycles_parents(
    balanced_manifest: SearchManifest,
) -> None:
    elites = build_initial_population(balanced_manifest)[:4]

    children = build_mutation_population(
        balanced_manifest,
        generation=1,
        ranked_elites=elites,
    )

    assert tuple(child.genome.coefficients.as_tuple() for child in children) == (
        _decimal_tuple("0.2", "0.75", "0.2", "0.25"),
        _decimal_tuple("0.5", "1.1", "0", "0.25"),
        _decimal_tuple("1.35", "0.55", "0.9", "0.7"),
        _decimal_tuple("1.2", "1.25", "0.75", "0.7"),
        _decimal_tuple("0.4", "0.75", "0.35", "0.25"),
        _decimal_tuple("0.7", "0.95", "0", "0.25"),
        _decimal_tuple("1.35", "0.5", "0.9", "0.8"),
        _decimal_tuple("1.15", "1.25", "0.95", "0.7"),
        _decimal_tuple("0.45", "0.75", "0.2", "0.25"),
        _decimal_tuple("0.5", "0.95", "0", "0.4"),
        _decimal_tuple("1.35", "0.5", "0.8", "0.7"),
        _decimal_tuple("1.15", "1.25", "0.8", "0.7"),
    )
    assert tuple(child.parent_identity for child in children) == tuple(
        elites[slot % len(elites)].identity for slot in range(12)
    )
    assert children[0].identity == "balanced-v3-candidate-g001-s000-7af3aa91006b"
    assert children[-1].identity == "balanced-v3-candidate-g001-s011-4a47e09e3847"

    for slot, child in enumerate(children):
        parent = elites[slot % len(elites)]
        differences = tuple(
            child_value - parent_value
            for child_value, parent_value in zip(
                child.genome.coefficients.as_tuple(),
                parent.genome.coefficients.as_tuple(),
                strict=True,
            )
            if child_value != parent_value
        )
        assert len(differences) == 1
        changed_steps = abs(differences[0] / Decimal("0.05"))
        assert Decimal(1) <= changed_steps <= balanced_manifest.algorithm.mutation_radius_steps


def test_mutation_is_independent_of_decimal_context_with_large_endpoints(
    tmp_path: Path,
) -> None:
    manifest = _load_large_grid_manifest(tmp_path)
    with localcontext() as context:
        context.prec = 50
        elites = build_initial_population(manifest)[: manifest.algorithm.elite_count]

    populations = []
    for precision in (8, 28, 50):
        with localcontext() as context:
            context.prec = precision
            populations.append(
                build_mutation_population(
                    manifest,
                    generation=1,
                    ranked_elites=elites,
                )
            )

    assert populations[0] == populations[1] == populations[2]
    assert any(
        child.genome.coefficients.liquidity_strength
        != elites[child.slot % len(elites)].genome.coefficients.liquidity_strength
        for child in populations[0]
    )


def test_mutation_requires_the_prior_ranked_elites(
    balanced_manifest: SearchManifest,
) -> None:
    elites = build_initial_population(balanced_manifest)[:4]

    with pytest.raises(ValueError, match="generation"):
        build_mutation_population(
            balanced_manifest,
            generation=0,
            ranked_elites=elites,
        )
    with pytest.raises(ValueError, match="exactly 4"):
        build_mutation_population(
            balanced_manifest,
            generation=1,
            ranked_elites=elites[:3],
        )

    wrong_personality = HeuristicCandidate(
        personality="aggressive",
        generation=0,
        slot=0,
        genome=elites[0].genome,
        parent_identity=None,
    )
    with pytest.raises(ValueError, match="balanced"):
        build_mutation_population(
            balanced_manifest,
            generation=1,
            ranked_elites=(wrong_personality, *elites[1:]),
        )


def test_candidate_spec_is_local_picklable_and_matches_a_direct_brain(
    balanced_manifest: SearchManifest,
) -> None:
    candidate = build_initial_population(balanced_manifest)[4]
    profile = candidate_profile(candidate)
    spec = candidate_bot_spec(candidate)

    assert profile.name == "balanced"
    assert (
        profile.liquidity_strength,
        profile.future_cash_weight,
        profile.objective_progress_weight,
        profile.bid_shading,
    ) == (1.0, 1.6, 0.35, 0.7)
    assert spec.name == candidate.identity
    assert spec.bot_id == candidate.identity
    assert candidate.identity not in BOT_SPECS_BY_NAME
    assert isinstance(spec.brain_factory, partial)

    restored_spec = pickle.loads(pickle.dumps(spec))
    worker_brain = restored_spec.make_brain(seed=8675309)
    direct_brain = HeuristicBotBrain(profile)
    bid_context = _context(hand=(int(Suit.ORE), int(Suit.SHEEP)))
    knowledge = _knowledge(private_cards=2)

    assert isinstance(worker_brain, HeuristicBotBrain)
    assert worker_brain.valuator.profile == profile
    assert worker_brain.choose_decision(
        bid_context,
        knowledge,
    ) == direct_brain.choose_decision(bid_context, knowledge)


def test_incumbent_candidate_preserves_the_v2_reveal_decision(
    balanced_manifest: SearchManifest,
) -> None:
    incumbent_candidate = build_initial_population(balanced_manifest)[0]
    candidate_brain = candidate_bot_spec(incumbent_candidate).make_brain(seed=1)
    reveal_context = _context(
        decision_kind="selectInfoToReveal",
        hand=(int(Suit.ORE), int(Suit.SHEEP)),
        legal_max=None,
    )
    knowledge = _knowledge(private_cards=2)

    assert candidate_brain.choose_decision(
        reveal_context,
        knowledge,
    ) == BalancedHeuristicV2Brain().choose_decision(reveal_context, knowledge)
    assert reveal_context.is_legal(candidate_brain.choose_decision(reveal_context, knowledge))


def test_all_generated_values_stay_on_the_named_manifest_grids(
    balanced_manifest: SearchManifest,
) -> None:
    initial = build_initial_population(balanced_manifest)
    children = build_mutation_population(
        balanced_manifest,
        generation=1,
        ranked_elites=initial[:4],
    )

    for candidate in (*initial, *children):
        for name, value, grid in zip(
            COEFFICIENT_NAMES,
            candidate.genome.coefficients.as_tuple(),
            balanced_manifest.coefficient_grids.as_tuple(),
            strict=True,
        ):
            assert grid.minimum <= value <= grid.maximum, name
            assert (value - grid.minimum) % grid.step == 0, name
