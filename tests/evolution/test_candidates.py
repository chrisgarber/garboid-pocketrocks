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
    PhaseAwareHeuristicBotBrain,
)
from garboid_pocketrocks.diagnostics.trace import PhaseAwareHeuristicBidExplanation
from garboid_pocketrocks.evolution.candidates import (
    CoefficientGenome,
    HeuristicCandidate,
    PhaseAwareHeuristicCandidate,
    PhaseCoefficientGenome,
    build_initial_population,
    build_mutation_population,
    candidate_bot_spec,
    candidate_profile,
    phase_candidate_profile,
)
from garboid_pocketrocks.evolution.manifest import (
    COEFFICIENT_NAMES,
    CoefficientGrid,
    CoefficientGrids,
    CoefficientValues,
    PhaseCoefficientValues,
    PhaseSearchManifest,
    SearchManifest,
    load_search_manifest,
    load_search_recipe,
)
from garboid_pocketrocks.knowledge import RulesetKnowledge
from garboid_pocketrocks.promotion.corpus import load_promotion_corpus

REPOSITORY_ROOT = Path(__file__).parents[2]

_PHASE_GENERATION_ZERO_GOLDENS = {
    "aggressive": (
        "aggressive-v4-candidate-g000-s000-e25140f48f36|1,1.95,0.15,0.4,1,1.95,0.15,0.4,1,1.95,0.15,0.4",
        "aggressive-v4-candidate-g000-s001-1ea7a0457b69|1.2,1.95,0.15,0.4,1,1.95,0.15,0.4,1,1.95,0.15,0.4",
        "aggressive-v4-candidate-g000-s002-54ebe4428e96|1,1.85,0.15,0.4,1,1.95,0.15,0.4,1,1.95,0.15,0.4",
        "aggressive-v4-candidate-g000-s003-805be32d38a8|1,1.95,0.3,0.4,1,1.95,0.15,0.4,1,1.95,0.15,0.4",
        "aggressive-v4-candidate-g000-s004-500e70c63e44|1,1.95,0.15,0.35,1,1.95,0.15,0.4,1,1.95,0.15,0.4",
        "aggressive-v4-candidate-g000-s005-82b875ca8080|1,1.95,0.15,0.4,0.95,1.95,0.15,0.4,1,1.95,0.15,0.4",
        "aggressive-v4-candidate-g000-s006-acd6a9e71d4b|1,1.95,0.15,0.4,1,1.9,0.15,0.4,1,1.95,0.15,0.4",
        "aggressive-v4-candidate-g000-s007-0fe92fa2a1f9|1,1.95,0.15,0.4,1,1.95,0.05,0.4,1,1.95,0.15,0.4",
        "aggressive-v4-candidate-g000-s008-3021c187769a|1,1.95,0.15,0.4,1,1.95,0.15,0.2,1,1.95,0.15,0.4",
        "aggressive-v4-candidate-g000-s009-7ed211ca7187|1,1.95,0.15,0.4,1,1.95,0.15,0.4,1.05,1.95,0.15,0.4",
        "aggressive-v4-candidate-g000-s010-ce4ec853de08|1,1.95,0.15,0.4,1,1.95,0.15,0.4,1,1.9,0.15,0.4",
        "aggressive-v4-candidate-g000-s011-8afbf9c9c68e|1,1.95,0.15,0.4,1,1.95,0.15,0.4,1,1.95,0.1,0.4",
        "aggressive-v4-candidate-g000-s012-11fa44297d6e|1,1.95,0.15,0.4,1,1.95,0.15,0.4,1,1.95,0.15,0.45",
        "aggressive-v4-candidate-g000-s013-d3cba6d069c2|0.85,0.95,0.7,0,1,1.95,0.15,0.4,1,1.95,0.15,0.4",
        "aggressive-v4-candidate-g000-s014-08f792b40310|1,1.95,0.15,0.4,1.1,0.55,0.35,0.6,1,1.95,0.15,0.4",
        "aggressive-v4-candidate-g000-s015-d91da38b80a6|1,1.95,0.15,0.4,1,1.95,0.15,0.4,0,0.8,0.45,0.45",
    ),
    "balanced": (
        "balanced-v4-candidate-g000-s000-4d0724fea3b3|0.25,1.55,0.3,0.35,0.25,1.55,0.3,0.35,0.25,1.55,0.3,0.35",
        "balanced-v4-candidate-g000-s001-298ba34d7ace|0.35,1.55,0.3,0.35,0.25,1.55,0.3,0.35,0.25,1.55,0.3,0.35",
        "balanced-v4-candidate-g000-s002-da6f26c73615|0.25,1.35,0.3,0.35,0.25,1.55,0.3,0.35,0.25,1.55,0.3,0.35",
        "balanced-v4-candidate-g000-s003-0c33f7a48c48|0.25,1.55,0.35,0.35,0.25,1.55,0.3,0.35,0.25,1.55,0.3,0.35",
        "balanced-v4-candidate-g000-s004-42a2393d77fd|0.25,1.55,0.3,0.4,0.25,1.55,0.3,0.35,0.25,1.55,0.3,0.35",
        "balanced-v4-candidate-g000-s005-067dc81a7610|0.25,1.55,0.3,0.35,0.3,1.55,0.3,0.35,0.25,1.55,0.3,0.35",
        "balanced-v4-candidate-g000-s006-3596c70a5674|0.25,1.55,0.3,0.35,0.25,1.65,0.3,0.35,0.25,1.55,0.3,0.35",
        "balanced-v4-candidate-g000-s007-a27dce705270|0.25,1.55,0.3,0.35,0.25,1.55,0.35,0.35,0.25,1.55,0.3,0.35",
        "balanced-v4-candidate-g000-s008-b0449fd4b5e3|0.25,1.55,0.3,0.35,0.25,1.55,0.3,0.25,0.25,1.55,0.3,0.35",
        "balanced-v4-candidate-g000-s009-abcf92dace0e|0.25,1.55,0.3,0.35,0.25,1.55,0.3,0.35,0.35,1.55,0.3,0.35",
        "balanced-v4-candidate-g000-s010-6467cd320b10|0.25,1.55,0.3,0.35,0.25,1.55,0.3,0.35,0.25,1.45,0.3,0.35",
        "balanced-v4-candidate-g000-s011-da57c7a85a31|0.25,1.55,0.3,0.35,0.25,1.55,0.3,0.35,0.25,1.55,0.2,0.35",
        "balanced-v4-candidate-g000-s012-263b9b1f7a76|0.25,1.55,0.3,0.35,0.25,1.55,0.3,0.35,0.25,1.55,0.3,0.45",
        "balanced-v4-candidate-g000-s013-4256840fff10|0.4,1.8,0.7,0.7,0.25,1.55,0.3,0.35,0.25,1.55,0.3,0.35",
        "balanced-v4-candidate-g000-s014-a6fb13247a68|0.25,1.55,0.3,0.35,0.75,0.45,0.9,0.7,0.25,1.55,0.3,0.35",
        "balanced-v4-candidate-g000-s015-559f50129824|0.25,1.55,0.3,0.35,0.25,1.55,0.3,0.35,1.4,0.95,0.05,0.45",
    ),
    "passive": (
        "passive-v4-candidate-g000-s000-52368fe901bf|1.5,1.8,0.95,0.45,1.5,1.8,0.95,0.45,1.5,1.8,0.95,0.45",
        "passive-v4-candidate-g000-s001-3fcf8f983348|1.45,1.8,0.95,0.45,1.5,1.8,0.95,0.45,1.5,1.8,0.95,0.45",
        "passive-v4-candidate-g000-s002-e237dc7e85bc|1.5,1.75,0.95,0.45,1.5,1.8,0.95,0.45,1.5,1.8,0.95,0.45",
        "passive-v4-candidate-g000-s003-464007c48545|1.5,1.8,0.85,0.45,1.5,1.8,0.95,0.45,1.5,1.8,0.95,0.45",
        "passive-v4-candidate-g000-s004-b1246f61b556|1.5,1.8,0.95,0.4,1.5,1.8,0.95,0.45,1.5,1.8,0.95,0.45",
        "passive-v4-candidate-g000-s005-23334ed25549|1.5,1.8,0.95,0.45,1.45,1.8,0.95,0.45,1.5,1.8,0.95,0.45",
        "passive-v4-candidate-g000-s006-16afcd308c68|1.5,1.8,0.95,0.45,1.5,1.6,0.95,0.45,1.5,1.8,0.95,0.45",
        "passive-v4-candidate-g000-s007-4496d86cf0a2|1.5,1.8,0.95,0.45,1.5,1.8,0.85,0.45,1.5,1.8,0.95,0.45",
        "passive-v4-candidate-g000-s008-09a8724adf27|1.5,1.8,0.95,0.45,1.5,1.8,0.95,0.6,1.5,1.8,0.95,0.45",
        "passive-v4-candidate-g000-s009-ed308f040e2c|1.5,1.8,0.95,0.45,1.5,1.8,0.95,0.45,1.3,1.8,0.95,0.45",
        "passive-v4-candidate-g000-s010-24b266474a0d|1.5,1.8,0.95,0.45,1.5,1.8,0.95,0.45,1.5,1.6,0.95,0.45",
        "passive-v4-candidate-g000-s011-91e3b3c0ed47|1.5,1.8,0.95,0.45,1.5,1.8,0.95,0.45,1.5,1.8,0.85,0.45",
        "passive-v4-candidate-g000-s012-82449e37dca6|1.5,1.8,0.95,0.45,1.5,1.8,0.95,0.45,1.5,1.8,0.95,0.4",
        "passive-v4-candidate-g000-s013-ec04672d278f|0.75,1.5,0.8,0.55,1.5,1.8,0.95,0.45,1.5,1.8,0.95,0.45",
        "passive-v4-candidate-g000-s014-e852a2916f70|1.5,1.8,0.95,0.45,0.9,0.1,1,0.35,1.5,1.8,0.95,0.45",
        "passive-v4-candidate-g000-s015-b5b37ae23032|1.5,1.8,0.95,0.45,1.5,1.8,0.95,0.45,0.15,0.15,0.35,0.15",
    ),
}

_BALANCED_GENERATION_ONE_GOLDEN = (
    "balanced-v4-candidate-g001-s000-298ba34d7ace|0.35,1.55,0.3,0.35,0.25,1.55,0.3,0.35,0.25,1.55,0.3,0.35",
    "balanced-v4-candidate-g001-s001-46f7b7db1a9c|0.35,1.7,0.3,0.35,0.25,1.55,0.3,0.35,0.25,1.55,0.3,0.35",
    "balanced-v4-candidate-g001-s002-916ab21f4ae3|0.25,1.35,0.25,0.35,0.25,1.55,0.3,0.35,0.25,1.55,0.3,0.35",
    "balanced-v4-candidate-g001-s003-cad1358f568d|0.25,1.55,0.35,0.4,0.25,1.55,0.3,0.35,0.25,1.55,0.3,0.35",
    "balanced-v4-candidate-g001-s004-b806b61ec567|0.25,1.55,0.3,0.35,0.35,1.55,0.3,0.35,0.25,1.55,0.3,0.35",
    "balanced-v4-candidate-g001-s005-5e726546e69d|0.35,1.55,0.3,0.35,0.25,1.6,0.3,0.35,0.25,1.55,0.3,0.35",
    "balanced-v4-candidate-g001-s006-aa17c121e4ad|0.25,1.35,0.3,0.35,0.25,1.55,0.25,0.35,0.25,1.55,0.3,0.35",
    "balanced-v4-candidate-g001-s007-55a45e1b8d43|0.25,1.55,0.35,0.35,0.25,1.55,0.3,0.4,0.25,1.55,0.3,0.35",
    "balanced-v4-candidate-g001-s008-14691706e2e5|0.25,1.55,0.3,0.35,0.25,1.55,0.3,0.35,0.1,1.55,0.3,0.35",
    "balanced-v4-candidate-g001-s009-71d2db00a248|0.35,1.55,0.3,0.35,0.25,1.55,0.3,0.35,0.25,1.65,0.3,0.35",
    "balanced-v4-candidate-g001-s010-1a99f5e6f9eb|0.25,1.35,0.3,0.35,0.25,1.55,0.3,0.35,0.25,1.55,0.4,0.35",
    "balanced-v4-candidate-g001-s011-936bf7e2e2d1|0.25,1.55,0.35,0.35,0.25,1.55,0.3,0.35,0.25,1.55,0.3,0.3",
    "balanced-v4-candidate-g001-s012-298ba34d7ace|0.35,1.55,0.3,0.35,0.25,1.55,0.3,0.35,0.25,1.55,0.3,0.35",
    "balanced-v4-candidate-g001-s013-bf08a5732157|0.35,1.5,0.3,0.35,0.25,1.55,0.3,0.35,0.25,1.55,0.3,0.35",
    "balanced-v4-candidate-g001-s014-493e793a029e|0.25,1.35,0.4,0.35,0.25,1.55,0.3,0.35,0.25,1.55,0.3,0.35",
    "balanced-v4-candidate-g001-s015-124f77c4da20|0.25,1.55,0.35,0.15,0.25,1.55,0.3,0.35,0.25,1.55,0.3,0.35",
)


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


@pytest.fixture(scope="module")
def balanced_phase_manifest() -> PhaseSearchManifest:
    corpus = load_promotion_corpus(
        REPOSITORY_ROOT / "configs/promotion/development-heuristic-v4-v1.json",
        registry=BOT_SPECS_BY_NAME,
    )
    recipe = load_search_recipe(
        REPOSITORY_ROOT / "configs/evolution/balanced-v4-search-v2.json",
        development_corpus=corpus,
    )
    assert isinstance(recipe, PhaseSearchManifest)
    return recipe


def _decimal_tuple(*values: str) -> tuple[Decimal, ...]:
    return tuple(Decimal(value) for value in values)


def _phase_values(values: tuple[Decimal, ...]) -> PhaseCoefficientValues:
    assert len(values) == 12
    return PhaseCoefficientValues(
        early=CoefficientValues(*values[0:4]),
        middle=CoefficientValues(*values[4:8]),
        late=CoefficientValues(*values[8:12]),
    )


def _phase_population_snapshot(
    population: tuple[PhaseAwareHeuristicCandidate, ...],
) -> tuple[str, ...]:
    return tuple(
        f"{candidate.identity}|"
        f"{','.join(str(value) for value in candidate.genome.experts.as_loci())}"
        for candidate in population
    )


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


def test_phase_genome_digest_binds_selector_and_all_twelve_labeled_values(
    balanced_phase_manifest: PhaseSearchManifest,
) -> None:
    first = PhaseCoefficientGenome(
        experts=balanced_phase_manifest.initial_experts,
        phase_selector=balanced_phase_manifest.phase_selector.kind,
    )
    equivalent = PhaseCoefficientGenome(
        experts=replace(
            balanced_phase_manifest.initial_experts,
            early=replace(
                balanced_phase_manifest.initial_experts.early,
                liquidity_strength=Decimal("0.250"),
            ),
        ),
        phase_selector=balanced_phase_manifest.phase_selector.kind,
    )
    assert first.digest == equivalent.digest
    for locus_index in range(12):
        changed_loci = list(first.experts.as_loci())
        changed_loci[locus_index] += Decimal("0.05")
        changed_locus = PhaseCoefficientGenome(
            experts=_phase_values(tuple(changed_loci)),
            phase_selector=balanced_phase_manifest.phase_selector.kind,
        )
        assert first.digest != changed_locus.digest, locus_index
    with pytest.raises(FrozenInstanceError):
        first.digest = "changed"  # type: ignore[misc]


def test_phase_genome_rejects_any_selector_except_the_fixed_public_rule(
    balanced_phase_manifest: PhaseSearchManifest,
) -> None:
    with pytest.raises(ValueError, match="fixed public resource selector"):
        PhaseCoefficientGenome(
            experts=balanced_phase_manifest.initial_experts,
            phase_selector="another-public-selector",
        )


def test_phase_generation_zero_has_stratified_locus_and_phase_proposals(
    balanced_phase_manifest: PhaseSearchManifest,
) -> None:
    population = build_initial_population(balanced_phase_manifest)

    assert len(population) == 16
    assert all(isinstance(candidate, PhaseAwareHeuristicCandidate) for candidate in population)
    assert population[0].genome.experts == balanced_phase_manifest.initial_experts
    assert population[0].identity.startswith("balanced-v4-candidate-g000-s000-")
    assert tuple(candidate.slot for candidate in population) == tuple(range(16))
    assert all(candidate.parent_identity is None for candidate in population)

    incumbent_loci = population[0].genome.experts.as_loci()
    for slot in range(1, 13):
        candidate_loci = population[slot].genome.experts.as_loci()
        changed_indices = tuple(
            index
            for index, (incumbent, proposed) in enumerate(
                zip(incumbent_loci, candidate_loci, strict=True)
            )
            if incumbent != proposed
        )
        assert changed_indices == (slot - 1,)

    for slot, phase_index in zip(range(13, 16), range(3), strict=True):
        candidate_experts = population[slot].genome.experts.as_tuple()
        incumbent_experts = population[0].genome.experts.as_tuple()
        assert candidate_experts[phase_index] != incumbent_experts[phase_index]
        assert tuple(
            candidate_experts[index] == incumbent_experts[index]
            for index in range(3)
            if index != phase_index
        ) == (True, True)

    assert build_initial_population(balanced_phase_manifest) == population


@pytest.mark.parametrize("personality", tuple(_PHASE_GENERATION_ZERO_GOLDENS))
def test_phase_generation_zero_full_genomes_and_identities_are_golden(
    personality: str,
) -> None:
    corpus = load_promotion_corpus(
        REPOSITORY_ROOT / "configs/promotion/development-heuristic-v4-v1.json",
        registry=BOT_SPECS_BY_NAME,
    )
    recipe = load_search_recipe(
        REPOSITORY_ROOT / f"configs/evolution/{personality}-v4-search-v2.json",
        development_corpus=corpus,
    )
    assert isinstance(recipe, PhaseSearchManifest)
    population = build_initial_population(recipe)

    assert _phase_population_snapshot(population) == _PHASE_GENERATION_ZERO_GOLDENS[personality]


def test_phase_later_generations_cover_all_loci_continuously(
    balanced_phase_manifest: PhaseSearchManifest,
) -> None:
    initial = build_initial_population(balanced_phase_manifest)
    elites = initial[: balanced_phase_manifest.algorithm.elite_count]
    changed_locus_counts = [0] * 12

    for generation in range(1, balanced_phase_manifest.algorithm.generation_count):
        children = build_mutation_population(
            balanced_phase_manifest,
            generation=generation,
            ranked_elites=elites,
        )
        assert (
            build_mutation_population(
                balanced_phase_manifest,
                generation=generation,
                ranked_elites=elites,
            )
            == children
        )
        assert tuple(child.parent_identity for child in children) == tuple(
            elites[slot % len(elites)].identity for slot in range(16)
        )
        for slot, child in enumerate(children):
            parent = elites[slot % len(elites)]
            changed_indices = tuple(
                index
                for index, (parent_value, child_value) in enumerate(
                    zip(
                        parent.genome.experts.as_loci(),
                        child.genome.experts.as_loci(),
                        strict=True,
                    )
                )
                if parent_value != child_value
            )
            expected_locus = ((generation - 1) * 16 + slot) % 12
            assert changed_indices == (expected_locus,)
            changed_locus_counts[expected_locus] += 1
            grid = balanced_phase_manifest.expert_coefficient_grids.as_loci()[expected_locus]
            child_value = child.genome.experts.as_loci()[expected_locus]
            assert grid.minimum <= child_value <= grid.maximum
            assert (child_value - grid.minimum) % grid.step == 0

    assert sorted(changed_locus_counts) == [14] * 4 + [15] * 8


def test_phase_generation_one_has_golden_genomes_and_identities(
    balanced_phase_manifest: PhaseSearchManifest,
) -> None:
    initial = build_initial_population(balanced_phase_manifest)
    children = build_mutation_population(
        balanced_phase_manifest,
        generation=1,
        ranked_elites=initial[: balanced_phase_manifest.algorithm.elite_count],
    )

    assert _phase_population_snapshot(children) == _BALANCED_GENERATION_ONE_GOLDEN


def test_phase_populations_are_independent_of_decimal_context(
    balanced_phase_manifest: PhaseSearchManifest,
) -> None:
    populations = []
    mutations = []
    for precision in (8, 28, 50):
        with localcontext() as context:
            context.prec = precision
            population = build_initial_population(balanced_phase_manifest)
            populations.append(population)
            mutations.append(
                build_mutation_population(
                    balanced_phase_manifest,
                    generation=1,
                    ranked_elites=population[: balanced_phase_manifest.algorithm.elite_count],
                )
            )

    assert populations[0] == populations[1] == populations[2]
    assert mutations[0] == mutations[1] == mutations[2]


def test_phase_mutation_rejects_a_parent_from_another_selector(
    balanced_phase_manifest: PhaseSearchManifest,
) -> None:
    elites = list(
        build_initial_population(balanced_phase_manifest)[
            : balanced_phase_manifest.algorithm.elite_count
        ]
    )
    object.__setattr__(
        elites[0].genome,
        "phase_selector",
        "another-public-selector",
    )

    with pytest.raises(ValueError, match="manifest phase selector"):
        build_mutation_population(
            balanced_phase_manifest,
            generation=1,
            ranked_elites=elites,
        )


def test_phase_candidate_spec_is_picklable_local_and_matches_direct_brain(
    balanced_phase_manifest: PhaseSearchManifest,
) -> None:
    candidate = build_initial_population(balanced_phase_manifest)[13]
    profile = candidate_profile(candidate)
    assert phase_candidate_profile(candidate) == profile
    for phase, expected_coefficients in zip(
        ("early", "middle", "late"),
        candidate.genome.experts.as_tuple(),
        strict=True,
    ):
        actual_expert = profile.profile_for_phase(phase)  # type: ignore[arg-type]
        assert (
            actual_expert.liquidity_strength,
            actual_expert.future_cash_weight,
            actual_expert.objective_progress_weight,
            actual_expert.bid_shading,
        ) == tuple(float(value) for value in expected_coefficients.as_tuple())
    spec = candidate_bot_spec(candidate)

    assert spec.name == candidate.identity
    assert spec.bot_id == candidate.identity
    assert candidate.identity not in BOT_SPECS_BY_NAME
    restored_spec = pickle.loads(pickle.dumps(spec))
    worker_brain = restored_spec.make_brain(seed=8675309)
    direct_brain = PhaseAwareHeuristicBotBrain(profile)
    bid_context = _context(hand=(int(Suit.ORE), int(Suit.SHEEP)))
    knowledge = _knowledge(private_cards=2)

    assert isinstance(worker_brain, PhaseAwareHeuristicBotBrain)
    assert worker_brain.profile == profile
    assert worker_brain.choose_decision(
        bid_context,
        knowledge,
    ) == direct_brain.choose_decision(bid_context, knowledge)

    for future_resources, expected_phase in (
        (10, "early"),
        (9, "middle"),
        (4, "late"),
    ):
        phase_context = _context_for_future_resources(future_resources)
        explained = worker_brain.choose_explained_decision(
            phase_context,
            _knowledge(),
            (),
        )
        assert isinstance(explained.explanation, PhaseAwareHeuristicBidExplanation)
        assert explained.explanation.selected_expert_phase == expected_phase


def _context_for_future_resources(future_resources: int) -> DecisionContext:
    already_won = 15 - future_resources - 1
    won_by_suit = (
        0,
        *(min(3, max(0, already_won - (3 * suit_index))) for suit_index in range(4)),
    )
    return replace(
        _context(),
        current_action_id=int(ActionId.AUCTION1),
        current_resource_ids=(int(Suit.BRICK), 0),
        won_resource_counts_by_seat=(
            won_by_suit,
            (0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0),
        ),
    )
