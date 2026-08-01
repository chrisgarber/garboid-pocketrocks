"""Build deterministic heuristic candidates without registering live bots."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from functools import partial
from typing import cast, overload

from garboid_pocketrocks.bots.base import BotSpec, BrainFactory
from garboid_pocketrocks.bots.heuristic import (
    HeuristicBotBrain,
    PhaseAwareHeuristicBotBrain,
)
from garboid_pocketrocks.evolution.manifest import (
    COEFFICIENT_NAMES,
    CoefficientGrid,
    CoefficientValues,
    Personality,
    PhaseCoefficientValues,
    PhaseSearchManifest,
    SearchManifest,
    SearchRecipe,
    decimal_from_grid_index,
    decimal_grid_index,
)
from garboid_pocketrocks.heuristics.phases import PHASE_SELECTOR_NAME
from garboid_pocketrocks.heuristics.profiles import (
    HeuristicProfile,
    PhaseAwareHeuristicProfile,
)
from garboid_pocketrocks.simulator.seeding import derive_seed


@dataclass(frozen=True, slots=True)
class CoefficientGenome:
    """One immutable set of exact search-grid coefficients."""

    coefficients: CoefficientValues
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, Decimal) and value.is_finite()
            for value in self.coefficients.as_tuple()
        ):
            raise ValueError("candidate coefficients must be finite Decimal values")
        object.__setattr__(self, "digest", _coefficient_digest(self.coefficients))


@dataclass(frozen=True, slots=True)
class HeuristicCandidate:
    """One separately identified proposal in a heuristic search."""

    personality: Personality
    generation: int
    slot: int
    genome: CoefficientGenome
    parent_identity: str | None
    identity: str = field(init=False)

    def __post_init__(self) -> None:
        if self.generation < 0:
            raise ValueError("candidate generation must be nonnegative")
        if self.slot < 0:
            raise ValueError("candidate slot must be nonnegative")
        if self.generation == 0 and self.parent_identity is not None:
            raise ValueError("generation-zero candidates cannot have a parent")
        if self.generation > 0 and self.parent_identity is None:
            raise ValueError("later-generation candidates must record a parent")
        object.__setattr__(
            self,
            "identity",
            (
                f"{self.personality}-v3-candidate-"
                f"g{self.generation:03d}-s{self.slot:03d}-{self.genome.digest[:12]}"
            ),
        )


@dataclass(frozen=True, slots=True)
class PhaseCoefficientGenome:
    """Three public-phase experts with one selector-bound content digest."""

    experts: PhaseCoefficientValues
    phase_selector: str
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.phase_selector != PHASE_SELECTOR_NAME:
            raise ValueError("phase-aware candidates must use the fixed public resource selector")
        if not all(
            isinstance(value, Decimal) and value.is_finite() for value in self.experts.as_loci()
        ):
            raise ValueError("phase-aware candidate coefficients must be finite Decimal values")
        object.__setattr__(
            self,
            "digest",
            _phase_coefficient_digest(
                self.experts,
                phase_selector=self.phase_selector,
            ),
        )


@dataclass(frozen=True, slots=True)
class PhaseAwareHeuristicCandidate:
    """One separately identified twelve-locus phase-aware proposal."""

    personality: Personality
    generation: int
    slot: int
    genome: PhaseCoefficientGenome
    parent_identity: str | None
    identity: str = field(init=False)

    def __post_init__(self) -> None:
        _validate_candidate_position(
            generation=self.generation,
            slot=self.slot,
            parent_identity=self.parent_identity,
        )
        object.__setattr__(
            self,
            "identity",
            (
                f"{self.personality}-v4-candidate-"
                f"g{self.generation:03d}-s{self.slot:03d}-{self.genome.digest[:12]}"
            ),
        )


type SearchCandidate = HeuristicCandidate | PhaseAwareHeuristicCandidate


@overload
def build_initial_population(manifest: SearchManifest) -> tuple[HeuristicCandidate, ...]: ...


@overload
def build_initial_population(
    manifest: PhaseSearchManifest,
) -> tuple[PhaseAwareHeuristicCandidate, ...]: ...


def build_initial_population(manifest: SearchRecipe) -> tuple[SearchCandidate, ...]:
    """Build generation zero from the predecessor followed by seeded proposals."""

    if isinstance(manifest, PhaseSearchManifest):
        return _build_phase_initial_population(manifest)

    candidates = [
        _candidate(
            manifest,
            generation=0,
            slot=0,
            coefficients=manifest.initial_coefficients,
            parent_identity=None,
        )
    ]
    for slot in range(1, manifest.algorithm.population_size):
        random_source = random.Random(_candidate_seed(manifest, generation=0, slot=slot))
        sampled_values = tuple(
            _sample_grid_value(grid, random_source)
            for grid in manifest.coefficient_grids.as_tuple()
        )
        candidates.append(
            _candidate(
                manifest,
                generation=0,
                slot=slot,
                coefficients=_coefficient_values(sampled_values),
                parent_identity=None,
            )
        )
    return tuple(candidates)


@overload
def build_mutation_population(
    manifest: SearchManifest,
    *,
    generation: int,
    ranked_elites: Sequence[HeuristicCandidate],
) -> tuple[HeuristicCandidate, ...]: ...


@overload
def build_mutation_population(
    manifest: PhaseSearchManifest,
    *,
    generation: int,
    ranked_elites: Sequence[PhaseAwareHeuristicCandidate],
) -> tuple[PhaseAwareHeuristicCandidate, ...]: ...


def build_mutation_population(
    manifest: SearchRecipe,
    *,
    generation: int,
    ranked_elites: Sequence[SearchCandidate],
) -> tuple[SearchCandidate, ...]:
    """Build one population by cycling ranked parents and changing one field."""

    if isinstance(manifest, PhaseSearchManifest):
        return _build_phase_mutation_population(
            manifest,
            generation=generation,
            ranked_elites=ranked_elites,
        )

    if not 1 <= generation < manifest.algorithm.generation_count:
        raise ValueError(
            f"mutation generation must be between 1 and {manifest.algorithm.generation_count - 1}"
        )
    candidate_elites = tuple(ranked_elites)
    expected_elites = manifest.algorithm.elite_count
    if len(candidate_elites) != expected_elites:
        raise ValueError(f"mutation needs exactly {expected_elites} ranked elites")
    if not all(isinstance(elite, HeuristicCandidate) for elite in candidate_elites):
        raise ValueError("scalar mutation parents must be scalar candidates")
    elites = cast(tuple[HeuristicCandidate, ...], candidate_elites)
    for elite in elites:
        if elite.personality != manifest.personality:
            raise ValueError(
                f"all mutation parents must have the {manifest.personality} personality"
            )
        if elite.generation >= generation:
            raise ValueError("mutation parents must come from an earlier generation")
        _validate_coefficients_on_grids(manifest, elite.genome.coefficients)

    children = []
    grids = manifest.coefficient_grids.as_tuple()
    for slot in range(manifest.algorithm.population_size):
        parent = elites[slot % len(elites)]
        random_source = random.Random(_candidate_seed(manifest, generation=generation, slot=slot))
        coefficient_index = random_source.randrange(len(COEFFICIENT_NAMES))
        grid = grids[coefficient_index]
        parent_values = parent.genome.coefficients.as_tuple()
        parent_grid_index = decimal_grid_index(
            parent_values[coefficient_index],
            minimum=grid.minimum,
            step=grid.step,
        )
        legal_offsets = _legal_mutation_offsets(
            parent_grid_index,
            grid=grid,
            radius=manifest.algorithm.mutation_radius_steps,
        )
        if not legal_offsets:
            coefficient_name = COEFFICIENT_NAMES[coefficient_index]
            raise ValueError(f"{coefficient_name} has no legal nonzero mutation")
        offset = random_source.choice(legal_offsets)
        child_values = list(parent_values)
        child_values[coefficient_index] = decimal_from_grid_index(
            parent_grid_index + offset,
            minimum=grid.minimum,
            step=grid.step,
        )
        children.append(
            _candidate(
                manifest,
                generation=generation,
                slot=slot,
                coefficients=_coefficient_values(child_values),
                parent_identity=parent.identity,
            )
        )
    return tuple(children)


@overload
def candidate_profile(candidate: HeuristicCandidate) -> HeuristicProfile: ...


@overload
def candidate_profile(
    candidate: PhaseAwareHeuristicCandidate,
) -> PhaseAwareHeuristicProfile: ...


def candidate_profile(
    candidate: SearchCandidate,
) -> HeuristicProfile | PhaseAwareHeuristicProfile:
    """Convert exact Decimal coefficients only at the existing profile boundary."""

    if isinstance(candidate, PhaseAwareHeuristicCandidate):
        return phase_candidate_profile(candidate)

    coefficients = candidate.genome.coefficients
    return _profile(candidate.personality, coefficients)


def phase_candidate_profile(
    candidate: PhaseAwareHeuristicCandidate,
) -> PhaseAwareHeuristicProfile:
    """Build the named three-expert profile represented by one v4 candidate."""

    experts = candidate.genome.experts
    return PhaseAwareHeuristicProfile(
        name=candidate.personality,
        early=_profile(candidate.personality, experts.early),
        middle=_profile(candidate.personality, experts.middle),
        late=_profile(candidate.personality, experts.late),
        phase_selector=candidate.genome.phase_selector,
    )


def build_candidate_brain(
    profile: HeuristicProfile,
    seed: int | None,
) -> HeuristicBotBrain:
    """Construct the ordinary heuristic brain for a local search worker."""

    del seed
    return HeuristicBotBrain(profile)


def build_phase_candidate_brain(
    profile: PhaseAwareHeuristicProfile,
    seed: int | None,
) -> PhaseAwareHeuristicBotBrain:
    """Construct a phase-aware heuristic brain for a local search worker."""

    del seed
    return PhaseAwareHeuristicBotBrain(profile)


def candidate_bot_spec(candidate: SearchCandidate) -> BotSpec:
    """Create a picklable local-only spec whose name is the candidate identity."""

    profile = candidate_profile(candidate)
    if isinstance(profile, PhaseAwareHeuristicProfile):
        brain_factory: BrainFactory = partial(build_phase_candidate_brain, profile)
    else:
        brain_factory = partial(build_candidate_brain, profile)
    return BotSpec.for_simulation(candidate.identity, brain_factory)


def _build_phase_initial_population(
    manifest: PhaseSearchManifest,
) -> tuple[PhaseAwareHeuristicCandidate, ...]:
    candidates = [
        _phase_candidate(
            manifest,
            generation=0,
            slot=0,
            experts=manifest.initial_experts,
            parent_identity=None,
        )
    ]
    incumbent_loci = manifest.initial_experts.as_loci()
    grids = manifest.expert_coefficient_grids.as_loci()
    radius = manifest.algorithm.mutation_radius_steps

    for slot in range(1, 13):
        random_source = random.Random(_candidate_seed(manifest, generation=0, slot=slot))
        locus_index = slot - 1
        proposal_loci = _mutate_one_phase_locus(
            incumbent_loci,
            grids=grids,
            locus_index=locus_index,
            radius=radius,
            random_source=random_source,
        )
        candidates.append(
            _phase_candidate(
                manifest,
                generation=0,
                slot=slot,
                experts=_phase_coefficient_values(proposal_loci),
                parent_identity=None,
            )
        )

    for slot in range(13, 16):
        random_source = random.Random(_candidate_seed(manifest, generation=0, slot=slot))
        phase_index = slot - 13
        phase_start = phase_index * len(COEFFICIENT_NAMES)
        broad_proposal_loci = list(incumbent_loci)
        phase_grids = grids[phase_start : phase_start + len(COEFFICIENT_NAMES)]
        incumbent_phase = incumbent_loci[phase_start : phase_start + len(COEFFICIENT_NAMES)]
        sampled_phase = _sample_broad_phase_proposal(
            incumbent_phase,
            grids=phase_grids,
            radius=radius,
            random_source=random_source,
        )
        broad_proposal_loci[phase_start : phase_start + len(COEFFICIENT_NAMES)] = sampled_phase
        candidates.append(
            _phase_candidate(
                manifest,
                generation=0,
                slot=slot,
                experts=_phase_coefficient_values(broad_proposal_loci),
                parent_identity=None,
            )
        )
    return tuple(candidates)


def _build_phase_mutation_population(
    manifest: PhaseSearchManifest,
    *,
    generation: int,
    ranked_elites: Sequence[SearchCandidate],
) -> tuple[PhaseAwareHeuristicCandidate, ...]:
    if not 1 <= generation < manifest.algorithm.generation_count:
        raise ValueError(
            f"mutation generation must be between 1 and {manifest.algorithm.generation_count - 1}"
        )
    elites = tuple(ranked_elites)
    expected_elites = manifest.algorithm.elite_count
    if len(elites) != expected_elites:
        raise ValueError(f"mutation needs exactly {expected_elites} ranked elites")
    for elite in elites:
        if not isinstance(elite, PhaseAwareHeuristicCandidate):
            raise ValueError("phase-aware mutation parents must be phase-aware candidates")
        if elite.personality != manifest.personality:
            raise ValueError(
                f"all mutation parents must have the {manifest.personality} personality"
            )
        if elite.genome.phase_selector != manifest.phase_selector.kind:
            raise ValueError("all mutation parents must use the manifest phase selector")
        if elite.generation >= generation:
            raise ValueError("mutation parents must come from an earlier generation")
        _validate_phase_coefficients_on_grids(manifest, elite.genome.experts)

    children = []
    grids = manifest.expert_coefficient_grids.as_loci()
    for slot in range(manifest.algorithm.population_size):
        parent = elites[slot % len(elites)]
        assert isinstance(parent, PhaseAwareHeuristicCandidate)
        random_source = random.Random(_candidate_seed(manifest, generation=generation, slot=slot))
        locus_index = ((generation - 1) * manifest.algorithm.population_size + slot) % len(grids)
        child_loci = _mutate_one_phase_locus(
            parent.genome.experts.as_loci(),
            grids=grids,
            locus_index=locus_index,
            radius=manifest.algorithm.mutation_radius_steps,
            random_source=random_source,
        )
        children.append(
            _phase_candidate(
                manifest,
                generation=generation,
                slot=slot,
                experts=_phase_coefficient_values(child_loci),
                parent_identity=parent.identity,
            )
        )
    return tuple(children)


def _sample_broad_phase_proposal(
    incumbent: Sequence[Decimal],
    *,
    grids: Sequence[CoefficientGrid],
    radius: int,
    random_source: random.Random,
) -> tuple[Decimal, ...]:
    """Sample all four phase values and guarantee the resulting phase differs.

    Each value is sampled independently from its full grid. An individual
    value may equal the incumbent; "broad" means the whole four-value phase is
    proposed together, not that all four values must change.
    """

    sampled = tuple(_sample_grid_value(grid, random_source) for grid in grids)
    if sampled != tuple(incumbent):
        return sampled
    return _mutate_one_phase_locus(
        incumbent,
        grids=grids,
        locus_index=0,
        radius=radius,
        random_source=random_source,
    )


def _mutate_one_phase_locus(
    loci: Sequence[Decimal],
    *,
    grids: Sequence[CoefficientGrid],
    locus_index: int,
    radius: int,
    random_source: random.Random,
) -> tuple[Decimal, ...]:
    grid = grids[locus_index]
    current_grid_index = decimal_grid_index(
        loci[locus_index],
        minimum=grid.minimum,
        step=grid.step,
    )
    legal_offsets = _legal_mutation_offsets(
        current_grid_index,
        grid=grid,
        radius=radius,
    )
    if not legal_offsets:
        phase_index, coefficient_index = divmod(locus_index, len(COEFFICIENT_NAMES))
        phase = ("early", "middle", "late")[phase_index]
        raise ValueError(
            f"{phase} {COEFFICIENT_NAMES[coefficient_index]} has no legal nonzero mutation"
        )
    proposal = list(loci)
    proposal[locus_index] = decimal_from_grid_index(
        current_grid_index + random_source.choice(legal_offsets),
        minimum=grid.minimum,
        step=grid.step,
    )
    return tuple(proposal)


def _candidate(
    manifest: SearchManifest,
    *,
    generation: int,
    slot: int,
    coefficients: CoefficientValues,
    parent_identity: str | None,
) -> HeuristicCandidate:
    _validate_coefficients_on_grids(manifest, coefficients)
    return HeuristicCandidate(
        personality=manifest.personality,
        generation=generation,
        slot=slot,
        genome=CoefficientGenome(coefficients),
        parent_identity=parent_identity,
    )


def _phase_candidate(
    manifest: PhaseSearchManifest,
    *,
    generation: int,
    slot: int,
    experts: PhaseCoefficientValues,
    parent_identity: str | None,
) -> PhaseAwareHeuristicCandidate:
    _validate_phase_coefficients_on_grids(manifest, experts)
    return PhaseAwareHeuristicCandidate(
        personality=manifest.personality,
        generation=generation,
        slot=slot,
        genome=PhaseCoefficientGenome(
            experts=experts,
            phase_selector=manifest.phase_selector.kind,
        ),
        parent_identity=parent_identity,
    )


def _candidate_seed(manifest: SearchRecipe, *, generation: int, slot: int) -> int:
    namespace = f"heuristic-evolution:{manifest.digest}:generation:{generation}"
    return derive_seed(manifest.search_seed, namespace, slot)


def _sample_grid_value(grid: CoefficientGrid, random_source: random.Random) -> Decimal:
    maximum_index = decimal_grid_index(
        grid.maximum,
        minimum=grid.minimum,
        step=grid.step,
    )
    sampled_index = random_source.randrange(maximum_index + 1)
    return decimal_from_grid_index(
        sampled_index,
        minimum=grid.minimum,
        step=grid.step,
    )


def _legal_mutation_offsets(
    current_index: int,
    *,
    grid: CoefficientGrid,
    radius: int,
) -> tuple[int, ...]:
    maximum_index = decimal_grid_index(
        grid.maximum,
        minimum=grid.minimum,
        step=grid.step,
    )
    return tuple(
        offset
        for offset in range(-radius, radius + 1)
        if offset != 0 and 0 <= current_index + offset <= maximum_index
    )


def _validate_coefficients_on_grids(
    manifest: SearchManifest,
    coefficients: CoefficientValues,
) -> None:
    for name, value, grid in zip(
        COEFFICIENT_NAMES,
        coefficients.as_tuple(),
        manifest.coefficient_grids.as_tuple(),
        strict=True,
    ):
        if not value.is_finite():
            raise ValueError(f"{name} must be finite")
        if not grid.minimum <= value <= grid.maximum:
            raise ValueError(f"{name} must stay within its configured grid")
        try:
            decimal_grid_index(
                value,
                minimum=grid.minimum,
                step=grid.step,
            )
        except ValueError as error:
            raise ValueError(f"{name} must align to its configured grid step") from error


def _validate_phase_coefficients_on_grids(
    manifest: PhaseSearchManifest,
    experts: PhaseCoefficientValues,
) -> None:
    for locus_index, (value, grid) in enumerate(
        zip(
            experts.as_loci(),
            manifest.expert_coefficient_grids.as_loci(),
            strict=True,
        )
    ):
        phase_index, coefficient_index = divmod(locus_index, len(COEFFICIENT_NAMES))
        phase = ("early", "middle", "late")[phase_index]
        name = COEFFICIENT_NAMES[coefficient_index]
        if not value.is_finite():
            raise ValueError(f"{phase} {name} must be finite")
        if not grid.minimum <= value <= grid.maximum:
            raise ValueError(f"{phase} {name} must stay within its configured grid")
        try:
            decimal_grid_index(
                value,
                minimum=grid.minimum,
                step=grid.step,
            )
        except ValueError as error:
            raise ValueError(f"{phase} {name} must align to its configured grid step") from error


def _coefficient_values(values: Sequence[Decimal]) -> CoefficientValues:
    if len(values) != len(COEFFICIENT_NAMES):
        raise ValueError("a heuristic genome needs exactly four coefficients")
    return CoefficientValues(
        liquidity_strength=values[0],
        future_cash_weight=values[1],
        objective_progress_weight=values[2],
        bid_shading=values[3],
    )


def _phase_coefficient_values(values: Sequence[Decimal]) -> PhaseCoefficientValues:
    expected_count = len(COEFFICIENT_NAMES) * 3
    if len(values) != expected_count:
        raise ValueError("a phase-aware heuristic genome needs exactly twelve coefficients")
    width = len(COEFFICIENT_NAMES)
    return PhaseCoefficientValues(
        early=_coefficient_values(values[0:width]),
        middle=_coefficient_values(values[width : width * 2]),
        late=_coefficient_values(values[width * 2 : width * 3]),
    )


def _profile(
    personality: Personality,
    coefficients: CoefficientValues,
) -> HeuristicProfile:
    return HeuristicProfile(
        name=personality,
        liquidity_strength=float(coefficients.liquidity_strength),
        future_cash_weight=float(coefficients.future_cash_weight),
        objective_progress_weight=float(coefficients.objective_progress_weight),
        bid_shading=float(coefficients.bid_shading),
    )


def _coefficient_digest(coefficients: CoefficientValues) -> str:
    payload = {
        name: _canonical_decimal_text(value)
        for name, value in zip(
            COEFFICIENT_NAMES,
            coefficients.as_tuple(),
            strict=True,
        )
    }
    encoded = (
        json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _phase_coefficient_digest(
    experts: PhaseCoefficientValues,
    *,
    phase_selector: str,
) -> str:
    payload = {
        "experts": {
            phase: {
                name: _canonical_decimal_text(value)
                for name, value in zip(
                    COEFFICIENT_NAMES,
                    coefficients.as_tuple(),
                    strict=True,
                )
            }
            for phase, coefficients in zip(
                ("early", "middle", "late"),
                experts.as_tuple(),
                strict=True,
            )
        },
        "phase_selector": phase_selector,
    }
    encoded = (
        json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_candidate_position(
    *,
    generation: int,
    slot: int,
    parent_identity: str | None,
) -> None:
    if generation < 0:
        raise ValueError("candidate generation must be nonnegative")
    if slot < 0:
        raise ValueError("candidate slot must be nonnegative")
    if generation == 0 and parent_identity is not None:
        raise ValueError("generation-zero candidates cannot have a parent")
    if generation > 0 and parent_identity is None:
        raise ValueError("later-generation candidates must record a parent")


def _canonical_decimal_text(value: Decimal) -> str:
    if value == 0:
        return "0"
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered
