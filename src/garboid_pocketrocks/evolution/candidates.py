"""Build deterministic heuristic candidates without registering live bots."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from functools import partial

from garboid_pocketrocks.bots.base import BotSpec
from garboid_pocketrocks.bots.heuristic import HeuristicBotBrain
from garboid_pocketrocks.evolution.manifest import (
    COEFFICIENT_NAMES,
    CoefficientGrid,
    CoefficientValues,
    Personality,
    SearchManifest,
    decimal_from_grid_index,
    decimal_grid_index,
)
from garboid_pocketrocks.heuristics.profiles import HeuristicProfile
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


def build_initial_population(manifest: SearchManifest) -> tuple[HeuristicCandidate, ...]:
    """Build generation zero: the v2 incumbent followed by seeded grid samples."""

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


def build_mutation_population(
    manifest: SearchManifest,
    *,
    generation: int,
    ranked_elites: Sequence[HeuristicCandidate],
) -> tuple[HeuristicCandidate, ...]:
    """Build one population by cycling ranked parents and changing one field."""

    if not 1 <= generation < manifest.algorithm.generation_count:
        raise ValueError(
            f"mutation generation must be between 1 and {manifest.algorithm.generation_count - 1}"
        )
    elites = tuple(ranked_elites)
    expected_elites = manifest.algorithm.elite_count
    if len(elites) != expected_elites:
        raise ValueError(f"mutation needs exactly {expected_elites} ranked elites")
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


def candidate_profile(candidate: HeuristicCandidate) -> HeuristicProfile:
    """Convert exact Decimal coefficients only at the existing profile boundary."""

    coefficients = candidate.genome.coefficients
    return HeuristicProfile(
        name=candidate.personality,
        liquidity_strength=float(coefficients.liquidity_strength),
        future_cash_weight=float(coefficients.future_cash_weight),
        objective_progress_weight=float(coefficients.objective_progress_weight),
        bid_shading=float(coefficients.bid_shading),
    )


def build_candidate_brain(
    profile: HeuristicProfile,
    seed: int | None,
) -> HeuristicBotBrain:
    """Construct the ordinary heuristic brain for a local search worker."""

    del seed
    return HeuristicBotBrain(profile)


def candidate_bot_spec(candidate: HeuristicCandidate) -> BotSpec:
    """Create a picklable local-only spec whose name is the candidate identity."""

    brain_factory = partial(build_candidate_brain, candidate_profile(candidate))
    return BotSpec.for_simulation(candidate.identity, brain_factory)


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


def _candidate_seed(manifest: SearchManifest, *, generation: int, slot: int) -> int:
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


def _coefficient_values(values: Sequence[Decimal]) -> CoefficientValues:
    if len(values) != len(COEFFICIENT_NAMES):
        raise ValueError("a heuristic genome needs exactly four coefficients")
    return CoefficientValues(
        liquidity_strength=values[0],
        future_cash_weight=values[1],
        objective_progress_weight=values[2],
        bid_shading=values[3],
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


def _canonical_decimal_text(value: Decimal) -> str:
    if value == 0:
        return "0"
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered
