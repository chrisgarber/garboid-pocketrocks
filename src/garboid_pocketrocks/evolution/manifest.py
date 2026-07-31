"""Load, validate, normalize, and hash heuristic search recipes."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal, NoReturn, cast

from garboid_pocketrocks.heuristics.phases import PHASE_SELECTOR_NAME, HeuristicPhase
from garboid_pocketrocks.heuristics.profiles import HEURISTIC_V2, HEURISTIC_V3, HeuristicProfile
from garboid_pocketrocks.promotion.corpus import PromotionCorpus

CoefficientName = Literal[
    "liquidity_strength",
    "future_cash_weight",
    "objective_progress_weight",
    "bid_shading",
]
Personality = Literal["aggressive", "balanced", "passive"]

COEFFICIENT_NAMES: tuple[CoefficientName, ...] = (
    "liquidity_strength",
    "future_cash_weight",
    "objective_progress_weight",
    "bid_shading",
)

_EXPECTED_MANIFEST_KEYS = {
    "schema_version",
    "name",
    "personality",
    "predecessor_name",
    "development_corpus",
    "search_seed",
    "algorithm",
    "initial_coefficients",
    "coefficient_grids",
}
_EXPECTED_CORPUS_KEYS = {"name", "digest"}
_EXPECTED_ALGORITHM_KEYS = {
    "name",
    "generation_count",
    "population_size",
    "elite_count",
    "mutation_radius_steps",
}
_EXPECTED_GRID_KEYS = {"minimum", "maximum", "step"}
_SEARCH_NAME_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_SUPPORTED_ALGORITHM = "mu-plus-lambda-v1"
_PHASES: tuple[HeuristicPhase, ...] = ("early", "middle", "late")
_EXPECTED_PHASE_MANIFEST_KEYS = {
    "schema_version",
    "name",
    "personality",
    "predecessor_name",
    "development_corpus",
    "search_seed",
    "algorithm",
    "phase_selector",
    "boundary_evidence",
    "initial_experts",
    "expert_coefficient_grids",
}
_EXPECTED_PHASE_KEYS: set[str] = set(_PHASES)
_EXPECTED_PHASE_SELECTOR_KEYS = {"kind", "early", "middle", "late"}
_EXPECTED_BOUNDARY_EVIDENCE_KEYS = {
    "report_path",
    "report_digest",
    "slices_path",
    "slices_digest",
}
_PHASE_SELECTOR_EARLY_RULE = "3*future>=2*total"
_PHASE_SELECTOR_MIDDLE_RULE = "3*future>=total"
_PHASE_SELECTOR_LATE_RULE = "otherwise"
_BOUNDARY_REPORT_PATH = "docs/benchmarks/2026-07-30-heuristic-v4-phase-boundaries.md"
_BOUNDARY_REPORT_DIGEST = "9961f26f32270dcebc98df443588e96cbde2f953858cd131c66a37aeecaa9b01"
_BOUNDARY_SLICES_PATH = (
    "docs/benchmarks/tournaments/"
    "2026-07-30-heuristic-v3-phase-boundaries-development/phase-boundary-slices.csv"
)
_BOUNDARY_SLICES_DIGEST = "4f8aa60edf31b28c746cb8004a4dd5468ee8ab1b26462550c914b2e3fa50d7ae"


@dataclass(frozen=True, slots=True)
class CoefficientValues:
    """The four existing heuristic policy coefficients."""

    liquidity_strength: Decimal
    future_cash_weight: Decimal
    objective_progress_weight: Decimal
    bid_shading: Decimal

    def as_tuple(self) -> tuple[Decimal, Decimal, Decimal, Decimal]:
        """Return coefficients in the single canonical field order."""

        return (
            self.liquidity_strength,
            self.future_cash_weight,
            self.objective_progress_weight,
            self.bid_shading,
        )


@dataclass(frozen=True, slots=True)
class CoefficientGrid:
    """The inclusive range and step allowed for one coefficient."""

    minimum: Decimal
    maximum: Decimal
    step: Decimal

    def as_tuple(self) -> tuple[Decimal, Decimal, Decimal]:
        """Return the grid as minimum, maximum, and step."""

        return (self.minimum, self.maximum, self.step)


@dataclass(frozen=True, slots=True)
class CoefficientGrids:
    """Named grids for exactly the four existing coefficient families."""

    liquidity_strength: CoefficientGrid
    future_cash_weight: CoefficientGrid
    objective_progress_weight: CoefficientGrid
    bid_shading: CoefficientGrid

    def as_tuple(self) -> tuple[CoefficientGrid, CoefficientGrid, CoefficientGrid, CoefficientGrid]:
        """Return grids in the single canonical coefficient order."""

        return (
            self.liquidity_strength,
            self.future_cash_weight,
            self.objective_progress_weight,
            self.bid_shading,
        )

    def as_tuples(
        self,
    ) -> tuple[
        tuple[Decimal, Decimal, Decimal],
        tuple[Decimal, Decimal, Decimal],
        tuple[Decimal, Decimal, Decimal],
        tuple[Decimal, Decimal, Decimal],
    ]:
        """Return each grid as a tuple in canonical coefficient order."""

        return tuple(grid.as_tuple() for grid in self.as_tuple())  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class DevelopmentCorpusBinding:
    """The exact development games a search recipe is allowed to use."""

    name: str
    digest: str


@dataclass(frozen=True, slots=True)
class SearchAlgorithm:
    """Deterministic mutation-only evolution settings."""

    name: str
    generation_count: int
    population_size: int
    elite_count: int
    mutation_radius_steps: int


@dataclass(frozen=True, slots=True)
class SearchManifest:
    """One validated, normalized, content-addressed search recipe."""

    schema_version: int
    name: str
    personality: Personality
    predecessor_name: str
    development_corpus: DevelopmentCorpusBinding
    search_seed: int
    algorithm: SearchAlgorithm
    initial_coefficients: CoefficientValues
    coefficient_grids: CoefficientGrids
    digest: str


@dataclass(frozen=True, slots=True)
class PhaseCoefficientValues:
    """The three experts' coefficients in public-resource phase order."""

    early: CoefficientValues
    middle: CoefficientValues
    late: CoefficientValues

    def as_tuple(self) -> tuple[CoefficientValues, CoefficientValues, CoefficientValues]:
        """Return experts in the canonical early, middle, late order."""

        return (self.early, self.middle, self.late)

    def as_loci(self) -> tuple[Decimal, ...]:
        """Return all twelve values, phase first and coefficient second."""

        return tuple(value for expert in self.as_tuple() for value in expert.as_tuple())


@dataclass(frozen=True, slots=True)
class PhaseCoefficientGrids:
    """The allowed coefficient grids for the three public-resource phases."""

    early: CoefficientGrids
    middle: CoefficientGrids
    late: CoefficientGrids

    def as_tuple(self) -> tuple[CoefficientGrids, CoefficientGrids, CoefficientGrids]:
        """Return expert grids in the canonical early, middle, late order."""

        return (self.early, self.middle, self.late)

    def as_loci(self) -> tuple[CoefficientGrid, ...]:
        """Return all twelve grids, phase first and coefficient second."""

        return tuple(grid for expert in self.as_tuple() for grid in expert.as_tuple())


@dataclass(frozen=True, slots=True)
class PhaseSelector:
    """The one public-information rule allowed for phase-aware v4 searches."""

    kind: str
    early: str
    middle: str
    late: str


@dataclass(frozen=True, slots=True)
class BoundaryEvidence:
    """Content-addressed development evidence chosen before v4 search."""

    report_path: str
    report_digest: str
    slices_path: str
    slices_digest: str


@dataclass(frozen=True, slots=True)
class PhaseSearchManifest:
    """One validated twelve-locus phase-aware search recipe."""

    schema_version: int
    name: str
    personality: Personality
    predecessor_name: str
    development_corpus: DevelopmentCorpusBinding
    search_seed: int
    algorithm: SearchAlgorithm
    phase_selector: PhaseSelector
    boundary_evidence: BoundaryEvidence
    initial_experts: PhaseCoefficientValues
    expert_coefficient_grids: PhaseCoefficientGrids
    digest: str


type SearchRecipe = SearchManifest | PhaseSearchManifest

_FIXED_PHASE_SEARCH_ALGORITHM = SearchAlgorithm(
    name=_SUPPORTED_ALGORITHM,
    generation_count=12,
    population_size=16,
    elite_count=4,
    mutation_radius_steps=4,
)
_ESTABLISHED_COEFFICIENT_GRIDS = CoefficientGrids(
    liquidity_strength=CoefficientGrid(
        minimum=Decimal("0"),
        maximum=Decimal("1.5"),
        step=Decimal("0.05"),
    ),
    future_cash_weight=CoefficientGrid(
        minimum=Decimal("0"),
        maximum=Decimal("2"),
        step=Decimal("0.05"),
    ),
    objective_progress_weight=CoefficientGrid(
        minimum=Decimal("0"),
        maximum=Decimal("1"),
        step=Decimal("0.05"),
    ),
    bid_shading=CoefficientGrid(
        minimum=Decimal("0"),
        maximum=Decimal("1"),
        step=Decimal("0.05"),
    ),
)


class SearchManifestError(ValueError):
    """Explain why a search manifest cannot be trusted."""

    code: str

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class _NonFiniteJsonNumber(ValueError):
    """Internal signal raised when JSON contains NaN or infinity."""


def load_search_manifest(
    path: Path,
    *,
    development_corpus: PromotionCorpus,
) -> SearchManifest:
    """Load one recipe and bind it to an already validated development corpus."""

    if development_corpus.recipe.purpose != "development":
        raise SearchManifestError(
            "held_out_corpus_forbidden",
            "Evolution may use only a development corpus; held-out games are the final exam.",
        )

    payload = _load_json_object(path)
    _reject_held_out_keys(payload)
    _require_exact_keys(
        payload,
        _EXPECTED_MANIFEST_KEYS,
        code="invalid_manifest_keys",
        subject="search manifest",
    )

    schema_version = _require_integer(
        payload["schema_version"],
        code="unsupported_schema",
        field_name="schema_version",
        minimum=1,
    )
    if schema_version != 1:
        raise SearchManifestError(
            "unsupported_schema",
            f"Search manifest schema version {schema_version} is not supported; expected 1.",
        )

    personality = _decode_personality(payload["personality"])
    name = _decode_search_name(payload["name"], personality=personality)
    predecessor_name = _decode_predecessor(
        payload["predecessor_name"],
        personality=personality,
    )
    corpus_binding = _decode_corpus_binding(
        payload["development_corpus"],
        development_corpus=development_corpus,
    )
    search_seed = _require_integer(
        payload["search_seed"],
        code="invalid_search_seed",
        field_name="search_seed",
        minimum=0,
    )
    algorithm = _decode_algorithm(payload["algorithm"])
    initial_coefficients = _decode_coefficient_values(payload["initial_coefficients"])
    coefficient_grids = _decode_coefficient_grids(payload["coefficient_grids"])
    _validate_initial_coefficients(
        initial_coefficients,
        grids=coefficient_grids,
        personality=personality,
    )

    manifest_without_digest = SearchManifest(
        schema_version=schema_version,
        name=name,
        personality=personality,
        predecessor_name=predecessor_name,
        development_corpus=corpus_binding,
        search_seed=search_seed,
        algorithm=algorithm,
        initial_coefficients=initial_coefficients,
        coefficient_grids=coefficient_grids,
        digest="",
    )
    digest = recompute_search_manifest_digest(manifest_without_digest)
    return SearchManifest(
        schema_version=manifest_without_digest.schema_version,
        name=manifest_without_digest.name,
        personality=manifest_without_digest.personality,
        predecessor_name=manifest_without_digest.predecessor_name,
        development_corpus=manifest_without_digest.development_corpus,
        search_seed=manifest_without_digest.search_seed,
        algorithm=manifest_without_digest.algorithm,
        initial_coefficients=manifest_without_digest.initial_coefficients,
        coefficient_grids=manifest_without_digest.coefficient_grids,
        digest=digest,
    )


def load_search_recipe(
    path: Path,
    *,
    development_corpus: PromotionCorpus,
) -> SearchRecipe:
    """Load either an immutable v1 recipe or a phase-aware v2 recipe."""

    if development_corpus.recipe.purpose != "development":
        raise SearchManifestError(
            "held_out_corpus_forbidden",
            "Evolution may use only a development corpus; held-out games are the final exam.",
        )

    payload = _load_json_object(path)
    _reject_held_out_keys(payload)
    schema_version = _require_integer(
        payload.get("schema_version"),
        code="unsupported_schema",
        field_name="schema_version",
        minimum=1,
    )
    if schema_version == 1:
        return load_search_manifest(path, development_corpus=development_corpus)
    if schema_version != 2:
        raise SearchManifestError(
            "unsupported_schema",
            f"Search manifest schema version {schema_version} is not supported; expected 1 or 2.",
        )
    return _decode_phase_search_manifest(
        payload,
        development_corpus=development_corpus,
    )


def _decode_phase_search_manifest(
    payload: dict[str, object],
    *,
    development_corpus: PromotionCorpus,
) -> PhaseSearchManifest:
    _require_exact_keys(
        payload,
        _EXPECTED_PHASE_MANIFEST_KEYS,
        code="invalid_manifest_keys",
        subject="phase search manifest",
    )
    personality = _decode_personality(payload["personality"])
    name = _decode_phase_search_name(payload["name"], personality=personality)
    predecessor_name = _decode_phase_predecessor(
        payload["predecessor_name"],
        personality=personality,
    )
    corpus_binding = _decode_corpus_binding(
        payload["development_corpus"],
        development_corpus=development_corpus,
    )
    search_seed = _require_integer(
        payload["search_seed"],
        code="invalid_search_seed",
        field_name="search_seed",
        minimum=0,
    )
    algorithm = _decode_algorithm(payload["algorithm"])
    if algorithm != _FIXED_PHASE_SEARCH_ALGORITHM:
        raise SearchManifestError(
            "wrong_phase_search_algorithm",
            "Schema-v2 searches require exactly 12 generations, a population of 16, "
            "4 elites, and a 4-step mutation radius.",
        )
    phase_selector = _decode_phase_selector(payload["phase_selector"])
    boundary_evidence = _decode_boundary_evidence(payload["boundary_evidence"])
    initial_experts = _decode_phase_coefficient_values(payload["initial_experts"])
    expert_grids = _decode_phase_coefficient_grids(payload["expert_coefficient_grids"])
    _validate_phase_initial_coefficients(
        initial_experts,
        grids=expert_grids,
        personality=personality,
    )

    without_digest = PhaseSearchManifest(
        schema_version=2,
        name=name,
        personality=personality,
        predecessor_name=predecessor_name,
        development_corpus=corpus_binding,
        search_seed=search_seed,
        algorithm=algorithm,
        phase_selector=phase_selector,
        boundary_evidence=boundary_evidence,
        initial_experts=initial_experts,
        expert_coefficient_grids=expert_grids,
        digest="",
    )
    return PhaseSearchManifest(
        schema_version=without_digest.schema_version,
        name=without_digest.name,
        personality=without_digest.personality,
        predecessor_name=without_digest.predecessor_name,
        development_corpus=without_digest.development_corpus,
        search_seed=without_digest.search_seed,
        algorithm=without_digest.algorithm,
        phase_selector=without_digest.phase_selector,
        boundary_evidence=without_digest.boundary_evidence,
        initial_experts=without_digest.initial_experts,
        expert_coefficient_grids=without_digest.expert_coefficient_grids,
        digest=recompute_phase_search_manifest_digest(without_digest),
    )


def search_manifest_payload(manifest: SearchManifest) -> dict[str, object]:
    """Return the normalized recipe without its derived digest."""

    return {
        "schema_version": manifest.schema_version,
        "name": manifest.name,
        "personality": manifest.personality,
        "predecessor_name": manifest.predecessor_name,
        "development_corpus": {
            "name": manifest.development_corpus.name,
            "digest": manifest.development_corpus.digest,
        },
        "search_seed": manifest.search_seed,
        "algorithm": {
            "name": manifest.algorithm.name,
            "generation_count": manifest.algorithm.generation_count,
            "population_size": manifest.algorithm.population_size,
            "elite_count": manifest.algorithm.elite_count,
            "mutation_radius_steps": manifest.algorithm.mutation_radius_steps,
        },
        "initial_coefficients": _coefficient_values_payload(manifest.initial_coefficients),
        "coefficient_grids": _coefficient_grids_payload(manifest.coefficient_grids),
    }


def recompute_search_manifest_digest(manifest: SearchManifest) -> str:
    """Hash the manifest's current normalized content, ignoring its stored digest."""

    return hashlib.sha256(_canonical_json_bytes(search_manifest_payload(manifest))).hexdigest()


def phase_search_manifest_payload(manifest: PhaseSearchManifest) -> dict[str, object]:
    """Return a normalized schema-v2 recipe without its derived digest."""

    return {
        "schema_version": manifest.schema_version,
        "name": manifest.name,
        "personality": manifest.personality,
        "predecessor_name": manifest.predecessor_name,
        "development_corpus": {
            "name": manifest.development_corpus.name,
            "digest": manifest.development_corpus.digest,
        },
        "search_seed": manifest.search_seed,
        "algorithm": {
            "name": manifest.algorithm.name,
            "generation_count": manifest.algorithm.generation_count,
            "population_size": manifest.algorithm.population_size,
            "elite_count": manifest.algorithm.elite_count,
            "mutation_radius_steps": manifest.algorithm.mutation_radius_steps,
        },
        "phase_selector": {
            "kind": manifest.phase_selector.kind,
            "early": manifest.phase_selector.early,
            "middle": manifest.phase_selector.middle,
            "late": manifest.phase_selector.late,
        },
        "boundary_evidence": {
            "report_path": manifest.boundary_evidence.report_path,
            "report_digest": manifest.boundary_evidence.report_digest,
            "slices_path": manifest.boundary_evidence.slices_path,
            "slices_digest": manifest.boundary_evidence.slices_digest,
        },
        "initial_experts": _phase_coefficient_values_payload(manifest.initial_experts),
        "expert_coefficient_grids": _phase_coefficient_grids_payload(
            manifest.expert_coefficient_grids
        ),
    }


def recompute_phase_search_manifest_digest(manifest: PhaseSearchManifest) -> str:
    """Hash the current normalized schema-v2 content, ignoring its stored digest."""

    payload_bytes = _canonical_json_bytes(phase_search_manifest_payload(manifest))
    return hashlib.sha256(payload_bytes).hexdigest()


def decimal_grid_index(
    value: Decimal,
    *,
    minimum: Decimal,
    step: Decimal,
) -> int:
    """Return the exact zero-based grid index without using Decimal context arithmetic."""

    _require_finite_grid_decimals(value=value, minimum=minimum, step=step)
    if step <= 0:
        raise ValueError("decimal grid step must be positive")
    if value < minimum:
        raise ValueError("decimal grid value must be at least its minimum")

    value_coefficient, value_exponent = _decimal_integer_and_exponent(value)
    minimum_coefficient, minimum_exponent = _decimal_integer_and_exponent(minimum)
    step_coefficient, step_exponent = _decimal_integer_and_exponent(step)
    common_exponent = min(value_exponent, minimum_exponent, step_exponent)
    scaled_value: int = value_coefficient * 10 ** (value_exponent - common_exponent)
    scaled_minimum: int = minimum_coefficient * 10 ** (minimum_exponent - common_exponent)
    scaled_step: int = step_coefficient * 10 ** (step_exponent - common_exponent)
    index, remainder = divmod(scaled_value - scaled_minimum, scaled_step)
    if remainder:
        raise ValueError("decimal grid value must align exactly to its step")
    return index


def decimal_from_grid_index(
    index: int,
    *,
    minimum: Decimal,
    step: Decimal,
) -> Decimal:
    """Return one exact grid value without using Decimal context arithmetic."""

    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError("decimal grid index must be an integer")
    _require_finite_grid_decimals(value=minimum, minimum=minimum, step=step)
    if step <= 0:
        raise ValueError("decimal grid step must be positive")

    minimum_coefficient, minimum_exponent = _decimal_integer_and_exponent(minimum)
    step_coefficient, step_exponent = _decimal_integer_and_exponent(step)
    common_exponent = min(minimum_exponent, step_exponent)
    scaled_minimum: int = minimum_coefficient * 10 ** (minimum_exponent - common_exponent)
    scaled_step: int = step_coefficient * 10 ** (step_exponent - common_exponent)
    scaled_value = scaled_minimum + scaled_step * index
    value = _decimal_from_integer_and_exponent(scaled_value, common_exponent)
    return Decimal(_decimal_string(value))


def _load_json_object(path: Path) -> dict[str, object]:
    def reject_nonfinite_number(value: str) -> NoReturn:
        raise _NonFiniteJsonNumber(value)

    def reject_duplicate_object_keys(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        decoded_object: dict[str, object] = {}
        for key, value in pairs:
            if key in decoded_object:
                raise SearchManifestError(
                    "duplicate_json_key",
                    f"{path} contains duplicate JSON key {key!r}; "
                    "the key appears more than once in the same object.",
                )
            decoded_object[key] = value
        return decoded_object

    try:
        decoded = cast(
            object,
            json.loads(
                path.read_text(encoding="utf-8"),
                parse_constant=reject_nonfinite_number,
                object_pairs_hook=reject_duplicate_object_keys,
            ),
        )
    except (json.JSONDecodeError, _NonFiniteJsonNumber) as error:
        raise SearchManifestError(
            "malformed_json",
            f"{path} must contain valid JSON with only finite numbers.",
        ) from error

    if not isinstance(decoded, dict):
        raise SearchManifestError(
            "invalid_manifest",
            f"{path} must contain one JSON object describing a search.",
        )
    return cast(dict[str, object], decoded)


def _reject_held_out_keys(value: object) -> None:
    if isinstance(value, dict):
        for key, nested_value in cast(dict[object, object], value).items():
            if isinstance(key, str):
                normalized_key = key.lower().replace("-", "_")
                if "held_out" in normalized_key or "heldout" in normalized_key:
                    raise SearchManifestError(
                        "held_out_key_forbidden",
                        f"Search manifests cannot contain held-out key {key!r}.",
                    )
            _reject_held_out_keys(nested_value)
    elif isinstance(value, list):
        for nested_value in cast(list[object], value):
            _reject_held_out_keys(nested_value)


def _require_exact_keys(
    payload: Mapping[str, object],
    expected: set[str],
    *,
    code: str,
    subject: str,
) -> None:
    missing = expected - set(payload)
    unknown = set(payload) - expected
    if missing or unknown:
        raise SearchManifestError(
            code,
            f"{subject} has missing keys {sorted(missing)} and unknown keys {sorted(unknown)}.",
        )


def _require_object(value: object, *, code: str, subject: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise SearchManifestError(code, f"{subject} must be one JSON object.")
    return cast(dict[str, object], value)


def _require_integer(
    value: object,
    *,
    code: str,
    field_name: str,
    minimum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SearchManifestError(
            code,
            f"{field_name} must be an integer greater than or equal to {minimum}.",
        )
    return value


def _require_string(
    value: object,
    *,
    code: str,
    field_name: str,
) -> str:
    if not isinstance(value, str):
        raise SearchManifestError(code, f"{field_name} must be a string.")
    return value


def _decode_digest(value: object, *, code: str, field_name: str) -> str:
    if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
        raise SearchManifestError(code, f"{field_name} must be a lowercase SHA-256 digest.")
    return value


def _decode_personality(value: object) -> Personality:
    if value == "aggressive":
        return "aggressive"
    if value == "balanced":
        return "balanced"
    if value == "passive":
        return "passive"
    raise SearchManifestError(
        "invalid_personality",
        "Personality must be 'aggressive', 'balanced', or 'passive'.",
    )


def _decode_search_name(value: object, *, personality: Personality) -> str:
    if not isinstance(value, str) or _SEARCH_NAME_PATTERN.fullmatch(value) is None:
        raise SearchManifestError(
            "invalid_search_name",
            "Search name must be a lowercase, hyphen-separated versioned name.",
        )
    expected_prefix = f"{personality}-v3-search-v"
    suffix = value.removeprefix(expected_prefix)
    if not value.startswith(expected_prefix) or not suffix.isdigit() or int(suffix) < 1:
        raise SearchManifestError(
            "invalid_search_name",
            f"Search name must identify the {personality} v3 search in versioned form.",
        )
    return value


def _decode_phase_search_name(value: object, *, personality: Personality) -> str:
    if not isinstance(value, str) or _SEARCH_NAME_PATTERN.fullmatch(value) is None:
        raise SearchManifestError(
            "invalid_search_name",
            "Search name must be a lowercase, hyphen-separated versioned name.",
        )
    expected = f"{personality}-v4-search-v2"
    if value != expected:
        raise SearchManifestError(
            "invalid_search_name",
            f"Schema-v2 search name must be exactly {expected!r}.",
        )
    return value


def _decode_predecessor(value: object, *, personality: Personality) -> str:
    expected = f"{personality}-v2"
    if value != expected:
        raise SearchManifestError(
            "wrong_predecessor",
            f"The {personality} search predecessor must be exactly {expected!r}.",
        )
    return value


def _decode_phase_predecessor(value: object, *, personality: Personality) -> str:
    expected = f"{personality}-v3"
    if value != expected:
        raise SearchManifestError(
            "wrong_predecessor",
            f"The {personality} v4 search predecessor must be exactly {expected!r}.",
        )
    return value


def _decode_corpus_binding(
    value: object,
    *,
    development_corpus: PromotionCorpus,
) -> DevelopmentCorpusBinding:
    payload = _require_object(
        value,
        code="invalid_development_corpus",
        subject="development_corpus",
    )
    _require_exact_keys(
        payload,
        _EXPECTED_CORPUS_KEYS,
        code="invalid_development_corpus_keys",
        subject="development_corpus",
    )
    name = payload["name"]
    if not isinstance(name, str) or _SEARCH_NAME_PATTERN.fullmatch(name) is None:
        raise SearchManifestError(
            "invalid_development_corpus_name",
            "Development corpus name must be lowercase and hyphen-separated.",
        )
    digest = payload["digest"]
    if not isinstance(digest, str) or _DIGEST_PATTERN.fullmatch(digest) is None:
        raise SearchManifestError(
            "invalid_development_corpus_digest",
            "Development corpus digest must be a lowercase SHA-256 digest.",
        )
    if name != development_corpus.recipe.name or digest != development_corpus.digest:
        raise SearchManifestError(
            "wrong_development_corpus",
            "The manifest development corpus name and digest must match the loaded corpus.",
        )
    return DevelopmentCorpusBinding(name=name, digest=digest)


def _decode_algorithm(value: object) -> SearchAlgorithm:
    payload = _require_object(value, code="invalid_algorithm", subject="algorithm")
    _require_exact_keys(
        payload,
        _EXPECTED_ALGORITHM_KEYS,
        code="invalid_algorithm_keys",
        subject="algorithm",
    )
    name = payload["name"]
    if name != _SUPPORTED_ALGORITHM:
        raise SearchManifestError(
            "unsupported_algorithm",
            f"Algorithm must be exactly {_SUPPORTED_ALGORITHM!r}.",
        )
    generation_count = _require_integer(
        payload["generation_count"],
        code="invalid_algorithm_setting",
        field_name="generation_count",
        minimum=1,
    )
    population_size = _require_integer(
        payload["population_size"],
        code="invalid_algorithm_setting",
        field_name="population_size",
        minimum=1,
    )
    elite_count = _require_integer(
        payload["elite_count"],
        code="invalid_algorithm_setting",
        field_name="elite_count",
        minimum=1,
    )
    mutation_radius_steps = _require_integer(
        payload["mutation_radius_steps"],
        code="invalid_algorithm_setting",
        field_name="mutation_radius_steps",
        minimum=1,
    )
    if elite_count > population_size:
        raise SearchManifestError(
            "invalid_algorithm_setting",
            "elite_count cannot exceed population_size.",
        )
    return SearchAlgorithm(
        name=name,
        generation_count=generation_count,
        population_size=population_size,
        elite_count=elite_count,
        mutation_radius_steps=mutation_radius_steps,
    )


def _decode_phase_selector(value: object) -> PhaseSelector:
    payload = _require_object(
        value,
        code="invalid_phase_selector",
        subject="phase_selector",
    )
    _require_exact_keys(
        payload,
        _EXPECTED_PHASE_SELECTOR_KEYS,
        code="invalid_phase_selector_keys",
        subject="phase_selector",
    )
    expected = PhaseSelector(
        kind=PHASE_SELECTOR_NAME,
        early=_PHASE_SELECTOR_EARLY_RULE,
        middle=_PHASE_SELECTOR_MIDDLE_RULE,
        late=_PHASE_SELECTOR_LATE_RULE,
    )
    decoded = PhaseSelector(
        kind=_require_string(
            payload["kind"],
            code="invalid_phase_selector",
            field_name="phase_selector.kind",
        ),
        early=_require_string(
            payload["early"],
            code="invalid_phase_selector",
            field_name="phase_selector.early",
        ),
        middle=_require_string(
            payload["middle"],
            code="invalid_phase_selector",
            field_name="phase_selector.middle",
        ),
        late=_require_string(
            payload["late"],
            code="invalid_phase_selector",
            field_name="phase_selector.late",
        ),
    )
    if decoded != expected:
        raise SearchManifestError(
            "invalid_phase_selector",
            "Schema-v2 searches must use the fixed public-resource horizon selector.",
        )
    return decoded


def _decode_boundary_evidence(value: object) -> BoundaryEvidence:
    payload = _require_object(
        value,
        code="invalid_boundary_evidence",
        subject="boundary_evidence",
    )
    _require_exact_keys(
        payload,
        _EXPECTED_BOUNDARY_EVIDENCE_KEYS,
        code="invalid_boundary_evidence_keys",
        subject="boundary_evidence",
    )
    decoded = BoundaryEvidence(
        report_path=_require_string(
            payload["report_path"],
            code="invalid_boundary_evidence",
            field_name="boundary_evidence.report_path",
        ),
        report_digest=_decode_digest(
            payload["report_digest"],
            code="invalid_boundary_evidence",
            field_name="boundary_evidence.report_digest",
        ),
        slices_path=_require_string(
            payload["slices_path"],
            code="invalid_boundary_evidence",
            field_name="boundary_evidence.slices_path",
        ),
        slices_digest=_decode_digest(
            payload["slices_digest"],
            code="invalid_boundary_evidence",
            field_name="boundary_evidence.slices_digest",
        ),
    )
    expected = BoundaryEvidence(
        report_path=_BOUNDARY_REPORT_PATH,
        report_digest=_BOUNDARY_REPORT_DIGEST,
        slices_path=_BOUNDARY_SLICES_PATH,
        slices_digest=_BOUNDARY_SLICES_DIGEST,
    )
    if decoded != expected:
        raise SearchManifestError(
            "wrong_boundary_evidence",
            "Schema-v2 searches must bind the committed pre-search boundary evidence.",
        )
    return decoded


def _decode_coefficient_values(
    value: object,
    *,
    subject: str = "initial_coefficients",
) -> CoefficientValues:
    payload = _require_object(
        value,
        code="invalid_coefficients",
        subject=subject,
    )
    _require_coefficient_names(payload)
    return CoefficientValues(
        liquidity_strength=_decode_decimal(
            payload["liquidity_strength"],
            field_name=f"{subject}.liquidity_strength",
        ),
        future_cash_weight=_decode_decimal(
            payload["future_cash_weight"],
            field_name=f"{subject}.future_cash_weight",
        ),
        objective_progress_weight=_decode_decimal(
            payload["objective_progress_weight"],
            field_name=f"{subject}.objective_progress_weight",
        ),
        bid_shading=_decode_decimal(
            payload["bid_shading"],
            field_name=f"{subject}.bid_shading",
        ),
    )


def _decode_coefficient_grids(
    value: object,
    *,
    subject: str = "coefficient_grids",
) -> CoefficientGrids:
    payload = _require_object(
        value,
        code="invalid_coefficient_grids",
        subject=subject,
    )
    _require_coefficient_names(payload)
    return CoefficientGrids(
        liquidity_strength=_decode_grid(
            payload["liquidity_strength"],
            coefficient_name="liquidity_strength",
            subject=subject,
        ),
        future_cash_weight=_decode_grid(
            payload["future_cash_weight"],
            coefficient_name="future_cash_weight",
            subject=subject,
        ),
        objective_progress_weight=_decode_grid(
            payload["objective_progress_weight"],
            coefficient_name="objective_progress_weight",
            subject=subject,
        ),
        bid_shading=_decode_grid(
            payload["bid_shading"],
            coefficient_name="bid_shading",
            subject=subject,
        ),
    )


def _decode_phase_coefficient_values(value: object) -> PhaseCoefficientValues:
    payload = _decode_phase_object(value, subject="initial_experts")
    return PhaseCoefficientValues(
        early=_decode_coefficient_values(
            payload["early"],
            subject="initial_experts.early",
        ),
        middle=_decode_coefficient_values(
            payload["middle"],
            subject="initial_experts.middle",
        ),
        late=_decode_coefficient_values(
            payload["late"],
            subject="initial_experts.late",
        ),
    )


def _decode_phase_coefficient_grids(value: object) -> PhaseCoefficientGrids:
    payload = _decode_phase_object(value, subject="expert_coefficient_grids")
    decoded = PhaseCoefficientGrids(
        early=_decode_coefficient_grids(
            payload["early"],
            subject="expert_coefficient_grids.early",
        ),
        middle=_decode_coefficient_grids(
            payload["middle"],
            subject="expert_coefficient_grids.middle",
        ),
        late=_decode_coefficient_grids(
            payload["late"],
            subject="expert_coefficient_grids.late",
        ),
    )
    if not decoded.early == decoded.middle == decoded.late:
        raise SearchManifestError(
            "unequal_expert_grids",
            "Every phase expert must use identical coefficient grids.",
        )
    if decoded.early != _ESTABLISHED_COEFFICIENT_GRIDS:
        raise SearchManifestError(
            "wrong_phase_coefficient_grids",
            "Schema-v2 searches must use the established four coefficient grids.",
        )
    return decoded


def _decode_phase_object(value: object, *, subject: str) -> dict[str, object]:
    payload = _require_object(
        value,
        code="invalid_phase_names",
        subject=subject,
    )
    _require_exact_keys(
        payload,
        _EXPECTED_PHASE_KEYS,
        code="invalid_phase_names",
        subject=subject,
    )
    return payload


def _require_coefficient_names(payload: Mapping[str, object]) -> None:
    expected = set(COEFFICIENT_NAMES)
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        unknown = sorted(set(payload) - expected)
        raise SearchManifestError(
            "invalid_coefficient_names",
            "Coefficient objects must contain exactly the four existing coefficient names; "
            f"missing {missing}, unknown {unknown}.",
        )


def _decode_grid(
    value: object,
    *,
    coefficient_name: CoefficientName,
    subject: str = "coefficient_grids",
) -> CoefficientGrid:
    field_name = f"{subject}.{coefficient_name}"
    payload = _require_object(
        value,
        code="invalid_coefficient_grid",
        subject=field_name,
    )
    _require_exact_keys(
        payload,
        _EXPECTED_GRID_KEYS,
        code="invalid_coefficient_grid_keys",
        subject=field_name,
    )
    minimum = _decode_decimal(
        payload["minimum"],
        field_name=f"{field_name}.minimum",
    )
    maximum = _decode_decimal(
        payload["maximum"],
        field_name=f"{field_name}.maximum",
    )
    step = _decode_decimal(
        payload["step"],
        field_name=f"{field_name}.step",
    )
    if (
        minimum > maximum
        or step <= 0
        or not _decimal_is_aligned_to_grid(maximum, minimum=minimum, step=step)
    ):
        raise SearchManifestError(
            "invalid_coefficient_grid",
            f"The {coefficient_name} grid needs minimum <= maximum, a positive step, "
            "and bounds aligned to that step.",
        )
    return CoefficientGrid(minimum=minimum, maximum=maximum, step=step)


def _decode_decimal(value: object, *, field_name: str) -> Decimal:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise SearchManifestError(
            "invalid_decimal",
            f"{field_name} must be a finite decimal string.",
        )
    try:
        decoded = Decimal(value)
    except InvalidOperation as error:
        raise SearchManifestError(
            "invalid_decimal",
            f"{field_name} must be a finite decimal string.",
        ) from error
    if not decoded.is_finite():
        raise SearchManifestError(
            "invalid_decimal",
            f"{field_name} must be a finite decimal string.",
        )
    return Decimal(_decimal_string(decoded))


def _validate_initial_coefficients(
    values: CoefficientValues,
    *,
    grids: CoefficientGrids,
    personality: Personality,
) -> None:
    for coefficient_name, coefficient_value, grid in zip(
        COEFFICIENT_NAMES,
        values.as_tuple(),
        grids.as_tuple(),
        strict=True,
    ):
        if (
            coefficient_value < grid.minimum
            or coefficient_value > grid.maximum
            or not _decimal_is_aligned_to_grid(
                coefficient_value,
                minimum=grid.minimum,
                step=grid.step,
            )
        ):
            raise SearchManifestError(
                "initial_coefficient_off_grid",
                f"Initial {coefficient_name} must lie exactly on its declared grid.",
            )

    expected = _profile_coefficient_values(getattr(HEURISTIC_V2, personality))
    if values != expected:
        raise SearchManifestError(
            "wrong_initial_coefficients",
            f"Initial coefficients for {personality} must exactly match heuristic v2.",
        )


def _validate_phase_initial_coefficients(
    values: PhaseCoefficientValues,
    *,
    grids: PhaseCoefficientGrids,
    personality: Personality,
) -> None:
    for phase, phase_values, phase_grids in zip(
        _PHASES,
        values.as_tuple(),
        grids.as_tuple(),
        strict=True,
    ):
        for coefficient_name, coefficient_value, grid in zip(
            COEFFICIENT_NAMES,
            phase_values.as_tuple(),
            phase_grids.as_tuple(),
            strict=True,
        ):
            if (
                coefficient_value < grid.minimum
                or coefficient_value > grid.maximum
                or not _decimal_is_aligned_to_grid(
                    coefficient_value,
                    minimum=grid.minimum,
                    step=grid.step,
                )
            ):
                raise SearchManifestError(
                    "initial_coefficient_off_grid",
                    f"Initial {phase} {coefficient_name} must lie exactly on its declared grid.",
                )

    expected = _profile_coefficient_values(getattr(HEURISTIC_V3, personality))
    for phase, phase_values in zip(_PHASES, values.as_tuple(), strict=True):
        if phase_values != expected:
            raise SearchManifestError(
                "wrong_initial_coefficients",
                f"Initial {phase} coefficients for {personality} must exactly match heuristic v3.",
            )


def _profile_coefficient_values(profile: HeuristicProfile) -> CoefficientValues:
    return CoefficientValues(
        liquidity_strength=Decimal(str(profile.liquidity_strength)),
        future_cash_weight=Decimal(str(profile.future_cash_weight)),
        objective_progress_weight=Decimal(str(profile.objective_progress_weight)),
        bid_shading=Decimal(str(profile.bid_shading)),
    )


def _coefficient_values_payload(values: CoefficientValues) -> dict[str, str]:
    return {
        coefficient_name: _decimal_string(value)
        for coefficient_name, value in zip(
            COEFFICIENT_NAMES,
            values.as_tuple(),
            strict=True,
        )
    }


def _coefficient_grids_payload(grids: CoefficientGrids) -> dict[str, object]:
    return {
        coefficient_name: {
            "minimum": _decimal_string(grid.minimum),
            "maximum": _decimal_string(grid.maximum),
            "step": _decimal_string(grid.step),
        }
        for coefficient_name, grid in zip(
            COEFFICIENT_NAMES,
            grids.as_tuple(),
            strict=True,
        )
    }


def _phase_coefficient_values_payload(values: PhaseCoefficientValues) -> dict[str, object]:
    return {
        phase: _coefficient_values_payload(phase_values)
        for phase, phase_values in zip(_PHASES, values.as_tuple(), strict=True)
    }


def _phase_coefficient_grids_payload(grids: PhaseCoefficientGrids) -> dict[str, object]:
    return {
        phase: _coefficient_grids_payload(phase_grids)
        for phase, phase_grids in zip(_PHASES, grids.as_tuple(), strict=True)
    }


def _decimal_string(value: Decimal) -> str:
    if value.is_zero():
        return "0"
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _decimal_is_aligned_to_grid(
    value: Decimal,
    *,
    minimum: Decimal,
    step: Decimal,
) -> bool:
    try:
        decimal_grid_index(value, minimum=minimum, step=step)
    except ValueError:
        return False
    return True


def _require_finite_grid_decimals(
    *,
    value: Decimal,
    minimum: Decimal,
    step: Decimal,
) -> None:
    if not all(
        isinstance(decimal_value, Decimal) and decimal_value.is_finite()
        for decimal_value in (value, minimum, step)
    ):
        raise ValueError("decimal grid values must be finite Decimal instances")


def _decimal_integer_and_exponent(value: Decimal) -> tuple[int, int]:
    representation = value.as_tuple()
    coefficient = 0
    for digit in representation.digits:
        coefficient = coefficient * 10 + digit
    if representation.sign:
        coefficient = -coefficient
    return coefficient, cast(int, representation.exponent)


def _decimal_from_integer_and_exponent(coefficient: int, exponent: int) -> Decimal:
    if coefficient == 0:
        return Decimal(0)
    sign = int(coefficient < 0)
    digits = tuple(int(digit) for digit in str(abs(coefficient)))
    return Decimal((sign, digits, exponent))


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")
