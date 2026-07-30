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

from garboid_pocketrocks.heuristics.profiles import HEURISTIC_V2, HeuristicProfile
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


def _decode_predecessor(value: object, *, personality: Personality) -> str:
    expected = f"{personality}-v2"
    if value != expected:
        raise SearchManifestError(
            "wrong_predecessor",
            f"The {personality} search predecessor must be exactly {expected!r}.",
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


def _decode_coefficient_values(value: object) -> CoefficientValues:
    payload = _require_object(
        value,
        code="invalid_coefficients",
        subject="initial_coefficients",
    )
    _require_coefficient_names(payload)
    return CoefficientValues(
        liquidity_strength=_decode_decimal(
            payload["liquidity_strength"],
            field_name="initial_coefficients.liquidity_strength",
        ),
        future_cash_weight=_decode_decimal(
            payload["future_cash_weight"],
            field_name="initial_coefficients.future_cash_weight",
        ),
        objective_progress_weight=_decode_decimal(
            payload["objective_progress_weight"],
            field_name="initial_coefficients.objective_progress_weight",
        ),
        bid_shading=_decode_decimal(
            payload["bid_shading"],
            field_name="initial_coefficients.bid_shading",
        ),
    )


def _decode_coefficient_grids(value: object) -> CoefficientGrids:
    payload = _require_object(
        value,
        code="invalid_coefficient_grids",
        subject="coefficient_grids",
    )
    _require_coefficient_names(payload)
    return CoefficientGrids(
        liquidity_strength=_decode_grid(
            payload["liquidity_strength"],
            coefficient_name="liquidity_strength",
        ),
        future_cash_weight=_decode_grid(
            payload["future_cash_weight"],
            coefficient_name="future_cash_weight",
        ),
        objective_progress_weight=_decode_grid(
            payload["objective_progress_weight"],
            coefficient_name="objective_progress_weight",
        ),
        bid_shading=_decode_grid(
            payload["bid_shading"],
            coefficient_name="bid_shading",
        ),
    )


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


def _decode_grid(value: object, *, coefficient_name: CoefficientName) -> CoefficientGrid:
    payload = _require_object(
        value,
        code="invalid_coefficient_grid",
        subject=f"coefficient_grids.{coefficient_name}",
    )
    _require_exact_keys(
        payload,
        _EXPECTED_GRID_KEYS,
        code="invalid_coefficient_grid_keys",
        subject=f"coefficient_grids.{coefficient_name}",
    )
    minimum = _decode_decimal(
        payload["minimum"],
        field_name=f"coefficient_grids.{coefficient_name}.minimum",
    )
    maximum = _decode_decimal(
        payload["maximum"],
        field_name=f"coefficient_grids.{coefficient_name}.maximum",
    )
    step = _decode_decimal(
        payload["step"],
        field_name=f"coefficient_grids.{coefficient_name}.step",
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
