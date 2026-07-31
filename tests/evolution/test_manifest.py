from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from garboid_pocketrocks.bots import BOT_SPECS_BY_NAME
from garboid_pocketrocks.evolution import manifest as manifest_module
from garboid_pocketrocks.evolution.manifest import (
    COEFFICIENT_NAMES,
    SearchManifest,
    SearchManifestError,
    decimal_from_grid_index,
    decimal_grid_index,
    load_search_manifest,
    recompute_search_manifest_digest,
    search_manifest_payload,
)
from garboid_pocketrocks.promotion.corpus import (
    PromotionCorpus,
    load_promotion_corpus,
)

REPOSITORY_ROOT = Path(__file__).parents[2]
DEVELOPMENT_CORPUS_PATH = REPOSITORY_ROOT / "configs/promotion/development-v1.json"
EVOLUTION_CONFIG_DIRECTORY = REPOSITORY_ROOT / "configs/evolution"
BOUNDARY_REPORT_PATH = "docs/benchmarks/2026-07-30-heuristic-v4-phase-boundaries.md"
BOUNDARY_REPORT_DIGEST = "9961f26f32270dcebc98df443588e96cbde2f953858cd131c66a37aeecaa9b01"
BOUNDARY_SLICES_PATH = (
    "docs/benchmarks/tournaments/"
    "2026-07-30-heuristic-v3-phase-boundaries-development/phase-boundary-slices.csv"
)
BOUNDARY_SLICES_DIGEST = "4f8aa60edf31b28c746cb8004a4dd5468ee8ab1b26462550c914b2e3fa50d7ae"
PHASES = ("early", "middle", "late")


@pytest.fixture(scope="module")
def development_corpus() -> PromotionCorpus:
    return load_promotion_corpus(
        DEVELOPMENT_CORPUS_PATH,
        registry=BOT_SPECS_BY_NAME,
    )


def _manifest_payload(
    *,
    personality: str = "balanced",
    predecessor_name: str = "balanced-v2",
    corpus_name: str = "development-v1",
    corpus_digest: str = "1" * 64,
    search_seed: object = 11002,
    algorithm: object | None = None,
    initial_coefficients: object | None = None,
    coefficient_grids: object | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "name": f"{personality}-v3-search-v1",
        "personality": personality,
        "predecessor_name": predecessor_name,
        "development_corpus": {
            "name": corpus_name,
            "digest": corpus_digest,
        },
        "search_seed": search_seed,
        "algorithm": (
            {
                "name": "mu-plus-lambda-v1",
                "generation_count": 8,
                "population_size": 12,
                "elite_count": 4,
                "mutation_radius_steps": 4,
            }
            if algorithm is None
            else algorithm
        ),
        "initial_coefficients": (
            {
                "liquidity_strength": "0.40",
                "future_cash_weight": "0.75",
                "objective_progress_weight": "0.20",
                "bid_shading": "0.25",
            }
            if initial_coefficients is None
            else initial_coefficients
        ),
        "coefficient_grids": (
            {
                "liquidity_strength": {
                    "minimum": "0.00",
                    "maximum": "1.50",
                    "step": "0.05",
                },
                "future_cash_weight": {
                    "minimum": "0.00",
                    "maximum": "2.00",
                    "step": "0.05",
                },
                "objective_progress_weight": {
                    "minimum": "0.00",
                    "maximum": "1.00",
                    "step": "0.05",
                },
                "bid_shading": {
                    "minimum": "0.00",
                    "maximum": "1.00",
                    "step": "0.05",
                },
            }
            if coefficient_grids is None
            else coefficient_grids
        ),
    }


def _phase_manifest_payload(
    *,
    personality: str = "balanced",
    predecessor_name: str = "balanced-v3",
    corpus_name: str = "development-v1",
    corpus_digest: str = "1" * 64,
    search_seed: object = 12002,
) -> dict[str, object]:
    initial_by_personality = {
        "aggressive": {
            "liquidity_strength": "1.00",
            "future_cash_weight": "1.95",
            "objective_progress_weight": "0.15",
            "bid_shading": "0.40",
        },
        "balanced": {
            "liquidity_strength": "0.25",
            "future_cash_weight": "1.55",
            "objective_progress_weight": "0.30",
            "bid_shading": "0.35",
        },
        "passive": {
            "liquidity_strength": "1.50",
            "future_cash_weight": "1.80",
            "objective_progress_weight": "0.95",
            "bid_shading": "0.45",
        },
    }
    grid = {
        "liquidity_strength": {
            "minimum": "0.00",
            "maximum": "1.50",
            "step": "0.05",
        },
        "future_cash_weight": {
            "minimum": "0.00",
            "maximum": "2.00",
            "step": "0.05",
        },
        "objective_progress_weight": {
            "minimum": "0.00",
            "maximum": "1.00",
            "step": "0.05",
        },
        "bid_shading": {
            "minimum": "0.00",
            "maximum": "1.00",
            "step": "0.05",
        },
    }
    initial = initial_by_personality[personality]
    return {
        "schema_version": 2,
        "name": f"{personality}-v4-search-v2",
        "personality": personality,
        "predecessor_name": predecessor_name,
        "development_corpus": {
            "name": corpus_name,
            "digest": corpus_digest,
        },
        "search_seed": search_seed,
        "algorithm": {
            "name": "mu-plus-lambda-v1",
            "generation_count": 12,
            "population_size": 16,
            "elite_count": 4,
            "mutation_radius_steps": 4,
        },
        "phase_selector": {
            "kind": "public-resource-horizon-v1",
            "early": "3*future>=2*total",
            "middle": "3*future>=total",
            "late": "otherwise",
        },
        "boundary_evidence": {
            "report_path": BOUNDARY_REPORT_PATH,
            "report_digest": BOUNDARY_REPORT_DIGEST,
            "slices_path": BOUNDARY_SLICES_PATH,
            "slices_digest": BOUNDARY_SLICES_DIGEST,
        },
        "initial_experts": {phase: dict(initial) for phase in PHASES},
        "expert_coefficient_grids": {phase: json.loads(json.dumps(grid)) for phase in PHASES},
    }


def _write_manifest(path: Path, payload: object, *, compact: bool = False) -> None:
    separators = (",", ":") if compact else None
    path.write_text(
        json.dumps(payload, indent=None if compact else 2, separators=separators) + "\n",
        encoding="utf-8",
    )


def _load_fixture(
    tmp_path: Path,
    development_corpus: PromotionCorpus,
    *,
    payload: dict[str, object] | None = None,
) -> SearchManifest:
    manifest_payload = _manifest_payload(
        corpus_name=development_corpus.recipe.name,
        corpus_digest=development_corpus.digest,
    )
    path = tmp_path / "balanced-v3-search-v1.json"
    _write_manifest(path, manifest_payload if payload is None else payload)
    return load_search_manifest(path, development_corpus=development_corpus)


def test_loads_exact_decimal_recipe_and_corpus_binding(
    tmp_path: Path,
    development_corpus: PromotionCorpus,
) -> None:
    manifest = _load_fixture(tmp_path, development_corpus)

    assert manifest.schema_version == 1
    assert manifest.name == "balanced-v3-search-v1"
    assert manifest.personality == "balanced"
    assert manifest.predecessor_name == "balanced-v2"
    assert manifest.development_corpus.name == development_corpus.recipe.name
    assert manifest.development_corpus.digest == development_corpus.digest
    assert manifest.search_seed == 11002
    assert manifest.algorithm.name == "mu-plus-lambda-v1"
    assert manifest.algorithm.generation_count == 8
    assert manifest.algorithm.population_size == 12
    assert manifest.algorithm.elite_count == 4
    assert manifest.algorithm.mutation_radius_steps == 4
    assert manifest.initial_coefficients.future_cash_weight == Decimal("0.75")
    assert manifest.coefficient_grids.liquidity_strength.maximum == Decimal("1.50")
    assert manifest.coefficient_grids.future_cash_weight.maximum == Decimal("2.00")
    assert manifest.coefficient_grids.objective_progress_weight.step == Decimal("0.05")
    assert manifest.coefficient_grids.bid_shading.minimum == Decimal("0.00")
    assert COEFFICIENT_NAMES == (
        "liquidity_strength",
        "future_cash_weight",
        "objective_progress_weight",
        "bid_shading",
    )


def test_manifest_and_nested_values_are_immutable(
    tmp_path: Path,
    development_corpus: PromotionCorpus,
) -> None:
    manifest = _load_fixture(tmp_path, development_corpus)

    with pytest.raises(FrozenInstanceError):
        manifest.name = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        manifest.algorithm.elite_count = 2  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        manifest.coefficient_grids.bid_shading.step = Decimal("0.10")  # type: ignore[misc]


def test_digest_uses_normalized_sorted_json(
    tmp_path: Path,
    development_corpus: PromotionCorpus,
) -> None:
    payload = _manifest_payload(
        corpus_name=development_corpus.recipe.name,
        corpus_digest=development_corpus.digest,
    )
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    _write_manifest(first_path, payload)
    reordered = dict(reversed(tuple(payload.items())))
    initial = reordered["initial_coefficients"]
    assert isinstance(initial, dict)
    reordered["initial_coefficients"] = dict(reversed(tuple(initial.items())))
    grids = reordered["coefficient_grids"]
    assert isinstance(grids, dict)
    reordered["coefficient_grids"] = dict(reversed(tuple(grids.items())))
    _write_manifest(second_path, reordered, compact=True)

    first = load_search_manifest(first_path, development_corpus=development_corpus)
    second = load_search_manifest(second_path, development_corpus=development_corpus)

    assert first.digest == second.digest
    assert len(first.digest) == 64
    int(first.digest, 16)
    encoded = (
        json.dumps(
            search_manifest_payload(first),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()
    assert first.digest == hashlib.sha256(encoded).hexdigest()


def test_recomputed_digest_detects_stale_manifest_content(
    tmp_path: Path,
    development_corpus: PromotionCorpus,
) -> None:
    manifest = _load_fixture(tmp_path, development_corpus)
    changed_manifests = (
        replace(manifest, search_seed=manifest.search_seed + 1),
        replace(
            manifest,
            algorithm=replace(
                manifest.algorithm,
                generation_count=manifest.algorithm.generation_count + 1,
            ),
        ),
        replace(
            manifest,
            coefficient_grids=replace(
                manifest.coefficient_grids,
                future_cash_weight=replace(
                    manifest.coefficient_grids.future_cash_weight,
                    maximum=Decimal("2.05"),
                ),
            ),
        ),
        replace(
            manifest,
            initial_coefficients=replace(
                manifest.initial_coefficients,
                bid_shading=Decimal("0.30"),
            ),
        ),
    )

    assert recompute_search_manifest_digest(manifest) == manifest.digest
    for changed_manifest in changed_manifests:
        assert changed_manifest.digest == manifest.digest
        assert recompute_search_manifest_digest(changed_manifest) != changed_manifest.digest


def test_equivalent_decimal_spellings_have_one_normalized_digest(
    tmp_path: Path,
    development_corpus: PromotionCorpus,
) -> None:
    first = _manifest_payload(
        corpus_name=development_corpus.recipe.name,
        corpus_digest=development_corpus.digest,
    )
    second = json.loads(json.dumps(first))
    second["initial_coefficients"]["liquidity_strength"] = "0.400"
    second["coefficient_grids"]["liquidity_strength"]["maximum"] = "1.500"
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    _write_manifest(first_path, first)
    _write_manifest(second_path, second)

    loaded_first = load_search_manifest(first_path, development_corpus=development_corpus)
    loaded_second = load_search_manifest(second_path, development_corpus=development_corpus)

    assert loaded_first.digest == loaded_second.digest
    assert search_manifest_payload(loaded_first) == search_manifest_payload(loaded_second)


def test_loading_is_independent_of_decimal_context_precision(
    tmp_path: Path,
    development_corpus: PromotionCorpus,
) -> None:
    payload = _manifest_payload(
        corpus_name=development_corpus.recipe.name,
        corpus_digest=development_corpus.digest,
    )
    grids = payload["coefficient_grids"]
    assert isinstance(grids, dict)
    future_cash_grid = grids["future_cash_weight"]
    assert isinstance(future_cash_grid, dict)
    future_cash_grid["maximum"] = "123456789.1500000000000000000"
    path = tmp_path / "manifest.json"
    _write_manifest(path, payload)

    loaded = []
    for precision in (8, 28, 50):
        with localcontext() as context:
            context.prec = precision
            loaded.append(load_search_manifest(path, development_corpus=development_corpus))

    assert (
        tuple(manifest.coefficient_grids.future_cash_weight.maximum for manifest in loaded)
        == (Decimal("123456789.15"),) * 3
    )
    assert len({manifest.digest for manifest in loaded}) == 1
    assert (
        len({json.dumps(search_manifest_payload(manifest), sort_keys=True) for manifest in loaded})
        == 1
    )


def test_exact_grid_helpers_are_independent_of_decimal_context_precision() -> None:
    minimum = Decimal("123456789.10")
    step = Decimal("0.05")
    value = Decimal("123456789.15")

    outcomes = []
    for precision in (8, 28, 50):
        with localcontext() as context:
            context.prec = precision
            index = decimal_grid_index(value, minimum=minimum, step=step)
            outcomes.append(
                (
                    index,
                    decimal_from_grid_index(index, minimum=minimum, step=step),
                )
            )

    assert outcomes == [(1, value)] * 3


def test_exact_grid_helpers_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="positive"):
        decimal_grid_index(Decimal("0.10"), minimum=Decimal(0), step=Decimal(0))
    with pytest.raises(ValueError, match="align"):
        decimal_grid_index(Decimal("0.11"), minimum=Decimal(0), step=Decimal("0.05"))
    with pytest.raises(ValueError, match="minimum"):
        decimal_grid_index(Decimal("-0.05"), minimum=Decimal(0), step=Decimal("0.05"))
    with pytest.raises(ValueError, match="integer"):
        decimal_from_grid_index(True, minimum=Decimal(0), step=Decimal("0.05"))


@pytest.mark.parametrize(
    ("field", "value", "expected_code"),
    (
        ("schema_version", 2, "unsupported_schema"),
        ("schema_version", True, "unsupported_schema"),
        ("name", "Balanced Search", "invalid_search_name"),
        ("personality", "reckless", "invalid_personality"),
        ("predecessor_name", "balanced-v1", "wrong_predecessor"),
        ("search_seed", -1, "invalid_search_seed"),
        ("search_seed", True, "invalid_search_seed"),
    ),
)
def test_invalid_top_level_values_have_stable_codes(
    tmp_path: Path,
    development_corpus: PromotionCorpus,
    field: str,
    value: object,
    expected_code: str,
) -> None:
    payload = _manifest_payload(
        corpus_name=development_corpus.recipe.name,
        corpus_digest=development_corpus.digest,
    )
    payload[field] = value
    path = tmp_path / "manifest.json"
    _write_manifest(path, payload)

    with pytest.raises(SearchManifestError) as captured:
        load_search_manifest(path, development_corpus=development_corpus)

    assert captured.value.code == expected_code
    assert str(captured.value)


@pytest.mark.parametrize(
    ("field", "value", "expected_code"),
    (
        ("name", "development-v2", "wrong_development_corpus"),
        ("digest", "2" * 64, "wrong_development_corpus"),
        ("digest", "not-a-digest", "invalid_development_corpus_digest"),
    ),
)
def test_manifest_must_match_the_loaded_development_corpus(
    tmp_path: Path,
    development_corpus: PromotionCorpus,
    field: str,
    value: object,
    expected_code: str,
) -> None:
    payload = _manifest_payload(
        corpus_name=development_corpus.recipe.name,
        corpus_digest=development_corpus.digest,
    )
    corpus_binding = payload["development_corpus"]
    assert isinstance(corpus_binding, dict)
    corpus_binding[field] = value
    path = tmp_path / "manifest.json"
    _write_manifest(path, payload)

    with pytest.raises(SearchManifestError) as captured:
        load_search_manifest(path, development_corpus=development_corpus)

    assert captured.value.code == expected_code


def test_a_held_out_corpus_cannot_be_loaded_as_development(
    tmp_path: Path,
    development_corpus: PromotionCorpus,
) -> None:
    manifest = _load_fixture(tmp_path, development_corpus)
    held_out_corpus = replace(
        development_corpus,
        recipe=replace(development_corpus.recipe, purpose="held_out"),
    )
    path = tmp_path / "manifest.json"
    _write_manifest(path, search_manifest_payload(manifest))

    with pytest.raises(SearchManifestError) as captured:
        load_search_manifest(path, development_corpus=held_out_corpus)

    assert captured.value.code == "held_out_corpus_forbidden"


@pytest.mark.parametrize(
    ("algorithm_field", "value", "expected_code"),
    (
        ("name", "random-search", "unsupported_algorithm"),
        ("generation_count", 0, "invalid_algorithm_setting"),
        ("generation_count", True, "invalid_algorithm_setting"),
        ("population_size", 0, "invalid_algorithm_setting"),
        ("elite_count", 0, "invalid_algorithm_setting"),
        ("elite_count", 13, "invalid_algorithm_setting"),
        ("mutation_radius_steps", 0, "invalid_algorithm_setting"),
    ),
)
def test_invalid_algorithm_settings_are_rejected(
    tmp_path: Path,
    development_corpus: PromotionCorpus,
    algorithm_field: str,
    value: object,
    expected_code: str,
) -> None:
    payload = _manifest_payload(
        corpus_name=development_corpus.recipe.name,
        corpus_digest=development_corpus.digest,
    )
    algorithm = payload["algorithm"]
    assert isinstance(algorithm, dict)
    algorithm[algorithm_field] = value
    path = tmp_path / "manifest.json"
    _write_manifest(path, payload)

    with pytest.raises(SearchManifestError) as captured:
        load_search_manifest(path, development_corpus=development_corpus)

    assert captured.value.code == expected_code


@pytest.mark.parametrize(
    ("decimal_value", "expected_code"),
    (
        (True, "invalid_decimal"),
        (0.4, "invalid_decimal"),
        ("", "invalid_decimal"),
        (" 0.40", "invalid_decimal"),
        ("NaN", "invalid_decimal"),
        ("Infinity", "invalid_decimal"),
        ("not-decimal", "invalid_decimal"),
    ),
)
def test_coefficients_require_finite_decimal_strings(
    tmp_path: Path,
    development_corpus: PromotionCorpus,
    decimal_value: object,
    expected_code: str,
) -> None:
    payload = _manifest_payload(
        corpus_name=development_corpus.recipe.name,
        corpus_digest=development_corpus.digest,
    )
    coefficients = payload["initial_coefficients"]
    assert isinstance(coefficients, dict)
    coefficients["liquidity_strength"] = decimal_value
    path = tmp_path / "manifest.json"
    _write_manifest(path, payload)

    with pytest.raises(SearchManifestError) as captured:
        load_search_manifest(path, development_corpus=development_corpus)

    assert captured.value.code == expected_code


@pytest.mark.parametrize(
    ("grid_update", "expected_code"),
    (
        ({"minimum": "1.00", "maximum": "0.00"}, "invalid_coefficient_grid"),
        ({"step": "0.00"}, "invalid_coefficient_grid"),
        ({"minimum": "0.01"}, "invalid_coefficient_grid"),
    ),
)
def test_invalid_grids_are_rejected(
    tmp_path: Path,
    development_corpus: PromotionCorpus,
    grid_update: dict[str, object],
    expected_code: str,
) -> None:
    payload = _manifest_payload(
        corpus_name=development_corpus.recipe.name,
        corpus_digest=development_corpus.digest,
    )
    grids = payload["coefficient_grids"]
    assert isinstance(grids, dict)
    liquidity_grid = grids["liquidity_strength"]
    assert isinstance(liquidity_grid, dict)
    liquidity_grid.update(grid_update)
    path = tmp_path / "manifest.json"
    _write_manifest(path, payload)

    with pytest.raises(SearchManifestError) as captured:
        load_search_manifest(path, development_corpus=development_corpus)

    assert captured.value.code == expected_code


@pytest.mark.parametrize(
    ("coefficient_name", "value"),
    (
        ("liquidity_strength", "0.41"),
        ("future_cash_weight", "2.05"),
    ),
)
def test_initial_coefficients_must_be_on_their_declared_grids(
    tmp_path: Path,
    development_corpus: PromotionCorpus,
    coefficient_name: str,
    value: str,
) -> None:
    payload = _manifest_payload(
        corpus_name=development_corpus.recipe.name,
        corpus_digest=development_corpus.digest,
    )
    coefficients = payload["initial_coefficients"]
    assert isinstance(coefficients, dict)
    coefficients[coefficient_name] = value
    path = tmp_path / "manifest.json"
    _write_manifest(path, payload)

    with pytest.raises(SearchManifestError) as captured:
        load_search_manifest(path, development_corpus=development_corpus)

    assert captured.value.code == "initial_coefficient_off_grid"


def test_initial_coefficients_must_exactly_match_v2(
    tmp_path: Path,
    development_corpus: PromotionCorpus,
) -> None:
    payload = _manifest_payload(
        corpus_name=development_corpus.recipe.name,
        corpus_digest=development_corpus.digest,
    )
    coefficients = payload["initial_coefficients"]
    assert isinstance(coefficients, dict)
    coefficients["liquidity_strength"] = "0.45"
    path = tmp_path / "manifest.json"
    _write_manifest(path, payload)

    with pytest.raises(SearchManifestError) as captured:
        load_search_manifest(path, development_corpus=development_corpus)

    assert captured.value.code == "wrong_initial_coefficients"


@pytest.mark.parametrize("section", ("initial_coefficients", "coefficient_grids"))
def test_unknown_coefficient_names_are_rejected(
    tmp_path: Path,
    development_corpus: PromotionCorpus,
    section: str,
) -> None:
    payload = _manifest_payload(
        corpus_name=development_corpus.recipe.name,
        corpus_digest=development_corpus.digest,
    )
    coefficients = payload[section]
    assert isinstance(coefficients, dict)
    coefficients["secret_aggression"] = coefficients["liquidity_strength"]
    path = tmp_path / "manifest.json"
    _write_manifest(path, payload)

    with pytest.raises(SearchManifestError) as captured:
        load_search_manifest(path, development_corpus=development_corpus)

    assert captured.value.code == "invalid_coefficient_names"


@pytest.mark.parametrize(
    "held_out_key",
    ("held_out_corpus", "held-out-seed", "heldout_digest"),
)
def test_any_held_out_key_is_forbidden_even_when_nested(
    tmp_path: Path,
    development_corpus: PromotionCorpus,
    held_out_key: str,
) -> None:
    payload = _manifest_payload(
        corpus_name=development_corpus.recipe.name,
        corpus_digest=development_corpus.digest,
    )
    algorithm = payload["algorithm"]
    assert isinstance(algorithm, dict)
    algorithm[held_out_key] = "never legal in search"
    path = tmp_path / "manifest.json"
    _write_manifest(path, payload)

    with pytest.raises(SearchManifestError) as captured:
        load_search_manifest(path, development_corpus=development_corpus)

    assert captured.value.code == "held_out_key_forbidden"


def test_missing_unknown_and_duplicate_keys_are_rejected(
    tmp_path: Path,
    development_corpus: PromotionCorpus,
) -> None:
    missing = _manifest_payload(
        corpus_name=development_corpus.recipe.name,
        corpus_digest=development_corpus.digest,
    )
    del missing["search_seed"]
    missing_path = tmp_path / "missing.json"
    _write_manifest(missing_path, missing)

    unknown = _manifest_payload(
        corpus_name=development_corpus.recipe.name,
        corpus_digest=development_corpus.digest,
    )
    unknown["notes"] = "not in schema version 1"
    unknown_path = tmp_path / "unknown.json"
    _write_manifest(unknown_path, unknown)

    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text(
        '{"schema_version":1,"schema_version":1}',
        encoding="utf-8",
    )

    expected = (
        (missing_path, "invalid_manifest_keys"),
        (unknown_path, "invalid_manifest_keys"),
        (duplicate_path, "duplicate_json_key"),
    )
    for path, expected_code in expected:
        with pytest.raises(SearchManifestError) as captured:
            load_search_manifest(path, development_corpus=development_corpus)
        assert captured.value.code == expected_code


@pytest.mark.parametrize("source", ("{", '{"search_seed": NaN}', "[]"))
def test_malformed_nonfinite_or_nonobject_json_is_rejected(
    tmp_path: Path,
    development_corpus: PromotionCorpus,
    source: str,
) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(source, encoding="utf-8")

    with pytest.raises(SearchManifestError) as captured:
        load_search_manifest(path, development_corpus=development_corpus)

    expected_code = "invalid_manifest" if source == "[]" else "malformed_json"
    assert captured.value.code == expected_code


def test_committed_manifests_bind_v2_and_the_fixed_design(
    development_corpus: PromotionCorpus,
) -> None:
    expected_initial = {
        "aggressive": ("0.75", "1.5", "0.25", "0.05"),
        "balanced": ("0.4", "0.75", "0.2", "0.25"),
        "passive": ("0.15", "0.6", "0.15", "0.3"),
    }

    for seed_offset, personality in enumerate(("aggressive", "balanced", "passive"), start=1):
        path = EVOLUTION_CONFIG_DIRECTORY / f"{personality}-v3-search-v1.json"
        manifest = load_search_manifest(path, development_corpus=development_corpus)

        assert manifest.name == f"{personality}-v3-search-v1"
        assert manifest.predecessor_name == f"{personality}-v2"
        assert manifest.search_seed == 11000 + seed_offset
        assert manifest.algorithm.generation_count == 8
        assert manifest.algorithm.population_size == 12
        assert manifest.algorithm.elite_count == 4
        assert manifest.algorithm.mutation_radius_steps == 4
        assert (
            tuple(map(str, manifest.initial_coefficients.as_tuple()))
            == expected_initial[personality]
        )
        assert manifest.coefficient_grids.as_tuples() == (
            (Decimal("0.00"), Decimal("1.50"), Decimal("0.05")),
            (Decimal("0.00"), Decimal("2.00"), Decimal("0.05")),
            (Decimal("0.00"), Decimal("1.00"), Decimal("0.05")),
            (Decimal("0.00"), Decimal("1.00"), Decimal("0.05")),
        )


def test_schema_v1_canonical_payload_bytes_and_digests_are_frozen(
    development_corpus: PromotionCorpus,
) -> None:
    expected = {
        "aggressive": (
            '{"algorithm":{"elite_count":4,"generation_count":8,"mutation_radius_steps":4,'
            '"name":"mu-plus-lambda-v1","population_size":12},"coefficient_grids":{'
            '"bid_shading":{"maximum":"1","minimum":"0","step":"0.05"},'
            '"future_cash_weight":{"maximum":"2","minimum":"0","step":"0.05"},'
            '"liquidity_strength":{"maximum":"1.5","minimum":"0","step":"0.05"},'
            '"objective_progress_weight":{"maximum":"1","minimum":"0","step":"0.05"}},'
            '"development_corpus":{"digest":'
            '"17c016350dbe717641b8cd499b0908e3bc0faa811a3b4f5e574f8713a5bf2b3d",'
            '"name":"development-v1"},"initial_coefficients":{"bid_shading":"0.05",'
            '"future_cash_weight":"1.5","liquidity_strength":"0.75",'
            '"objective_progress_weight":"0.25"},"name":"aggressive-v3-search-v1",'
            '"personality":"aggressive","predecessor_name":"aggressive-v2",'
            '"schema_version":1,"search_seed":11001}\n',
            "627eb77836f8dceace745a8fb7f60573e2dad05aa47a423d902850c32a98f5e0",
        ),
        "balanced": (
            '{"algorithm":{"elite_count":4,"generation_count":8,"mutation_radius_steps":4,'
            '"name":"mu-plus-lambda-v1","population_size":12},"coefficient_grids":{'
            '"bid_shading":{"maximum":"1","minimum":"0","step":"0.05"},'
            '"future_cash_weight":{"maximum":"2","minimum":"0","step":"0.05"},'
            '"liquidity_strength":{"maximum":"1.5","minimum":"0","step":"0.05"},'
            '"objective_progress_weight":{"maximum":"1","minimum":"0","step":"0.05"}},'
            '"development_corpus":{"digest":'
            '"17c016350dbe717641b8cd499b0908e3bc0faa811a3b4f5e574f8713a5bf2b3d",'
            '"name":"development-v1"},"initial_coefficients":{"bid_shading":"0.25",'
            '"future_cash_weight":"0.75","liquidity_strength":"0.4",'
            '"objective_progress_weight":"0.2"},"name":"balanced-v3-search-v1",'
            '"personality":"balanced","predecessor_name":"balanced-v2",'
            '"schema_version":1,"search_seed":11002}\n',
            "da9e2162eec9dd934dc80e59d9950b49c74a3a4cd4d72e6273134b502e705152",
        ),
        "passive": (
            '{"algorithm":{"elite_count":4,"generation_count":8,"mutation_radius_steps":4,'
            '"name":"mu-plus-lambda-v1","population_size":12},"coefficient_grids":{'
            '"bid_shading":{"maximum":"1","minimum":"0","step":"0.05"},'
            '"future_cash_weight":{"maximum":"2","minimum":"0","step":"0.05"},'
            '"liquidity_strength":{"maximum":"1.5","minimum":"0","step":"0.05"},'
            '"objective_progress_weight":{"maximum":"1","minimum":"0","step":"0.05"}},'
            '"development_corpus":{"digest":'
            '"17c016350dbe717641b8cd499b0908e3bc0faa811a3b4f5e574f8713a5bf2b3d",'
            '"name":"development-v1"},"initial_coefficients":{"bid_shading":"0.3",'
            '"future_cash_weight":"0.6","liquidity_strength":"0.15",'
            '"objective_progress_weight":"0.15"},"name":"passive-v3-search-v1",'
            '"personality":"passive","predecessor_name":"passive-v2",'
            '"schema_version":1,"search_seed":11003}\n',
            "bf533a434a4208e7b018606c53488fcc3a09499b6da2fcb4b1d020346001a9c1",
        ),
    }

    for personality, (expected_bytes, expected_digest) in expected.items():
        manifest = load_search_manifest(
            EVOLUTION_CONFIG_DIRECTORY / f"{personality}-v3-search-v1.json",
            development_corpus=development_corpus,
        )
        canonical_bytes = (
            json.dumps(
                search_manifest_payload(manifest),
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        )
        assert canonical_bytes == expected_bytes
        assert manifest.digest == expected_digest
        assert hashlib.sha256(canonical_bytes.encode()).hexdigest() == expected_digest


def test_loads_schema_v2_as_explicit_phase_recipe(
    tmp_path: Path,
    development_corpus: PromotionCorpus,
) -> None:
    payload = _phase_manifest_payload(
        corpus_name=development_corpus.recipe.name,
        corpus_digest=development_corpus.digest,
    )
    path = tmp_path / "balanced-v4-search-v2.json"
    _write_manifest(path, payload)

    recipe = manifest_module.load_search_recipe(path, development_corpus=development_corpus)

    assert isinstance(recipe, manifest_module.PhaseSearchManifest)
    assert recipe.schema_version == 2
    assert recipe.name == "balanced-v4-search-v2"
    assert recipe.predecessor_name == "balanced-v3"
    assert recipe.phase_selector == manifest_module.PhaseSelector(
        kind="public-resource-horizon-v1",
        early="3*future>=2*total",
        middle="3*future>=total",
        late="otherwise",
    )
    assert recipe.boundary_evidence.report_path == BOUNDARY_REPORT_PATH
    assert recipe.boundary_evidence.report_digest == BOUNDARY_REPORT_DIGEST
    assert recipe.boundary_evidence.slices_path == BOUNDARY_SLICES_PATH
    assert recipe.boundary_evidence.slices_digest == BOUNDARY_SLICES_DIGEST
    assert (
        tuple(map(str, recipe.initial_experts.as_loci()))
        == (
            "0.25",
            "1.55",
            "0.3",
            "0.35",
        )
        * 3
    )
    assert recipe.expert_coefficient_grids.early == recipe.expert_coefficient_grids.middle
    assert recipe.expert_coefficient_grids.middle == recipe.expert_coefficient_grids.late
    assert len(recipe.expert_coefficient_grids.as_loci()) == 12
    assert manifest_module.recompute_phase_search_manifest_digest(recipe) == recipe.digest
    normalized_payload = manifest_module.phase_search_manifest_payload(recipe)
    assert normalized_payload["phase_selector"] == payload["phase_selector"]
    assert normalized_payload["boundary_evidence"] == payload["boundary_evidence"]
    assert normalized_payload["initial_experts"] == {
        phase: {
            coefficient: str(Decimal(value).normalize())
            for coefficient, value in payload["initial_experts"][phase].items()
        }
        for phase in PHASES
    }


def test_load_search_manifest_remains_schema_v1_only(
    tmp_path: Path,
    development_corpus: PromotionCorpus,
) -> None:
    path = tmp_path / "balanced-v4-search-v2.json"
    _write_manifest(
        path,
        _phase_manifest_payload(
            corpus_name=development_corpus.recipe.name,
            corpus_digest=development_corpus.digest,
        ),
    )

    with pytest.raises(SearchManifestError) as captured:
        load_search_manifest(path, development_corpus=development_corpus)

    assert captured.value.code == "invalid_manifest_keys"


@pytest.mark.parametrize(
    ("mutate", "expected_code"),
    (
        (
            lambda payload: payload["phase_selector"].__setitem__("early", "future>=total"),
            "invalid_phase_selector",
        ),
        (
            lambda payload: payload.__setitem__("predecessor_name", "balanced-v2"),
            "wrong_predecessor",
        ),
        (
            lambda payload: payload["initial_experts"].__delitem__("late"),
            "invalid_phase_names",
        ),
        (
            lambda payload: payload["initial_experts"].__setitem__(
                "opening", payload["initial_experts"]["early"]
            ),
            "invalid_phase_names",
        ),
        (
            lambda payload: payload["initial_experts"]["middle"].__setitem__(
                "liquidity_strength", True
            ),
            "invalid_decimal",
        ),
        (
            lambda payload: payload["initial_experts"]["middle"].__setitem__(
                "liquidity_strength", "NaN"
            ),
            "invalid_decimal",
        ),
        (
            lambda payload: payload["initial_experts"]["middle"].__setitem__(
                "liquidity_strength", "0.26"
            ),
            "initial_coefficient_off_grid",
        ),
        (
            lambda payload: payload["initial_experts"]["middle"].__setitem__(
                "liquidity_strength", "0.30"
            ),
            "wrong_initial_coefficients",
        ),
        (
            lambda payload: payload["expert_coefficient_grids"]["late"][
                "liquidity_strength"
            ].__setitem__("maximum", "1.45"),
            "unequal_expert_grids",
        ),
        (
            lambda payload: payload["boundary_evidence"].__setitem__("report_digest", "0" * 64),
            "wrong_boundary_evidence",
        ),
        (
            lambda payload: payload["algorithm"].__setitem__("held_out_seed", 999),
            "held_out_key_forbidden",
        ),
    ),
)
def test_schema_v2_rejects_noncanonical_or_unsafe_content(
    tmp_path: Path,
    development_corpus: PromotionCorpus,
    mutate: object,
    expected_code: str,
) -> None:
    payload = _phase_manifest_payload(
        corpus_name=development_corpus.recipe.name,
        corpus_digest=development_corpus.digest,
    )
    assert callable(mutate)
    mutate(payload)
    path = tmp_path / "manifest.json"
    _write_manifest(path, payload)

    with pytest.raises(SearchManifestError) as captured:
        manifest_module.load_search_recipe(path, development_corpus=development_corpus)

    assert captured.value.code == expected_code


def test_schema_v2_rejects_duplicate_and_unknown_nested_keys(
    tmp_path: Path,
    development_corpus: PromotionCorpus,
) -> None:
    payload = _phase_manifest_payload(
        corpus_name=development_corpus.recipe.name,
        corpus_digest=development_corpus.digest,
    )
    selector = payload["phase_selector"]
    assert isinstance(selector, dict)
    selector["notes"] = "not part of the fixed selector"
    unknown_path = tmp_path / "unknown.json"
    _write_manifest(unknown_path, payload)

    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text(
        '{"schema_version":2,"schema_version":2}',
        encoding="utf-8",
    )

    for path, code in (
        (unknown_path, "invalid_phase_selector_keys"),
        (duplicate_path, "duplicate_json_key"),
    ):
        with pytest.raises(SearchManifestError) as captured:
            manifest_module.load_search_recipe(path, development_corpus=development_corpus)
        assert captured.value.code == code


def test_committed_schema_v2_manifests_bind_v3_and_fixed_evidence(
    development_corpus: PromotionCorpus,
) -> None:
    expected_initial = {
        "aggressive": ("1", "1.95", "0.15", "0.4"),
        "balanced": ("0.25", "1.55", "0.3", "0.35"),
        "passive": ("1.5", "1.8", "0.95", "0.45"),
    }

    for seed_offset, personality in enumerate(("aggressive", "balanced", "passive"), start=1):
        recipe = manifest_module.load_search_recipe(
            EVOLUTION_CONFIG_DIRECTORY / f"{personality}-v4-search-v2.json",
            development_corpus=development_corpus,
        )
        assert isinstance(recipe, manifest_module.PhaseSearchManifest)
        assert recipe.name == f"{personality}-v4-search-v2"
        assert recipe.predecessor_name == f"{personality}-v3"
        assert recipe.search_seed == 12000 + seed_offset
        assert recipe.algorithm.generation_count == 12
        assert recipe.algorithm.population_size == 16
        assert recipe.algorithm.elite_count == 4
        assert recipe.algorithm.mutation_radius_steps == 4
        assert (
            tuple(map(str, recipe.initial_experts.early.as_tuple()))
            == expected_initial[personality]
        )
        assert recipe.initial_experts.early == recipe.initial_experts.middle
        assert recipe.initial_experts.middle == recipe.initial_experts.late
        assert recipe.boundary_evidence.report_digest == BOUNDARY_REPORT_DIGEST
        assert recipe.boundary_evidence.slices_digest == BOUNDARY_SLICES_DIGEST


@pytest.mark.parametrize(
    ("algorithm_field", "alternate_value"),
    (
        ("generation_count", 11),
        ("population_size", 15),
        ("elite_count", 3),
        ("mutation_radius_steps", 3),
    ),
)
def test_schema_v2_rejects_an_alternative_algorithm_budget(
    tmp_path: Path,
    development_corpus: PromotionCorpus,
    algorithm_field: str,
    alternate_value: int,
) -> None:
    payload = _phase_manifest_payload(
        corpus_name=development_corpus.recipe.name,
        corpus_digest=development_corpus.digest,
    )
    algorithm = payload["algorithm"]
    assert isinstance(algorithm, dict)
    algorithm[algorithm_field] = alternate_value
    path = tmp_path / "manifest.json"
    _write_manifest(path, payload)

    with pytest.raises(SearchManifestError) as captured:
        manifest_module.load_search_recipe(path, development_corpus=development_corpus)

    assert captured.value.code == "wrong_phase_search_algorithm"


def test_schema_v2_rejects_alternative_grids_even_when_all_phases_match(
    tmp_path: Path,
    development_corpus: PromotionCorpus,
) -> None:
    payload = _phase_manifest_payload(
        corpus_name=development_corpus.recipe.name,
        corpus_digest=development_corpus.digest,
    )
    phase_grids = payload["expert_coefficient_grids"]
    assert isinstance(phase_grids, dict)
    for phase in PHASES:
        grids = phase_grids[phase]
        assert isinstance(grids, dict)
        liquidity_grid = grids["liquidity_strength"]
        assert isinstance(liquidity_grid, dict)
        liquidity_grid["maximum"] = "1.55"
    path = tmp_path / "manifest.json"
    _write_manifest(path, payload)

    with pytest.raises(SearchManifestError) as captured:
        manifest_module.load_search_recipe(path, development_corpus=development_corpus)

    assert captured.value.code == "wrong_phase_coefficient_grids"


def test_phase_coefficient_loci_are_phase_first_then_coefficient_name() -> None:
    values = manifest_module.PhaseCoefficientValues(
        early=manifest_module.CoefficientValues(
            Decimal("1"),
            Decimal("2"),
            Decimal("3"),
            Decimal("4"),
        ),
        middle=manifest_module.CoefficientValues(
            Decimal("5"),
            Decimal("6"),
            Decimal("7"),
            Decimal("8"),
        ),
        late=manifest_module.CoefficientValues(
            Decimal("9"),
            Decimal("10"),
            Decimal("11"),
            Decimal("12"),
        ),
    )

    assert values.as_loci() == tuple(Decimal(value) for value in range(1, 13))


def test_committed_boundary_evidence_content_matches_pinned_digests() -> None:
    expected = (
        (BOUNDARY_REPORT_PATH, BOUNDARY_REPORT_DIGEST),
        (BOUNDARY_SLICES_PATH, BOUNDARY_SLICES_DIGEST),
    )

    for relative_path, expected_digest in expected:
        content = (REPOSITORY_ROOT / relative_path).read_bytes()
        assert hashlib.sha256(content).hexdigest() == expected_digest


def test_schema_v2_canonical_payload_digests_are_frozen(
    development_corpus: PromotionCorpus,
) -> None:
    expected = {
        "aggressive": (
            2120,
            "71c06a1a246e81c935156ff818f66dcac454719168fabdd4af63ba94249ca69b",
        ),
        "balanced": (
            2123,
            "e1f1bed8f09aef9193ffeb0ed3e0be822be96df7fd69985c9e4111f5c725933c",
        ),
        "passive": (
            2117,
            "334579f896a0d4281c8926bb4cc5d9bffd9b3c63b8be3d0ae3375699792d4bc6",
        ),
    }

    for personality, (expected_size, expected_digest) in expected.items():
        recipe = manifest_module.load_search_recipe(
            EVOLUTION_CONFIG_DIRECTORY / f"{personality}-v4-search-v2.json",
            development_corpus=development_corpus,
        )
        assert isinstance(recipe, manifest_module.PhaseSearchManifest)
        canonical_bytes = (
            json.dumps(
                manifest_module.phase_search_manifest_payload(recipe),
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode()
        assert len(canonical_bytes) == expected_size
        assert hashlib.sha256(canonical_bytes).hexdigest() == expected_digest
        assert recipe.digest == expected_digest


def test_schema_v2_errors_identify_the_nested_phase_field(
    tmp_path: Path,
    development_corpus: PromotionCorpus,
) -> None:
    payload = _phase_manifest_payload(
        corpus_name=development_corpus.recipe.name,
        corpus_digest=development_corpus.digest,
    )
    initial_experts = payload["initial_experts"]
    assert isinstance(initial_experts, dict)
    middle = initial_experts["middle"]
    assert isinstance(middle, dict)
    middle["liquidity_strength"] = True
    path = tmp_path / "manifest.json"
    _write_manifest(path, payload)

    with pytest.raises(SearchManifestError) as captured:
        manifest_module.load_search_recipe(path, development_corpus=development_corpus)

    assert captured.value.code == "invalid_decimal"
    assert "initial_experts.middle.liquidity_strength" in str(captured.value)
