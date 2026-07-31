from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from garboid_pocketrocks.bots import BOT_SPECS_BY_NAME
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
) -> None:
    expected_initial = {
        "aggressive": ("0.75", "1.5", "0.25", "0.05"),
        "balanced": ("0.4", "0.75", "0.2", "0.25"),
        "passive": ("0.15", "0.6", "0.15", "0.3"),
    }

    for seed_offset, personality in enumerate(("aggressive", "balanced", "passive"), start=1):
        path = EVOLUTION_CONFIG_DIRECTORY / f"{personality}-v3-search-v1.json"
        development_corpus = load_promotion_corpus(
            REPOSITORY_ROOT
            / f"configs/promotion/development-{personality}-v3-broad-v1.json",
            registry=BOT_SPECS_BY_NAME,
        )
        manifest = load_search_manifest(path, development_corpus=development_corpus)

        assert manifest.name == f"{personality}-v3-search-v1"
        assert manifest.predecessor_name == f"{personality}-v2"
        assert manifest.development_corpus.name == (
            f"development-{personality}-v3-broad-v1"
        )
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
