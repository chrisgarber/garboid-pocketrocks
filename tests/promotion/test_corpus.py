from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from garboid_pocketrocks.bots import BOT_SPECS_BY_NAME
from garboid_pocketrocks.promotion.corpus import (
    PromotionCorpus,
    PromotionCorpusError,
    corpus_snapshot_payload,
    load_promotion_corpus,
    validate_corpus_separation,
)
from garboid_pocketrocks.simulator.seeding import derive_seed


def _recipe_payload(
    *,
    name: str = "fixture-development-v1",
    purpose: str = "development",
    root_seed: object = 17,
    repetitions: object = 2,
    charts: object | None = None,
    player_counts: object | None = None,
    opponent_names: object | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "name": name,
        "purpose": purpose,
        "root_seed": root_seed,
        "repetitions_per_seat_cell": repetitions,
        "charts": list("abcde") if charts is None else charts,
        "player_counts": [3, 4, 5] if player_counts is None else player_counts,
        "opponent_names": (
            ["random", "aggressive-v1", "balanced-v1", "passive-v1"]
            if opponent_names is None
            else opponent_names
        ),
    }


def _write_recipe(path: Path, payload: dict[str, object], *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    separators = (",", ":") if compact else None
    path.write_text(
        json.dumps(payload, indent=None if compact else 2, separators=separators) + "\n",
        encoding="utf-8",
    )


def _load_fixture(
    tmp_path: Path,
    *,
    filename: str = "fixture.json",
    payload: dict[str, object] | None = None,
) -> PromotionCorpus:
    path = tmp_path / filename
    _write_recipe(path, _recipe_payload() if payload is None else payload)
    return load_promotion_corpus(path, registry=BOT_SPECS_BY_NAME)


def test_loads_normalizes_and_exactly_expands_a_recipe(tmp_path: Path) -> None:
    corpus = _load_fixture(tmp_path)

    assert corpus.recipe.name == "fixture-development-v1"
    assert corpus.recipe.purpose == "development"
    assert corpus.recipe.charts == ("A", "B", "C", "D", "E")
    assert corpus.recipe.player_counts == (3, 4, 5)
    assert len(corpus.cases) == 5 * (3 + 4 + 5) * 2
    assert len(set(corpus.engine_seeds)) == len(corpus.cases)

    first = corpus.cases[0]
    assert first.case_id == "fixture-development-v1:A:3:seat-0:repeat-0"
    assert first.engine_seed == derive_seed(
        17,
        "promotion-corpus:fixture-development-v1",
        0,
    )
    assert first.opponent_names_by_seat == (None, "passive-v1", "random")

    second = corpus.cases[1]
    assert second.case_id == "fixture-development-v1:A:3:seat-1:repeat-0"
    assert second.opponent_names_by_seat == ("random", None, "aggressive-v1")

    for case in corpus.cases:
        assert case.chart in "ABCDE"
        assert len(case.opponent_names_by_seat) == case.player_count
        assert case.opponent_names_by_seat[case.focal_seat] is None
        opponents = tuple(name for name in case.opponent_names_by_seat if name is not None)
        assert len(opponents) == case.player_count - 1
        assert len(set(opponents)) == len(opponents)
        assert all(name in BOT_SPECS_BY_NAME for name in opponents)


def test_recipe_and_expanded_values_are_immutable(tmp_path: Path) -> None:
    corpus = _load_fixture(tmp_path)

    with pytest.raises(FrozenInstanceError):
        corpus.recipe.name = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        corpus.cases[0].engine_seed = 99  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        corpus.digest = "changed"  # type: ignore[misc]
    assert isinstance(corpus.recipe.charts, tuple)
    assert isinstance(corpus.cases, tuple)
    assert isinstance(corpus.cases[0].opponent_names_by_seat, tuple)


def test_digest_is_stable_across_source_json_formatting_and_key_order(tmp_path: Path) -> None:
    payload = _recipe_payload()
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    _write_recipe(first_path, payload)
    _write_recipe(second_path, dict(reversed(tuple(payload.items()))), compact=True)

    first = load_promotion_corpus(first_path, registry=BOT_SPECS_BY_NAME)
    second = load_promotion_corpus(second_path, registry=BOT_SPECS_BY_NAME)

    assert first.digest == second.digest
    assert len(first.digest) == 64
    int(first.digest, 16)


def test_snapshot_contains_normalized_recipe_cases_and_their_digest(tmp_path: Path) -> None:
    corpus = _load_fixture(tmp_path)

    snapshot = corpus_snapshot_payload(corpus)

    assert snapshot["recipe"] == {
        "schema_version": 1,
        "name": "fixture-development-v1",
        "purpose": "development",
        "root_seed": 17,
        "repetitions_per_seat_cell": 2,
        "charts": ["A", "B", "C", "D", "E"],
        "player_counts": [3, 4, 5],
        "opponent_names": ["random", "aggressive-v1", "balanced-v1", "passive-v1"],
    }
    cases = snapshot["cases"]
    assert isinstance(cases, list)
    assert cases[0] == {
        "case_id": "fixture-development-v1:A:3:seat-0:repeat-0",
        "chart": "A",
        "player_count": 3,
        "focal_seat": 0,
        "engine_seed": corpus.cases[0].engine_seed,
        "opponent_names_by_seat": [None, "passive-v1", "random"],
    }
    assert snapshot["digest"] == corpus.digest

    expanded_payload = dict(snapshot)
    del expanded_payload["digest"]
    encoded = (
        json.dumps(expanded_payload, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")
    assert corpus.digest == hashlib.sha256(encoded).hexdigest()


@pytest.mark.parametrize(
    ("field", "value", "expected_code"),
    (
        ("schema_version", 2, "unsupported_schema"),
        ("schema_version", True, "unsupported_schema"),
        ("name", "", "invalid_corpus_name"),
        ("name", "Fixture Development", "invalid_corpus_name"),
        ("purpose", "final", "invalid_purpose"),
        ("root_seed", -1, "invalid_root_seed"),
        ("root_seed", True, "invalid_root_seed"),
        ("repetitions_per_seat_cell", 0, "invalid_repetitions"),
        ("repetitions_per_seat_cell", True, "invalid_repetitions"),
        ("charts", ["A", "F"], "unsupported_chart"),
        ("charts", ["A", "a"], "unsupported_chart"),
        ("charts", ["AA"], "unsupported_chart"),
        ("player_counts", [2, 3], "unsupported_player_count"),
        ("player_counts", [3, 3], "unsupported_player_count"),
        ("player_counts", [3, True], "unsupported_player_count"),
        (
            "opponent_names",
            ["random", "aggressive-v1", "balanced-v1", "not-registered"],
            "unknown_opponent",
        ),
        (
            "opponent_names",
            ["random", "aggressive-v1", "balanced-v1", "random"],
            "duplicate_opponent",
        ),
        (
            "opponent_names",
            ["random", "aggressive-v1", "balanced-v1"],
            "insufficient_opponents",
        ),
    ),
)
def test_malformed_recipe_fields_have_stable_error_codes(
    tmp_path: Path,
    field: str,
    value: object,
    expected_code: str,
) -> None:
    payload = _recipe_payload()
    payload[field] = value
    path = tmp_path / "fixture.json"
    _write_recipe(path, payload)

    with pytest.raises(PromotionCorpusError) as captured:
        load_promotion_corpus(path, registry=BOT_SPECS_BY_NAME)

    assert captured.value.code == expected_code
    assert str(captured.value)


@pytest.mark.parametrize("payload", ([], None, "recipe"))
def test_recipe_root_must_be_a_json_object(tmp_path: Path, payload: object) -> None:
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PromotionCorpusError) as captured:
        load_promotion_corpus(path, registry=BOT_SPECS_BY_NAME)

    assert captured.value.code == "invalid_recipe"


def test_missing_and_unknown_recipe_keys_are_not_ignored(tmp_path: Path) -> None:
    missing = _recipe_payload()
    del missing["root_seed"]
    missing_path = tmp_path / "missing.json"
    _write_recipe(missing_path, missing)

    unknown = _recipe_payload()
    unknown["notes"] = "not part of schema version 1"
    unknown_path = tmp_path / "unknown.json"
    _write_recipe(unknown_path, unknown)

    for path in (missing_path, unknown_path):
        with pytest.raises(PromotionCorpusError) as captured:
            load_promotion_corpus(path, registry=BOT_SPECS_BY_NAME)
        assert captured.value.code == "invalid_recipe_keys"
        assert "missing keys" in str(captured.value)
        assert "unknown keys" in str(captured.value)


@pytest.mark.parametrize("source", ("{", '{"root_seed": NaN}'))
def test_malformed_or_nonfinite_json_is_rejected(tmp_path: Path, source: str) -> None:
    path = tmp_path / "fixture.json"
    path.write_text(source, encoding="utf-8")

    with pytest.raises(PromotionCorpusError) as captured:
        load_promotion_corpus(path, registry=BOT_SPECS_BY_NAME)

    assert captured.value.code == "malformed_json"
    assert "valid JSON" in str(captured.value)


@pytest.mark.parametrize(
    ("duplicate_key", "replacement"),
    (("root_seed", 18), ("purpose", "held_out")),
)
def test_duplicate_recipe_keys_are_rejected_before_a_value_can_be_overwritten(
    tmp_path: Path,
    duplicate_key: str,
    replacement: object,
) -> None:
    payload = _recipe_payload()
    original_entry = f'"{duplicate_key}": {json.dumps(payload[duplicate_key])}'
    duplicate_entry = f'{original_entry}, "{duplicate_key}": {json.dumps(replacement)}'
    source = json.dumps(payload).replace(original_entry, duplicate_entry)
    assert source.count(f'"{duplicate_key}"') == 2
    path = tmp_path / "fixture.json"
    path.write_text(source, encoding="utf-8")

    with pytest.raises(PromotionCorpusError) as captured:
        load_promotion_corpus(path, registry=BOT_SPECS_BY_NAME)

    assert captured.value.code == "duplicate_json_key"
    assert duplicate_key in str(captured.value)
    assert "appears more than once" in str(captured.value)


def test_duplicate_keys_in_nested_json_objects_are_also_rejected(tmp_path: Path) -> None:
    payload = _recipe_payload()
    source = json.dumps(
        {
            **payload,
            "metadata": {"explanation": "first"},
        }
    ).replace(
        '"explanation": "first"',
        '"explanation": "first", "explanation": "second"',
    )
    path = tmp_path / "fixture.json"
    path.write_text(source, encoding="utf-8")

    with pytest.raises(PromotionCorpusError) as captured:
        load_promotion_corpus(path, registry=BOT_SPECS_BY_NAME)

    assert captured.value.code == "duplicate_json_key"
    assert "explanation" in str(captured.value)


@pytest.mark.parametrize(
    ("filename", "purpose"),
    (("development-v1.json", "held_out"), ("held-out-v1.json", "development")),
)
def test_versioned_corpus_filename_must_match_its_purpose(
    tmp_path: Path,
    filename: str,
    purpose: str,
) -> None:
    payload = _recipe_payload(purpose=purpose)
    path = tmp_path / "configs" / "promotion" / filename
    _write_recipe(path, payload)

    with pytest.raises(PromotionCorpusError) as captured:
        load_promotion_corpus(path, registry=BOT_SPECS_BY_NAME)

    assert captured.value.code == "invalid_purpose"
    assert "filename" in str(captured.value)


def test_duplicate_expanded_seed_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "garboid_pocketrocks.promotion.corpus.derive_seed",
        lambda root_seed, namespace, index: 1,
    )

    with pytest.raises(PromotionCorpusError) as captured:
        _load_fixture(tmp_path)

    assert captured.value.code == "duplicate_engine_seed"
    assert "engine seed" in str(captured.value)


def test_valid_development_and_held_out_corpora_are_separate(tmp_path: Path) -> None:
    development = _load_fixture(tmp_path, filename="development-fixture.json")
    held_out = _load_fixture(
        tmp_path,
        filename="held-out-fixture.json",
        payload=_recipe_payload(
            name="fixture-held-out-v1",
            purpose="held_out",
            root_seed=18,
        ),
    )

    validate_corpus_separation(development, held_out)


def test_overlapping_development_and_held_out_seeds_are_rejected(tmp_path: Path) -> None:
    development = _load_fixture(tmp_path, filename="development-fixture.json")
    held_out = _load_fixture(
        tmp_path,
        filename="held-out-fixture.json",
        payload=_recipe_payload(
            name="fixture-held-out-v1",
            purpose="held_out",
            root_seed=18,
        ),
    )
    overlapping_case = replace(
        held_out.cases[0],
        engine_seed=development.cases[0].engine_seed,
    )
    overlapping_held_out = replace(
        held_out,
        cases=(overlapping_case, *held_out.cases[1:]),
    )

    with pytest.raises(PromotionCorpusError) as captured:
        validate_corpus_separation(development, overlapping_held_out)

    assert captured.value.code == "corpus_seed_overlap"
    assert "not used for tuning" in str(captured.value)


def test_separation_requires_corpora_in_their_named_roles(tmp_path: Path) -> None:
    development = _load_fixture(tmp_path, filename="development-fixture.json")
    held_out = _load_fixture(
        tmp_path,
        filename="held-out-fixture.json",
        payload=_recipe_payload(
            name="fixture-held-out-v1",
            purpose="held_out",
            root_seed=18,
        ),
    )

    with pytest.raises(PromotionCorpusError) as captured:
        validate_corpus_separation(held_out, development)

    assert captured.value.code == "invalid_purpose"


def test_separation_requires_distinct_corpus_names(tmp_path: Path) -> None:
    development = _load_fixture(tmp_path, filename="development-fixture.json")
    held_out = _load_fixture(
        tmp_path,
        filename="held-out-fixture.json",
        payload=_recipe_payload(
            name="fixture-held-out-v1",
            purpose="held_out",
            root_seed=18,
        ),
    )
    same_name = replace(
        held_out,
        recipe=replace(held_out.recipe, name=development.recipe.name),
    )

    with pytest.raises(PromotionCorpusError) as captured:
        validate_corpus_separation(development, same_name)

    assert captured.value.code == "duplicate_corpus_name"


def test_committed_corpora_have_exact_coverage_and_disjoint_seeds() -> None:
    development = load_promotion_corpus(
        Path("configs/promotion/development-v1.json"),
        registry=BOT_SPECS_BY_NAME,
    )
    held_out = load_promotion_corpus(
        Path("configs/promotion/held-out-v1.json"),
        registry=BOT_SPECS_BY_NAME,
    )

    assert development.recipe.name == "development-v1"
    assert development.recipe.purpose == "development"
    assert development.recipe.repetitions_per_seat_cell == 4
    assert len(development.cases) == 240
    assert held_out.recipe.name == "held-out-v1"
    assert held_out.recipe.purpose == "held_out"
    assert held_out.recipe.repetitions_per_seat_cell == 8
    assert len(held_out.cases) == 480
    expected_opponents = (
        "aggressive-v1",
        "balanced-v1",
        "passive-v1",
        "aggressive-v2",
        "balanced-v2",
        "passive-v2",
        "vector_ppo_large_v1_g350k",
    )
    assert development.recipe.opponent_names == expected_opponents
    assert held_out.recipe.opponent_names == expected_opponents
    assert development.digest == "3baf37660bb33ac2571ba62a09873a74cccbe6d7491f063e5d4a3e641fd24f4c"
    assert held_out.digest == "8b5a42d944f5c79486fc0a78333c35acee72322bdd062b1f80a6a247cf7a5164"
    validate_corpus_separation(development, held_out)
