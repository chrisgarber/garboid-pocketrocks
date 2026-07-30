"""Load immutable recipes for the fixed games used by the promotion gate."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NoReturn, cast

from pocketrocks.sim.constants import VALUE_CHARTS

from garboid_pocketrocks.bots import BotSpec
from garboid_pocketrocks.simulator.seeding import derive_seed

CorpusPurpose = Literal["development", "held_out"]

_EXPECTED_RECIPE_KEYS = {
    "schema_version",
    "name",
    "purpose",
    "root_seed",
    "repetitions_per_seat_cell",
    "charts",
    "player_counts",
    "opponent_names",
}
_CORPUS_NAME_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_SUPPORTED_PLAYER_COUNTS = (3, 4, 5)
_MINIMUM_OPPONENTS = max(_SUPPORTED_PLAYER_COUNTS) - 1


@dataclass(frozen=True, slots=True)
class PromotionCorpusRecipe:
    """A versioned recipe for a fixed set of promotion games."""

    schema_version: int
    name: str
    purpose: CorpusPurpose
    root_seed: int
    repetitions_per_seat_cell: int
    charts: tuple[str, ...]
    player_counts: tuple[int, ...]
    opponent_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PromotionCase:
    """One chart, player count, focal seat, opponent lineup, and engine seed."""

    case_id: str
    chart: str
    player_count: int
    focal_seat: int
    engine_seed: int
    opponent_names_by_seat: tuple[str | None, ...]


@dataclass(frozen=True, slots=True)
class PromotionCorpus:
    """An immutable normalized recipe and all cases expanded from it."""

    recipe: PromotionCorpusRecipe
    cases: tuple[PromotionCase, ...]
    digest: str

    @property
    def engine_seeds(self) -> tuple[int, ...]:
        """Return expanded engine seeds in case order."""

        return tuple(case.engine_seed for case in self.cases)


class PromotionCorpusError(ValueError):
    """Explain why a promotion corpus cannot be used safely."""

    code: str

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class _NonFiniteJsonNumber(ValueError):
    """Internal signal raised when JSON contains NaN or infinity."""


def load_promotion_corpus(
    path: Path,
    *,
    registry: Mapping[str, BotSpec],
) -> PromotionCorpus:
    """Load, validate, expand, and hash one promotion corpus recipe."""

    payload = _load_json_object(path)
    _require_exact_keys(payload, _EXPECTED_RECIPE_KEYS, subject="promotion corpus recipe")
    recipe = _decode_recipe(payload, path=path, registry=registry)
    cases = _expand_cases(recipe)
    _require_unique_engine_seeds(cases, corpus_name=recipe.name)

    expanded_payload = _expanded_payload(recipe, cases)
    digest = hashlib.sha256(_canonical_json_bytes(expanded_payload)).hexdigest()
    return PromotionCorpus(recipe=recipe, cases=cases, digest=digest)


def validate_corpus_separation(
    development: PromotionCorpus,
    held_out: PromotionCorpus,
) -> None:
    """Require development and held-out corpora to occupy distinct roles and seeds."""

    if development.recipe.purpose != "development":
        raise PromotionCorpusError(
            "invalid_purpose",
            "The development corpus must have purpose 'development'.",
        )
    if held_out.recipe.purpose != "held_out":
        raise PromotionCorpusError(
            "invalid_purpose",
            "The held-out corpus must have purpose 'held_out'.",
        )
    if development.recipe.name == held_out.recipe.name:
        raise PromotionCorpusError(
            "duplicate_corpus_name",
            "Development and held-out corpora must have different versioned names.",
        )

    _require_unique_engine_seeds(development.cases, corpus_name=development.recipe.name)
    _require_unique_engine_seeds(held_out.cases, corpus_name=held_out.recipe.name)
    overlapping_seeds = set(development.engine_seeds) & set(held_out.engine_seeds)
    if overlapping_seeds:
        raise PromotionCorpusError(
            "corpus_seed_overlap",
            "The held-out corpus (the games not used for tuning) shares "
            f"{len(overlapping_seeds)} engine seed(s) with the development corpus.",
        )


def corpus_snapshot_payload(corpus: PromotionCorpus) -> dict[str, object]:
    """Return the normalized expanded corpus and its canonical digest."""

    payload = _expanded_payload(corpus.recipe, corpus.cases)
    return {**payload, "digest": corpus.digest}


def _load_json_object(path: Path) -> dict[str, object]:
    def reject_nonfinite_number(value: str) -> NoReturn:
        raise _NonFiniteJsonNumber(value)

    try:
        decoded = cast(
            object,
            json.loads(
                path.read_text(encoding="utf-8"),
                parse_constant=reject_nonfinite_number,
            ),
        )
    except (json.JSONDecodeError, _NonFiniteJsonNumber) as error:
        raise PromotionCorpusError(
            "malformed_json",
            f"{path} must contain valid JSON with only finite numbers.",
        ) from error

    if not isinstance(decoded, dict):
        raise PromotionCorpusError(
            "invalid_recipe",
            f"{path} must contain one JSON object describing a promotion corpus.",
        )
    return cast(dict[str, object], decoded)


def _require_exact_keys(
    payload: Mapping[str, object],
    expected: set[str],
    *,
    subject: str,
) -> None:
    missing = expected - set(payload)
    unknown = set(payload) - expected
    if missing or unknown:
        raise PromotionCorpusError(
            "invalid_recipe_keys",
            f"{subject} has missing keys {sorted(missing)} and unknown keys {sorted(unknown)}.",
        )


def _decode_recipe(
    payload: Mapping[str, object],
    *,
    path: Path,
    registry: Mapping[str, BotSpec],
) -> PromotionCorpusRecipe:
    schema_version = _require_integer(
        payload["schema_version"],
        code="unsupported_schema",
        field_name="schema_version",
        minimum=1,
    )
    if schema_version != 1:
        raise PromotionCorpusError(
            "unsupported_schema",
            f"Promotion corpus schema version {schema_version} is not supported; expected 1.",
        )

    name_value = payload["name"]
    if not isinstance(name_value, str) or _CORPUS_NAME_PATTERN.fullmatch(name_value) is None:
        raise PromotionCorpusError(
            "invalid_corpus_name",
            "The corpus name must be a nonempty lowercase, hyphen-separated versioned name.",
        )

    purpose = _decode_purpose(payload["purpose"])
    _validate_filename_purpose(path, purpose)
    root_seed = _require_integer(
        payload["root_seed"],
        code="invalid_root_seed",
        field_name="root_seed",
        minimum=0,
    )
    repetitions = _require_integer(
        payload["repetitions_per_seat_cell"],
        code="invalid_repetitions",
        field_name="repetitions_per_seat_cell",
        minimum=1,
    )
    charts = _decode_charts(payload["charts"])
    player_counts = _decode_player_counts(payload["player_counts"])
    opponent_names = _decode_opponents(payload["opponent_names"], registry=registry)

    return PromotionCorpusRecipe(
        schema_version=schema_version,
        name=name_value,
        purpose=purpose,
        root_seed=root_seed,
        repetitions_per_seat_cell=repetitions,
        charts=charts,
        player_counts=player_counts,
        opponent_names=opponent_names,
    )


def _require_integer(
    value: object,
    *,
    code: str,
    field_name: str,
    minimum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PromotionCorpusError(
            code,
            f"{field_name} must be an integer greater than or equal to {minimum}.",
        )
    return value


def _decode_purpose(value: object) -> CorpusPurpose:
    if value == "development":
        return "development"
    if value == "held_out":
        return "held_out"
    raise PromotionCorpusError(
        "invalid_purpose",
        "Corpus purpose must be 'development' or 'held_out'.",
    )


def _validate_filename_purpose(path: Path, purpose: CorpusPurpose) -> None:
    expected: CorpusPurpose | None = None
    if path.name.startswith("development-"):
        expected = "development"
    elif path.name.startswith("held-out-"):
        expected = "held_out"
    if expected is not None and purpose != expected:
        raise PromotionCorpusError(
            "invalid_purpose",
            f"The filename {path.name!r} identifies a {expected!r} corpus, "
            f"but the recipe declares purpose {purpose!r}.",
        )


def _decode_charts(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise PromotionCorpusError(
            "unsupported_chart",
            "Corpus charts must be a nonempty list of chart names A through E.",
        )

    raw_charts = cast(list[object], value)
    if any(not isinstance(raw_chart, str) or len(raw_chart) != 1 for raw_chart in raw_charts):
        raise PromotionCorpusError(
            "unsupported_chart",
            "Every corpus chart must be a one-character name from A through E.",
        )
    charts = tuple(cast(str, raw_chart).upper() for raw_chart in raw_charts)
    if any(chart not in VALUE_CHARTS for chart in charts) or len(set(charts)) != len(charts):
        raise PromotionCorpusError(
            "unsupported_chart",
            "Corpus charts must be unique names from A through E.",
        )
    return charts


def _decode_player_counts(value: object) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise PromotionCorpusError(
            "unsupported_player_count",
            "Corpus player counts must be a nonempty list containing only 3, 4, or 5.",
        )

    raw_counts = cast(list[object], value)
    if any(
        isinstance(raw_count, bool)
        or not isinstance(raw_count, int)
        or raw_count not in _SUPPORTED_PLAYER_COUNTS
        for raw_count in raw_counts
    ):
        raise PromotionCorpusError(
            "unsupported_player_count",
            "Every corpus player count must be 3, 4, or 5.",
        )
    player_counts = tuple(cast(int, raw_count) for raw_count in raw_counts)
    if len(set(player_counts)) != len(player_counts):
        raise PromotionCorpusError(
            "unsupported_player_count",
            "Corpus player counts must not contain duplicates.",
        )
    return player_counts


def _decode_opponents(
    value: object,
    *,
    registry: Mapping[str, BotSpec],
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise PromotionCorpusError(
            "insufficient_opponents",
            f"Corpus opponent_names must be a list of at least {_MINIMUM_OPPONENTS} names.",
        )

    raw_names = cast(list[object], value)
    if any(not isinstance(raw_name, str) or not raw_name for raw_name in raw_names):
        raise PromotionCorpusError(
            "unknown_opponent",
            "Every corpus opponent must have a nonempty registered bot name.",
        )
    opponent_names = tuple(cast(str, raw_name) for raw_name in raw_names)
    if len(set(opponent_names)) != len(opponent_names):
        raise PromotionCorpusError(
            "duplicate_opponent",
            "Corpus opponent names must be distinct.",
        )
    if len(opponent_names) < _MINIMUM_OPPONENTS:
        raise PromotionCorpusError(
            "insufficient_opponents",
            f"Five-player cases require at least {_MINIMUM_OPPONENTS} distinct opponents.",
        )
    unknown_names = tuple(name for name in opponent_names if name not in registry)
    if unknown_names:
        raise PromotionCorpusError(
            "unknown_opponent",
            f"Corpus opponent names are not registered: {list(unknown_names)}.",
        )
    return opponent_names


def _expand_cases(recipe: PromotionCorpusRecipe) -> tuple[PromotionCase, ...]:
    cases: list[PromotionCase] = []
    for repetition in range(recipe.repetitions_per_seat_cell):
        for chart_index, chart in enumerate(recipe.charts):
            for player_count in recipe.player_counts:
                for focal_seat in range(player_count):
                    case_index = len(cases)
                    engine_seed = derive_seed(
                        recipe.root_seed,
                        f"promotion-corpus:{recipe.name}",
                        case_index,
                    )
                    rotation = (repetition + chart_index + player_count + focal_seat) % len(
                        recipe.opponent_names
                    )
                    rotated_opponents = (
                        recipe.opponent_names[rotation:] + recipe.opponent_names[:rotation]
                    )
                    selected_opponents = iter(rotated_opponents[: player_count - 1])
                    opponent_names_by_seat = tuple(
                        None if seat == focal_seat else next(selected_opponents)
                        for seat in range(player_count)
                    )
                    cases.append(
                        PromotionCase(
                            case_id=(
                                f"{recipe.name}:{chart}:{player_count}:"
                                f"seat-{focal_seat}:repeat-{repetition}"
                            ),
                            chart=chart,
                            player_count=player_count,
                            focal_seat=focal_seat,
                            engine_seed=engine_seed,
                            opponent_names_by_seat=opponent_names_by_seat,
                        )
                    )
    return tuple(cases)


def _require_unique_engine_seeds(
    cases: tuple[PromotionCase, ...],
    *,
    corpus_name: str,
) -> None:
    engine_seeds = tuple(case.engine_seed for case in cases)
    if len(set(engine_seeds)) != len(engine_seeds):
        raise PromotionCorpusError(
            "duplicate_engine_seed",
            f"Promotion corpus {corpus_name!r} expands to a duplicate engine seed.",
        )


def _expanded_payload(
    recipe: PromotionCorpusRecipe,
    cases: tuple[PromotionCase, ...],
) -> dict[str, object]:
    recipe_payload: dict[str, object] = {
        "schema_version": recipe.schema_version,
        "name": recipe.name,
        "purpose": recipe.purpose,
        "root_seed": recipe.root_seed,
        "repetitions_per_seat_cell": recipe.repetitions_per_seat_cell,
        "charts": list(recipe.charts),
        "player_counts": list(recipe.player_counts),
        "opponent_names": list(recipe.opponent_names),
    }
    case_payloads: list[dict[str, object]] = [
        {
            "case_id": case.case_id,
            "chart": case.chart,
            "player_count": case.player_count,
            "focal_seat": case.focal_seat,
            "engine_seed": case.engine_seed,
            "opponent_names_by_seat": list(case.opponent_names_by_seat),
        }
        for case in cases
    ]
    return {"recipe": recipe_payload, "cases": case_payloads}


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")
