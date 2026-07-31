"""Load stable tournament artifacts without retaining raw decision traces."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast


class TournamentDatasetError(ValueError):
    """Raised when a tournament artifact set is malformed or incomplete."""


@dataclass(frozen=True, slots=True)
class TournamentDataset:
    """A tournament summary plus optional public game and decision evidence."""

    source_dir: Path
    summary: dict[str, Any]
    games: tuple[dict[str, Any], ...]
    game_details: tuple[dict[str, Any], ...]
    decision_traces_path: Path | None

    @classmethod
    def load(cls, source_dir: Path) -> TournamentDataset:
        source_dir = source_dir.resolve()
        summary_path = source_dir / "summary.json"
        if not summary_path.is_file():
            raise TournamentDatasetError(f"missing tournament summary: {summary_path}")
        summary = _json_object(summary_path)
        if summary.get("schema_version") != 1:
            raise TournamentDatasetError("unsupported tournament summary schema version")
        if not isinstance(summary.get("leaderboard"), list):
            raise TournamentDatasetError("tournament summary is missing its leaderboard")

        games_path = source_dir / "game-summaries.jsonl"
        details_path = source_dir / "game-details.jsonl"
        traces_path = source_dir / "decision-traces.jsonl"
        return cls(
            source_dir=source_dir,
            summary=summary,
            games=tuple(_json_lines(games_path)) if games_path.is_file() else (),
            game_details=tuple(_json_lines(details_path)) if details_path.is_file() else (),
            decision_traces_path=traces_path if traces_path.is_file() else None,
        )

    def iter_decision_traces(self) -> Iterator[dict[str, Any]]:
        if self.decision_traces_path is None:
            return
        yield from _json_lines(self.decision_traces_path)


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TournamentDatasetError(f"could not read {path.name}: {error}") from error
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TournamentDatasetError(f"{path.name} must contain one JSON object")
    return dict(cast(Mapping[str, Any], value))


def _json_lines(path: Path) -> Iterator[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise TournamentDatasetError(
                        f"invalid JSON in {path.name} line {line_number}: {error.msg}"
                    ) from error
                if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
                    raise TournamentDatasetError(
                        f"{path.name} line {line_number} must be a JSON object"
                    )
                yield dict(cast(Mapping[str, Any], value))
    except OSError as error:
        raise TournamentDatasetError(f"could not read {path.name}: {error}") from error
