from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import replace
from typing import Any, cast

import pytest

from garboid_pocketrocks.diagnostics.reporting import render_decision_artifacts

from .test_analysis import _build, _game, _trace


def test_decision_artifacts_are_sanitized_canonical_and_order_independent() -> None:
    first_game = _game(game_index=0, chart="A", decision_counts=(1, 0, 0))
    second_game = _game(game_index=1, chart="E", decision_counts=(1, 0, 0))
    first_trace = _trace(first_game, seat=0)
    second_trace = _trace(second_game, seat=0)
    report = _build((second_trace, first_trace), (second_game, first_game))

    canonical = render_decision_artifacts(decision_report=report)
    reversed_sources = render_decision_artifacts(
        decision_report=replace(report, slices=tuple(reversed(report.slices))),
    )

    assert canonical == reversed_sources
    assert canonical.game_summaries_jsonl.endswith("\n")
    assert canonical.decision_traces_jsonl.endswith("\n")
    assert canonical.decision_slices_csv.endswith("\r\n")

    game_payloads = tuple(json.loads(line) for line in canonical.game_summaries_jsonl.splitlines())
    assert [payload["game_index"] for payload in game_payloads] == [0, 1]
    assert set(game_payloads[0]) == {
        "game_index",
        "player_count",
        "ruleset_name",
        "bot_names",
        "bot_ids",
        "scores",
        "decision_counts",
        "fault_counts",
    }
    assert "seed" not in canonical.game_summaries_jsonl

    trace_payloads = tuple(
        json.loads(line) for line in canonical.decision_traces_jsonl.splitlines()
    )
    assert [payload["game_index"] for payload in trace_payloads] == [0, 1]
    assert "root_seed" not in canonical.decision_traces_jsonl
    assert "engine_seed" not in canonical.decision_traces_jsonl

    slice_rows = tuple(csv.DictReader(io.StringIO(canonical.decision_slices_csv)))
    assert [row["chart"] for row in slice_rows] == ["A", "E"]
    assert json.loads(slice_rows[0]["opponent_bot_ids"]) == ["beta", "gamma"]
    assert sum(int(row["decision_count"]) for row in slice_rows) == 2


def test_render_api_cannot_combine_a_valid_report_with_unrelated_sources() -> None:
    game = _game(decision_counts=(1, 0, 0))
    trace = _trace(game, seat=0)
    report = _build((trace,), (game,))
    unrelated_game = _game(game_index=99, decision_counts=(0, 0, 0))
    render_with_unsupported_arguments = cast(Any, render_decision_artifacts)

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        render_with_unsupported_arguments(
            decision_report=report,
            game_summaries=(unrelated_game,),
            decision_traces=(),
        )


def test_nonfinite_game_summary_fails_json_rendering() -> None:
    game = _game(decision_counts=(1, 0, 0))
    trace = _trace(game, seat=0)
    report = _build((trace,), (game,))
    nonfinite_score = replace(
        game.scores[0],
        final_money=cast(Any, math.nan),
    )
    nonfinite_game = replace(game, scores=(nonfinite_score, *game.scores[1:]))

    with pytest.raises(ValueError, match="finite JSON"):
        render_decision_artifacts(
            decision_report=replace(report, game_summaries=(nonfinite_game,)),
        )
