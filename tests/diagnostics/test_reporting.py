from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import replace
from typing import Any, cast

import pytest

from garboid_pocketrocks.diagnostics.analysis import DecisionAnalysisError, DecisionReport
from garboid_pocketrocks.diagnostics.reporting import (
    render_decision_artifacts,
    render_opponent_model_summary,
)
from garboid_pocketrocks.diagnostics.trace import OpponentAwareHeuristicBidExplanation
from garboid_pocketrocks.heuristics.opponent_bids import (
    OPPONENT_BID_MODEL_NAME,
    CompetitiveBidPoint,
    OpponentBidDistribution,
)

from .test_analysis import _build, _game, _trace


def _opponent_aware_explanation(
    *,
    model_config_digest: str = "a" * 64,
) -> OpponentAwareHeuristicBidExplanation:
    return OpponentAwareHeuristicBidExplanation(
        resource_value=3.0,
        objective_completion_value=0.0,
        objective_progress_value=0.0,
        terminal_cash_value=0.0,
        liquidity_value=0.0,
        future_cash_value=0.0,
        total_value=3.0,
        reservation_bid=3,
        chosen_bid=3,
        public_game_phase="early",
        model_name=OPPONENT_BID_MODEL_NAME,
        model_config_digest=model_config_digest,
        opponent_distributions=tuple(
            OpponentBidDistribution(
                opponent_seat=seat,
                legal_max_amount=15,
                probabilities_by_amount=(1.0,) + (0.0,) * 15,
                prior_only=True,
                history_round_count=0,
                effective_history_weight=0.0,
            )
            for seat in (1, 2)
        ),
        competitive_bid_points=tuple(
            CompetitiveBidPoint(
                effective_bid=bid,
                win_probability=probability,
                win_delta=float(bid),
                expected_surplus=probability * bid,
            )
            for bid, probability in enumerate((0.1, 0.2, 0.3, 0.4))
        ),
    )


def _opponent_report(
    game_count: int,
    *,
    model_config_digest: str = "a" * 64,
) -> DecisionReport:
    games = tuple(_game(game_index=index, decision_counts=(1, 0, 0)) for index in range(game_count))
    explanation = _opponent_aware_explanation(
        model_config_digest=model_config_digest,
    )
    traces = tuple(
        replace(
            ordinary_trace,
            context=replace(ordinary_trace.context, cash_by_seat=(3, 15, 15)),
            explanation=explanation,
        )
        for game in games
        for ordinary_trace in (_trace(game, seat=0),)
    )
    return _build(traces, games)


def test_opponent_model_summary_uses_plain_aggregate_labels_without_identifiers() -> None:
    report = _opponent_report(30)

    rendered = render_opponent_model_summary(decision_report=report)
    payload = json.loads(rendered.summary_json)

    assert payload["schema_version"] == 1
    assert payload["report_kind"] == "public_opponent_bid_model_summary"
    assert payload["privacy"] == {
        "aggregation": "model_configuration_cohort",
        "minimum_distinct_games": 30,
        "contains_game_or_decision_identifiers": False,
        "contains_per_decision_tables": False,
    }
    summary = payload["model_summaries"][0]
    assert summary["cohort"] == {
        "distinct_game_count": 30,
        "decision_count": 30,
        "opponent_forecast_count": 60,
    }
    assert summary["chosen_actions"] == {
        "pass_count": 0,
        "positive_bid_count": 30,
    }
    predicted_bids = summary["predicted_opponent_bid_distribution"]
    assert [item["effective_bid"] for item in predicted_bids] == list(range(16))
    assert sum(item["mean_predicted_probability"] for item in predicted_bids) == 1.0
    competitive_bids = summary["expected_surplus_by_legal_bid"]
    assert [item["effective_bid"] for item in competitive_bids] == [0, 1, 2, 3]
    assert [item["chosen_count"] for item in competitive_bids] == [0, 0, 0, 30]
    assert competitive_bids[3]["mean_expected_surplus"] == pytest.approx(1.2)

    assert "### Predicted opponent bids" in rendered.summary_markdown
    assert "### Expected surplus by legal bid" in rendered.summary_markdown
    assert "### Chosen actions" in rendered.summary_markdown
    assert "Pass (effective bid 0)" in rendered.summary_markdown
    combined = rendered.summary_json + rendered.summary_markdown
    for forbidden in (
        "game_index",
        "step_index",
        "seat",
        "probabilities_by_amount",
        "public_history",
    ):
        assert forbidden not in combined
    assert '"seed"' not in rendered.summary_json


def test_opponent_model_summary_fails_closed_below_thirty_distinct_games() -> None:
    report = _opponent_report(29)

    with pytest.raises(DecisionAnalysisError, match="29 distinct games; at least 30"):
        render_opponent_model_summary(decision_report=report)


def test_opponent_model_summary_counts_games_instead_of_decisions_for_privacy() -> None:
    game = _game(decision_counts=(30, 0, 0))
    explanation = _opponent_aware_explanation()
    traces = tuple(
        replace(
            ordinary_trace,
            context=replace(ordinary_trace.context, cash_by_seat=(3, 15, 15)),
            explanation=explanation,
        )
        for step_index in range(30)
        for ordinary_trace in (_trace(game, seat=0, step_index=step_index),)
    )
    report = _build(traces, (game,))

    with pytest.raises(DecisionAnalysisError, match="1 distinct games; at least 30"):
        render_opponent_model_summary(decision_report=report)


def test_local_trace_json_retains_complete_public_opponent_reasoning() -> None:
    report = _opponent_report(30)

    trace_payload = json.loads(
        render_decision_artifacts(decision_report=report).decision_traces_jsonl.splitlines()[0]
    )
    explanation = trace_payload["explanation"]

    assert trace_payload["schema_version"] == 2
    assert explanation["opponent_distributions"][0]["probabilities_by_amount"] == [
        1.0,
        *([0.0] * 15),
    ]
    assert [
        point["expected_surplus"] for point in explanation["competitive_bid_points"]
    ] == pytest.approx([0.0, 0.2, 0.6, 1.2])
    assert trace_payload["selected_action"] == {"action_kind": "submitBid", "value": 3}


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
