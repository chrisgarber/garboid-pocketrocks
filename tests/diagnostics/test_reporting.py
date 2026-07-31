from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import replace
from typing import Any, cast

import pytest

from garboid_pocketrocks.diagnostics.reporting import (
    render_decision_artifacts,
    render_decision_slices_csv,
)

from .test_analysis import _build, _game, _phase_aware_trace, _trace


def test_decision_artifacts_are_sanitized_canonical_and_order_independent() -> None:
    first_game = _game(game_index=0, chart="A", decision_counts=(1, 0, 0))
    second_game = _game(game_index=1, chart="E", decision_counts=(1, 0, 0))
    first_trace = _trace(first_game, seat=0)
    second_trace = _trace(second_game, seat=0)
    report = _build((second_trace, first_trace), (second_game, first_game))
    forward_report = _build((first_trace, second_trace), (first_game, second_game))

    canonical = render_decision_artifacts(decision_report=report)
    forward_sources = render_decision_artifacts(decision_report=forward_report)

    assert canonical == forward_sources
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


def test_nonfinite_decision_trace_fails_json_rendering() -> None:
    game = _game(decision_counts=(1, 0, 0))
    trace = _phase_aware_trace(
        game,
        seat=0,
        future_biddable_resources=10,
    )
    report = _build((trace,), (game,))
    assert trace.explanation is not None
    object.__setattr__(trace.explanation, "resource_value", math.inf)

    with pytest.raises(ValueError, match="finite JSON"):
        render_decision_artifacts(decision_report=report)


def test_v1_decision_slice_csv_bytes_remain_unchanged() -> None:
    game = _game(decision_counts=(1, 0, 0))
    report = _build((_trace(game, seat=0),), (game,))
    expected = (
        "bot_name,bot_id,game_phase,chart,player_count,decision_kind,"
        "auction_action,selected_action_kind,future_biddable_resources,"
        "total_biddable_resources,actor_owned_objectives,"
        "opponent_owned_objectives,unclaimed_objectives,seat,opponent_bot_ids,"
        "decision_count,pass_count,selected_value_count,selected_value_sum,"
        "eventual_final_money_sum,eventual_normalized_finish_sum,"
        "outright_win_decision_count,tied_first_decision_count,"
        "decisions_from_faulted_game_seat\r\n"
        "alpha,alpha,early,A,3,submitBid,Auction1,submitBid,14,15,0,0,2,0,"
        '"[""beta"",""gamma""]",1,0,1,3,30,1.0,1,0,0\r\n'
    )

    assert report.schema_version == 1
    assert (
        render_decision_slices_csv(
            report.slices,
            schema_version=report.schema_version,
        )
        == expected
    )
    assert render_decision_artifacts(decision_report=report).decision_slices_csv == expected


def test_v2_csv_adds_nullable_selected_expert_phase_and_is_deterministic() -> None:
    game = _game(decision_counts=(1, 1, 0))
    phase_aware = _phase_aware_trace(
        game,
        seat=0,
        future_biddable_resources=10,
    )
    ordinary = _trace(game, seat=1)
    report = _build((ordinary, phase_aware), (game,))

    rendered = render_decision_slices_csv(
        tuple(reversed(report.slices)),
        schema_version=report.schema_version,
    )
    canonical = render_decision_artifacts(decision_report=report).decision_slices_csv
    rows = tuple(csv.DictReader(io.StringIO(rendered)))

    assert rendered == canonical
    assert tuple(rows[0])[:5] == (
        "bot_name",
        "bot_id",
        "game_phase",
        "selected_expert_phase",
        "chart",
    )
    assert {row["bot_id"]: row["selected_expert_phase"] for row in rows} == {
        "alpha": "early",
        "beta": "",
    }


@pytest.mark.parametrize("schema_version", (0, 3))
def test_slice_csv_rejects_unknown_report_schema(schema_version: int) -> None:
    game = _game(decision_counts=(1, 0, 0))
    report = _build((_trace(game, seat=0),), (game,))

    with pytest.raises(ValueError, match="schema version"):
        render_decision_slices_csv(
            report.slices,
            schema_version=schema_version,
        )


@pytest.mark.parametrize("schema_version", (True, False, 1.0, 2.0))
def test_slice_csv_requires_an_exact_integer_schema_version(
    schema_version: object,
) -> None:
    game = _game(decision_counts=(1, 0, 0))
    report = _build((_trace(game, seat=0),), (game,))

    with pytest.raises(ValueError, match="schema version"):
        render_decision_slices_csv(
            report.slices,
            schema_version=cast(Any, schema_version),
        )


def test_slice_csv_rejects_expert_content_under_schema_v1() -> None:
    game = _game(decision_counts=(1, 0, 0))
    report = _build(
        (_phase_aware_trace(game, seat=0, future_biddable_resources=10),),
        (game,),
    )

    with pytest.raises(ValueError, match="schema v1|expert phase"):
        render_decision_slices_csv(report.slices, schema_version=1)


def test_slice_csv_rejects_schema_v2_without_expert_content() -> None:
    game = _game(decision_counts=(1, 0, 0))
    report = _build((_trace(game, seat=0),), (game,))

    with pytest.raises(ValueError, match="schema v2|expert phase"):
        render_decision_slices_csv(report.slices, schema_version=2)


@pytest.mark.parametrize("selected_expert_phase", ("bogus", "", False, 0, ["early"]))
def test_slice_csv_rejects_an_invalid_nullable_expert_phase(
    selected_expert_phase: object,
) -> None:
    game = _game(decision_counts=(1, 0, 0))
    report = _build(
        (_phase_aware_trace(game, seat=0, future_biddable_resources=10),),
        (game,),
    )
    invalid_slice = replace(
        report.slices[0],
        selected_expert_phase=cast(Any, selected_expert_phase),
    )

    with pytest.raises(ValueError, match="expert phase"):
        render_decision_slices_csv((invalid_slice,), schema_version=2)


def test_artifact_rendering_rejects_report_trace_schema_disagreement() -> None:
    game = _game(decision_counts=(1, 0, 0))
    phase_report = _build(
        (_phase_aware_trace(game, seat=0, future_biddable_resources=10),),
        (game,),
    )
    legacy_report = _build((_trace(game, seat=0),), (game,))

    with pytest.raises(ValueError, match="schema|trace"):
        render_decision_artifacts(
            decision_report=replace(phase_report, schema_version=1),
        )
    with pytest.raises(ValueError, match="schema|trace"):
        render_decision_artifacts(
            decision_report=replace(legacy_report, schema_version=2),
        )


def test_artifact_rendering_rejects_report_slice_schema_disagreement() -> None:
    game = _game(decision_counts=(1, 0, 0))
    report = _build(
        (_phase_aware_trace(game, seat=0, future_biddable_resources=10),),
        (game,),
    )
    missing_phase = replace(report.slices[0], selected_expert_phase=None)

    with pytest.raises(ValueError, match="schema v2|expert phase"):
        render_decision_artifacts(
            decision_report=replace(report, slices=(missing_phase,)),
        )


def test_artifact_rendering_rejects_a_relabeled_expert_slice() -> None:
    game = _game(decision_counts=(1, 0, 0))
    report = _build(
        (_phase_aware_trace(game, seat=0, future_biddable_resources=10),),
        (game,),
    )
    relabeled = replace(report.slices[0], selected_expert_phase="middle")

    with pytest.raises(ValueError, match="canonical|slice"):
        render_decision_artifacts(
            decision_report=replace(report, slices=(relabeled,)),
        )


def test_artifact_rendering_rejects_tampered_phase_outcome_money() -> None:
    game = _game(decision_counts=(1, 0, 0))
    report = _build(
        (_phase_aware_trace(game, seat=0, future_biddable_resources=10),),
        (game,),
    )
    tampered_outcome = replace(
        report.phase_outcomes[0],
        eventual_final_money_sum=999,
    )

    with pytest.raises(ValueError, match="canonical|phase outcome"):
        render_decision_artifacts(
            decision_report=replace(
                report,
                phase_outcomes=(tampered_outcome, *report.phase_outcomes[1:]),
            ),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("game_count", 999),
        ("game_seat_count", 999),
        ("trace_decision_count", 999),
        ("game_summary_decision_count", 999),
        ("slice_decision_count", 999),
        ("selected_expert_decision_count", 999),
    ),
)
def test_artifact_rendering_rejects_tampered_reconciliation(
    field: str,
    value: int,
) -> None:
    game = _game(decision_counts=(1, 0, 0))
    report = _build(
        (_phase_aware_trace(game, seat=0, future_biddable_resources=10),),
        (game,),
    )

    with pytest.raises(ValueError, match="canonical|reconciliation"):
        render_decision_artifacts(
            decision_report=replace(
                report,
                reconciliation=replace(report.reconciliation, **{field: value}),
            ),
        )


def test_artifact_rendering_rejects_tampered_ordinary_slice_count() -> None:
    game = _game(decision_counts=(1, 0, 0))
    report = _build((_trace(game, seat=0),), (game,))
    tampered_slice = replace(report.slices[0], decision_count=999)

    with pytest.raises(ValueError, match="canonical|slice"):
        render_decision_artifacts(
            decision_report=replace(report, slices=(tampered_slice,)),
        )


@pytest.mark.parametrize(
    "field",
    (
        "player_count",
        "future_biddable_resources",
        "total_biddable_resources",
        "actor_owned_objectives",
        "opponent_owned_objectives",
        "unclaimed_objectives",
        "seat",
        "decision_count",
        "pass_count",
        "selected_value_count",
        "selected_value_sum",
        "eventual_final_money_sum",
        "outright_win_decision_count",
        "tied_first_decision_count",
        "decisions_from_faulted_game_seat",
    ),
)
def test_artifact_rendering_rejects_float_values_for_slice_integer_fields(
    field: str,
) -> None:
    game = _game(decision_counts=(1, 0, 0))
    report = _build((_trace(game, seat=0),), (game,))
    item = report.slices[0]
    tampered_slice = replace(
        item,
        **cast(Any, {field: float(getattr(item, field))}),
    )

    with pytest.raises(ValueError, match="exact integer"):
        render_decision_artifacts(
            decision_report=replace(report, slices=(tampered_slice,)),
        )


@pytest.mark.parametrize("value", (True, 1.0))
def test_artifact_rendering_rejects_bool_or_float_slice_counts(
    value: object,
) -> None:
    game = _game(decision_counts=(1, 0, 0))
    report = _build((_trace(game, seat=0),), (game,))
    tampered_slice = replace(
        report.slices[0],
        decision_count=cast(Any, value),
    )

    with pytest.raises(ValueError, match="exact integer"):
        render_decision_artifacts(
            decision_report=replace(report, slices=(tampered_slice,)),
        )


@pytest.mark.parametrize("value", (1, math.inf, math.nan))
def test_artifact_rendering_requires_a_finite_exact_slice_float(
    value: object,
) -> None:
    game = _game(decision_counts=(1, 0, 0))
    report = _build((_trace(game, seat=0),), (game,))
    tampered_slice = replace(
        report.slices[0],
        eventual_normalized_finish_sum=cast(Any, value),
    )

    with pytest.raises(ValueError, match="finite float"):
        render_decision_artifacts(
            decision_report=replace(report, slices=(tampered_slice,)),
        )


@pytest.mark.parametrize(
    "field",
    (
        "decision_count",
        "eventual_final_money_sum",
        "outright_win_decision_count",
        "tied_first_decision_count",
        "decisions_from_faulted_game_seat",
    ),
)
def test_artifact_rendering_rejects_float_values_for_phase_outcome_integer_fields(
    field: str,
) -> None:
    game = _game(decision_counts=(1, 0, 0))
    report = _build(
        (_phase_aware_trace(game, seat=0, future_biddable_resources=10),),
        (game,),
    )
    outcome = report.phase_outcomes[0]
    tampered_outcome = replace(
        outcome,
        **cast(Any, {field: float(getattr(outcome, field))}),
    )

    with pytest.raises(ValueError, match="exact integer"):
        render_decision_artifacts(
            decision_report=replace(
                report,
                phase_outcomes=(tampered_outcome, *report.phase_outcomes[1:]),
            ),
        )


@pytest.mark.parametrize("value", (True, 1.0))
def test_artifact_rendering_rejects_bool_or_float_phase_outcome_counts(
    value: object,
) -> None:
    game = _game(decision_counts=(1, 0, 0))
    report = _build(
        (_phase_aware_trace(game, seat=0, future_biddable_resources=10),),
        (game,),
    )
    tampered_outcome = replace(
        report.phase_outcomes[0],
        decision_count=cast(Any, value),
    )

    with pytest.raises(ValueError, match="exact integer"):
        render_decision_artifacts(
            decision_report=replace(
                report,
                phase_outcomes=(tampered_outcome, *report.phase_outcomes[1:]),
            ),
        )


@pytest.mark.parametrize("value", (1, math.inf, math.nan))
def test_artifact_rendering_requires_a_finite_exact_phase_outcome_float(
    value: object,
) -> None:
    game = _game(decision_counts=(1, 0, 0))
    report = _build(
        (_phase_aware_trace(game, seat=0, future_biddable_resources=10),),
        (game,),
    )
    tampered_outcome = replace(
        report.phase_outcomes[0],
        eventual_normalized_finish_sum=cast(Any, value),
    )

    with pytest.raises(ValueError, match="finite float"):
        render_decision_artifacts(
            decision_report=replace(
                report,
                phase_outcomes=(tampered_outcome, *report.phase_outcomes[1:]),
            ),
        )


@pytest.mark.parametrize(
    "field",
    (
        "game_count",
        "game_seat_count",
        "trace_decision_count",
        "game_summary_decision_count",
        "slice_decision_count",
        "selected_expert_decision_count",
    ),
)
def test_artifact_rendering_rejects_float_values_for_reconciliation_integer_fields(
    field: str,
) -> None:
    game = _game(decision_counts=(1, 0, 0))
    report = _build((_trace(game, seat=0),), (game,))
    reconciliation = report.reconciliation
    tampered_reconciliation = replace(
        reconciliation,
        **cast(Any, {field: float(getattr(reconciliation, field))}),
    )

    with pytest.raises(ValueError, match="exact integer"):
        render_decision_artifacts(
            decision_report=replace(
                report,
                reconciliation=tampered_reconciliation,
            ),
        )


@pytest.mark.parametrize("value", (True, 1.0))
def test_artifact_rendering_rejects_bool_or_float_reconciliation_counts(
    value: object,
) -> None:
    game = _game(decision_counts=(1, 0, 0))
    report = _build((_trace(game, seat=0),), (game,))
    tampered_reconciliation = replace(
        report.reconciliation,
        trace_decision_count=cast(Any, value),
    )

    with pytest.raises(ValueError, match="exact integer"):
        render_decision_artifacts(
            decision_report=replace(
                report,
                reconciliation=tampered_reconciliation,
            ),
        )
