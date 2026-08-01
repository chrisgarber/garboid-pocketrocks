"""Deterministic, privacy-safe rendering for decision diagnostic artifacts."""

from __future__ import annotations

import csv
import io
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from garboid_pocketrocks.diagnostics.game_detail import PublicGameDetail
from garboid_pocketrocks.diagnostics.trace import decision_trace_payload
from garboid_pocketrocks.simulator.monte_carlo import GameSummary
from garboid_pocketrocks.simulator.session import SessionScore

if TYPE_CHECKING:
    from garboid_pocketrocks.diagnostics.analysis import (
        DecisionReport,
        DecisionSlice,
        PhaseOutcome,
    )

GAME_SUMMARIES_NAME = "game-summaries.jsonl"
GAME_DETAILS_NAME = "game-details.jsonl"
DECISION_TRACES_NAME = "decision-traces.jsonl"
DECISION_SLICES_NAME = "decision-slices.csv"

_SLICE_FIELDS = (
    "bot_name",
    "bot_id",
    "game_phase",
    "chart",
    "player_count",
    "decision_kind",
    "auction_action",
    "selected_action_kind",
    "future_biddable_resources",
    "total_biddable_resources",
    "actor_owned_objectives",
    "opponent_owned_objectives",
    "unclaimed_objectives",
    "seat",
    "opponent_bot_ids",
    "decision_count",
    "pass_count",
    "selected_value_count",
    "selected_value_sum",
    "eventual_final_money_sum",
    "eventual_normalized_finish_sum",
    "outright_win_decision_count",
    "tied_first_decision_count",
    "decisions_from_faulted_game_seat",
)
_SLICE_FIELDS_V2 = (
    "bot_name",
    "bot_id",
    "game_phase",
    "selected_expert_phase",
    *_SLICE_FIELDS[3:],
)
_EXPERT_PHASES = frozenset(("early", "middle", "late"))
_SLICE_INTEGER_FIELDS = (
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
)
_PHASE_OUTCOME_INTEGER_FIELDS = (
    "decision_count",
    "eventual_final_money_sum",
    "outright_win_decision_count",
    "tied_first_decision_count",
    "decisions_from_faulted_game_seat",
)
_RECONCILIATION_INTEGER_FIELDS = (
    "game_count",
    "game_seat_count",
    "trace_decision_count",
    "game_summary_decision_count",
    "slice_decision_count",
    "selected_expert_decision_count",
)


@dataclass(frozen=True, slots=True)
class RenderedDecisionArtifacts:
    """In-memory diagnostic contents ready for one transactional write."""

    game_summaries_jsonl: str
    game_details_jsonl: str
    decision_traces_jsonl: str
    decision_slices_csv: str

    def named_contents(self) -> tuple[tuple[str, str], ...]:
        """Return stable artifact names paired with their rendered contents."""

        return (
            (GAME_SUMMARIES_NAME, self.game_summaries_jsonl),
            (GAME_DETAILS_NAME, self.game_details_jsonl),
            (DECISION_TRACES_NAME, self.decision_traces_jsonl),
            (DECISION_SLICES_NAME, self.decision_slices_csv),
        )


def render_decision_artifacts(
    *,
    decision_report: DecisionReport,
) -> RenderedDecisionArtifacts:
    """Render canonical public diagnostics without seeds or hidden game state."""

    _validate_report_schema(decision_report)
    return RenderedDecisionArtifacts(
        game_summaries_jsonl=_render_json_lines(
            _game_summary_payload(summary) for summary in decision_report.game_summaries
        ),
        game_details_jsonl=_render_json_lines(
            _game_detail_payload(detail) for detail in decision_report.game_details
        ),
        decision_traces_jsonl=_render_json_lines(
            decision_trace_payload(trace) for trace in decision_report.decision_traces
        ),
        decision_slices_csv=render_decision_slices_csv(
            decision_report.slices,
            schema_version=decision_report.schema_version,
        ),
    )


def _game_detail_payload(detail: PublicGameDetail) -> dict[str, object]:
    return {
        "schema_version": 1,
        "game_index": detail.game_index,
        "chart": detail.chart,
        "player_count": detail.player_count,
        "bot_names": list(detail.bot_names),
        "bot_ids": list(detail.bot_ids),
        "value_chart": list(detail.value_chart),
        "turns": [
            {
                "turn_index": turn.turn_index,
                "action": turn.action,
                "raw_bids": list(turn.raw_bids),
                "effective_bids": list(turn.effective_bids),
                "winner_seat": turn.winner_seat,
                "paid": turn.paid,
                "bundle_suits": list(turn.bundle_suits),
                "claimed_objective_ids": list(turn.claimed_objective_ids),
            }
            for turn in detail.turns
        ],
        "scores": [
            {
                "seat": score.seat,
                "cash": score.cash,
                "items_value": score.items_value,
                "objectives_value": score.objectives_value,
                "investments_value": score.investments_value,
                "loans_value": score.loans_value,
                "total": score.total,
            }
            for score in detail.scores
        ],
    }


def _game_summary_payload(summary: GameSummary) -> dict[str, object]:
    return {
        "game_index": summary.game_index,
        "player_count": summary.player_count,
        "ruleset_name": summary.ruleset_name,
        "bot_names": list(summary.bot_names),
        "bot_ids": list(summary.bot_ids),
        "scores": [
            _session_score_payload(score)
            for score in sorted(summary.scores, key=lambda item: item.seat)
        ],
        "decision_counts": list(summary.decision_counts),
        "fault_counts": list(summary.fault_counts),
    }


def _session_score_payload(score: SessionScore) -> dict[str, object]:
    return {
        "seat": score.seat,
        "final_money": score.final_money,
        "rank": score.rank,
    }


def _render_json_lines(payloads: Iterable[Mapping[str, object]]) -> str:
    try:
        return "".join(
            json.dumps(
                payload,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
            for payload in payloads
        )
    except ValueError as error:
        raise ValueError("Decision artifacts must contain only finite JSON numbers.") from error


def _slice_sort_key(item: DecisionSlice) -> tuple[object, ...]:
    return (
        item.bot_name,
        item.bot_id,
        item.game_phase,
        item.selected_expert_phase or "",
        item.chart,
        item.player_count,
        item.decision_kind,
        item.auction_action,
        item.selected_action_kind,
        item.future_biddable_resources,
        item.total_biddable_resources,
        item.actor_owned_objectives,
        item.opponent_owned_objectives,
        item.unclaimed_objectives,
        item.seat,
        item.opponent_bot_ids,
    )


def render_decision_slices_csv(
    slices: Sequence[DecisionSlice],
    *,
    schema_version: int,
) -> str:
    """Render canonical slice rows for the report schema that owns them."""

    _validate_slice_schema(slices, schema_version=schema_version)
    fields: tuple[str, ...]
    if schema_version == 1:
        fields = _SLICE_FIELDS
    elif schema_version == 2:
        fields = _SLICE_FIELDS_V2
    else:
        raise ValueError(f"Unsupported decision report schema version {schema_version}.")

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for item in sorted(slices, key=_slice_sort_key):
        if not math.isfinite(item.eventual_normalized_finish_sum):
            raise ValueError("Decision artifacts must contain only finite slice values.")
        writer.writerow(
            _slice_payload(
                item,
                include_selected_expert_phase=schema_version == 2,
            )
        )
    return stream.getvalue()


def _validate_report_schema(decision_report: DecisionReport) -> None:
    from garboid_pocketrocks.diagnostics.analysis import (
        _canonical_decision_evidence,
    )

    schema_version = _validated_schema_version(decision_report.schema_version)
    _validate_slice_schema(
        decision_report.slices,
        schema_version=schema_version,
    )
    _validate_phase_outcome_field_types(decision_report.phase_outcomes)
    _require_exact_integer_fields(
        decision_report.reconciliation,
        _RECONCILIATION_INTEGER_FIELDS,
        label="Decision reconciliation",
    )
    canonical = _canonical_decision_evidence(
        decision_report.decision_traces,
        game_summaries=decision_report.game_summaries,
        game_details=decision_report.game_details,
    )
    expected_schema_version = 2 if canonical.reconciliation.selected_expert_decision_count else 1
    if schema_version != expected_schema_version:
        raise ValueError("Decision report schema version does not agree with its decision traces.")

    if decision_report.game_summaries != canonical.game_summaries:
        raise ValueError("Decision report game summaries are not in canonical order.")
    if decision_report.game_details != canonical.game_details:
        raise ValueError("Decision report game details are not in canonical order.")
    if decision_report.decision_traces != canonical.decision_traces:
        raise ValueError("Decision report decision traces are not in canonical order.")
    if decision_report.slices != canonical.slices:
        raise ValueError("Decision report slices do not match canonical public evidence.")
    if decision_report.phase_outcomes != canonical.phase_outcomes:
        raise ValueError("Decision report phase outcomes do not match canonical public evidence.")
    if decision_report.reconciliation != canonical.reconciliation:
        raise ValueError("Decision report reconciliation does not match canonical public evidence.")


def _validate_slice_schema(
    slices: Sequence[DecisionSlice],
    *,
    schema_version: int,
) -> None:
    validated_schema_version = _validated_schema_version(schema_version)
    _validate_slice_field_types(slices)
    has_selected_expert_phase = False
    for item in slices:
        phase = item.selected_expert_phase
        if phase is not None and (type(phase) is not str or phase not in _EXPERT_PHASES):
            raise ValueError(
                "Decision slice selected expert phase must be null, early, middle, or late."
            )
        has_selected_expert_phase |= phase is not None
    if validated_schema_version == 1 and has_selected_expert_phase:
        raise ValueError("Decision slice schema v1 cannot contain an expert phase.")
    if validated_schema_version == 2 and not has_selected_expert_phase:
        raise ValueError("Decision slice schema v2 requires an expert phase.")


def _validate_slice_field_types(slices: Sequence[DecisionSlice]) -> None:
    for index, item in enumerate(slices):
        label = f"Decision slice {index}"
        _require_exact_integer_fields(item, _SLICE_INTEGER_FIELDS, label=label)
        _require_finite_float(
            item.eventual_normalized_finish_sum,
            field_name=f"{label}.eventual_normalized_finish_sum",
        )


def _validate_phase_outcome_field_types(
    outcomes: Sequence[PhaseOutcome],
) -> None:
    for index, outcome in enumerate(outcomes):
        label = f"Phase outcome {index}"
        _require_exact_integer_fields(
            outcome,
            _PHASE_OUTCOME_INTEGER_FIELDS,
            label=label,
        )
        _require_finite_float(
            outcome.eventual_normalized_finish_sum,
            field_name=f"{label}.eventual_normalized_finish_sum",
        )


def _require_exact_integer_fields(
    item: object,
    field_names: Sequence[str],
    *,
    label: str,
) -> None:
    for field_name in field_names:
        if type(getattr(item, field_name)) is not int:
            raise ValueError(f"{label}.{field_name} must be an exact integer.")


def _require_finite_float(value: object, *, field_name: str) -> None:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{field_name} must be a finite float.")


def _validated_schema_version(schema_version: object) -> int:
    if type(schema_version) is not int or schema_version not in (1, 2):
        raise ValueError(f"Unsupported decision report schema version {schema_version!r}.")
    return schema_version


def _slice_payload(
    item: DecisionSlice,
    *,
    include_selected_expert_phase: bool,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "bot_name": item.bot_name,
        "bot_id": item.bot_id,
        "game_phase": item.game_phase,
        "chart": item.chart,
        "player_count": item.player_count,
        "decision_kind": item.decision_kind,
        "auction_action": item.auction_action,
        "selected_action_kind": item.selected_action_kind,
        "future_biddable_resources": item.future_biddable_resources,
        "total_biddable_resources": item.total_biddable_resources,
        "actor_owned_objectives": item.actor_owned_objectives,
        "opponent_owned_objectives": item.opponent_owned_objectives,
        "unclaimed_objectives": item.unclaimed_objectives,
        "seat": item.seat,
        "opponent_bot_ids": json.dumps(
            item.opponent_bot_ids,
            separators=(",", ":"),
        ),
        "decision_count": item.decision_count,
        "pass_count": item.pass_count,
        "selected_value_count": item.selected_value_count,
        "selected_value_sum": item.selected_value_sum,
        "eventual_final_money_sum": item.eventual_final_money_sum,
        "eventual_normalized_finish_sum": item.eventual_normalized_finish_sum,
        "outright_win_decision_count": item.outright_win_decision_count,
        "tied_first_decision_count": item.tied_first_decision_count,
        "decisions_from_faulted_game_seat": item.decisions_from_faulted_game_seat,
    }
    if include_selected_expert_phase:
        payload["selected_expert_phase"] = (
            "" if item.selected_expert_phase is None else item.selected_expert_phase
        )
    return payload
