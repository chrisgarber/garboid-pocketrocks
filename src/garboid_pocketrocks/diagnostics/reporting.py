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
    from garboid_pocketrocks.diagnostics.analysis import DecisionReport, DecisionSlice

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

    ordered_slices = sorted(decision_report.slices, key=_slice_sort_key)
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
        decision_slices_csv=_render_decision_slices(ordered_slices),
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


def _render_decision_slices(slices: Sequence[DecisionSlice]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=_SLICE_FIELDS)
    writer.writeheader()
    for item in slices:
        if not math.isfinite(item.eventual_normalized_finish_sum):
            raise ValueError("Decision artifacts must contain only finite slice values.")
        writer.writerow(_slice_payload(item))
    return stream.getvalue()


def _slice_payload(item: DecisionSlice) -> dict[str, object]:
    return {
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
