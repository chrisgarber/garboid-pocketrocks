"""Additive decision slices with strict reconciliation to ordinary results."""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from pocketrocks.sim.constants import ACTION_WIRE_IDS

from garboid_pocketrocks.diagnostics.game_detail import PublicGameDetail
from garboid_pocketrocks.diagnostics.trace import DecisionTrace
from garboid_pocketrocks.knowledge import (
    canonical_knowledge,
    ruleset_name,
    value_chart_from_ruleset_name,
)
from garboid_pocketrocks.simulator.monte_carlo import (
    BotStatistics,
    GameSummary,
)
from garboid_pocketrocks.simulator.session import SessionScore
from garboid_pocketrocks.tournament.analysis import (
    ConditionStatistics,
    TournamentAnalysis,
    TournamentBotRow,
)

type GamePhase = Literal["early", "middle", "late"]

_ACTION_NAME_BY_ID = {wire_id: name for name, wire_id in ACTION_WIRE_IDS.items()}


class DecisionAnalysisError(ValueError):
    """Raised when diagnostic evidence disagrees with ordinary simulation results."""


@dataclass(frozen=True, slots=True)
class DecisionSlice:
    """One observed combination of public decision conditions and additive measures."""

    bot_name: str
    bot_id: str
    game_phase: GamePhase
    chart: str
    player_count: int
    decision_kind: str
    auction_action: str
    selected_action_kind: str
    future_biddable_resources: int
    total_biddable_resources: int
    actor_owned_objectives: int
    opponent_owned_objectives: int
    unclaimed_objectives: int
    seat: int
    opponent_bot_ids: tuple[str, ...]
    decision_count: int
    pass_count: int
    selected_value_count: int
    selected_value_sum: int
    eventual_final_money_sum: int
    eventual_normalized_finish_sum: float
    outright_win_decision_count: int
    tied_first_decision_count: int
    decisions_from_faulted_game_seat: int


@dataclass(frozen=True, slots=True)
class DecisionReconciliation:
    """Totals proving traces, slices, games, and tournament results agree."""

    game_count: int
    game_seat_count: int
    trace_decision_count: int
    game_summary_decision_count: int
    slice_decision_count: int


@dataclass(frozen=True, slots=True)
class DecisionReport:
    """Validated additive diagnostic results for one tournament."""

    schema_version: int
    game_summaries: tuple[GameSummary, ...]
    game_details: tuple[PublicGameDetail, ...]
    decision_traces: tuple[DecisionTrace, ...]
    slices: tuple[DecisionSlice, ...]
    reconciliation: DecisionReconciliation


@dataclass(frozen=True, order=True, slots=True)
class _SliceKey:
    bot_name: str
    bot_id: str
    game_phase: GamePhase
    chart: str
    player_count: int
    decision_kind: str
    auction_action: str
    selected_action_kind: str
    future_biddable_resources: int
    total_biddable_resources: int
    actor_owned_objectives: int
    opponent_owned_objectives: int
    unclaimed_objectives: int
    seat: int
    opponent_bot_ids: tuple[str, ...]


@dataclass(slots=True)
class _SliceAccumulator:
    decision_count: int = 0
    pass_count: int = 0
    selected_value_count: int = 0
    selected_value_sum: int = 0
    eventual_final_money_sum: int = 0
    eventual_normalized_finish_values: list[float] | None = None
    outright_win_decision_count: int = 0
    tied_first_decision_count: int = 0
    decisions_from_faulted_game_seat: int = 0

    def __post_init__(self) -> None:
        if self.eventual_normalized_finish_values is None:
            self.eventual_normalized_finish_values = []


@dataclass(frozen=True, slots=True)
class _BotAggregate:
    bot_name: str
    games: int
    outright_wins: int
    first_place_ties: int
    rank_counts: tuple[int, ...]
    final_money_samples: tuple[int, ...]
    score_margins: tuple[int, ...]
    decision_counts: tuple[int, ...]
    faults: int
    normalized_finishes: tuple[float, ...]
    winning_money: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _ConditionAggregate:
    games: int
    outright_wins: int
    first_place_ties: int
    normalized_finishes: tuple[float, ...]
    final_money: tuple[int, ...]


def build_decision_report(
    traces: Sequence[DecisionTrace],
    *,
    game_summaries: Sequence[GameSummary],
    game_details: Sequence[PublicGameDetail] = (),
    bot_statistics: Sequence[BotStatistics],
    tournament_analysis: TournamentAnalysis,
) -> DecisionReport:
    """Validate every source and group traces into deterministic additive slices."""

    ordered_games = tuple(sorted(game_summaries, key=lambda game: game.game_index))
    games_by_index = _validated_games_by_index(ordered_games)
    ordered_details = tuple(sorted(game_details, key=lambda game: game.game_index))
    _validate_game_details(ordered_details, games_by_index=games_by_index)
    ordered_traces = tuple(
        sorted(
            traces,
            key=lambda trace: (
                -1 if trace.game_index is None else trace.game_index,
                trace.step_index,
                trace.seat,
            ),
        )
    )
    _require_unique_trace_keys(ordered_traces)

    traces_per_game_seat: Counter[tuple[int, int]] = Counter()
    slices: dict[_SliceKey, _SliceAccumulator] = {}
    for trace in ordered_traces:
        game = _matching_game(trace, games_by_index)
        _validate_trace_against_game(trace, game)
        traces_per_game_seat[game.game_index, trace.seat] += 1
        key = _slice_key(trace)
        accumulator = slices.setdefault(key, _SliceAccumulator())
        _record_slice(accumulator, trace, game)

    expected_decisions = _validate_decision_counts(
        ordered_games,
        traces_per_game_seat=traces_per_game_seat,
    )
    bot_aggregates, condition_aggregates = _ordinary_aggregates(ordered_games)
    _validate_bot_statistics(bot_statistics, expected=bot_aggregates)
    _validate_tournament_rows(tournament_analysis.rows, expected=bot_aggregates)
    _validate_condition_statistics(
        tournament_analysis.condition_statistics,
        expected=condition_aggregates,
    )

    frozen_slices = tuple(_freeze_slice(key, slices[key]) for key in sorted(slices))
    slice_decisions = sum(item.decision_count for item in frozen_slices)
    if slice_decisions != len(ordered_traces) or slice_decisions != expected_decisions:
        raise DecisionAnalysisError(
            "slice decision total does not match traces and game-summary decisions"
        )
    return DecisionReport(
        schema_version=1,
        game_summaries=ordered_games,
        game_details=ordered_details,
        decision_traces=ordered_traces,
        slices=frozen_slices,
        reconciliation=DecisionReconciliation(
            game_count=len(ordered_games),
            game_seat_count=sum(game.player_count for game in ordered_games),
            trace_decision_count=len(ordered_traces),
            game_summary_decision_count=expected_decisions,
            slice_decision_count=slice_decisions,
        ),
    )


def _validate_game_details(
    details: tuple[PublicGameDetail, ...],
    *,
    games_by_index: dict[int, GameSummary],
) -> None:
    """Require a complete, identity-consistent ledger when details are supplied."""

    if not details:
        return
    if tuple(detail.game_index for detail in details) != tuple(sorted(games_by_index)):
        raise DecisionAnalysisError("game details must cover every game summary exactly once")
    for detail in details:
        game = games_by_index[detail.game_index]
        if (
            detail.player_count != game.player_count
            or detail.bot_ids != game.bot_ids
            or detail.bot_names != game.bot_names
        ):
            raise DecisionAnalysisError(
                f"game detail {detail.game_index} does not match game-summary identities"
            )
        if tuple(score.seat for score in detail.scores) != tuple(range(game.player_count)):
            raise DecisionAnalysisError(
                f"game detail {detail.game_index} score rows do not cover every seat"
            )
        expected_money = {score.seat: score.final_money for score in game.scores}
        if any(score.total != expected_money[score.seat] for score in detail.scores):
            raise DecisionAnalysisError(
                f"game detail {detail.game_index} score totals disagree with game summary"
            )


def _validated_games_by_index(
    games: tuple[GameSummary, ...],
) -> dict[int, GameSummary]:
    games_by_index: dict[int, GameSummary] = {}
    for game in games:
        if game.game_index in games_by_index:
            raise DecisionAnalysisError(f"duplicate game summary index {game.game_index}")
        if (
            len(game.bot_names) != game.player_count
            or len(game.bot_ids) != game.player_count
            or len(game.decision_counts) != game.player_count
            or len(game.fault_counts) != game.player_count
        ):
            raise DecisionAnalysisError(
                f"game summary {game.game_index} seat data does not match player count"
            )
        score_seats = tuple(sorted(score.seat for score in game.scores))
        if score_seats != tuple(range(game.player_count)):
            raise DecisionAnalysisError(
                f"game summary {game.game_index} scores do not cover every seat"
            )
        if any(count < 0 for count in (*game.decision_counts, *game.fault_counts)):
            raise DecisionAnalysisError(f"game summary {game.game_index} contains a negative count")
        games_by_index[game.game_index] = game
    return games_by_index


def _require_unique_trace_keys(traces: tuple[DecisionTrace, ...]) -> None:
    seen: set[tuple[int | None, int, int]] = set()
    for trace in traces:
        key = (trace.game_index, trace.step_index, trace.seat)
        if key in seen:
            raise DecisionAnalysisError(f"duplicate decision trace key {key}")
        seen.add(key)


def _matching_game(
    trace: DecisionTrace,
    games_by_index: dict[int, GameSummary],
) -> GameSummary:
    if trace.game_index is None or trace.game_index not in games_by_index:
        raise DecisionAnalysisError(f"decision trace game {trace.game_index!r} has no game summary")
    return games_by_index[trace.game_index]


def _validate_trace_against_game(trace: DecisionTrace, game: GameSummary) -> None:
    label = f"decision trace game {game.game_index} step {trace.step_index} seat {trace.seat}"
    if trace.context.player_count != game.player_count:
        raise DecisionAnalysisError(f"{label} player count does not match game summary")
    if not 0 <= trace.seat < game.player_count:
        raise DecisionAnalysisError(f"{label} seat is outside the game")
    try:
        expected_ruleset = ruleset_name(
            trace.chart,
            objectives_enabled=bool(trace.context.objective_ids),
        )
    except ValueError as error:
        raise DecisionAnalysisError(f"{label} has an unknown chart {trace.chart!r}") from error
    if expected_ruleset != game.ruleset_name:
        raise DecisionAnalysisError(f"{label} chart does not match game summary")
    expected_knowledge = canonical_knowledge(
        game.player_count,
        value_chart=trace.chart,
        objectives_enabled=bool(trace.context.objective_ids),
    )
    if trace.context.value_chart != expected_knowledge.value_chart:
        raise DecisionAnalysisError(f"{label} public context chart does not match trace chart")
    if trace.bot_names_by_seat != game.bot_names or trace.bot_ids_by_seat != game.bot_ids:
        raise DecisionAnalysisError(f"{label} lineup identity does not match game summary")
    if trace.bot_name != game.bot_names[trace.seat] or trace.bot_id != game.bot_ids[trace.seat]:
        raise DecisionAnalysisError(f"{label} bot identity does not match game summary")

    scores_by_seat = {score.seat: score for score in game.scores}
    score = scores_by_seat[trace.seat]
    expected_tie = score.rank == 1 and sum(item.rank == 1 for item in game.scores) > 1
    if (
        trace.outcome.rank != score.rank
        or trace.outcome.final_money != score.final_money
        or trace.outcome.first_place_tied != expected_tie
    ):
        raise DecisionAnalysisError(f"{label} public outcome does not match game summary")
    if trace.selection_source == "fault_fallback" and game.fault_counts[trace.seat] == 0:
        raise DecisionAnalysisError(f"{label} fault fallback has no recorded game-seat fault")


def _validate_decision_counts(
    games: tuple[GameSummary, ...],
    *,
    traces_per_game_seat: Counter[tuple[int, int]],
) -> int:
    expected_total = 0
    for game in games:
        for seat, expected in enumerate(game.decision_counts):
            actual = traces_per_game_seat[game.game_index, seat]
            if actual != expected:
                raise DecisionAnalysisError(
                    f"game {game.game_index} seat {seat} decision count "
                    f"is {actual}, expected {expected}"
                )
            expected_total += expected
    return expected_total


def _slice_key(trace: DecisionTrace) -> _SliceKey:
    action_id = trace.context.current_action_id
    if action_id is None:
        raise DecisionAnalysisError("decision trace has no current auction action")
    try:
        auction_action = _ACTION_NAME_BY_ID[action_id]
    except KeyError as error:
        raise DecisionAnalysisError(
            f"decision trace has unknown current auction action {action_id!r}"
        ) from error
    future_resources, total_resources = _cash_horizon(trace)
    actor_objectives, opponent_objectives, unclaimed_objectives = _objective_state(trace)
    return _SliceKey(
        bot_name=trace.bot_name,
        bot_id=trace.bot_id,
        game_phase=_game_phase(trace.turn_index),
        chart=trace.chart,
        player_count=trace.context.player_count,
        decision_kind=trace.context.decision_kind,
        auction_action=auction_action,
        selected_action_kind=trace.selected_action.action_kind,
        future_biddable_resources=future_resources,
        total_biddable_resources=total_resources,
        actor_owned_objectives=actor_objectives,
        opponent_owned_objectives=opponent_objectives,
        unclaimed_objectives=unclaimed_objectives,
        seat=trace.seat,
        opponent_bot_ids=tuple(
            sorted(
                bot_id for seat, bot_id in enumerate(trace.bot_ids_by_seat) if seat != trace.seat
            )
        ),
    )


def _game_phase(turn_index: int) -> GamePhase:
    if turn_index < 0:
        raise DecisionAnalysisError("decision trace turn index must be nonnegative")
    one_based_turn = turn_index + 1
    if one_based_turn <= 5:
        return "early"
    if one_based_turn <= 12:
        return "middle"
    return "late"


def _cash_horizon(trace: DecisionTrace) -> tuple[int, int]:
    context = trace.context
    knowledge = canonical_knowledge(
        context.player_count,
        value_chart=trace.chart,
        objectives_enabled=bool(context.objective_ids),
    )
    total_biddable = sum(knowledge.resource_counts) - (
        context.player_count * knowledge.private_cards_per_player
    )
    already_won = sum(sum(row) for row in context.won_resource_counts_by_seat)
    currently_offered = (
        sum(resource_id > 0 for resource_id in context.current_resource_ids)
        if context.decision_kind == "submitBid"
        else 0
    )
    future_biddable = total_biddable - already_won - currently_offered
    if future_biddable < 0:
        raise DecisionAnalysisError("decision trace public resource counts exceed the ruleset")
    return future_biddable, total_biddable


def _objective_state(trace: DecisionTrace) -> tuple[int, int, int]:
    context = trace.context
    if len(context.owned_objective_ids_by_seat) != context.player_count:
        raise DecisionAnalysisError(
            "decision trace objective ownership does not match player count"
        )
    active = set(context.objective_ids)
    claimed = [
        objective_id
        for objectives in context.owned_objective_ids_by_seat
        for objective_id in objectives
    ]
    if len(active) != len(context.objective_ids):
        raise DecisionAnalysisError("decision trace active objective IDs must be unique")
    if len(set(claimed)) != len(claimed) or any(
        objective_id not in active for objective_id in claimed
    ):
        raise DecisionAnalysisError(
            "decision trace claimed objectives do not match active objectives"
        )
    actor_owned = len(context.owned_objective_ids_by_seat[trace.seat])
    opponent_owned = len(claimed) - actor_owned
    return actor_owned, opponent_owned, len(active) - len(claimed)


def _record_slice(
    accumulator: _SliceAccumulator,
    trace: DecisionTrace,
    game: GameSummary,
) -> None:
    accumulator.decision_count += 1
    accumulator.pass_count += int(trace.selected_action.action_kind == "pass")
    selected_value = trace.selected_action.value
    if selected_value is not None:
        accumulator.selected_value_count += 1
        accumulator.selected_value_sum += selected_value
    accumulator.eventual_final_money_sum += trace.outcome.final_money
    normalized_finish = (game.player_count - trace.outcome.rank) / (game.player_count - 1)
    assert accumulator.eventual_normalized_finish_values is not None
    accumulator.eventual_normalized_finish_values.append(normalized_finish)
    accumulator.outright_win_decision_count += int(
        trace.outcome.rank == 1 and not trace.outcome.first_place_tied
    )
    accumulator.tied_first_decision_count += int(trace.outcome.first_place_tied)
    accumulator.decisions_from_faulted_game_seat += int(game.fault_counts[trace.seat] > 0)


def _freeze_slice(key: _SliceKey, accumulator: _SliceAccumulator) -> DecisionSlice:
    assert accumulator.eventual_normalized_finish_values is not None
    return DecisionSlice(
        bot_name=key.bot_name,
        bot_id=key.bot_id,
        game_phase=key.game_phase,
        chart=key.chart,
        player_count=key.player_count,
        decision_kind=key.decision_kind,
        auction_action=key.auction_action,
        selected_action_kind=key.selected_action_kind,
        future_biddable_resources=key.future_biddable_resources,
        total_biddable_resources=key.total_biddable_resources,
        actor_owned_objectives=key.actor_owned_objectives,
        opponent_owned_objectives=key.opponent_owned_objectives,
        unclaimed_objectives=key.unclaimed_objectives,
        seat=key.seat,
        opponent_bot_ids=key.opponent_bot_ids,
        decision_count=accumulator.decision_count,
        pass_count=accumulator.pass_count,
        selected_value_count=accumulator.selected_value_count,
        selected_value_sum=accumulator.selected_value_sum,
        eventual_final_money_sum=accumulator.eventual_final_money_sum,
        eventual_normalized_finish_sum=math.fsum(accumulator.eventual_normalized_finish_values),
        outright_win_decision_count=accumulator.outright_win_decision_count,
        tied_first_decision_count=accumulator.tied_first_decision_count,
        decisions_from_faulted_game_seat=accumulator.decisions_from_faulted_game_seat,
    )


def _ordinary_aggregates(
    games: tuple[GameSummary, ...],
) -> tuple[
    dict[str, _BotAggregate],
    dict[tuple[str, int, str], _ConditionAggregate],
]:
    bot_items: dict[str, list[tuple[GameSummary, int, SessionScore]]] = defaultdict(list)
    names: dict[str, str] = {}
    for game in games:
        scores_by_seat = {score.seat: score for score in game.scores}
        for seat, (bot_id, bot_name) in enumerate(zip(game.bot_ids, game.bot_names, strict=True)):
            previous_name = names.setdefault(bot_id, bot_name)
            if previous_name != bot_name:
                raise DecisionAnalysisError(
                    f"bot identity {bot_id!r} has inconsistent names in game summaries"
                )
            bot_items[bot_id].append((game, seat, scores_by_seat[seat]))

    rank_count = max((game.player_count for game in games), default=0)
    bots: dict[str, _BotAggregate] = {}
    for bot_id, items in bot_items.items():
        bots[bot_id] = _bot_aggregate(
            names[bot_id],
            items,
            rank_count=rank_count,
        )

    condition_items: dict[
        tuple[str, int, str],
        list[tuple[GameSummary, SessionScore]],
    ] = defaultdict(list)
    for bot_id, items in bot_items.items():
        for game, _seat, score in items:
            try:
                chart = value_chart_from_ruleset_name(game.ruleset_name)
            except ValueError as error:
                raise DecisionAnalysisError(
                    f"game summary {game.game_index} has an unknown ruleset"
                ) from error
            condition_items[chart, game.player_count, bot_id].append((game, score))
    conditions = {key: _condition_aggregate(items) for key, items in condition_items.items()}
    return bots, conditions


def _bot_aggregate(
    bot_name: str,
    items: list[tuple[GameSummary, int, SessionScore]],
    *,
    rank_count: int,
) -> _BotAggregate:
    return _BotAggregate(
        bot_name=bot_name,
        games=len(items),
        outright_wins=sum(_is_outright_win(game, score) for game, _, score in items),
        first_place_ties=sum(_is_tied_first(game, score) for game, _, score in items),
        rank_counts=tuple(
            sum(score.rank == rank for _, _, score in items) for rank in range(1, rank_count + 1)
        ),
        final_money_samples=tuple(score.final_money for _, _, score in items),
        score_margins=tuple(
            score.final_money - max(other.final_money for other in game.scores)
            for game, _, score in items
        ),
        decision_counts=tuple(game.decision_counts[seat] for game, seat, _ in items),
        faults=sum(game.fault_counts[seat] for game, seat, _ in items),
        normalized_finishes=tuple(
            (game.player_count - score.rank) / (game.player_count - 1) for game, _, score in items
        ),
        winning_money=tuple(score.final_money for game, _, score in items if score.rank == 1),
    )


def _condition_aggregate(
    items: list[tuple[GameSummary, SessionScore]],
) -> _ConditionAggregate:
    return _ConditionAggregate(
        games=len(items),
        outright_wins=sum(_is_outright_win(game, score) for game, score in items),
        first_place_ties=sum(_is_tied_first(game, score) for game, score in items),
        normalized_finishes=tuple(
            (game.player_count - score.rank) / (game.player_count - 1) for game, score in items
        ),
        final_money=tuple(score.final_money for _, score in items),
    )


def _is_outright_win(game: GameSummary, score: SessionScore) -> bool:
    return score.rank == 1 and sum(item.rank == 1 for item in game.scores) == 1


def _is_tied_first(game: GameSummary, score: SessionScore) -> bool:
    return score.rank == 1 and sum(item.rank == 1 for item in game.scores) > 1


def _validate_bot_statistics(
    statistics_rows: Sequence[BotStatistics],
    *,
    expected: dict[str, _BotAggregate],
) -> None:
    actual = _unique_by_bot_id(statistics_rows, source="bot statistics")
    if set(actual) != set(expected):
        raise DecisionAnalysisError("bot statistics identities do not match game summaries")
    for bot_id, expected_row in expected.items():
        row = actual[bot_id]
        mismatches = []
        if row.bot_name != expected_row.bot_name:
            mismatches.append("name")
        if row.games != expected_row.games:
            mismatches.append("games")
        if row.outright_wins != expected_row.outright_wins:
            mismatches.append("outright wins")
        if row.first_place_ties != expected_row.first_place_ties:
            mismatches.append("first-place ties")
        if row.rank_counts != expected_row.rank_counts:
            mismatches.append("rank counts")
        if sorted(row.final_money_samples) != sorted(expected_row.final_money_samples):
            mismatches.append("final money")
        if sorted(row.score_margins) != sorted(expected_row.score_margins):
            mismatches.append("score margins")
        if sorted(row.decision_counts) != sorted(expected_row.decision_counts):
            mismatches.append("decision counts")
        if row.faults != expected_row.faults:
            mismatches.append("faults")
        if mismatches:
            raise DecisionAnalysisError(
                f"bot statistics for {bot_id!r} do not match game summaries: "
                + ", ".join(mismatches)
            )


def _validate_tournament_rows(
    rows: Sequence[TournamentBotRow],
    *,
    expected: dict[str, _BotAggregate],
) -> None:
    actual = _unique_by_bot_id(rows, source="tournament row")
    if set(actual) != set(expected):
        raise DecisionAnalysisError("tournament row identities do not match game summaries")
    for bot_id, expected_row in expected.items():
        row = actual[bot_id]
        expected_winning_money = (
            float(statistics.mean(expected_row.winning_money))
            if expected_row.winning_money
            else None
        )
        if (
            row.bot_name != expected_row.bot_name
            or row.games != expected_row.games
            or row.outright_wins != expected_row.outright_wins
            or row.first_place_ties != expected_row.first_place_ties
            or row.faults != expected_row.faults
            or not _same_float(
                row.mean_normalized_finish,
                statistics.mean(expected_row.normalized_finishes),
            )
            or not _same_float(
                row.mean_final_money,
                statistics.mean(expected_row.final_money_samples),
            )
            or not _same_optional_float(row.mean_winning_money, expected_winning_money)
        ):
            raise DecisionAnalysisError(
                f"tournament row for {bot_id!r} does not match game summaries"
            )


def _validate_condition_statistics(
    rows: Sequence[ConditionStatistics],
    *,
    expected: dict[tuple[str, int, str], _ConditionAggregate],
) -> None:
    actual: dict[tuple[str, int, str], ConditionStatistics] = {}
    for row in rows:
        key = (row.chart, row.player_count, row.bot_id)
        if key in actual:
            raise DecisionAnalysisError(f"duplicate condition statistics row {key}")
        actual[key] = row
    if set(actual) != set(expected):
        raise DecisionAnalysisError("condition statistics identities do not match game summaries")
    for key, expected_row in expected.items():
        row = actual[key]
        if (
            row.games != expected_row.games
            or row.outright_wins != expected_row.outright_wins
            or row.first_place_ties != expected_row.first_place_ties
            or not _same_float(
                row.mean_normalized_finish,
                statistics.mean(expected_row.normalized_finishes),
            )
            or not _same_float(
                row.mean_final_money,
                statistics.mean(expected_row.final_money),
            )
        ):
            raise DecisionAnalysisError(
                f"condition statistics for {key} do not match game summaries"
            )


def _unique_by_bot_id[Row: (BotStatistics, TournamentBotRow)](
    rows: Sequence[Row],
    *,
    source: str,
) -> dict[str, Row]:
    result: dict[str, Row] = {}
    for row in rows:
        if row.bot_id in result:
            raise DecisionAnalysisError(f"duplicate {source} for bot {row.bot_id!r}")
        result[row.bot_id] = row
    return result


def _same_float(first: float, second: float) -> bool:
    return math.isclose(first, second, rel_tol=1e-12, abs_tol=1e-12)


def _same_optional_float(first: float | None, second: float | None) -> bool:
    if first is None or second is None:
        return first is second
    return _same_float(first, second)
