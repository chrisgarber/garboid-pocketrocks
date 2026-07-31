from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import replace

import pytest
from pocketrocks.sim.constants import ACTION_WIRE_IDS

from garboid_pocketrocks.diagnostics.analysis import (
    DecisionAnalysisError,
    DecisionReport,
    build_decision_report,
)
from garboid_pocketrocks.diagnostics.trace import (
    DecisionTrace,
    PendingDecisionTrace,
    PublicDecisionContext,
    PublicDecisionOutcome,
    RecordedAction,
)
from garboid_pocketrocks.knowledge import canonical_knowledge, ruleset_name
from garboid_pocketrocks.simulator.monte_carlo import (
    BehaviorStatistics,
    BotStatistics,
    GameSummary,
)
from garboid_pocketrocks.simulator.session import SessionScore
from garboid_pocketrocks.tournament.analysis import (
    ConditionStatistics,
    TournamentAnalysis,
    TournamentBotRow,
)


def _game(
    *,
    game_index: int = 0,
    chart: str = "A",
    bot_ids: tuple[str, ...] = ("alpha", "beta", "gamma"),
    final_money: tuple[int, ...] = (30, 20, 10),
    ranks: tuple[int, ...] = (1, 2, 3),
    decision_counts: tuple[int, ...] = (1, 1, 1),
    fault_counts: tuple[int, ...] | None = None,
) -> GameSummary:
    player_count = len(bot_ids)
    return GameSummary(
        game_index=game_index,
        root_seed=41,
        seed=101 + game_index,
        player_count=player_count,
        ruleset_name=ruleset_name(chart),
        bot_names=bot_ids,
        bot_ids=bot_ids,
        scores=tuple(
            SessionScore(seat=seat, final_money=money, rank=rank)
            for seat, (money, rank) in enumerate(zip(final_money, ranks, strict=True))
        ),
        decision_counts=decision_counts,
        fault_counts=(0,) * player_count if fault_counts is None else fault_counts,
    )


def _trace(
    game: GameSummary,
    *,
    seat: int,
    step_index: int = 0,
    turn_index: int = 0,
    decision_kind: str = "submitBid",
    action_id: int = ACTION_WIRE_IDS["Auction1"],
    selected_action: RecordedAction | None = None,
    current_resource_ids: tuple[int, int] = (3, 0),
    won_resource_counts_by_seat: tuple[tuple[int, ...], ...] | None = None,
    objective_ids: tuple[int, ...] = (1, 10),
    owned_objective_ids_by_seat: tuple[tuple[int, ...], ...] | None = None,
    selection_source: str = "policy",
) -> DecisionTrace:
    player_count = game.player_count
    knowledge = canonical_knowledge(player_count, value_chart=game.ruleset_name[5])
    score = game.scores[seat]
    first_place_tied = score.rank == 1 and sum(item.rank == 1 for item in game.scores) > 1
    won_counts = (
        ((0, 0, 0, 0, 0),) * player_count
        if won_resource_counts_by_seat is None
        else won_resource_counts_by_seat
    )
    owned_objectives = (
        ((),) * player_count if owned_objective_ids_by_seat is None else owned_objective_ids_by_seat
    )
    legal_actions: tuple[RecordedAction, ...]
    if decision_kind == "selectInfoToReveal":
        legal_actions = (
            RecordedAction("pass"),
            RecordedAction("selectInfoToReveal", 0),
        )
        legal_max_amount = None
        revealable_count = 1
        if selected_action is None:
            selected_action = RecordedAction("selectInfoToReveal", 0)
    else:
        legal_actions = (
            RecordedAction("pass"),
            RecordedAction("submitBid", 1),
            RecordedAction("submitBid", 2),
            RecordedAction("submitBid", 3),
        )
        legal_max_amount = 3
        revealable_count = 0
        if selected_action is None:
            selected_action = RecordedAction("submitBid", 3)
    pending = PendingDecisionTrace(
        game_index=game.game_index,
        chart=game.ruleset_name[5],
        step_index=step_index,
        turn_index=turn_index,
        seat=seat,
        bot_name=game.bot_names[seat],
        bot_id=game.bot_ids[seat],
        bot_names_by_seat=game.bot_names,
        bot_ids_by_seat=game.bot_ids,
        context=PublicDecisionContext(
            decision_kind=decision_kind,  # type: ignore[arg-type]
            player_count=player_count,
            starting_cash=knowledge.starting_cash,
            value_chart=knowledge.value_chart,
            objective_ids=objective_ids,
            current_action_id=action_id,
            current_resource_ids=current_resource_ids,
            cash_by_seat=(knowledge.starting_cash,) * player_count,
            tiebreak_seat=0,
            won_resource_counts_by_seat=won_counts,
            revealed_info_counts_by_seat=((0, 0, 0, 0, 0),) * player_count,
            owned_objective_ids_by_seat=owned_objectives,
            bot_seat=seat,
            legal_max_amount=legal_max_amount,
            revealable_count=revealable_count,
        ),
        public_history=(),
        legal_actions=legal_actions,
        selected_action=selected_action,
        explanation=None,
        selection_source=selection_source,  # type: ignore[arg-type]
    )
    return DecisionTrace.from_pending(
        pending,
        PublicDecisionOutcome(
            rank=score.rank,
            first_place_tied=first_place_tied,
            final_money=score.final_money,
        ),
    )


def _ordinary_results(
    games: tuple[GameSummary, ...],
) -> tuple[tuple[BotStatistics, ...], TournamentAnalysis]:
    appearances: dict[str, list[tuple[GameSummary, int, SessionScore]]] = defaultdict(list)
    names: dict[str, str] = {}
    for game in games:
        scores_by_seat = {score.seat: score for score in game.scores}
        for seat, (bot_id, bot_name) in enumerate(zip(game.bot_ids, game.bot_names, strict=True)):
            names[bot_id] = bot_name
            appearances[bot_id].append((game, seat, scores_by_seat[seat]))

    rank_count = max(game.player_count for game in games)
    empty_behavior = BehaviorStatistics(0, 0, (), (), (0,) * 6, 0, 0)
    bot_statistics: list[BotStatistics] = []
    tournament_rows: list[TournamentBotRow] = []
    for display_rank, bot_id in enumerate(sorted(appearances), start=1):
        items = appearances[bot_id]
        ranks = tuple(
            sum(score.rank == rank for _, _, score in items) for rank in range(1, rank_count + 1)
        )
        wins = sum(
            score.rank == 1 and sum(item.rank == 1 for item in game.scores) == 1
            for game, _, score in items
        )
        ties = sum(
            score.rank == 1 and sum(item.rank == 1 for item in game.scores) > 1
            for game, _, score in items
        )
        money = tuple(score.final_money for _, _, score in items)
        faults = sum(game.fault_counts[seat] for game, seat, _ in items)
        decision_counts = tuple(game.decision_counts[seat] for game, seat, _ in items)
        normalized_finishes = tuple(
            (game.player_count - score.rank) / (game.player_count - 1) for game, _, score in items
        )
        winning_money = tuple(score.final_money for _, _, score in items if score.rank == 1)
        bot_statistics.append(
            BotStatistics(
                bot_name=names[bot_id],
                bot_id=bot_id,
                games=len(items),
                outright_wins=wins,
                first_place_ties=ties,
                rank_counts=ranks,
                final_money_samples=money,
                score_margins=tuple(
                    score.final_money - max(item.final_money for item in game.scores)
                    for game, _, score in items
                ),
                per_seat=(),
                per_ruleset=(),
                decision_counts=decision_counts,
                faults=faults,
                behavior=empty_behavior,
            )
        )
        tournament_rows.append(
            TournamentBotRow(
                rank=display_rank,
                bot_name=names[bot_id],
                bot_id=bot_id,
                worth=1.0,
                log_worth=0.0,
                pl_rating=1500.0,
                games=len(items),
                outright_wins=wins,
                first_place_ties=ties,
                mean_normalized_finish=float(statistics.mean(normalized_finishes)),
                mean_final_money=float(statistics.mean(money)),
                mean_winning_money=(
                    float(statistics.mean(winning_money)) if winning_money else None
                ),
                faults=faults,
            )
        )

    conditions: dict[tuple[str, int, str], list[tuple[GameSummary, SessionScore]]] = defaultdict(
        list
    )
    for bot_id, items in appearances.items():
        for game, _, score in items:
            conditions[game.ruleset_name[5], game.player_count, bot_id].append((game, score))
    condition_statistics = tuple(
        ConditionStatistics(
            chart=chart,
            player_count=player_count,
            bot_id=bot_id,
            games=len(items),
            outright_wins=sum(
                score.rank == 1 and sum(item.rank == 1 for item in game.scores) == 1
                for game, score in items
            ),
            first_place_ties=sum(
                score.rank == 1 and sum(item.rank == 1 for item in game.scores) > 1
                for game, score in items
            ),
            mean_normalized_finish=float(
                statistics.mean(
                    (game.player_count - score.rank) / (game.player_count - 1)
                    for game, score in items
                )
            ),
            mean_final_money=float(statistics.mean(score.final_money for _, score in items)),
        )
        for (chart, player_count, bot_id), items in sorted(conditions.items())
    )
    analysis = TournamentAnalysis(
        rows=tuple(tournament_rows),
        condition_statistics=condition_statistics,
        calibration=(),
        pair_outcomes=sum(game.player_count * (game.player_count - 1) // 2 for game in games),
    )
    return tuple(bot_statistics), analysis


def _build(
    traces: tuple[DecisionTrace, ...],
    games: tuple[GameSummary, ...],
) -> DecisionReport:
    bot_statistics, tournament_analysis = _ordinary_results(games)
    return build_decision_report(
        traces,
        game_summaries=games,
        bot_statistics=bot_statistics,
        tournament_analysis=tournament_analysis,
    )


def test_report_owns_the_exact_validated_sources_in_canonical_order() -> None:
    first_game = _game(game_index=0, chart="A", decision_counts=(1, 0, 0))
    second_game = _game(game_index=1, chart="E", decision_counts=(1, 0, 0))
    first_trace = _trace(first_game, seat=0)
    second_trace = _trace(second_game, seat=0)

    report = _build((second_trace, first_trace), (second_game, first_game))

    assert report.game_summaries == (first_game, second_game)
    assert report.decision_traces == (first_trace, second_trace)


def test_slices_keep_every_dimension_and_add_decision_weighted_measures() -> None:
    game = _game(decision_counts=(2, 1, 1), fault_counts=(1, 0, 0))
    owned = ((1,), (), ())
    alpha_first = _trace(
        game,
        seat=0,
        step_index=0,
        owned_objective_ids_by_seat=owned,
        selection_source="fault_fallback",
    )
    alpha_second = replace(alpha_first, step_index=1)
    beta = _trace(
        game,
        seat=1,
        selected_action=RecordedAction("pass"),
        owned_objective_ids_by_seat=owned,
    )
    gamma = _trace(
        game,
        seat=2,
        decision_kind="selectInfoToReveal",
        selected_action=RecordedAction("selectInfoToReveal", 0),
        owned_objective_ids_by_seat=owned,
        won_resource_counts_by_seat=(
            (1, 0, 0, 0, 0),
            (0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0),
        ),
    )

    report = _build((gamma, alpha_second, beta, alpha_first), (game,))

    alpha = next(item for item in report.slices if item.bot_id == "alpha")
    assert (
        alpha.game_phase,
        alpha.chart,
        alpha.player_count,
        alpha.decision_kind,
        alpha.auction_action,
        alpha.selected_action_kind,
    ) == ("early", "A", 3, "submitBid", "Auction1", "submitBid")
    assert (
        alpha.future_biddable_resources,
        alpha.total_biddable_resources,
    ) == (14, 15)
    assert (
        alpha.actor_owned_objectives,
        alpha.opponent_owned_objectives,
        alpha.unclaimed_objectives,
    ) == (1, 0, 1)
    assert alpha.seat == 0
    assert alpha.opponent_bot_ids == ("beta", "gamma")
    assert (
        alpha.decision_count,
        alpha.pass_count,
        alpha.selected_value_count,
        alpha.selected_value_sum,
    ) == (2, 0, 2, 6)
    assert alpha.eventual_final_money_sum == 60
    assert alpha.eventual_normalized_finish_sum == 2.0
    assert alpha.outright_win_decision_count == 2
    assert alpha.tied_first_decision_count == 0
    assert alpha.decisions_from_faulted_game_seat == 2

    beta_slice = next(item for item in report.slices if item.bot_id == "beta")
    assert beta_slice.pass_count == 1
    gamma_slice = next(item for item in report.slices if item.bot_id == "gamma")
    assert gamma_slice.decision_kind == "selectInfoToReveal"
    assert gamma_slice.selected_action_kind == "selectInfoToReveal"
    assert gamma_slice.selected_value_count == 1
    assert gamma_slice.selected_value_sum == 0
    assert gamma_slice.future_biddable_resources == 14

    assert report.reconciliation.game_count == 1
    assert report.reconciliation.game_seat_count == 3
    assert report.reconciliation.trace_decision_count == 4
    assert report.reconciliation.game_summary_decision_count == 4
    assert report.reconciliation.slice_decision_count == 4


@pytest.mark.parametrize(
    ("turn_index", "expected"),
    ((0, "early"), (4, "early"), (5, "middle"), (11, "middle"), (12, "late")),
)
def test_game_phase_uses_documented_one_based_turn_boundaries(
    turn_index: int,
    expected: str,
) -> None:
    game = _game(decision_counts=(1, 0, 0))

    report = _build((_trace(game, seat=0, turn_index=turn_index),), (game,))

    assert report.slices[0].game_phase == expected


def test_cash_horizon_uses_action_aware_public_resource_accounting() -> None:
    game = _game(decision_counts=(1, 1, 1))
    won_before_auction = (
        (1, 1, 1, 0, 0),
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
    )
    won_after_auction_one = (
        (2, 1, 1, 0, 0),
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
    )
    auction_one = _trace(
        game,
        seat=0,
        turn_index=12,
        action_id=ACTION_WIRE_IDS["Auction1"],
        current_resource_ids=(3, 4),
        won_resource_counts_by_seat=won_before_auction,
    )
    auction_two = _trace(
        game,
        seat=1,
        turn_index=12,
        action_id=ACTION_WIRE_IDS["Auction2"],
        current_resource_ids=(3, 4),
        won_resource_counts_by_seat=won_before_auction,
    )
    reveal_after_auction_one = _trace(
        game,
        seat=2,
        turn_index=12,
        decision_kind="selectInfoToReveal",
        action_id=ACTION_WIRE_IDS["Auction1"],
        current_resource_ids=(3, 4),
        won_resource_counts_by_seat=won_after_auction_one,
    )

    report = _build(
        (auction_one, auction_two, reveal_after_auction_one),
        (game,),
    )
    slices_by_bot = {item.bot_id: item for item in report.slices}

    assert (
        slices_by_bot["alpha"].future_biddable_resources,
        slices_by_bot["alpha"].total_biddable_resources,
    ) == (11, 15)
    assert (
        slices_by_bot["beta"].future_biddable_resources,
        slices_by_bot["beta"].total_biddable_resources,
    ) == (10, 15)
    assert (
        slices_by_bot["gamma"].future_biddable_resources,
        slices_by_bot["gamma"].total_biddable_resources,
    ) == (11, 15)
    assert {item.game_phase for item in report.slices} == {"late"}


def test_chart_players_seat_and_opponent_composition_are_canonical_dimensions() -> None:
    game = _game(
        chart="E",
        bot_ids=("zeta", "alpha", "delta", "beta"),
        final_money=(40, 20, 40, 10),
        ranks=(1, 3, 1, 4),
        decision_counts=(0, 0, 1, 0),
    )

    report = _build((_trace(game, seat=2),), (game,))

    row = report.slices[0]
    assert (row.chart, row.player_count, row.seat) == ("E", 4, 2)
    assert row.opponent_bot_ids == ("alpha", "beta", "zeta")
    assert row.tied_first_decision_count == 1


def test_duplicate_trace_keys_are_rejected_before_aggregation() -> None:
    game = _game(decision_counts=(1, 0, 0))
    trace = _trace(game, seat=0)
    bot_statistics, analysis = _ordinary_results((game,))

    with pytest.raises(DecisionAnalysisError, match="duplicate decision trace key"):
        build_decision_report(
            (trace, trace),
            game_summaries=(game,),
            bot_statistics=bot_statistics,
            tournament_analysis=analysis,
        )


@pytest.mark.parametrize(
    ("mismatch", "message"),
    (
        ("game", "no game summary"),
        ("chart", "chart"),
        ("unknown_chart", "unknown chart"),
        ("context_chart", "public context chart"),
        ("identity", "identity"),
    ),
)
def test_trace_must_match_its_game_summary(mismatch: str, message: str) -> None:
    game = _game(decision_counts=(1, 0, 0))
    trace = _trace(game, seat=0)
    if mismatch == "game":
        trace = replace(trace, game_index=99)
    elif mismatch == "chart":
        trace = replace(trace, chart="E")
    elif mismatch == "unknown_chart":
        trace = replace(trace, chart="Z")
    elif mismatch == "context_chart":
        trace = replace(
            trace,
            context=replace(
                trace.context,
                value_chart=canonical_knowledge(3, value_chart="E").value_chart,
            ),
        )
    elif mismatch == "identity":
        trace = replace(
            trace,
            bot_id="wrong",
            bot_ids_by_seat=("wrong", *trace.bot_ids_by_seat[1:]),
        )
    bot_statistics, analysis = _ordinary_results((game,))

    with pytest.raises(DecisionAnalysisError, match=message):
        build_decision_report(
            (trace,),
            game_summaries=(game,),
            bot_statistics=bot_statistics,
            tournament_analysis=analysis,
        )


@pytest.mark.parametrize(
    ("rank", "first_place_tied", "final_money"),
    (
        (2, False, 30),
        (1, True, 30),
        (1, False, 31),
    ),
)
def test_every_trace_outcome_field_must_match_its_game_summary(
    rank: int,
    first_place_tied: bool,
    final_money: int,
) -> None:
    game = _game(decision_counts=(1, 0, 0))
    trace = _trace(game, seat=0)
    trace = replace(
        trace,
        outcome=PublicDecisionOutcome(
            rank=rank,
            first_place_tied=first_place_tied,
            final_money=final_money,
        ),
    )
    bot_statistics, analysis = _ordinary_results((game,))

    with pytest.raises(DecisionAnalysisError, match="public outcome"):
        build_decision_report(
            (trace,),
            game_summaries=(game,),
            bot_statistics=bot_statistics,
            tournament_analysis=analysis,
        )


def test_trace_counts_must_match_every_game_seat_decision_count() -> None:
    game = _game(decision_counts=(2, 0, 0))
    bot_statistics, analysis = _ordinary_results((game,))

    with pytest.raises(DecisionAnalysisError, match="decision count"):
        build_decision_report(
            (_trace(game, seat=0),),
            game_summaries=(game,),
            bot_statistics=bot_statistics,
            tournament_analysis=analysis,
        )


def test_fault_fallback_must_reconcile_with_a_recorded_game_seat_fault() -> None:
    game = _game(decision_counts=(1, 0, 0))
    bot_statistics, analysis = _ordinary_results((game,))

    with pytest.raises(DecisionAnalysisError, match="fault fallback"):
        build_decision_report(
            (_trace(game, seat=0, selection_source="fault_fallback"),),
            game_summaries=(game,),
            bot_statistics=bot_statistics,
            tournament_analysis=analysis,
        )


def test_bot_statistics_must_reconcile_with_deduplicated_game_seats() -> None:
    game = _game(decision_counts=(1, 0, 0))
    bot_statistics, analysis = _ordinary_results((game,))
    wrong = (replace(bot_statistics[0], games=bot_statistics[0].games + 1), *bot_statistics[1:])

    with pytest.raises(DecisionAnalysisError, match="bot statistics"):
        build_decision_report(
            (_trace(game, seat=0),),
            game_summaries=(game,),
            bot_statistics=wrong,
            tournament_analysis=analysis,
        )


def test_tournament_rows_and_condition_statistics_must_reconcile() -> None:
    game = _game(decision_counts=(1, 0, 0))
    bot_statistics, analysis = _ordinary_results((game,))
    wrong_row = replace(analysis.rows[0], mean_final_money=999.0)
    with pytest.raises(DecisionAnalysisError, match="tournament row"):
        build_decision_report(
            (_trace(game, seat=0),),
            game_summaries=(game,),
            bot_statistics=bot_statistics,
            tournament_analysis=replace(analysis, rows=(wrong_row, *analysis.rows[1:])),
        )

    wrong_condition = replace(analysis.condition_statistics[0], outright_wins=99)
    with pytest.raises(DecisionAnalysisError, match="condition statistics"):
        build_decision_report(
            (_trace(game, seat=0),),
            game_summaries=(game,),
            bot_statistics=bot_statistics,
            tournament_analysis=replace(
                analysis,
                condition_statistics=(
                    wrong_condition,
                    *analysis.condition_statistics[1:],
                ),
            ),
        )
