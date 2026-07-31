"""Run synchronous bot brains over homogeneous SDK batch-engine chunks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from pocketrocks import BotDecision
from pocketrocks.sim import BatchSimEngine, BatchTurnOutcome, RevealRecord, ScoreRow, TurnRecord
from pocketrocks.sim.constants import ACTION_WIRE_IDS

from garboid_pocketrocks.adapters.public_history import (
    PublicAuctionResolved,
    PublicEvent,
    PublicEventKind,
    PublicGameSetup,
    PublicInformationRevealed,
    PublicTurnOpened,
)
from garboid_pocketrocks.bots.base import BotBrain, BotSpec
from garboid_pocketrocks.diagnostics.trace import PendingDecisionTrace
from garboid_pocketrocks.knowledge import RulesetKnowledge, canonical_knowledge
from garboid_pocketrocks.simulator.batch_context import build_batch_context
from garboid_pocketrocks.simulator.bot_execution import (
    BotFault,
    FaultMode,
    execute_brain_decision,
    initialize_brains,
)
from garboid_pocketrocks.simulator.replay import MatchReplay
from garboid_pocketrocks.simulator.runner import (
    MatchResult,
    _build_pending_decision_trace,
    _finalize_decision_traces,
)
from garboid_pocketrocks.simulator.session import (
    SessionResult,
    SessionScore,
)

_ACTION_BY_WIRE_ID = {wire_id: action for action, wire_id in ACTION_WIRE_IDS.items()}
_RESOURCE_GRANTS = {
    ACTION_WIRE_IDS["Auction1"]: 1,
    ACTION_WIRE_IDS["Auction2"]: 2,
}


class BatchGameJob(Protocol):
    @property
    def game_index(self) -> int: ...

    @property
    def seed(self) -> int: ...

    @property
    def player_count(self) -> int: ...

    @property
    def value_chart(self) -> str: ...

    @property
    def objectives_enabled(self) -> bool: ...

    @property
    def lineup(self) -> tuple[BotSpec, ...]: ...

    @property
    def fault_mode(self) -> FaultMode: ...

    @property
    def capture_decision_traces(self) -> bool: ...


@dataclass(slots=True)
class _BatchRunState:
    jobs: tuple[BatchGameJob, ...]
    player_count: int
    engine: BatchSimEngine
    brains: list[tuple[BotBrain | None, ...]]
    faults: list[list[BotFault]]
    knowledge: list[RulesetKnowledge]
    histories: list[list[PublicEvent]]
    decisions: list[list[tuple[int, tuple[tuple[int, BotDecision], ...]]]]
    pending_traces: list[list[PendingDecisionTrace]]
    turns: list[list[TurnRecord]]
    step_indices: list[int]


@dataclass(slots=True)
class _PendingBatchTurn:
    action_ids: NDArray[np.uint8]
    active_rows: NDArray[np.int64]
    resources_before: NDArray[np.uint8]
    objective_claimants_before: NDArray[np.int8]
    turn_indices: NDArray[np.int64]
    legal_max_bids: NDArray[np.int16]
    raw_bids: NDArray[np.int16]


def _validate_batch_jobs(jobs: tuple[BatchGameJob, ...]) -> int:
    player_count = jobs[0].player_count
    if any(job.player_count != player_count for job in jobs):
        raise ValueError("batch match jobs must share one player count")
    if any(len(job.lineup) != player_count for job in jobs):
        raise ValueError("batch match lineup length must match player count")
    return player_count


def _initialize_batch(jobs: tuple[BatchGameJob, ...]) -> _BatchRunState:
    player_count = _validate_batch_jobs(jobs)
    engine = BatchSimEngine.start(
        player_count=player_count,
        seeds=tuple(job.seed for job in jobs),
        value_charts=tuple(job.value_chart for job in jobs),
        objectives_enabled=tuple(job.objectives_enabled for job in jobs),
    )
    brains: list[tuple[BotBrain | None, ...]] = []
    faults: list[list[BotFault]] = [[] for _ in jobs]
    knowledge: list[RulesetKnowledge] = []
    histories: list[list[PublicEvent]] = []
    for row, job in enumerate(jobs):
        row_brains, construction_faults = initialize_brains(
            job.lineup,
            seed=job.seed,
            fault_mode=job.fault_mode,
        )
        brains.append(row_brains)
        faults[row].extend(construction_faults)
        game_knowledge = canonical_knowledge(
            player_count,
            value_chart=job.value_chart,
            objectives_enabled=job.objectives_enabled,
        )
        knowledge.append(game_knowledge)
        histories.append(
            [
                PublicGameSetup(
                    kind=PublicEventKind.GAME_SETUP,
                    player_count=player_count,
                    starting_cash=game_knowledge.starting_cash,
                    value_chart=game_knowledge.value_chart,
                    initial_tiebreak_seat=int(engine.tiebreak_seats[row]),
                    objective_ids=tuple(
                        int(value) for value in engine.objective_ids[row] if value > 0
                    ),
                )
            ]
        )
    return _BatchRunState(
        jobs=jobs,
        player_count=player_count,
        engine=engine,
        brains=brains,
        faults=faults,
        knowledge=knowledge,
        histories=histories,
        decisions=[[] for _ in jobs],
        pending_traces=[[] for _ in jobs],
        turns=[[] for _ in jobs],
        step_indices=[0] * len(jobs),
    )


def _prepare_next_turn(state: _BatchRunState) -> _PendingBatchTurn | None:
    action_ids = state.engine.flip_actions()
    active_rows = np.flatnonzero(action_ids)
    if not len(active_rows):
        return None
    return _PendingBatchTurn(
        action_ids=action_ids,
        active_rows=active_rows,
        resources_before=state.engine.upcoming.copy(),
        objective_claimants_before=state.engine.objective_claimants.copy(),
        turn_indices=state.engine.turn_indices.astype(np.int64, copy=True),
        legal_max_bids=state.engine.legal_max_bids(),
        raw_bids=np.zeros((len(state.jobs), state.player_count), dtype=np.int16),
    )


def _record_turn_opened(state: _BatchRunState, pending: _PendingBatchTurn) -> None:
    for raw_row in pending.active_rows:
        row = int(raw_row)
        state.histories[row].append(
            PublicTurnOpened(
                kind=PublicEventKind.TURN_OPENED,
                action_id=int(pending.action_ids[row]),
                resource_ids=(
                    int(pending.resources_before[row, 0]),
                    int(pending.resources_before[row, 1]),
                ),
            )
        )


def _collect_bid_decisions(state: _BatchRunState, pending: _PendingBatchTurn) -> None:
    for raw_row in pending.active_rows:
        row = int(raw_row)
        job = state.jobs[row]
        action_id = int(pending.action_ids[row])
        resource_ids = (
            int(pending.resources_before[row, 0]),
            int(pending.resources_before[row, 1]),
        )
        recorded: list[tuple[int, BotDecision]] = []
        for seat in range(state.player_count):
            context = build_batch_context(
                state.engine,
                row=row,
                seat=seat,
                decision_kind="submitBid",
                action_id=action_id,
                resource_ids=resource_ids,
                turn_index=int(pending.turn_indices[row]),
                legal_max_amount=int(pending.legal_max_bids[row, seat]),
            )
            history = tuple(state.histories[row])
            execution = execute_brain_decision(
                brain=state.brains[row][seat],
                context=context,
                knowledge=state.knowledge[row],
                history=history,
                fault_mode=job.fault_mode,
                faults=state.faults[row],
                turn_index=int(pending.turn_indices[row]),
                seat=seat,
                bot_name=job.lineup[seat].name,
                request_explanation=job.capture_decision_traces,
            )
            decision = execution.decision
            recorded.append((seat, decision))
            if job.capture_decision_traces:
                state.pending_traces[row].append(
                    _build_pending_decision_trace(
                        game_index=job.game_index,
                        chart=job.value_chart,
                        step_index=state.step_indices[row],
                        turn_index=int(pending.turn_indices[row]),
                        seat=seat,
                        lineup=job.lineup,
                        context=context,
                        public_history=history,
                        execution=execution,
                    )
                )
            if decision.action_kind == "submitBid":
                assert decision.value is not None
                pending.raw_bids[row, seat] = decision.value
        state.decisions[row].append((state.step_indices[row], tuple(recorded)))
        state.step_indices[row] += 1


def _resolve_bids(
    state: _BatchRunState,
    pending: _PendingBatchTurn,
) -> BatchTurnOutcome:
    return state.engine.resolve_bids(pending.raw_bids)


def _record_bid_outcomes(
    state: _BatchRunState,
    pending: _PendingBatchTurn,
    outcome: BatchTurnOutcome,
) -> None:
    for raw_row in pending.active_rows:
        row = int(raw_row)
        state.histories[row].append(
            PublicAuctionResolved(
                kind=PublicEventKind.AUCTION_RESOLVED,
                bids_by_seat=tuple(int(value) for value in outcome.effective_bids[row]),
            )
        )
    for raw_row in pending.active_rows:
        row = int(raw_row)
        action_id = int(pending.action_ids[row])
        winner = int(outcome.winner_seats[row])
        upcoming_before = tuple(
            int(resource) for resource in pending.resources_before[row] if resource > 0
        )
        previously_claimed = {
            int(state.engine.objective_ids[row, index])
            for index, claimant in enumerate(pending.objective_claimants_before[row])
            if claimant >= 0
        }
        claimed = tuple(
            int(objective_id)
            for index, objective_id in enumerate(state.engine.objective_ids[row])
            if (
                objective_id > 0
                and state.engine.objective_claimants[row, index] == winner
                and int(objective_id) not in previously_claimed
            )
        )
        grant_count = _RESOURCE_GRANTS.get(action_id, 0)
        state.turns[row].append(
            TurnRecord(
                turn_index=int(pending.turn_indices[row]),
                action=_ACTION_BY_WIRE_ID[action_id],
                upcoming_before=upcoming_before,
                raw_bids=tuple(int(value) for value in pending.raw_bids[row]),
                effective_bids=tuple(int(value) for value in outcome.effective_bids[row]),
                winner_seat=winner,
                paid=int(outcome.paid[row]),
                bundle_suits=upcoming_before[:grant_count],
                claimed_objective_wire_ids=claimed,
                reveal=None,
            )
        )


def _resolve_reveals(
    state: _BatchRunState,
    pending: _PendingBatchTurn,
    outcome: BatchTurnOutcome,
) -> tuple[tuple[int, RevealRecord], ...]:
    reveal_indices = np.full(len(state.jobs), -1, dtype=np.int16)
    resolved_reveals: list[tuple[int, RevealRecord]] = []
    for raw_row in pending.active_rows:
        row = int(raw_row)
        mode = int(outcome.reveal_modes[row])
        if mode == 0:
            continue
        winner = int(outcome.winner_seats[row])
        reveal_index = 0
        auto = mode == 1
        if mode == 2:
            job = state.jobs[row]
            context = build_batch_context(
                state.engine,
                row=row,
                seat=winner,
                decision_kind="selectInfoToReveal",
                action_id=int(pending.action_ids[row]),
                resource_ids=(
                    int(pending.resources_before[row, 0]),
                    int(pending.resources_before[row, 1]),
                ),
                turn_index=int(pending.turn_indices[row]),
                legal_max_amount=None,
            )
            history = tuple(state.histories[row])
            execution = execute_brain_decision(
                brain=state.brains[row][winner],
                context=context,
                knowledge=state.knowledge[row],
                history=history,
                fault_mode=job.fault_mode,
                faults=state.faults[row],
                turn_index=int(state.engine.turn_indices[row]),
                seat=winner,
                bot_name=job.lineup[winner].name,
                request_explanation=job.capture_decision_traces,
            )
            decision = execution.decision
            if job.capture_decision_traces:
                state.pending_traces[row].append(
                    _build_pending_decision_trace(
                        game_index=job.game_index,
                        chart=job.value_chart,
                        step_index=state.step_indices[row],
                        turn_index=int(state.engine.turn_indices[row]),
                        seat=winner,
                        lineup=job.lineup,
                        context=context,
                        public_history=history,
                        execution=execution,
                    )
                )
            state.decisions[row].append((state.step_indices[row], ((winner, decision),)))
            state.step_indices[row] += 1
            if decision.action_kind == "selectInfoToReveal":
                assert decision.value is not None
                reveal_index = decision.value
        reveal_indices[row] = reveal_index
        resolved_reveals.append(
            (
                row,
                RevealRecord(
                    seat=winner,
                    suit=int(state.engine.hand_cards[row, winner, reveal_index]),
                    auto=auto,
                ),
            )
        )
    state.engine.apply_reveals(reveal_indices)
    return tuple(resolved_reveals)


def _record_reveals(
    state: _BatchRunState,
    reveals: tuple[tuple[int, RevealRecord], ...],
) -> None:
    for row, reveal in reveals:
        state.histories[row].append(
            PublicInformationRevealed(
                kind=PublicEventKind.INFORMATION_REVEALED,
                seat=reveal.seat,
                suit_id=reveal.suit,
            )
        )
        state.turns[row][-1] = replace(state.turns[row][-1], reveal=reveal)


def _build_match_results(state: _BatchRunState) -> tuple[MatchResult, ...]:
    scores = state.engine.scores()
    rankings = state.engine.rankings()
    results: list[MatchResult] = []
    for row, job in enumerate(state.jobs):
        score_rows = tuple(
            ScoreRow(
                seat=seat,
                name=job.lineup[seat].name,
                cash=int(scores.cash[row, seat]),
                items_value=int(scores.items[row, seat]),
                objectives_value=int(scores.objectives[row, seat]),
                investments_value=int(scores.investments[row, seat]),
                loans_value=int(scores.loans[row, seat]),
                total=int(scores.total[row, seat]),
            )
            for seat in range(state.player_count)
        )
        session_result = SessionResult(
            scores=tuple(
                SessionScore(
                    seat=seat,
                    final_money=int(scores.total[row, seat]),
                    rank=1
                    + sum(
                        int(scores.total[row, other]) > int(scores.total[row, seat])
                        for other in range(state.player_count)
                    ),
                )
                for seat in range(state.player_count)
            ),
            rows=score_rows,
            ranking=tuple(int(seat) for seat in rankings[row]),
        )
        replay = MatchReplay(
            schema_version=2,
            player_count=state.player_count,
            seed=job.seed,
            value_chart=job.value_chart.upper(),
            objectives_enabled=job.objectives_enabled,
            root_seed=None,
            game_index=None,
            bot_names=tuple(spec.name for spec in job.lineup),
            decisions=tuple(state.decisions[row]),
            turns=tuple(state.turns[row]),
            result=session_result,
        )
        results.append(
            MatchResult(
                result=session_result,
                events=(),
                turns=tuple(state.turns[row]),
                faults=tuple(state.faults[row]),
                replay=replay,
                decision_traces=_finalize_decision_traces(
                    state.pending_traces[row],
                    session_result,
                ),
            )
        )
    return tuple(results)


def run_batch_matches(jobs: Sequence[BatchGameJob]) -> tuple[MatchResult, ...]:
    """Run a nonempty homogeneous-player-count chunk in deterministic row order."""

    batch_jobs = tuple(jobs)
    if not batch_jobs:
        return ()
    state = _initialize_batch(batch_jobs)

    while pending := _prepare_next_turn(state):
        _record_turn_opened(state, pending)
        _collect_bid_decisions(state, pending)
        outcome = _resolve_bids(state, pending)
        _record_bid_outcomes(state, pending, outcome)
        reveals = _resolve_reveals(state, pending, outcome)
        _record_reveals(state, reveals)

    return _build_match_results(state)
