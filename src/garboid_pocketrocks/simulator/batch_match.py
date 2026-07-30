"""Run synchronous bot brains over homogeneous SDK batch-engine chunks."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol

import numpy as np
from pocketrocks import BotDecision, DecisionContext
from pocketrocks.sim import BatchSimEngine, RevealRecord, ScoreRow, TurnRecord
from pocketrocks.sim.constants import ACTION_WIRE_IDS

from garboid_pocketrocks.bots.base import BotBrain, BotSpec
from garboid_pocketrocks.knowledge import RulesetKnowledge, canonical_knowledge
from garboid_pocketrocks.simulator.batch_context import build_batch_context
from garboid_pocketrocks.simulator.replay import MatchReplay
from garboid_pocketrocks.simulator.runner import (
    BotFault,
    FaultMode,
    MatchResult,
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


def run_batch_matches(jobs: Sequence[BatchGameJob]) -> tuple[MatchResult, ...]:
    """Run a nonempty homogeneous-player-count chunk in deterministic row order."""

    if not jobs:
        return ()
    player_count = jobs[0].player_count
    if any(job.player_count != player_count for job in jobs):
        raise ValueError("batch match jobs must share one player count")
    if any(len(job.lineup) != player_count for job in jobs):
        raise ValueError("batch match lineup length must match player count")

    engine = BatchSimEngine.start(
        player_count=player_count,
        seeds=tuple(job.seed for job in jobs),
        value_charts=tuple(job.value_chart for job in jobs),
        objectives_enabled=tuple(job.objectives_enabled for job in jobs),
    )
    brains: list[list[BotBrain | None]] = []
    faults: list[list[BotFault]] = [[] for _ in jobs]
    knowledge: list[RulesetKnowledge] = []
    decisions: list[list[tuple[int, tuple[tuple[int, BotDecision], ...]]]] = [[] for _ in jobs]
    turns: list[list[TurnRecord]] = [[] for _ in jobs]
    step_indices = [0] * len(jobs)

    for row, job in enumerate(jobs):
        brain_rng = random.Random(job.seed)
        row_brains: list[BotBrain | None] = []
        for seat, spec in enumerate(job.lineup):
            try:
                row_brains.append(spec.make_brain(seed=brain_rng.randrange(2**63)))
            except Exception as error:
                if job.fault_mode is FaultMode.RAISE:
                    raise
                row_brains.append(None)
                _record_fault(
                    faults[row],
                    turn_index=0,
                    seat=seat,
                    bot_name=spec.name,
                    error=error,
                )
        brains.append(row_brains)
        knowledge.append(
            canonical_knowledge(
                player_count,
                value_chart=job.value_chart,
                objectives_enabled=job.objectives_enabled,
            )
        )

    while True:
        action_ids = engine.flip_actions()
        active_rows = np.flatnonzero(action_ids)
        if not len(active_rows):
            break
        resources_before = engine.upcoming.copy()
        objective_claimants_before = engine.objective_claimants.copy()
        turn_indices = engine.turn_indices.astype(np.int64, copy=True)
        legal = engine.legal_max_bids()
        raw_bids = np.zeros((len(jobs), player_count), dtype=np.int16)

        for raw_row in active_rows:
            row = int(raw_row)
            job = jobs[row]
            action_id = int(action_ids[row])
            resource_ids = (
                int(resources_before[row, 0]),
                int(resources_before[row, 1]),
            )
            recorded: list[tuple[int, BotDecision]] = []
            for seat in range(player_count):
                context = build_batch_context(
                    engine,
                    row=row,
                    seat=seat,
                    decision_kind="submitBid",
                    action_id=action_id,
                    resource_ids=resource_ids,
                    turn_index=int(turn_indices[row]),
                    legal_max_amount=int(legal[row, seat]),
                )
                decision = _choose_decision(
                    brain=brains[row][seat],
                    context=context,
                    knowledge=knowledge[row],
                    fault_mode=job.fault_mode,
                    faults=faults[row],
                    turn_index=int(turn_indices[row]),
                    seat=seat,
                    bot_name=job.lineup[seat].name,
                )
                recorded.append((seat, decision))
                if decision.action_kind == "submitBid":
                    assert decision.value is not None
                    raw_bids[row, seat] = decision.value
            decisions[row].append((step_indices[row], tuple(recorded)))
            step_indices[row] += 1

        outcome = engine.resolve_bids(raw_bids)
        for raw_row in active_rows:
            row = int(raw_row)
            action_id = int(action_ids[row])
            winner = int(outcome.winner_seats[row])
            upcoming_before = tuple(
                int(resource) for resource in resources_before[row] if resource > 0
            )
            previously_claimed = {
                int(engine.objective_ids[row, index])
                for index, claimant in enumerate(objective_claimants_before[row])
                if claimant >= 0
            }
            claimed = tuple(
                int(objective_id)
                for index, objective_id in enumerate(engine.objective_ids[row])
                if (
                    objective_id > 0
                    and engine.objective_claimants[row, index] == winner
                    and int(objective_id) not in previously_claimed
                )
            )
            grant_count = _RESOURCE_GRANTS.get(action_id, 0)
            turns[row].append(
                TurnRecord(
                    turn_index=int(turn_indices[row]),
                    action=_ACTION_BY_WIRE_ID[action_id],
                    upcoming_before=upcoming_before,
                    raw_bids=tuple(int(value) for value in raw_bids[row]),
                    effective_bids=tuple(int(value) for value in outcome.effective_bids[row]),
                    winner_seat=winner,
                    paid=int(outcome.paid[row]),
                    bundle_suits=upcoming_before[:grant_count],
                    claimed_objective_wire_ids=claimed,
                    reveal=None,
                )
            )

        reveal_indices = np.full(len(jobs), -1, dtype=np.int16)
        for raw_row in active_rows:
            row = int(raw_row)
            mode = int(outcome.reveal_modes[row])
            if mode == 0:
                continue
            winner = int(outcome.winner_seats[row])
            reveal_index = 0
            auto = mode == 1
            if mode == 2:
                job = jobs[row]
                context = build_batch_context(
                    engine,
                    row=row,
                    seat=winner,
                    decision_kind="selectInfoToReveal",
                    action_id=int(action_ids[row]),
                    resource_ids=(
                        int(resources_before[row, 0]),
                        int(resources_before[row, 1]),
                    ),
                    turn_index=int(turn_indices[row]),
                    legal_max_amount=None,
                )
                decision = _choose_decision(
                    brain=brains[row][winner],
                    context=context,
                    knowledge=knowledge[row],
                    fault_mode=job.fault_mode,
                    faults=faults[row],
                    turn_index=int(engine.turn_indices[row]),
                    seat=winner,
                    bot_name=job.lineup[winner].name,
                )
                decisions[row].append((step_indices[row], ((winner, decision),)))
                step_indices[row] += 1
                assert decision.action_kind == "selectInfoToReveal"
                assert decision.value is not None
                reveal_index = decision.value
            reveal_indices[row] = reveal_index
            revealed_suit = int(engine.hand_cards[row, winner, reveal_index])
            turns[row][-1] = replace(
                turns[row][-1],
                reveal=RevealRecord(
                    seat=winner,
                    suit=revealed_suit,
                    auto=auto,
                ),
            )
        engine.apply_reveals(reveal_indices)

    scores = engine.scores()
    rankings = engine.rankings()
    results: list[MatchResult] = []
    for row, job in enumerate(jobs):
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
            for seat in range(player_count)
        )
        session_result = SessionResult(
            scores=tuple(
                SessionScore(
                    seat=seat,
                    final_money=int(scores.total[row, seat]),
                    rank=1
                    + sum(
                        int(scores.total[row, other]) > int(scores.total[row, seat])
                        for other in range(player_count)
                    ),
                )
                for seat in range(player_count)
            ),
            rows=score_rows,
            ranking=tuple(int(seat) for seat in rankings[row]),
        )
        replay = MatchReplay(
            schema_version=2,
            player_count=player_count,
            seed=job.seed,
            value_chart=job.value_chart.upper(),
            objectives_enabled=job.objectives_enabled,
            root_seed=None,
            game_index=None,
            bot_names=tuple(spec.name for spec in job.lineup),
            decisions=tuple(decisions[row]),
            turns=tuple(turns[row]),
            result=session_result,
        )
        results.append(
            MatchResult(
                result=session_result,
                events=(),
                turns=tuple(turns[row]),
                faults=tuple(faults[row]),
                replay=replay,
            )
        )
    return tuple(results)


def _choose_decision(
    *,
    brain: BotBrain | None,
    context: DecisionContext,
    knowledge: RulesetKnowledge,
    fault_mode: FaultMode,
    faults: list[BotFault],
    turn_index: int,
    seat: int,
    bot_name: str,
) -> BotDecision:
    if brain is None:
        return _fallback(context)
    try:
        decision = brain.choose_decision(context, knowledge)
        context.validate(decision)
        return decision
    except Exception as error:
        if fault_mode is FaultMode.RAISE:
            raise
        _record_fault(
            faults,
            turn_index=turn_index,
            seat=seat,
            bot_name=bot_name,
            error=error,
        )
        return _fallback(context)


def _fallback(context: DecisionContext) -> BotDecision:
    if context.decision_kind == "selectInfoToReveal":
        return BotDecision.select_info_to_reveal(0)
    return BotDecision.submit_bid(0)


def _record_fault(
    faults: list[BotFault],
    *,
    turn_index: int,
    seat: int,
    bot_name: str,
    error: Exception,
) -> None:
    faults.append(
        BotFault(
            turn_index=turn_index,
            seat=seat,
            bot_name=bot_name,
            error_type=type(error).__name__,
            message=str(error),
        )
    )
