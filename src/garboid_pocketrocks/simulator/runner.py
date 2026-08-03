from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace

from pocketrocks import BotDecision, DecisionContext
from pocketrocks.sim import TurnRecord

from garboid_pocketrocks.adapters.public_history import (
    PublicHistory,
    public_history_from_sdk_events,
)
from garboid_pocketrocks.bots.base import BotSpec
from garboid_pocketrocks.diagnostics.trace import (
    DecisionTrace,
    PendingDecisionTrace,
    PublicDecisionOutcome,
    RecordedAction,
    legal_actions_for_context,
    public_context_from_sdk,
)
from garboid_pocketrocks.knowledge import canonical_knowledge
from garboid_pocketrocks.simulator.bot_execution import BotFault as BotFault
from garboid_pocketrocks.simulator.bot_execution import (
    DecisionExecution,
    execute_brain_decision,
    initialize_brains,
)
from garboid_pocketrocks.simulator.bot_execution import FaultMode as FaultMode
from garboid_pocketrocks.simulator.replay import MatchReplay
from garboid_pocketrocks.simulator.session import SdkGameSession, SessionResult


@dataclass(frozen=True, slots=True)
class MatchResult:
    result: SessionResult
    events: tuple[object, ...]
    turns: tuple[TurnRecord, ...]
    faults: tuple[BotFault, ...]
    replay: MatchReplay
    decision_traces: tuple[DecisionTrace, ...] = ()


class MatchRunner:
    @staticmethod
    def run(
        lineup: Sequence[BotSpec],
        *,
        player_count: int,
        seed: int,
        value_chart: str = "A",
        objectives_enabled: bool = True,
        fault_mode: FaultMode = FaultMode.RAISE,
        game_index: int | None = None,
        capture_decision_traces: bool = False,
    ) -> MatchResult:
        if len(lineup) != player_count:
            raise ValueError(f"lineup has {len(lineup)} bots but player_count is {player_count}")
        brains, construction_faults = initialize_brains(
            lineup,
            seed=seed,
            fault_mode=fault_mode,
        )
        faults = list(construction_faults)

        session = SdkGameSession.start(
            player_count=player_count,
            seed=seed,
            value_chart=value_chart,
            objectives_enabled=objectives_enabled,
            player_names=tuple(spec.name for spec in lineup),
        )
        knowledge = canonical_knowledge(
            player_count,
            value_chart=value_chart,
            objectives_enabled=objectives_enabled,
        )
        decisions: list[tuple[int, tuple[tuple[int, BotDecision], ...]]] = []
        pending_traces: list[PendingDecisionTrace] = []
        step_index = 0
        while not session.terminated:
            decisions_by_seat: dict[int, BotDecision] = {}
            history = public_history_from_sdk_events(session.events)
            for seat, context in session.pending.contexts:
                execution = execute_brain_decision(
                    brain=brains[seat],
                    context=context,
                    knowledge=knowledge,
                    history=history,
                    fault_mode=fault_mode,
                    faults=faults,
                    turn_index=session.snapshot.turn_index,
                    seat=seat,
                    bot_name=lineup[seat].name,
                    request_explanation=capture_decision_traces,
                )
                decision = execution.decision
                decisions_by_seat[seat] = decision
                if capture_decision_traces:
                    pending_traces.append(
                        _build_pending_decision_trace(
                            game_index=game_index,
                            chart=value_chart,
                            step_index=step_index,
                            turn_index=session.snapshot.turn_index,
                            seat=seat,
                            lineup=lineup,
                            context=context,
                            public_history=history,
                            execution=execution,
                        )
                    )
            recorded = tuple(sorted(decisions_by_seat.items()))
            decisions.append((step_index, recorded))
            session.step(decisions_by_seat)
            step_index += 1

        result = session.result
        assert result is not None
        replay = MatchReplay(
            schema_version=2,
            player_count=player_count,
            seed=seed,
            value_chart=value_chart.upper(),
            objectives_enabled=objectives_enabled,
            root_seed=None,
            game_index=None,
            bot_names=tuple(spec.name for spec in lineup),
            decisions=tuple(decisions),
            turns=session.history,
            result=result,
        )
        return MatchResult(
            result=result,
            events=session.events,
            turns=session.history,
            faults=tuple(faults),
            replay=replay,
            decision_traces=_finalize_decision_traces(pending_traces, result),
        )


def _build_pending_decision_trace(
    *,
    game_index: int | None,
    chart: str,
    step_index: int,
    turn_index: int,
    seat: int,
    lineup: Sequence[BotSpec],
    context: DecisionContext,
    public_history: PublicHistory,
    execution: DecisionExecution,
) -> PendingDecisionTrace:
    return PendingDecisionTrace(
        game_index=game_index,
        chart=chart.upper(),
        step_index=step_index,
        turn_index=turn_index,
        seat=seat,
        bot_name=lineup[seat].name,
        bot_id=lineup[seat].bot_id,
        bot_names_by_seat=tuple(spec.name for spec in lineup),
        bot_ids_by_seat=tuple(spec.bot_id for spec in lineup),
        context=public_context_from_sdk(
            dataclass_replace(
                context,
                owned_objective_ids_by_seat=tuple(
                    tuple(sorted(objective_ids))
                    for objective_ids in context.owned_objective_ids_by_seat
                ),
            )
        ),
        public_history=public_history,
        legal_actions=legal_actions_for_context(context),
        selected_action=RecordedAction.from_decision(execution.decision),
        explanation=execution.explanation,
        selection_source=execution.selection_source,
        result_metrics=execution.result_metrics,
    )


def _finalize_decision_traces(
    pending_traces: Sequence[PendingDecisionTrace],
    result: SessionResult,
) -> tuple[DecisionTrace, ...]:
    scores_by_seat = {score.seat: score for score in result.scores}
    first_place_count = sum(score.rank == 1 for score in result.scores)
    return tuple(
        DecisionTrace.from_pending(
            pending,
            PublicDecisionOutcome(
                rank=scores_by_seat[pending.seat].rank,
                final_money=scores_by_seat[pending.seat].final_money,
                first_place_tied=(scores_by_seat[pending.seat].rank == 1 and first_place_count > 1),
            ),
        )
        for pending in sorted(
            pending_traces,
            key=lambda trace: (trace.step_index, trace.seat),
        )
    )
