from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from pocketrocks import BotDecision, DecisionContext
from pocketrocks.sim import TurnRecord

from garboid_pocketrocks.bots.base import BotBrain, BotSpec
from garboid_pocketrocks.knowledge import canonical_knowledge
from garboid_pocketrocks.simulator.replay import MatchReplay
from garboid_pocketrocks.simulator.session import SdkGameSession, SessionResult


class FaultMode(StrEnum):
    RAISE = "raise"
    RECORD_AND_PASS = "record_and_pass"


@dataclass(frozen=True, slots=True)
class BotFault:
    turn_index: int
    seat: int
    bot_name: str
    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class MatchResult:
    result: SessionResult
    events: tuple[object, ...]
    turns: tuple[TurnRecord, ...]
    faults: tuple[BotFault, ...]
    replay: MatchReplay


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
    ) -> MatchResult:
        if len(lineup) != player_count:
            raise ValueError(f"lineup has {len(lineup)} bots but player_count is {player_count}")
        brain_rng = random.Random(seed)
        brains: list[BotBrain | None] = []
        faults: list[BotFault] = []
        for seat, spec in enumerate(lineup):
            try:
                brains.append(spec.make_brain(seed=brain_rng.randrange(2**63)))
            except Exception as error:
                if fault_mode is FaultMode.RAISE:
                    raise
                brains.append(None)
                _record_fault(
                    faults,
                    turn_index=0,
                    seat=seat,
                    bot_name=spec.name,
                    error=error,
                )

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
        step_index = 0
        while not session.terminated:
            decisions_by_seat: dict[int, BotDecision] = {}
            for seat, context in session.pending.contexts:
                brain = brains[seat]
                if brain is None:
                    decision = _fallback(context)
                else:
                    try:
                        decision = brain.choose_decision(context, knowledge)
                        context.validate(decision)
                    except Exception as error:
                        if fault_mode is FaultMode.RAISE:
                            raise
                        _record_fault(
                            faults,
                            turn_index=session.snapshot.turn_index,
                            seat=seat,
                            bot_name=lineup[seat].name,
                            error=error,
                        )
                        decision = _fallback(context)
                decisions_by_seat[seat] = decision
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
        )


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
