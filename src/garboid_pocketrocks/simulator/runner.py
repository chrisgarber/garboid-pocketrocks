from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from pocketrocks import BotDecision

from garboid_pocketrocks.bots.base import BotBrain, BotSpec
from garboid_pocketrocks.rules import Ruleset
from garboid_pocketrocks.simulator.engine import GameEngine
from garboid_pocketrocks.simulator.events import EventKind, GameEvent
from garboid_pocketrocks.simulator.model import GameResult
from garboid_pocketrocks.simulator.replay import MatchReplay


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
    result: GameResult
    events: tuple[GameEvent, ...]
    faults: tuple[BotFault, ...]
    replay: MatchReplay


class MatchRunner:
    @staticmethod
    def run(
        lineup: Sequence[BotSpec],
        *,
        ruleset: Ruleset,
        player_count: int,
        seed: int,
        fault_mode: FaultMode = FaultMode.RAISE,
    ) -> MatchResult:
        if len(lineup) != player_count:
            raise ValueError(f"lineup has {len(lineup)} bots but player_count is {player_count}")
        brain_rng = random.Random(seed)
        brains: list[BotBrain | None] = []
        faults: list[BotFault] = []
        external_events: list[GameEvent] = []
        for seat, spec in enumerate(lineup):
            try:
                brains.append(spec.make_brain(seed=brain_rng.randrange(2**63)))
            except Exception as error:
                if fault_mode is FaultMode.RAISE:
                    raise
                brains.append(None)
                _record_fault(
                    faults,
                    external_events,
                    turn_index=0,
                    seat=seat,
                    bot_name=spec.name,
                    error=error,
                )

        transition = GameEngine.start(
            ruleset,
            player_count=player_count,
            seed=seed,
        )
        events = list(transition.events)
        decisions: list[tuple[int, tuple[tuple[int, BotDecision], ...]]] = []
        step_index = 0
        while transition.result is None:
            assert transition.pending is not None
            decisions_by_seat: dict[int, BotDecision] = {}
            for seat, context in transition.pending.contexts:
                brain = brains[seat]
                if brain is None:
                    decision = BotDecision.pass_turn()
                else:
                    try:
                        decision = brain.choose_decision(
                            context,
                            ruleset.knowledge(player_count),
                        )
                        context.validate(decision)
                    except Exception as error:
                        if fault_mode is FaultMode.RAISE:
                            raise
                        _record_fault(
                            faults,
                            external_events,
                            turn_index=transition.state.turn_index,
                            seat=seat,
                            bot_name=lineup[seat].name,
                            error=error,
                        )
                        decision = BotDecision.pass_turn()
                decisions_by_seat[seat] = decision
            recorded = tuple(sorted(decisions_by_seat.items()))
            decisions.append((step_index, recorded))
            transition = GameEngine.step(transition.state, decisions_by_seat)
            events.extend(external_events)
            external_events.clear()
            events.extend(transition.events)
            step_index += 1

        replay = MatchReplay(
            schema_version=1,
            ruleset=ruleset,
            player_count=player_count,
            seed=seed,
            root_seed=None,
            game_index=None,
            bot_names=tuple(spec.name for spec in lineup),
            decisions=tuple(decisions),
            events=tuple(events),
        )
        return MatchResult(
            result=transition.result,
            events=tuple(events),
            faults=tuple(faults),
            replay=replay,
        )


def _record_fault(
    faults: list[BotFault],
    events: list[GameEvent],
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
    events.extend(
        (
            GameEvent(
                EventKind.BOT_FAULT,
                turn_index=turn_index,
                seat=seat,
            ),
            GameEvent(
                EventKind.FALLBACK_APPLIED,
                turn_index=turn_index,
                seat=seat,
            ),
        )
    )
