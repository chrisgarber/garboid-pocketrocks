from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pocketrocks import BotDecision
from pocketrocks.sim import TurnRecord

from garboid_pocketrocks.adapters.public_history import public_history_from_sdk_events
from garboid_pocketrocks.bots.base import BotSpec
from garboid_pocketrocks.knowledge import canonical_knowledge
from garboid_pocketrocks.simulator.bot_execution import BotFault as BotFault
from garboid_pocketrocks.simulator.bot_execution import FaultMode as FaultMode
from garboid_pocketrocks.simulator.bot_execution import (
    choose_brain_decision,
    initialize_brains,
)
from garboid_pocketrocks.simulator.replay import MatchReplay
from garboid_pocketrocks.simulator.session import SdkGameSession, SessionResult


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
        step_index = 0
        while not session.terminated:
            decisions_by_seat: dict[int, BotDecision] = {}
            history = public_history_from_sdk_events(session.events)
            for seat, context in session.pending.contexts:
                decision = choose_brain_decision(
                    brain=brains[seat],
                    context=context,
                    knowledge=knowledge,
                    history=history,
                    fault_mode=fault_mode,
                    faults=faults,
                    turn_index=session.snapshot.turn_index,
                    seat=seat,
                    bot_name=lineup[seat].name,
                )
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
