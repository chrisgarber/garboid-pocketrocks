from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Literal

from pocketrocks import BotDecision, DecisionContext
from pocketrocks.exceptions import InvalidBotDecision
from pocketrocks.sim import ScoreRow, SimEngine, TurnRecord
from pocketrocks.sim.context import build_sim_context

from garboid_pocketrocks.simulator.errors import (
    ActingSeatsError,
    IllegalDecisionError,
    InvalidPhaseError,
)

type Seat = int
type DecisionKind = Literal["submitBid", "selectInfoToReveal"]


@dataclass(frozen=True, slots=True)
class PlayerSnapshot:
    seat: Seat
    cash: int
    hand_suits: tuple[int, ...]
    won_suits: tuple[int, ...]
    revealed_suits: tuple[int, ...]
    loans: tuple[int, ...]
    investments: tuple[tuple[int, int], ...]
    objective_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SessionSnapshot:
    turn_index: int
    tiebreak_seat: Seat
    current_action: str | None
    game_over: bool
    players: tuple[PlayerSnapshot, ...]


@dataclass(frozen=True, slots=True)
class SessionScore:
    seat: Seat
    final_money: int
    rank: int


@dataclass(frozen=True, slots=True)
class SessionResult:
    scores: tuple[SessionScore, ...]
    rows: tuple[ScoreRow, ...]
    ranking: tuple[Seat, ...]


@dataclass(frozen=True, slots=True)
class PendingDecisions:
    decision_kind: DecisionKind
    contexts: tuple[tuple[Seat, DecisionContext], ...]

    @property
    def acting_seats(self) -> tuple[Seat, ...]:
        return tuple(seat for seat, _context in self.contexts)

    @property
    def contexts_by_seat(self) -> Mapping[Seat, DecisionContext]:
        return MappingProxyType(dict(self.contexts))


@dataclass(frozen=True, slots=True)
class SessionTransition:
    before: SessionSnapshot
    snapshot: SessionSnapshot
    pending: PendingDecisions | None
    result: SessionResult | None
    decisions: tuple[tuple[Seat, BotDecision], ...]
    events: tuple[object, ...]
    turn_records: tuple[TurnRecord, ...]

    @property
    def terminated(self) -> bool:
        return self.result is not None


class SdkGameSession:
    """Synchronous per-decision orchestration over the SDK rules engine."""

    _DECISION_BUDGET_MS = 60_000

    def __init__(self, engine: SimEngine) -> None:
        self._engine = engine
        self._pending: PendingDecisions | None = None
        self._result: SessionResult | None = None
        self._advance_to_bid_or_terminal()

    @classmethod
    def start(
        cls,
        *,
        player_count: int,
        seed: str | int,
        value_chart: str = "A",
        objectives_enabled: bool = True,
        player_names: Sequence[str] | None = None,
    ) -> SdkGameSession:
        return cls(
            SimEngine(
                player_count,
                str(seed),
                value_chart=value_chart,
                objectives_enabled=objectives_enabled,
                player_names=player_names,
            )
        )

    @property
    def sdk_engine(self) -> SimEngine:
        return self._engine

    @property
    def pending(self) -> PendingDecisions:
        if self._pending is None:
            raise InvalidPhaseError("terminal session has no pending decisions")
        return self._pending

    @property
    def snapshot(self) -> SessionSnapshot:
        return _snapshot(self._engine)

    @property
    def result(self) -> SessionResult | None:
        return self._result

    @property
    def terminated(self) -> bool:
        return self._result is not None

    @property
    def history(self) -> tuple[TurnRecord, ...]:
        return tuple(self._engine.history)

    @property
    def events(self) -> tuple[object, ...]:
        return tuple(self._engine.events)

    def step(
        self,
        decisions_by_seat: Mapping[Seat, BotDecision],
    ) -> SessionTransition:
        if self._pending is None:
            raise InvalidPhaseError("cannot step a terminal session")
        pending = self._pending
        actual = set(decisions_by_seat)
        expected = set(pending.acting_seats)
        if actual != expected:
            raise ActingSeatsError(
                f"decision_kind={pending.decision_kind} expected seats={sorted(expected)} "
                f"received seats={sorted(actual)}"
            )
        for seat, context in pending.contexts:
            decision = decisions_by_seat[seat]
            try:
                context.validate(decision)
            except InvalidBotDecision as error:
                raise IllegalDecisionError(
                    f"decision_kind={pending.decision_kind} seat={seat} "
                    f"decision={decision}: {error}"
                ) from error

        before = self.snapshot
        event_count = len(self._engine.events)
        history_before = tuple(self._engine.history)
        decisions = tuple(sorted(decisions_by_seat.items()))
        if pending.decision_kind == "submitBid":
            self._resolve_bids(decisions_by_seat)
        else:
            self._resolve_reveal(decisions_by_seat)
        history_after = tuple(self._engine.history)
        if len(history_after) > len(history_before):
            changed_turns = history_after[len(history_before) :]
        elif history_after and history_after != history_before:
            changed_turns = (history_after[-1],)
        else:
            changed_turns = ()
        return SessionTransition(
            before=before,
            snapshot=self.snapshot,
            pending=self._pending,
            result=self._result,
            decisions=decisions,
            events=tuple(self._engine.events[event_count:]),
            turn_records=changed_turns,
        )

    def _resolve_bids(self, decisions_by_seat: Mapping[Seat, BotDecision]) -> None:
        bids = tuple(
            _bid_amount(decisions_by_seat[seat]) for seat in range(len(self._engine.players))
        )
        outcome = self._engine.resolve(bids)
        if outcome.reveal_needed == "auto":
            self._engine.apply_reveal(outcome.winner_seat, 0, auto=True)
            self._advance_to_bid_or_terminal()
        elif outcome.reveal_needed == "choice":
            self._pending = self._reveal_pending(outcome.winner_seat)
        else:
            self._advance_to_bid_or_terminal()

    def _resolve_reveal(self, decisions_by_seat: Mapping[Seat, BotDecision]) -> None:
        seat = self.pending.acting_seats[0]
        decision = decisions_by_seat[seat]
        index = decision.value if decision.action_kind == "selectInfoToReveal" else 0
        assert index is not None
        self._engine.apply_reveal(seat, index, auto=False)
        self._advance_to_bid_or_terminal()

    def _advance_to_bid_or_terminal(self) -> None:
        action = self._engine.flip_action()
        if action is None:
            self._pending = None
            self._result = _result(self._engine)
            return
        self._pending = PendingDecisions(
            decision_kind="submitBid",
            contexts=tuple(
                (
                    seat,
                    _deterministic_context(
                        build_sim_context(
                            self._engine,
                            seat,
                            "submitBid",
                            budget_ms=self._DECISION_BUDGET_MS,
                            turn_index=self._engine.turn_index,
                        )
                    ),
                )
                for seat in range(len(self._engine.players))
            ),
        )

    def _reveal_pending(self, seat: Seat) -> PendingDecisions:
        return PendingDecisions(
            decision_kind="selectInfoToReveal",
            contexts=(
                (
                    seat,
                    _deterministic_context(
                        build_sim_context(
                            self._engine,
                            seat,
                            "selectInfoToReveal",
                            budget_ms=self._DECISION_BUDGET_MS,
                            turn_index=self._engine.turn_index - 1,
                        )
                    ),
                ),
            ),
        )


def _bid_amount(decision: BotDecision) -> int:
    if decision.action_kind != "submitBid":
        return 0
    assert decision.value is not None
    return decision.value


def _deterministic_context(context: DecisionContext) -> DecisionContext:
    return replace(
        context,
        deadline_at=2**63 - 1,
        received_at=0,
    )


def _snapshot(engine: SimEngine) -> SessionSnapshot:
    return SessionSnapshot(
        turn_index=engine.turn_index,
        tiebreak_seat=engine.tiebreak_seat,
        current_action=engine.current_action,
        game_over=engine.game_over,
        players=tuple(
            PlayerSnapshot(
                seat=player.seat,
                cash=player.cash,
                hand_suits=tuple(player.hand_suits),
                won_suits=tuple(player.won_suits),
                revealed_suits=tuple(player.revealed_suits),
                loans=tuple(player.loans),
                investments=tuple(player.investments),
                objective_ids=tuple(player.objective_wire_ids),
            )
            for player in engine.players
        ),
    )


def _result(engine: SimEngine) -> SessionResult:
    rows = tuple(engine.score())
    ranking = tuple(engine.ranking())
    return SessionResult(
        scores=tuple(
            SessionScore(
                seat=row.seat,
                final_money=row.total,
                rank=1 + sum(other.total > row.total for other in rows),
            )
            for row in rows
        ),
        rows=rows,
        ranking=ranking,
    )
