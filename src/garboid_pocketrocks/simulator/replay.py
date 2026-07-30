from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from pocketrocks import BotDecision
from pocketrocks.sim import RevealRecord, ScoreRow, TurnRecord
from pocketrocks.types import decisionActionKind

from garboid_pocketrocks.simulator.errors import SimulationError
from garboid_pocketrocks.simulator.session import (
    SdkGameSession,
    SessionResult,
    SessionScore,
)


class ReplayDivergence(SimulationError):
    """Raised when recorded SDK decisions no longer reproduce a replay."""


@dataclass(frozen=True, slots=True)
class MatchReplay:
    schema_version: int
    player_count: int
    seed: int
    value_chart: str
    objectives_enabled: bool
    root_seed: int | None
    game_index: int | None
    bot_names: tuple[str, ...]
    decisions: tuple[tuple[int, tuple[tuple[int, BotDecision], ...]], ...]
    turns: tuple[TurnRecord, ...]
    result: SessionResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "configuration": {
                "player_count": self.player_count,
                "seed": self.seed,
                "value_chart": self.value_chart,
                "objectives_enabled": self.objectives_enabled,
            },
            "root_seed": self.root_seed,
            "game_index": self.game_index,
            "bot_names": list(self.bot_names),
            "decisions": [
                {
                    "step": step,
                    "by_seat": [
                        {
                            "seat": seat,
                            "action_kind": decision.action_kind,
                            "value": decision.value,
                        }
                        for seat, decision in decisions
                    ],
                }
                for step, decisions in self.decisions
            ],
            "turns": [_turn_to_dict(turn) for turn in self.turns],
            "result": _result_to_dict(self.result),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MatchReplay:
        version = int(payload["schema_version"])
        if version == 1:
            raise ReplayDivergence("replay schema version 1 used the removed project game engine")
        if version != 2:
            raise ReplayDivergence(f"unsupported replay schema version {version}")
        configuration = cast(dict[str, Any], payload["configuration"])
        return cls(
            schema_version=version,
            player_count=int(configuration["player_count"]),
            seed=int(configuration["seed"]),
            value_chart=str(configuration["value_chart"]),
            objectives_enabled=bool(configuration["objectives_enabled"]),
            root_seed=_optional_int(payload.get("root_seed")),
            game_index=_optional_int(payload.get("game_index")),
            bot_names=tuple(str(name) for name in cast(list[object], payload["bot_names"])),
            decisions=tuple(
                (
                    int(item["step"]),
                    tuple(
                        (
                            int(by_seat["seat"]),
                            BotDecision(
                                action_kind=cast(
                                    decisionActionKind,
                                    by_seat["action_kind"],
                                ),
                                value=_optional_int(by_seat.get("value")),
                            ),
                        )
                        for by_seat in cast(list[dict[str, Any]], item["by_seat"])
                    ),
                )
                for item in cast(list[dict[str, Any]], payload["decisions"])
            ),
            turns=tuple(
                _turn_from_dict(item) for item in cast(list[dict[str, Any]], payload["turns"])
            ),
            result=_result_from_dict(cast(dict[str, Any], payload["result"])),
        )


@dataclass(frozen=True, slots=True)
class ReplayedMatch:
    events: tuple[object, ...]
    turns: tuple[TurnRecord, ...]
    result: SessionResult


def save_replay(replay: MatchReplay, path: Path) -> None:
    path.write_text(
        json.dumps(replay.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_replay(path: Path) -> MatchReplay:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ReplayDivergence("replay root must be a JSON object")
    return MatchReplay.from_dict(payload)


def replay_match(replay: MatchReplay) -> ReplayedMatch:
    session = SdkGameSession.start(
        player_count=replay.player_count,
        seed=replay.seed,
        value_chart=replay.value_chart,
        objectives_enabled=replay.objectives_enabled,
        player_names=replay.bot_names,
    )
    for expected_step, (recorded_step, decisions) in enumerate(replay.decisions):
        if recorded_step != expected_step:
            raise ReplayDivergence(f"expected decision step {expected_step}, got {recorded_step}")
        recorded_seats = tuple(seat for seat, _decision in decisions)
        if recorded_seats != session.pending.acting_seats:
            raise ReplayDivergence(
                f"decision step {expected_step} expected seats "
                f"{session.pending.acting_seats}, got {recorded_seats}"
            )
        try:
            session.step(dict(decisions))
        except SimulationError as error:
            raise ReplayDivergence(
                f"decision step {expected_step} is no longer valid: {error}"
            ) from error
    if not session.terminated or session.result is None:
        raise ReplayDivergence("recorded decisions ended before the SDK game terminated")
    if session.history != replay.turns:
        raise ReplayDivergence("SDK turn history differs from replay")
    if session.result != replay.result:
        raise ReplayDivergence("SDK terminal result differs from replay")
    return ReplayedMatch(
        events=session.events,
        turns=session.history,
        result=session.result,
    )


def _turn_to_dict(turn: TurnRecord) -> dict[str, Any]:
    return {
        "turn_index": turn.turn_index,
        "action": turn.action,
        "upcoming_before": list(turn.upcoming_before),
        "raw_bids": list(turn.raw_bids),
        "effective_bids": list(turn.effective_bids),
        "winner_seat": turn.winner_seat,
        "paid": turn.paid,
        "bundle_suits": list(turn.bundle_suits),
        "claimed_objective_wire_ids": list(turn.claimed_objective_wire_ids),
        "reveal": (
            {
                "seat": turn.reveal.seat,
                "suit": turn.reveal.suit,
                "auto": turn.reveal.auto,
            }
            if turn.reveal is not None
            else None
        ),
    }


def _turn_from_dict(payload: dict[str, Any]) -> TurnRecord:
    reveal_payload = payload.get("reveal")
    reveal = (
        RevealRecord(
            seat=int(reveal_payload["seat"]),
            suit=int(reveal_payload["suit"]),
            auto=bool(reveal_payload["auto"]),
        )
        if isinstance(reveal_payload, dict)
        else None
    )
    return TurnRecord(
        turn_index=int(payload["turn_index"]),
        action=str(payload["action"]),
        upcoming_before=tuple(int(value) for value in payload["upcoming_before"]),
        raw_bids=tuple(int(value) for value in payload["raw_bids"]),
        effective_bids=tuple(int(value) for value in payload["effective_bids"]),
        winner_seat=int(payload["winner_seat"]),
        paid=int(payload["paid"]),
        bundle_suits=tuple(int(value) for value in payload["bundle_suits"]),
        claimed_objective_wire_ids=tuple(
            int(value) for value in payload["claimed_objective_wire_ids"]
        ),
        reveal=reveal,
    )


def _result_to_dict(result: SessionResult) -> dict[str, Any]:
    return {
        "scores": [
            {
                "seat": score.seat,
                "final_money": score.final_money,
                "rank": score.rank,
            }
            for score in result.scores
        ],
        "rows": [
            {
                "seat": row.seat,
                "name": row.name,
                "cash": row.cash,
                "items_value": row.items_value,
                "objectives_value": row.objectives_value,
                "investments_value": row.investments_value,
                "loans_value": row.loans_value,
                "total": row.total,
            }
            for row in result.rows
        ],
        "ranking": list(result.ranking),
    }


def _result_from_dict(payload: dict[str, Any]) -> SessionResult:
    return SessionResult(
        scores=tuple(
            SessionScore(
                seat=int(score["seat"]),
                final_money=int(score["final_money"]),
                rank=int(score["rank"]),
            )
            for score in cast(list[dict[str, Any]], payload["scores"])
        ),
        rows=tuple(
            ScoreRow(
                seat=int(row["seat"]),
                name=str(row["name"]),
                cash=int(row["cash"]),
                items_value=int(row["items_value"]),
                objectives_value=int(row["objectives_value"]),
                investments_value=int(row["investments_value"]),
                loans_value=int(row["loans_value"]),
                total=int(row["total"]),
            )
            for row in cast(list[dict[str, Any]], payload["rows"])
        ),
        ranking=tuple(int(seat) for seat in payload["ranking"]),
    )


def _optional_int(value: object) -> int | None:
    return None if value is None else int(cast(int | str, value))
