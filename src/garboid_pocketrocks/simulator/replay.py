from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from pocketrocks import ActionId, BotDecision
from pocketrocks.types import decisionActionKind

from garboid_pocketrocks.rules import PlayerSetup, Ruleset
from garboid_pocketrocks.simulator.engine import GameEngine
from garboid_pocketrocks.simulator.errors import SimulationError
from garboid_pocketrocks.simulator.events import EventKind, GameEvent
from garboid_pocketrocks.simulator.model import GameResult, Score


class ReplayDivergence(SimulationError):
    """Raised when recorded decisions no longer reproduce a replay."""


@dataclass(frozen=True, slots=True)
class MatchReplay:
    schema_version: int
    ruleset: Ruleset
    player_count: int
    seed: int
    root_seed: int | None
    game_index: int | None
    bot_names: tuple[str, ...]
    decisions: tuple[tuple[int, tuple[tuple[int, BotDecision], ...]], ...]
    events: tuple[GameEvent, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "ruleset": _ruleset_to_dict(self.ruleset),
            "player_count": self.player_count,
            "seed": self.seed,
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
            "events": [_event_to_dict(event) for event in self.events],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MatchReplay:
        version = int(payload["schema_version"])
        if version != 1:
            raise ReplayDivergence(f"unsupported replay schema version {version}")
        return cls(
            schema_version=version,
            ruleset=_ruleset_from_dict(cast(dict[str, Any], payload["ruleset"])),
            player_count=int(payload["player_count"]),
            seed=int(payload["seed"]),
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
                        for by_seat in cast(
                            list[dict[str, Any]],
                            item["by_seat"],
                        )
                    ),
                )
                for item in cast(list[dict[str, Any]], payload["decisions"])
            ),
            events=tuple(
                _event_from_dict(item) for item in cast(list[dict[str, Any]], payload["events"])
            ),
        )


@dataclass(frozen=True, slots=True)
class ReplayedMatch:
    events: tuple[GameEvent, ...]
    result: GameResult


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
    transition = GameEngine.start(
        replay.ruleset,
        player_count=replay.player_count,
        seed=replay.seed,
    )
    regenerated_events = list(transition.events)
    for expected_step, (recorded_step, decisions) in enumerate(replay.decisions):
        if recorded_step != expected_step:
            raise ReplayDivergence(f"expected decision step {expected_step}, got {recorded_step}")
        if transition.pending is None:
            raise ReplayDivergence(f"decision step {expected_step} has no pending batch")
        recorded_seats = tuple(seat for seat, _ in decisions)
        if recorded_seats != transition.pending.acting_seats:
            raise ReplayDivergence(
                f"decision step {expected_step} expected seats "
                f"{transition.pending.acting_seats}, got {recorded_seats}"
            )
        transition = GameEngine.step(transition.state, dict(decisions))
        regenerated_events.extend(transition.events)
    if transition.result is None:
        raise ReplayDivergence("recorded decisions ended before the game terminated")
    recorded_engine_events = tuple(
        event
        for event in replay.events
        if event.kind not in (EventKind.BOT_FAULT, EventKind.FALLBACK_APPLIED)
    )
    if tuple(regenerated_events) != recorded_engine_events:
        raise ReplayDivergence("regenerated event stream differs from replay")
    return ReplayedMatch(events=replay.events, result=transition.result)


def _optional_int(value: object) -> int | None:
    return None if value is None else int(cast(int | str, value))


def _ruleset_to_dict(ruleset: Ruleset) -> dict[str, Any]:
    return {
        "name": ruleset.name,
        "resource_counts": list(ruleset.resource_counts),
        "action_counts": list(ruleset.action_counts),
        "player_setups": [
            {
                "player_count": setup.player_count,
                "starting_cash": setup.starting_cash,
                "private_cards_per_player": setup.private_cards_per_player,
            }
            for setup in ruleset.player_setups
        ],
        "value_chart": list(ruleset.value_chart),
        "objective_pool": list(ruleset.objective_pool),
        "active_objective_count": ruleset.active_objective_count,
        "objectives_enabled": ruleset.objectives_enabled,
    }


def _ruleset_from_dict(payload: dict[str, Any]) -> Ruleset:
    return Ruleset(
        name=str(payload["name"]),
        resource_counts=tuple(int(value) for value in payload["resource_counts"]),
        action_counts=tuple(int(value) for value in payload["action_counts"]),
        player_setups=tuple(
            PlayerSetup(
                player_count=int(setup["player_count"]),
                starting_cash=int(setup["starting_cash"]),
                private_cards_per_player=int(setup["private_cards_per_player"]),
            )
            for setup in payload["player_setups"]
        ),
        value_chart=tuple(int(value) for value in payload["value_chart"]),
        objective_pool=tuple(int(value) for value in payload["objective_pool"]),
        active_objective_count=int(payload["active_objective_count"]),
        objectives_enabled=bool(payload["objectives_enabled"]),
    )


def _event_to_dict(event: GameEvent) -> dict[str, Any]:
    return {
        "kind": event.kind.value,
        "turn_index": event.turn_index,
        "seat": event.seat,
        "action_id": int(event.action_id) if event.action_id is not None else None,
        "amount": event.amount,
        "resource_ids": (list(event.resource_ids) if event.resource_ids is not None else None),
        "objective_ids": (list(event.objective_ids) if event.objective_ids is not None else None),
        "scores": (
            [
                {
                    "seat": score.seat,
                    "final_money": score.final_money,
                    "rank": score.rank,
                }
                for score in event.scores
            ]
            if event.scores is not None
            else None
        ),
        "automatic": event.automatic,
    }


def _event_from_dict(payload: dict[str, Any]) -> GameEvent:
    scores_payload = payload.get("scores")
    return GameEvent(
        kind=EventKind(str(payload["kind"])),
        turn_index=_optional_int(payload.get("turn_index")),
        seat=_optional_int(payload.get("seat")),
        action_id=(
            ActionId(int(payload["action_id"])) if payload.get("action_id") is not None else None
        ),
        amount=_optional_int(payload.get("amount")),
        resource_ids=(
            tuple(int(value) for value in payload["resource_ids"])
            if payload.get("resource_ids") is not None
            else None
        ),
        objective_ids=(
            tuple(int(value) for value in payload["objective_ids"])
            if payload.get("objective_ids") is not None
            else None
        ),
        scores=(
            tuple(
                Score(
                    seat=int(score["seat"]),
                    final_money=int(score["final_money"]),
                    rank=int(score["rank"]),
                )
                for score in cast(list[dict[str, Any]], scores_payload)
            )
            if scores_payload is not None
            else None
        ),
        automatic=(bool(payload["automatic"]) if payload.get("automatic") is not None else None),
    )
