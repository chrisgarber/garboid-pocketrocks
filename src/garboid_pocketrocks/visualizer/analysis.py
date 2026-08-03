"""Two aggregate engines for tournament-wide and single-bot insight reports."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from pocketrocks.sim.constants import ACTION_WIRE_IDS

from garboid_pocketrocks.visualizer.loading import TournamentDataset

_ACTION_BY_WIRE_ID = {wire_id: name for name, wire_id in ACTION_WIRE_IDS.items()}
_DISPLAY_ACTION = {
    "Auction1": "Auction 1 resource",
    "Auction2": "Auction 2 resources",
    "Loan10": "Loan $10",
    "Loan20": "Loan $20",
    "Invest5": "Invest $5",
    "Invest10": "Invest $10",
}
_INVESTMENT_VALUE = {"Invest5": 5, "Invest10": 10}


@dataclass(slots=True)
class _PairAccumulator:
    games: int = 0
    score: float = 0.0


@dataclass(slots=True)
class _ObjectiveAccumulator:
    games: int = 0
    games_with_objective: int = 0
    objectives: int = 0


@dataclass(slots=True)
class _CashAccumulator:
    requests: int = 0
    passes: int = 0
    cash_zero: int = 0
    hard_constrained: int = 0
    cap_binding: int = 0
    cash_total: int = 0
    bid_total: int = 0

    def record(self, *, cash: int, legal_maximum: int, bid: int) -> None:
        self.requests += 1
        self.passes += int(bid == 0)
        self.cash_zero += int(cash == 0)
        self.hard_constrained += int(legal_maximum == 0)
        self.cap_binding += int(legal_maximum > 0 and bid == legal_maximum)
        self.cash_total += cash
        self.bid_total += bid


@dataclass(slots=True)
class _BotAccumulator:
    pairs: dict[str, _PairAccumulator] = field(
        default_factory=lambda: defaultdict(_PairAccumulator)
    )
    objectives: dict[str, _ObjectiveAccumulator] = field(
        default_factory=lambda: defaultdict(_ObjectiveAccumulator)
    )
    profits: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    investment_prices: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    loans: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    action_wins: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    cash_by_phase: dict[str, _CashAccumulator] = field(
        default_factory=lambda: defaultdict(_CashAccumulator)
    )
    cash_by_turn: dict[int, _CashAccumulator] = field(
        default_factory=lambda: defaultdict(_CashAccumulator)
    )
    cash_by_action: dict[str, _CashAccumulator] = field(
        default_factory=lambda: defaultdict(_CashAccumulator)
    )
    score_components: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))


class TournamentInsightsEngine:
    """Build field-wide rating, matchup, condition, and calibration insights."""

    def __init__(self, dataset: TournamentDataset) -> None:
        self.dataset = dataset

    def build(self) -> dict[str, Any]:
        summary = self.dataset.summary
        leaderboard = [dict(row) for row in summary["leaderboard"]]
        names = {str(row["bot_id"]): str(row["bot_name"]) for row in leaderboard}
        pair_accumulators = _pairwise_accumulators(self.dataset.games)
        matchups = [
            {
                "bot_id": bot_id,
                "bot_name": names.get(bot_id, bot_id),
                "opponent_id": opponent_id,
                "opponent_name": names.get(opponent_id, opponent_id),
                "games": accumulator.games,
                "score": accumulator.score / accumulator.games,
                "lower": _wilson(accumulator.score, accumulator.games)[0],
                "upper": _wilson(accumulator.score, accumulator.games)[1],
            }
            for (bot_id, opponent_id), accumulator in sorted(pair_accumulators.items())
        ]
        conditions = []
        for row in summary.get("condition_statistics", []):
            games = int(row["games"])
            wins = int(row["outright_wins"])
            lower, upper = _wilson(wins, games)
            conditions.append(
                {
                    **row,
                    "bot_name": names.get(str(row["bot_id"]), str(row["bot_id"])),
                    "win_rate": wins / games if games else 0.0,
                    "win_rate_lower": lower,
                    "win_rate_upper": upper,
                }
            )
        return {
            "configuration": summary.get("configuration", {}),
            "leaderboard": leaderboard,
            "matchups": matchups,
            "conditions": conditions,
            "calibration": summary.get("calibration", []),
            "pair_outcomes": summary.get("pair_outcomes", 0),
            "availability": {
                "game_summaries": bool(self.dataset.games),
                "game_details": bool(self.dataset.game_details),
                "decision_traces": self.dataset.decision_traces_path is not None,
            },
        }


class BotInsightsEngine:
    """Aggregate behavior once and expose a focused report for any field bot."""

    def __init__(self, dataset: TournamentDataset) -> None:
        self.dataset = dataset
        leaderboard = dataset.summary["leaderboard"]
        self._rows = {str(row["bot_id"]): dict(row) for row in leaderboard}
        self._names = {bot_id: str(row["bot_name"]) for bot_id, row in self._rows.items()}
        self._bots = {bot_id: _BotAccumulator() for bot_id in self._rows}
        self._consume_game_summaries()
        self._consume_game_details()
        self._consume_decision_traces()

    def build_all(self) -> dict[str, dict[str, Any]]:
        return {bot_id: self.build(bot_id) for bot_id in self._rows}

    def build(self, bot_id: str) -> dict[str, Any]:
        if bot_id not in self._rows:
            raise KeyError(f"unknown tournament bot {bot_id!r}")
        row = self._rows[bot_id]
        accumulator = self._bots[bot_id]
        conditions = [
            item
            for item in self.dataset.summary.get("condition_statistics", [])
            if str(item["bot_id"]) == bot_id
        ]
        value_charts = []
        for item in conditions:
            games = int(item["games"])
            wins = int(item["outright_wins"])
            lower, upper = _wilson(wins, games)
            value_charts.append(
                {
                    "chart": item["chart"],
                    "player_count": item["player_count"],
                    "games": games,
                    "win_rate": wins / games if games else 0.0,
                    "lower": lower,
                    "upper": upper,
                    "mean_finish": item["mean_normalized_finish"],
                    "mean_money": item["mean_final_money"],
                }
            )
        pair_rows = []
        for opponent_id, pair in sorted(
            accumulator.pairs.items(), key=lambda item: (-item[1].score / item[1].games, item[0])
        ):
            lower, upper = _wilson(pair.score, pair.games)
            pair_rows.append(
                {
                    "opponent_id": opponent_id,
                    "opponent_name": self._names.get(opponent_id, opponent_id),
                    "games": pair.games,
                    "score": pair.score / pair.games,
                    "lower": lower,
                    "upper": upper,
                }
            )
        objective_rows = []
        for opponent_id, objective in sorted(accumulator.objectives.items()):
            lower, upper = _wilson(objective.games_with_objective, objective.games)
            objective_rows.append(
                {
                    "opponent_id": opponent_id,
                    "opponent_name": self._names.get(opponent_id, opponent_id),
                    "games": objective.games,
                    "games_with_objective": objective.games_with_objective,
                    "game_rate": (
                        objective.games_with_objective / objective.games if objective.games else 0.0
                    ),
                    "lower": lower,
                    "upper": upper,
                    "objectives_per_100_games": (
                        100 * objective.objectives / objective.games if objective.games else 0.0
                    ),
                }
            )
        return {
            "bot_id": bot_id,
            "bot_name": self._names[bot_id],
            "summary": row,
            "opponents": pair_rows,
            "objectives": objective_rows,
            "auction_profit": [
                {"action": _DISPLAY_ACTION.get(action, action), **_distribution(values)}
                for action, values in sorted(accumulator.profits.items())
            ],
            "investment_prices": [
                {
                    "action": _DISPLAY_ACTION.get(action, action),
                    "fixed_profit": _INVESTMENT_VALUE[action],
                    **_distribution(values),
                }
                for action, values in sorted(accumulator.investment_prices.items())
            ],
            "loans": [
                {
                    "action": _DISPLAY_ACTION.get(action, action),
                    "principal": 10 if action == "Loan10" else 20,
                    "mean_upfront_liquidity": (10 if action == "Loan10" else 20)
                    - statistics.mean(values),
                    **_distribution(values),
                }
                for action, values in sorted(accumulator.loans.items())
            ],
            "action_wins": [
                {
                    "action": _DISPLAY_ACTION.get(action, action),
                    "wins": wins,
                    "per_100_games": 100 * wins / int(row["games"]) if row["games"] else 0.0,
                }
                for action, wins in sorted(accumulator.action_wins.items())
            ],
            "value_charts": value_charts,
            "cash_by_phase": [
                {"phase": phase, **_cash_payload(cash)}
                for phase, cash in sorted(
                    accumulator.cash_by_phase.items(),
                    key=lambda item: ("early", "middle", "late").index(item[0]),
                )
            ],
            "cash_by_turn": [
                {"turn": turn, **_cash_payload(cash)}
                for turn, cash in sorted(accumulator.cash_by_turn.items())
            ],
            "bidding_by_action": [
                {"action": _DISPLAY_ACTION.get(action, action), **_cash_payload(cash)}
                for action, cash in sorted(accumulator.cash_by_action.items())
            ],
            "score_components": [
                {
                    "component": component,
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                }
                for component, values in sorted(accumulator.score_components.items())
                if values
            ],
            "availability": {
                "opponents": bool(accumulator.pairs),
                "objectives_and_auctions": bool(self.dataset.game_details),
                "cash_pressure": self.dataset.decision_traces_path is not None,
            },
        }

    def _consume_game_summaries(self) -> None:
        pairwise = _pairwise_accumulators(self.dataset.games)
        for (bot_id, opponent_id), pair in pairwise.items():
            if bot_id in self._bots:
                self._bots[bot_id].pairs[opponent_id] = pair

    def _consume_game_details(self) -> None:
        for game in self.dataset.game_details:
            bot_ids = tuple(str(value) for value in game["bot_ids"])
            chart = tuple(int(value) for value in game["value_chart"])
            objectives_by_seat: dict[int, int] = defaultdict(int)
            for turn in game["turns"]:
                winner = int(turn["winner_seat"])
                bot_id = bot_ids[winner]
                if bot_id not in self._bots:
                    continue
                action = str(turn["action"])
                accumulator = self._bots[bot_id]
                accumulator.action_wins[action] += 1
                objectives_by_seat[winner] += len(turn["claimed_objective_ids"])
                paid = int(turn["paid"])
                if action in ("Loan10", "Loan20"):
                    accumulator.loans[action].append(paid)
                elif action in ("Auction1", "Auction2"):
                    gross_value = sum(chart[int(suit) - 1] for suit in turn["bundle_suits"])
                    accumulator.profits[action].append(gross_value - paid)
                elif action in _INVESTMENT_VALUE:
                    accumulator.investment_prices[action].append(paid)

            for seat, bot_id in enumerate(bot_ids):
                if bot_id not in self._bots:
                    continue
                objective_count = objectives_by_seat[seat]
                for opponent_id in bot_ids:
                    if opponent_id == bot_id:
                        continue
                    objective = self._bots[bot_id].objectives[opponent_id]
                    objective.games += 1
                    objective.games_with_objective += int(objective_count > 0)
                    objective.objectives += objective_count
            for score in game["scores"]:
                seat = int(score["seat"])
                bot_id = bot_ids[seat]
                if bot_id not in self._bots:
                    continue
                for key in (
                    "cash",
                    "items_value",
                    "objectives_value",
                    "investments_value",
                    "loans_value",
                ):
                    self._bots[bot_id].score_components[key].append(int(score[key]))

    def _consume_decision_traces(self) -> None:
        for trace in self.dataset.iter_decision_traces():
            bot_id = str(trace["bot_id"])
            if bot_id not in self._bots:
                continue
            context = trace["context"]
            if context["decision_kind"] != "submitBid":
                continue
            seat = int(trace["seat"])
            cash = int(context["cash_by_seat"][seat])
            legal_maximum = int(context["legal_max_amount"])
            selected = trace["selected_action"]
            bid = int(selected["value"] or 0) if selected["action_kind"] == "submitBid" else 0
            turn = int(trace["turn_index"]) + 1
            phase = "early" if turn <= 5 else "middle" if turn <= 12 else "late"
            action = _ACTION_BY_WIRE_ID.get(int(context["current_action_id"]), "Unknown")
            accumulator = self._bots[bot_id]
            accumulator.cash_by_phase[phase].record(cash=cash, legal_maximum=legal_maximum, bid=bid)
            accumulator.cash_by_turn[turn].record(cash=cash, legal_maximum=legal_maximum, bid=bid)
            accumulator.cash_by_action[action].record(
                cash=cash, legal_maximum=legal_maximum, bid=bid
            )


def _pairwise_accumulators(
    games: tuple[dict[str, Any], ...],
) -> dict[tuple[str, str], _PairAccumulator]:
    pairs: dict[tuple[str, str], _PairAccumulator] = defaultdict(_PairAccumulator)
    for game in games:
        bot_ids = tuple(str(value) for value in game["bot_ids"])
        ranks = {int(score["seat"]): int(score["rank"]) for score in game["scores"]}
        for seat, bot_id in enumerate(bot_ids):
            for opponent_seat, opponent_id in enumerate(bot_ids):
                if seat == opponent_seat:
                    continue
                pair = pairs[bot_id, opponent_id]
                pair.games += 1
                pair.score += (
                    1.0
                    if ranks[seat] < ranks[opponent_seat]
                    else 0.5
                    if ranks[seat] == ranks[opponent_seat]
                    else 0.0
                )
    return dict(pairs)


def _wilson(successes: float, trials: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0:
        return (0.0, 0.0)
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    spread = (
        z
        * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials))
        / denominator
    )
    return (max(0.0, center - spread), min(1.0, center + spread))


def _quantile(values: list[int], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _distribution(values: list[int]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "positive_rate": sum(value > 0 for value in values) / len(values),
        "p10": _quantile(values, 0.10),
        "p25": _quantile(values, 0.25),
        "median": _quantile(values, 0.50),
        "p75": _quantile(values, 0.75),
        "p90": _quantile(values, 0.90),
    }


def _cash_payload(cash: _CashAccumulator) -> dict[str, float | int]:
    requests = cash.requests
    return {
        "requests": requests,
        "pass_rate": cash.passes / requests if requests else 0.0,
        "cash_zero_rate": cash.cash_zero / requests if requests else 0.0,
        "hard_constrained_rate": cash.hard_constrained / requests if requests else 0.0,
        "cap_binding_rate": cash.cap_binding / requests if requests else 0.0,
        "mean_cash": cash.cash_total / requests if requests else 0.0,
        "mean_bid": cash.bid_total / requests if requests else 0.0,
    }
