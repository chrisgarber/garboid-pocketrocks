"""Reproduce the early/middle/late two-resource auction diagnostic dataset."""

from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import batched

from pocketrocks import ActionId

from garboid_pocketrocks.bots import (
    AggressiveHeuristicBot,
    BalancedHeuristicBot,
    BotSpec,
    PassiveHeuristicBot,
)
from garboid_pocketrocks.heuristics.profiles import (
    AGGRESSIVE_PROFILE,
    BALANCED_PROFILE,
    PASSIVE_PROFILE,
)
from garboid_pocketrocks.heuristics.valuation import HeuristicValuator
from garboid_pocketrocks.rules import live_ruleset
from garboid_pocketrocks.simulator.engine import GameEngine
from garboid_pocketrocks.simulator.events import EventKind
from garboid_pocketrocks.simulator.model import Phase
from garboid_pocketrocks.simulator.monte_carlo import (
    GameJob,
    MonteCarloConfig,
    MonteCarloRunner,
    _execute_job,
)
from garboid_pocketrocks.simulator.sampling import WeightedRulesetSampler

GAMES = 100_000
ROOT_SEED = 20260729
BOT_CLASSES = (
    AggressiveHeuristicBot,
    BalancedHeuristicBot,
    PassiveHeuristicBot,
)
PROFILES = {
    "aggressive": AGGRESSIVE_PROFILE,
    "balanced": BALANCED_PROFILE,
    "passive": PASSIVE_PROFILE,
}
RESOURCE_ACTIONS = (ActionId.AUCTION1, ActionId.AUCTION2)


def band(turn: int) -> str:
    if turn <= 5:
        return "early_1_5"
    if turn <= 12:
        return "middle_6_12"
    return "late_13_plus"


@dataclass
class BidBucket:
    requests: int = 0
    cash_total: int = 0
    legal_max_total: int = 0
    bid_total: int = 0
    reservation_total: int = 0
    gross_total: float = 0.0
    resource_total: float = 0.0
    objective_total: float = 0.0
    liquidity_cost_total: float = 0.0
    cash_zero: int = 0
    passes: int = 0
    cap_binding: int = 0

    def merge(self, other: BidBucket) -> None:
        for field in self.__dataclass_fields__:
            setattr(self, field, getattr(self, field) + getattr(other, field))

    def freeze(self) -> dict[str, float | int]:
        n = self.requests or 1
        return {
            "requests": self.requests,
            "mean_cash": round(self.cash_total / n, 4),
            "mean_legal_max": round(self.legal_max_total / n, 4),
            "mean_bid": round(self.bid_total / n, 4),
            "mean_reservation": round(self.reservation_total / n, 4),
            "mean_gross_value": round(self.gross_total / n, 4),
            "mean_resource_value": round(self.resource_total / n, 4),
            "mean_objective_value": round(self.objective_total / n, 4),
            "mean_liquidity_cost_at_chosen_bid": round(self.liquidity_cost_total / n, 4),
            "cash_zero_rate": round(100 * self.cash_zero / n, 4),
            "pass_rate": round(100 * self.passes / n, 4),
            "cap_binding_rate_when_cash_positive": round(
                100 * self.cap_binding / max(self.requests - self.cash_zero, 1), 4
            ),
        }


@dataclass
class WinBucket:
    wins: int = 0
    payments: int = 0

    def merge(self, other: WinBucket) -> None:
        self.wins += other.wins
        self.payments += other.payments

    def freeze(self, games: int) -> dict[str, float | int]:
        return {
            "wins": self.wins,
            "mean_winning_bid": round(self.payments / self.wins if self.wins else 0, 4),
            "wins_per_game": round(self.wins / games, 4),
            "payments_per_game": round(self.payments / games, 4),
        }


@dataclass
class Chunk:
    bids: dict[tuple[str, ActionId, str], BidBucket]
    auction2_turns: dict[tuple[str, int], BidBucket]
    wins: dict[tuple[str, ActionId, str], WinBucket]
    market_wins: dict[tuple[ActionId, str], WinBucket]
    terminal_cash: dict[str, int]
    games: int


def analyze_chunk(jobs: tuple[GameJob, ...]) -> Chunk:
    bids: defaultdict[tuple[str, ActionId, str], BidBucket] = defaultdict(BidBucket)
    auction2_turns: defaultdict[tuple[str, int], BidBucket] = defaultdict(BidBucket)
    wins: defaultdict[tuple[str, ActionId, str], WinBucket] = defaultdict(WinBucket)
    market_wins: defaultdict[tuple[ActionId, str], WinBucket] = defaultdict(WinBucket)
    terminal_cash: defaultdict[str, int] = defaultdict(int)
    valuators = {name: HeuristicValuator(profile) for name, profile in PROFILES.items()}

    for job in jobs:
        completed = _execute_job(job)
        transition = GameEngine.start(job.ruleset, player_count=3, seed=job.seed)
        knowledge = job.ruleset.knowledge(3)
        for _, recorded_decisions in completed.match.replay.decisions:
            assert transition.pending is not None
            decisions = dict(recorded_decisions)
            if transition.state.phase is Phase.BIDDING:
                assert transition.state.current_action is not None
                action = transition.state.current_action.action_id
                turn = transition.state.turn_index + 1
                if action in RESOURCE_ACTIONS:
                    for seat, context in transition.pending.contexts:
                        bot = job.lineup[seat].name
                        result = valuators[bot].evaluate_bid(context, knowledge)
                        submitted = decisions[seat].value or 0
                        chosen = result.points[result.chosen_bid].breakdown
                        bucket = bids[(bot, action, band(turn))]
                        bucket.requests += 1
                        cash = context.cash_by_seat[seat]
                        bucket.cash_total += cash
                        bucket.legal_max_total += context.legal_max_amount or 0
                        bucket.bid_total += submitted
                        bucket.reservation_total += result.reservation_bid
                        bucket.resource_total += chosen.resource
                        bucket.objective_total += (
                            chosen.objective_completion + chosen.objective_progress
                        )
                        bucket.gross_total += (
                            chosen.resource
                            + chosen.objective_completion
                            + chosen.objective_progress
                        )
                        bucket.liquidity_cost_total += -chosen.liquidity
                        bucket.cash_zero += int(cash == 0)
                        bucket.passes += int(submitted == 0)
                        bucket.cap_binding += int(
                            cash > 0
                            and result.reservation_bid == context.legal_max_amount
                            and result.points[-1].win_delta >= 0
                        )
                        if action is ActionId.AUCTION2:
                            auction2_turns[(bot, turn)].merge(
                                bucket.__class__(
                                    requests=1,
                                    cash_total=cash,
                                    legal_max_total=context.legal_max_amount or 0,
                                    bid_total=submitted,
                                    reservation_total=result.reservation_bid,
                                    gross_total=(
                                        chosen.resource
                                        + chosen.objective_completion
                                        + chosen.objective_progress
                                    ),
                                    resource_total=chosen.resource,
                                    objective_total=(
                                        chosen.objective_completion + chosen.objective_progress
                                    ),
                                    liquidity_cost_total=-chosen.liquidity,
                                    cash_zero=int(cash == 0),
                                    passes=int(submitted == 0),
                                    cap_binding=int(
                                        cash > 0
                                        and result.reservation_bid == context.legal_max_amount
                                        and result.points[-1].win_delta >= 0
                                    ),
                                )
                            )

            transition = GameEngine.step(transition.state, decisions)
            for event in transition.events:
                if event.kind is EventKind.AUCTION_RESOLVED and event.action_id in RESOURCE_ACTIONS:
                    assert event.seat is not None
                    assert event.amount is not None
                    assert event.turn_index is not None
                    bot = job.lineup[event.seat].name
                    key = (bot, event.action_id, band(event.turn_index + 1))
                    wins[key].wins += 1
                    wins[key].payments += event.amount
                    market = market_wins[(event.action_id, band(event.turn_index + 1))]
                    market.wins += 1
                    market.payments += event.amount

        for seat, player in enumerate(transition.state.players):
            terminal_cash[job.lineup[seat].name] += player.cash

    return Chunk(
        bids=dict(bids),
        auction2_turns=dict(auction2_turns),
        wins=dict(wins),
        market_wins=dict(market_wins),
        terminal_cash=dict(terminal_cash),
        games=len(jobs),
    )


def main() -> None:
    charts = tuple(live_ruleset(chart) for chart in "ABCDE")
    config = MonteCarloConfig(
        bot_specs=tuple(BotSpec.from_bot_class(bot) for bot in BOT_CLASSES),
        games=GAMES,
        player_counts=(3,),
        ruleset_sampler=WeightedRulesetSampler(tuple((ruleset, 1) for ruleset in charts)),
        root_seed=ROOT_SEED,
    )
    chunks = tuple(
        tuple(chunk) for chunk in batched(MonteCarloRunner.plan(config), 25, strict=False)
    )
    bids: defaultdict[tuple[str, ActionId, str], BidBucket] = defaultdict(BidBucket)
    auction2_turns: defaultdict[tuple[str, int], BidBucket] = defaultdict(BidBucket)
    wins: defaultdict[tuple[str, ActionId, str], WinBucket] = defaultdict(WinBucket)
    market_wins: defaultdict[tuple[ActionId, str], WinBucket] = defaultdict(WinBucket)
    terminal_cash: defaultdict[str, int] = defaultdict(int)
    total_games = 0
    with ProcessPoolExecutor(max_workers=16) as executor:
        for result in executor.map(analyze_chunk, chunks):
            total_games += result.games
            for key, value in result.bids.items():
                bids[key].merge(value)
            for key, value in result.auction2_turns.items():
                auction2_turns[key].merge(value)
            for key, value in result.wins.items():
                wins[key].merge(value)
            for key, value in result.market_wins.items():
                market_wins[key].merge(value)
            for key, value in result.terminal_cash.items():
                terminal_cash[key] += value

    bands = ("early_1_5", "middle_6_12", "late_13_plus")
    payload = {
        "configuration": {
            "games": GAMES,
            "players": 3,
            "bots": tuple(PROFILES),
            "charts": tuple("ABCDE"),
            "chart_sampling": "equal-weight deterministic sampling",
            "root_seed": ROOT_SEED,
            "workers": 16,
        },
        "games": total_games,
        "auction2_by_band": {
            name: {period: bids[(name, ActionId.AUCTION2, period)].freeze() for period in bands}
            for name in PROFILES
        },
        "auction2_market_by_band": {
            period: market_wins[(ActionId.AUCTION2, period)].freeze(total_games) for period in bands
        },
        "resource_payments_per_game": {
            name: {
                action.name: {
                    period: wins[(name, action, period)].freeze(total_games) for period in bands
                }
                for action in RESOURCE_ACTIONS
            }
            for name in PROFILES
        },
        "mean_terminal_cash": {
            name: round(terminal_cash[name] / total_games, 4) for name in PROFILES
        },
        "auction2_by_turn": {
            name: {
                str(turn): auction2_turns[(name, turn)].freeze()
                for turn in range(1, 23)
                if auction2_turns[(name, turn)].requests
            }
            for name in PROFILES
        },
    }
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
