"""Reproduce auction-turn, information-asymmetry, and value-chart datasets."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from itertools import batched

from pocketrocks import ActionId, DecisionContext

from garboid_pocketrocks.bots import (
    AggressiveHeuristicBot,
    BalancedHeuristicBot,
    BotSpec,
    PassiveHeuristicBot,
)
from garboid_pocketrocks.heuristics.belief import build_belief
from garboid_pocketrocks.heuristics.reveals import _expected_price
from garboid_pocketrocks.rules import RulesetKnowledge, live_ruleset
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

BOT_CLASSES = (
    AggressiveHeuristicBot,
    BalancedHeuristicBot,
    PassiveHeuristicBot,
)
BOT_NAMES = tuple(bot.BOT_NAME for bot in BOT_CLASSES)
RESOURCE_ACTIONS = (ActionId.AUCTION1, ActionId.AUCTION2)
RESOURCE_ACTION_NAMES = {
    ActionId.AUCTION1: "Auction 1 resource",
    ActionId.AUCTION2: "Auction 2 resources",
}
ACTION_NAMES = {
    ActionId.AUCTION1: "Auction 1 resource",
    ActionId.AUCTION2: "Auction 2 resources",
    ActionId.LOAN10: "Loan $10",
    ActionId.LOAN20: "Loan $20",
    ActionId.INVEST5: "Invest $5",
    ActionId.INVEST10: "Invest $10",
}
LOAN_ACTIONS = (ActionId.LOAN10, ActionId.LOAN20)
GAMES = 100_000
ROOT_SEED = 20260729
CHUNK_SIZE = 25


@dataclass
class NumberBucket:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.total_sq += value * value

    def merge(self, other: NumberBucket) -> None:
        self.count += other.count
        self.total += other.total
        self.total_sq += other.total_sq

    def freeze(self) -> dict[str, float | int]:
        mean = self.total / self.count if self.count else 0.0
        variance = max(0.0, self.total_sq / self.count - mean * mean) if self.count else 0.0
        return {
            "count": self.count,
            "mean": round(mean, 4),
            "stddev": round(math.sqrt(variance), 4),
        }


@dataclass
class IntegerBucket:
    count: int = 0
    total: int = 0
    histogram: Counter[int] = field(default_factory=Counter)

    def add(self, value: int) -> None:
        self.count += 1
        self.total += value
        self.histogram[value] += 1

    def merge(self, other: IntegerBucket) -> None:
        self.count += other.count
        self.total += other.total
        self.histogram.update(other.histogram)

    def quantile(self, probability: float) -> int:
        target = math.ceil(probability * self.count)
        seen = 0
        for value, count in sorted(self.histogram.items()):
            seen += count
            if seen >= target:
                return value
        return 0

    def freeze(self) -> dict[str, float | int]:
        positive_count = self.count - self.histogram[0]
        positive_total = sum(value * count for value, count in self.histogram.items() if value > 0)
        return {
            "count": self.count,
            "mean": round(self.total / self.count if self.count else 0.0, 4),
            "mean_paid": round(
                positive_total / positive_count if positive_count else 0.0,
                4,
            ),
            "free_rate": round(
                100 * self.histogram[0] / self.count if self.count else 0.0,
                4,
            ),
            "p10": self.quantile(0.10),
            "p25": self.quantile(0.25),
            "median": self.quantile(0.50),
            "p75": self.quantile(0.75),
            "p90": self.quantile(0.90),
            "max": max(self.histogram) if self.histogram else 0,
        }


@dataclass
class InfoBucket:
    count: int = 0
    bid_per_card_total: float = 0.0
    premium_per_card_total: float = 0.0

    def add(self, *, bid_per_card: float, premium_per_card: float) -> None:
        self.count += 1
        self.bid_per_card_total += bid_per_card
        self.premium_per_card_total += premium_per_card

    def merge(self, other: InfoBucket) -> None:
        self.count += other.count
        self.bid_per_card_total += other.bid_per_card_total
        self.premium_per_card_total += other.premium_per_card_total

    def freeze(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "mean_bid_per_card": round(
                self.bid_per_card_total / self.count if self.count else 0.0,
                4,
            ),
            "mean_private_premium_per_card": round(
                self.premium_per_card_total / self.count if self.count else 0.0,
                4,
            ),
        }


@dataclass
class ChunkSummary:
    submitted_by_turn: dict[tuple[str, ActionId, int], NumberBucket]
    winning_by_turn: dict[tuple[ActionId, int], IntegerBucket]
    winning_by_chart: dict[tuple[str, ActionId], IntegerBucket]
    winning_by_chart_bot: dict[tuple[str, str, ActionId], IntegerBucket]
    bot_scores_by_chart: dict[tuple[str, str], IntegerBucket]
    winning_scores_by_chart: dict[str, IntegerBucket]
    info: dict[tuple[str, str, int], InfoBucket]
    all_winning: dict[tuple[str, ActionId], IntegerBucket]
    loan_winning_by_turn: dict[tuple[str, ActionId, int], IntegerBucket]


def _offered_resource_counts(
    context: DecisionContext,
    action: ActionId,
) -> tuple[int, ...]:
    counts = [0] * 5
    if action in RESOURCE_ACTIONS:
        for resource_id in context.current_resource_ids:
            if resource_id:
                counts[resource_id - 1] += 1
    return tuple(counts)


def _public_expected_prices(
    context: object,
    knowledge: RulesetKnowledge,
) -> tuple[float, ...]:
    revealed_by_suit = tuple(
        sum(row[index] for row in context.revealed_info_counts_by_seat) for index in range(5)
    )
    won_by_suit = tuple(
        sum(row[index] for row in context.won_resource_counts_by_seat) for index in range(5)
    )
    offered_by_suit = [0] * 5
    for resource_id in context.current_resource_ids:
        if resource_id:
            offered_by_suit[resource_id - 1] += 1
    unseen_by_suit = tuple(
        total - revealed - won - offered
        for total, revealed, won, offered in zip(
            knowledge.resource_counts,
            revealed_by_suit,
            won_by_suit,
            offered_by_suit,
            strict=True,
        )
    )
    hidden_slots = sum(
        knowledge.private_cards_per_player - sum(row)
        for row in context.revealed_info_counts_by_seat
    )
    unseen_population = sum(unseen_by_suit)
    return tuple(
        _expected_price(
            known_reveals=revealed,
            unseen_suit_count=unseen,
            unseen_population=unseen_population,
            hidden_slots=hidden_slots,
            value_chart=context.value_chart,
        )
        for revealed, unseen in zip(revealed_by_suit, unseen_by_suit, strict=True)
    )


def _analyze_chunk(jobs: tuple[GameJob, ...]) -> ChunkSummary:
    submitted_by_turn: defaultdict[tuple[str, ActionId, int], NumberBucket] = defaultdict(
        NumberBucket
    )
    winning_by_turn: defaultdict[tuple[ActionId, int], IntegerBucket] = defaultdict(IntegerBucket)
    winning_by_chart: defaultdict[tuple[str, ActionId], IntegerBucket] = defaultdict(IntegerBucket)
    winning_by_chart_bot: defaultdict[tuple[str, str, ActionId], IntegerBucket] = defaultdict(
        IntegerBucket
    )
    bot_scores_by_chart: defaultdict[tuple[str, str], IntegerBucket] = defaultdict(IntegerBucket)
    winning_scores_by_chart: defaultdict[str, IntegerBucket] = defaultdict(IntegerBucket)
    info: defaultdict[tuple[str, str, int], InfoBucket] = defaultdict(InfoBucket)
    all_winning: defaultdict[tuple[str, ActionId], IntegerBucket] = defaultdict(IntegerBucket)
    loan_winning_by_turn: defaultdict[tuple[str, ActionId, int], IntegerBucket] = defaultdict(
        IntegerBucket
    )

    for job in jobs:
        completed = _execute_job(job)
        replay = completed.match.replay
        transition = GameEngine.start(
            job.ruleset,
            player_count=job.player_count,
            seed=job.seed,
        )
        knowledge = job.ruleset.knowledge(job.player_count)
        chart = job.ruleset.name.removeprefix("live-")

        for _, recorded_decisions in replay.decisions:
            assert transition.pending is not None
            decisions = dict(recorded_decisions)
            if transition.state.phase is Phase.BIDDING:
                assert transition.state.current_action is not None
                action = transition.state.current_action.action_id
                turn = transition.state.turn_index + 1
                if action in RESOURCE_ACTIONS:
                    contexts = transition.pending.contexts_by_seat
                    offered_counts = _offered_resource_counts(contexts[0], action)
                    offered_suits = {
                        index + 1 for index, count in enumerate(offered_counts) if count
                    }
                    offered_count = sum(offered_counts)
                    for seat in range(job.player_count):
                        context = contexts[seat]
                        decision = decisions[seat]
                        bid = decision.value or 0
                        bot = job.lineup[seat].name
                        submitted_by_turn[(bot, action, turn)].add(bid)

                        hidden_matches = sum(
                            suit_id in offered_suits for suit_id in context.current_hand_suit_ids
                        )
                        hidden_bucket = min(hidden_matches, 3)
                        belief = build_belief(context, knowledge)
                        public_prices = _public_expected_prices(context, knowledge)
                        private_value = sum(
                            count * suit.expected_terminal_price
                            for count, suit in zip(
                                offered_counts,
                                belief.suits,
                                strict=True,
                            )
                        )
                        public_value = sum(
                            count * price
                            for count, price in zip(
                                offered_counts,
                                public_prices,
                                strict=True,
                            )
                        )
                        info[(chart, bot, hidden_bucket)].add(
                            bid_per_card=bid / offered_count,
                            premium_per_card=(private_value - public_value) / offered_count,
                        )

            transition = GameEngine.step(transition.state, decisions)
            for event in transition.events:
                if event.kind is not EventKind.AUCTION_RESOLVED:
                    continue
                assert event.action_id is not None
                assert event.turn_index is not None
                assert event.amount is not None
                assert event.seat is not None
                winner_bot = job.lineup[event.seat].name
                all_winning[(winner_bot, event.action_id)].add(event.amount)
                if event.action_id in LOAN_ACTIONS:
                    loan_winning_by_turn[(winner_bot, event.action_id, event.turn_index + 1)].add(
                        event.amount
                    )
                if event.action_id not in RESOURCE_ACTIONS:
                    continue
                action = event.action_id
                turn = event.turn_index + 1
                winning_by_turn[(action, turn)].add(event.amount)
                winning_by_chart[(chart, action)].add(event.amount)
                winning_by_chart_bot[(chart, winner_bot, action)].add(event.amount)

        scores = completed.match.result.scores
        winning_money = max(score.final_money for score in scores)
        winning_scores_by_chart[chart].add(winning_money)
        for score in scores:
            bot_scores_by_chart[(chart, job.lineup[score.seat].name)].add(score.final_money)

    return ChunkSummary(
        submitted_by_turn=dict(submitted_by_turn),
        winning_by_turn=dict(winning_by_turn),
        winning_by_chart=dict(winning_by_chart),
        winning_by_chart_bot=dict(winning_by_chart_bot),
        bot_scores_by_chart=dict(bot_scores_by_chart),
        winning_scores_by_chart=dict(winning_scores_by_chart),
        info=dict(info),
        all_winning=dict(all_winning),
        loan_winning_by_turn=dict(loan_winning_by_turn),
    )


def _merge_number(
    target: defaultdict[tuple, NumberBucket],
    source: dict[tuple, NumberBucket],
) -> None:
    for key, bucket in source.items():
        target[key].merge(bucket)


def _merge_integer(
    target: defaultdict[tuple, IntegerBucket],
    source: dict[tuple, IntegerBucket],
) -> None:
    for key, bucket in source.items():
        target[key].merge(bucket)


def _merge_info(
    target: defaultdict[tuple, InfoBucket],
    source: dict[tuple, InfoBucket],
) -> None:
    for key, bucket in source.items():
        target[key].merge(bucket)


def main() -> None:
    charts = tuple(live_ruleset(chart) for chart in "ABCDE")
    config = MonteCarloConfig(
        bot_specs=tuple(BotSpec.from_bot_class(bot) for bot in BOT_CLASSES),
        games=GAMES,
        player_counts=(3,),
        ruleset_sampler=WeightedRulesetSampler(tuple((ruleset, 1) for ruleset in charts)),
        root_seed=ROOT_SEED,
    )
    jobs = MonteCarloRunner.plan(config)
    chunks = tuple(tuple(chunk) for chunk in batched(jobs, CHUNK_SIZE, strict=False))

    submitted_by_turn: defaultdict[tuple, NumberBucket] = defaultdict(NumberBucket)
    winning_by_turn: defaultdict[tuple, IntegerBucket] = defaultdict(IntegerBucket)
    winning_by_chart: defaultdict[tuple, IntegerBucket] = defaultdict(IntegerBucket)
    winning_by_chart_bot: defaultdict[tuple, IntegerBucket] = defaultdict(IntegerBucket)
    bot_scores_by_chart: defaultdict[tuple, IntegerBucket] = defaultdict(IntegerBucket)
    winning_scores_by_chart: defaultdict[tuple, IntegerBucket] = defaultdict(IntegerBucket)
    info: defaultdict[tuple, InfoBucket] = defaultdict(InfoBucket)
    all_winning: defaultdict[tuple, IntegerBucket] = defaultdict(IntegerBucket)
    loan_winning_by_turn: defaultdict[tuple, IntegerBucket] = defaultdict(IntegerBucket)

    with ProcessPoolExecutor(max_workers=16) as executor:
        for summary in executor.map(_analyze_chunk, chunks):
            _merge_number(submitted_by_turn, summary.submitted_by_turn)
            _merge_integer(winning_by_turn, summary.winning_by_turn)
            _merge_integer(winning_by_chart, summary.winning_by_chart)
            _merge_integer(winning_by_chart_bot, summary.winning_by_chart_bot)
            _merge_integer(bot_scores_by_chart, summary.bot_scores_by_chart)
            _merge_integer(winning_scores_by_chart, summary.winning_scores_by_chart)
            _merge_info(info, summary.info)
            _merge_integer(all_winning, summary.all_winning)
            _merge_integer(
                loan_winning_by_turn,
                summary.loan_winning_by_turn,
            )

    turns = range(1, 23)
    payload = {
        "configuration": {
            "games": GAMES,
            "players": 3,
            "bots": BOT_NAMES,
            "charts": tuple("ABCDE"),
            "root_seed": ROOT_SEED,
            "workers": 16,
        },
        "by_turn": {
            RESOURCE_ACTION_NAMES[action]: {
                "market_winning_bid": [
                    {"turn": turn, **winning_by_turn[(action, turn)].freeze()} for turn in turns
                ],
                "submitted_by_bot": {
                    bot: [
                        {"turn": turn, **submitted_by_turn[(bot, action, turn)].freeze()}
                        for turn in turns
                    ]
                    for bot in BOT_NAMES
                },
            }
            for action in RESOURCE_ACTIONS
        },
        "by_chart": {
            chart: {
                "winning_score": winning_scores_by_chart[chart].freeze(),
                "bot_scores": {
                    bot: bot_scores_by_chart[(chart, bot)].freeze() for bot in BOT_NAMES
                },
                "auction_prices": {
                    RESOURCE_ACTION_NAMES[action]: {
                        "market": winning_by_chart[(chart, action)].freeze(),
                        "by_winner": {
                            bot: winning_by_chart_bot[(chart, bot, action)].freeze()
                            for bot in BOT_NAMES
                        },
                    }
                    for action in RESOURCE_ACTIONS
                },
            }
            for chart in "ABCDE"
        },
        "information_asymmetry": {
            chart: {
                bot: {str(hidden): info[(chart, bot, hidden)].freeze() for hidden in range(4)}
                for bot in BOT_NAMES
            }
            for chart in "ABCDE"
        },
        "all_action_winning_bids": {
            bot: {ACTION_NAMES[action]: all_winning[(bot, action)].freeze() for action in ActionId}
            for bot in BOT_NAMES
        },
        "loan_winning_by_turn": {
            ACTION_NAMES[action]: {
                bot: [
                    {
                        "turn": turn,
                        **loan_winning_by_turn[(bot, action, turn)].freeze(),
                    }
                    for turn in turns
                ]
                for bot in BOT_NAMES
            }
            for action in LOAN_ACTIONS
        },
    }
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
