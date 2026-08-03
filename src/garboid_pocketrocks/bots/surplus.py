from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb

from pocketrocks import OBJECTIVES, ActionId, BotDecision, DecisionContext

from garboid_pocketrocks.adapters.public_history import (
    PublicAuctionResolved,
    PublicHistory,
    PublicTurnOpened,
)
from garboid_pocketrocks.bots.base import BotSpec
from garboid_pocketrocks.knowledge import RulesetKnowledge


@dataclass(frozen=True, slots=True)
class SurplusPolicy:
    """Frozen behavior switches for one surplus bot generation."""

    use_posterior_values: bool = False
    use_objective_values: bool = False
    use_opponent_objective_threat: bool = False
    use_objective_progress: bool = False
    bid_investments: bool = False
    bid_liquidity_loans: bool = False
    manage_liquidity: bool = False
    use_action_liquidity_demand: bool = False
    use_market_prices: bool = False
    resource_value_numerator: int = 1
    resource_value_denominator: int = 1
    objective_value_numerator: int = 1
    objective_value_denominator: int = 1
    opponent_objective_numerator: int = 0
    opponent_objective_denominator: int = 1
    objective_progress_numerator: int = 0
    objective_progress_denominator: int = 1
    resource_reserve_numerator: int = 0
    resource_reserve_denominator: int = 1
    investment_reserve_numerator: int = 0
    investment_reserve_denominator: int = 1
    objective_reserve_release_numerator: int = 0
    objective_reserve_release_denominator: int = 1
    loan_trigger_numerator: int = 0
    loan_trigger_denominator: int = 1
    loan_fee_numerator: int = 0
    loan_fee_denominator: int = 1
    loan_opening_fee_numerator: int = 0
    loan_opening_fee_denominator: int = 1
    auction1_fallback_price: int = 5
    auction2_fallback_price: int = 10
    market_quantile_numerator: int = 3
    market_quantile_denominator: int = 4

    def __post_init__(self) -> None:
        if self.resource_value_numerator < 0 or self.resource_value_denominator <= 0:
            raise ValueError("resource value multiplier must be nonnegative")
        if self.objective_value_numerator < 0 or self.objective_value_denominator <= 0:
            raise ValueError("objective value multiplier must be nonnegative")
        if (
            self.opponent_objective_numerator < 0
            or self.opponent_objective_denominator <= 0
        ):
            raise ValueError("opponent objective multiplier must be nonnegative")
        for name, numerator, denominator in (
            (
                "objective progress",
                self.objective_progress_numerator,
                self.objective_progress_denominator,
            ),
            (
                "resource reserve",
                self.resource_reserve_numerator,
                self.resource_reserve_denominator,
            ),
            (
                "investment reserve",
                self.investment_reserve_numerator,
                self.investment_reserve_denominator,
            ),
            (
                "objective reserve release",
                self.objective_reserve_release_numerator,
                self.objective_reserve_release_denominator,
            ),
            (
                "loan trigger",
                self.loan_trigger_numerator,
                self.loan_trigger_denominator,
            ),
            (
                "loan fee",
                self.loan_fee_numerator,
                self.loan_fee_denominator,
            ),
            (
                "loan opening fee",
                self.loan_opening_fee_numerator,
                self.loan_opening_fee_denominator,
            ),
        ):
            if numerator < 0 or denominator <= 0:
                raise ValueError(f"{name} multiplier must be nonnegative")
        if self.auction1_fallback_price < 0 or self.auction2_fallback_price < 0:
            raise ValueError("fallback auction prices must be nonnegative")
        if not 0 <= self.market_quantile_numerator <= self.market_quantile_denominator:
            raise ValueError("market quantile must be between zero and one")


class SurplusBrain:
    """Bid a shaded fraction of independently estimated terminal value.

    The family starts from a deliberately small economic hypothesis: cash and
    resources use the same final-score unit, so a resource is worth buying only
    below its estimated score contribution. Later generations add information
    and opportunity values without changing the released earlier policies.
    """

    def __init__(self, policy: SurplusPolicy) -> None:
        self._policy = policy

    def choose_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
        history: PublicHistory = (),
    ) -> BotDecision:
        if context.decision_kind != "submitBid":
            if context.revealable_count <= 0:
                return BotDecision.pass_turn()
            return BotDecision.select_info_to_reveal(0)

        if self._policy.bid_investments and context.current_action_id in (
            ActionId.INVEST5,
            ActionId.INVEST10,
        ):
            payout = 5 if context.current_action_id == ActionId.INVEST5 else 10
            amount = min(payout, context.legal_max_amount or 0)
            if self._policy.use_market_prices:
                amount = min(amount, self._market_price_cap(context, history))
            if self._policy.manage_liquidity:
                amount = min(
                    amount,
                    self._spendable_cash(
                        context,
                        ruleset,
                        history,
                        reserve_numerator=self._policy.investment_reserve_numerator,
                        reserve_denominator=self._policy.investment_reserve_denominator,
                    ),
                )
            return BotDecision.submit_bid(amount) if amount > 0 else BotDecision.pass_turn()

        if self._policy.bid_liquidity_loans and context.current_action_id in (
            ActionId.LOAN10,
            ActionId.LOAN20,
        ):
            return self._loan_decision(context, ruleset, history)

        if context.current_action_id not in (ActionId.AUCTION1, ActionId.AUCTION2):
            return BotDecision.pass_turn()

        resource_value = sum(
            (
                (
                    self._posterior_resource_value(context, ruleset, suit_id)
                    if self._policy.use_posterior_values
                    else Fraction(self._visible_resource_value(context, suit_id))
                )
                for suit_id in self._awarded_resource_ids(context)
            ),
            start=Fraction(),
        )
        awarded_resource_ids = self._awarded_resource_ids(context)
        estimated_value = resource_value * Fraction(
            self._policy.resource_value_numerator,
            self._policy.resource_value_denominator,
        )
        immediate_objective_value = 0
        if self._policy.use_objective_values:
            immediate_objective_value = self._new_objective_value(
                context,
                awarded_resource_ids,
            )
            estimated_value += immediate_objective_value * Fraction(
                self._policy.objective_value_numerator,
                self._policy.objective_value_denominator,
            )
        if self._policy.use_opponent_objective_threat:
            estimated_value += self._opponent_objective_threat(
                context,
                self._awarded_resource_ids(context),
            ) * Fraction(
                self._policy.opponent_objective_numerator,
                self._policy.opponent_objective_denominator,
            )
        if self._policy.use_objective_progress:
            estimated_value += self._objective_progress_value(
                context,
                awarded_resource_ids,
            ) * Fraction(
                self._policy.objective_progress_numerator,
                self._policy.objective_progress_denominator,
            )
        bid = int(estimated_value * Fraction(context.player_count - 1, context.player_count))
        if self._policy.use_market_prices:
            bid = min(bid, self._market_price_cap(context, history))
        if self._policy.manage_liquidity:
            spendable = self._spendable_cash(
                context,
                ruleset,
                history,
                reserve_numerator=self._policy.resource_reserve_numerator,
                reserve_denominator=self._policy.resource_reserve_denominator,
                awarded_resource_count=len(awarded_resource_ids),
            )
            reserve_release = int(
                immediate_objective_value
                * Fraction(
                    self._policy.objective_reserve_release_numerator,
                    self._policy.objective_reserve_release_denominator,
                )
            )
            bid = min(bid, spendable + reserve_release)
        legal_max = context.legal_max_amount or 0
        amount = min(bid, legal_max)
        return BotDecision.submit_bid(amount) if amount > 0 else BotDecision.pass_turn()

    @staticmethod
    def _visible_resource_value(context: DecisionContext, suit_id: int) -> int:
        private_count = context.current_hand_suit_ids.count(suit_id)
        public_count = context.revealed_info_counts_by_suit[suit_id - 1]
        return context.value_chart[min(private_count + public_count, 5)]

    @staticmethod
    def _awarded_resource_ids(context: DecisionContext) -> tuple[int, ...]:
        count = 1 if context.current_action_id == ActionId.AUCTION1 else 2
        return tuple(suit_id for suit_id in context.current_resource_ids[:count] if suit_id)

    @staticmethod
    def _posterior_resource_value(
        context: DecisionContext,
        ruleset: RulesetKnowledge,
        suit_id: int,
    ) -> Fraction:
        """Expected score value after removing every publicly known card.

        The 30-card deck contains six cards of each suit. The acting bot knows
        its own unrevealed hand, all public reveals, every resource already won,
        and both currently face-up resources. The only uncertain split is then
        between opponent hands and the future resource deck, which is exactly a
        hypergeometric distribution.
        """

        known_info_counts = tuple(
            context.current_hand_suit_ids.count(candidate)
            + context.revealed_info_counts_by_suit[candidate - 1]
            for candidate in range(1, 6)
        )
        seen_resource_counts = tuple(
            context.won_resource_counts_by_suit[candidate - 1]
            + context.current_resource_ids.count(candidate)
            for candidate in range(1, 6)
        )
        remaining_cards = (
            sum(ruleset.resource_counts)
            - sum(known_info_counts)
            - sum(seen_resource_counts)
        )
        opponent_hidden_cards = sum(
            max(0, ruleset.private_cards_per_player - sum(revealed_counts))
            for seat, revealed_counts in enumerate(context.revealed_info_counts_by_seat)
            if seat != context.bot_seat
        )
        remaining_suit_cards = (
            ruleset.resource_counts[suit_id - 1]
            - known_info_counts[suit_id - 1]
            - seen_resource_counts[suit_id - 1]
        )
        if (
            remaining_cards < 0
            or opponent_hidden_cards < 0
            or opponent_hidden_cards > remaining_cards
            or remaining_suit_cards < 0
            or remaining_suit_cards > remaining_cards
        ):
            return Fraction(
                context.value_chart[min(known_info_counts[suit_id - 1], 5)]
            )

        denominator = comb(remaining_cards, opponent_hidden_cards)
        if denominator == 0:
            return Fraction(
                context.value_chart[min(known_info_counts[suit_id - 1], 5)]
            )
        minimum = max(0, opponent_hidden_cards - (remaining_cards - remaining_suit_cards))
        maximum = min(remaining_suit_cards, opponent_hidden_cards)
        numerator = sum(
            comb(remaining_suit_cards, hidden_count)
            * comb(
                remaining_cards - remaining_suit_cards,
                opponent_hidden_cards - hidden_count,
            )
            * context.value_chart[
                min(known_info_counts[suit_id - 1] + hidden_count, 5)
            ]
            for hidden_count in range(minimum, maximum + 1)
        )
        return Fraction(numerator, denominator)

    @classmethod
    def _new_objective_value(
        cls,
        context: DecisionContext,
        awarded_suits: tuple[int, ...],
    ) -> int:
        return cls._new_objective_value_for_seat(
            context,
            awarded_suits,
            context.bot_seat,
        )

    @classmethod
    def _new_objective_value_for_seat(
        cls,
        context: DecisionContext,
        awarded_suits: tuple[int, ...],
        seat: int,
    ) -> int:
        claimed = {
            objective_id
            for owned_ids in context.owned_objective_ids_by_seat
            for objective_id in owned_ids
        }
        counts_before = context.won_resource_counts_by_seat[seat]
        counts_after = list(counts_before)
        for suit_id in awarded_suits:
            counts_after[suit_id - 1] += 1
        return sum(
            OBJECTIVES[objective_id].payout
            for objective_id in context.objective_ids
            if objective_id not in claimed
            and not cls._objective_met(objective_id, counts_before)
            and cls._objective_met(objective_id, tuple(counts_after))
        )

    @classmethod
    def _opponent_objective_threat(
        cls,
        context: DecisionContext,
        awarded_suits: tuple[int, ...],
    ) -> int:
        """Largest objective payout a rival could claim from this bundle."""

        return max(
            (
                cls._new_objective_value_for_seat(
                    context,
                    awarded_suits,
                    seat,
                )
                for seat in range(context.player_count)
                if seat != context.bot_seat
            ),
            default=0,
        )

    @classmethod
    def _objective_progress_value(
        cls,
        context: DecisionContext,
        awarded_suits: tuple[int, ...],
    ) -> Fraction:
        """Value partial progress without duplicating immediate completion value."""

        claimed = {
            objective_id
            for owned_ids in context.owned_objective_ids_by_seat
            for objective_id in owned_ids
        }
        counts_before = context.won_resource_counts_by_seat[context.bot_seat]
        counts_after = list(counts_before)
        for suit_id in awarded_suits:
            counts_after[suit_id - 1] += 1

        value = Fraction()
        for objective_id in context.objective_ids:
            if objective_id in claimed:
                continue
            before = cls._objective_distance(objective_id, counts_before)
            after = cls._objective_distance(objective_id, tuple(counts_after))
            if before <= 0 or after <= 0 or after >= before:
                continue
            objective = OBJECTIVES[objective_id]
            requirement_size = cls._objective_requirement_size(objective_id)
            value += Fraction(objective.payout * (before - after), requirement_size)
        return value

    @staticmethod
    def _objective_requirement_size(objective_id: int) -> int:
        objective = OBJECTIVES[objective_id]
        if objective.pattern == "same2":
            return 2
        if objective.pattern in ("same3", "different3"):
            return 3
        if objective.pattern in ("different4", "twoPairs4"):
            return 4
        assert objective.requirement is not None
        return sum(objective.requirement)

    @classmethod
    def _objective_distance(cls, objective_id: int, counts: tuple[int, ...]) -> int:
        objective = OBJECTIVES[objective_id]
        if objective.pattern == "same2":
            return max(0, 2 - max(counts))
        if objective.pattern == "same3":
            return max(0, 3 - max(counts))
        if objective.pattern == "different3":
            return max(0, 3 - sum(count > 0 for count in counts))
        if objective.pattern == "different4":
            return max(0, 4 - sum(count > 0 for count in counts))
        if objective.pattern == "twoPairs4":
            return sum(sorted(max(0, 2 - count) for count in counts)[:2])
        assert objective.requirement is not None
        return sum(
            max(0, required - count)
            for count, required in zip(counts, objective.requirement, strict=True)
        )

    @staticmethod
    def _objective_met(objective_id: int, counts: tuple[int, ...]) -> bool:
        objective = OBJECTIVES[objective_id]
        if objective.pattern == "same2":
            return any(count >= 2 for count in counts)
        if objective.pattern == "same3":
            return any(count >= 3 for count in counts)
        if objective.pattern == "different3":
            return sum(count > 0 for count in counts) >= 3
        if objective.pattern == "different4":
            return sum(count > 0 for count in counts) >= 4
        if objective.pattern == "twoPairs4":
            return sum(count >= 2 for count in counts) >= 2
        assert objective.requirement is not None
        return all(
            count >= required
            for count, required in zip(counts, objective.requirement, strict=True)
        )

    def _market_price_cap(self, context: DecisionContext, history: PublicHistory) -> int:
        """Upper-quartile rival price plus one for the current action type."""

        return min(
            context.legal_max_amount or 0,
            self._market_price_target(
                context,
                history,
                action_id=context.current_action_id,
                fallback=context.legal_max_amount or 0,
            ),
        )

    def _market_price_target(
        self,
        context: DecisionContext,
        history: PublicHistory,
        *,
        action_id: int | None,
        fallback: int,
    ) -> int:
        """Return an uncapped upper-quartile rival price plus one."""

        rival_prices: list[int] = []
        opened_action_id: int | None = None
        for event in history:
            if isinstance(event, PublicTurnOpened):
                opened_action_id = event.action_id
            elif isinstance(event, PublicAuctionResolved):
                if opened_action_id == action_id:
                    rival_prices.append(
                        max(
                            bid
                            for seat, bid in enumerate(event.bids_by_seat)
                            if seat != context.bot_seat
                        )
                    )
                opened_action_id = None
        if not rival_prices:
            return fallback
        ordered = sorted(rival_prices)
        scaled_index = self._policy.market_quantile_numerator * (len(ordered) - 1)
        quantile_index = (
            scaled_index + self._policy.market_quantile_denominator - 1
        ) // self._policy.market_quantile_denominator
        return ordered[quantile_index] + 1

    @staticmethod
    def _future_resource_count(
        context: DecisionContext,
        ruleset: RulesetKnowledge,
        *,
        awarded_resource_count: int = 0,
    ) -> int:
        total_resources = sum(ruleset.resource_counts) - (
            context.player_count * ruleset.private_cards_per_player
        )
        already_won = sum(sum(counts) for counts in context.won_resource_counts_by_seat)
        return max(0, total_resources - already_won - awarded_resource_count)

    def _liquidity_basis(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
        history: PublicHistory,
        *,
        awarded_resource_count: int = 0,
    ) -> Fraction:
        if not self._policy.use_action_liquidity_demand:
            return Fraction(
                self._future_resource_count(
                    context,
                    ruleset,
                    awarded_resource_count=awarded_resource_count,
                )
            )
        return self._projected_resource_spend(
            context,
            ruleset,
            history,
            awarded_resource_count=awarded_resource_count,
        )

    def _projected_resource_spend(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
        history: PublicHistory,
        *,
        awarded_resource_count: int = 0,
    ) -> Fraction:
        """Expected market cost of the remaining resource deck.

        Public turn openings reveal which action cards have already left the
        shuffled deck. The remaining Auction1/Auction2 capacity determines the
        expected mix of one- and two-card purchases; public rival prices supply
        their clearing-price estimates.
        """

        remaining_action_counts = list(ruleset.action_counts)
        for event in history:
            if isinstance(event, PublicTurnOpened):
                index = event.action_id - 1
                if 0 <= index < len(remaining_action_counts):
                    remaining_action_counts[index] = max(
                        0,
                        remaining_action_counts[index] - 1,
                    )
        auction1_count = remaining_action_counts[int(ActionId.AUCTION1) - 1]
        auction2_count = remaining_action_counts[int(ActionId.AUCTION2) - 1]
        resource_capacity = auction1_count + (2 * auction2_count)
        future_resources = self._future_resource_count(
            context,
            ruleset,
            awarded_resource_count=awarded_resource_count,
        )
        if resource_capacity <= 0 or future_resources <= 0:
            return Fraction()

        auction1_price = self._market_price_target(
            context,
            history,
            action_id=ActionId.AUCTION1,
            fallback=self._policy.auction1_fallback_price,
        )
        auction2_price = self._market_price_target(
            context,
            history,
            action_id=ActionId.AUCTION2,
            fallback=self._policy.auction2_fallback_price,
        )
        cost_per_resource = Fraction(
            (auction1_count * auction1_price) + (auction2_count * auction2_price),
            resource_capacity,
        )
        return min(future_resources, resource_capacity) * cost_per_resource

    def _spendable_cash(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
        history: PublicHistory,
        *,
        reserve_numerator: int,
        reserve_denominator: int,
        awarded_resource_count: int = 0,
    ) -> int:
        basis = self._liquidity_basis(
            context,
            ruleset,
            history,
            awarded_resource_count=awarded_resource_count,
        )
        reserve = self._ceil_fraction(
            basis * Fraction(reserve_numerator, reserve_denominator)
        )
        cash = context.cash_by_seat[context.bot_seat]
        return max(0, cash - reserve)

    def _loan_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
        history: PublicHistory,
    ) -> BotDecision:
        principal = 10 if context.current_action_id == ActionId.LOAN10 else 20
        basis = self._liquidity_basis(context, ruleset, history)
        target_cash = self._ceil_fraction(
            basis
            * Fraction(
                self._policy.loan_trigger_numerator,
                self._policy.loan_trigger_denominator,
            )
        )
        cash = context.cash_by_seat[context.bot_seat]
        if cash >= target_cash:
            return BotDecision.pass_turn()
        amount = int(
            principal
            * Fraction(
                self._policy.loan_fee_numerator,
                self._policy.loan_fee_denominator,
            )
        )
        if (
            self._policy.use_action_liquidity_demand
            and self._policy.loan_opening_fee_numerator > 0
        ):
            opening_amount = int(
                principal
                * Fraction(
                    self._policy.loan_opening_fee_numerator,
                    self._policy.loan_opening_fee_denominator,
                )
            )
            amount = min(
                amount,
                self._market_price_target(
                    context,
                    history,
                    action_id=context.current_action_id,
                    fallback=opening_amount,
                ),
            )
        elif self._policy.use_market_prices:
            amount = min(amount, self._market_price_cap(context, history))
        amount = min(amount, context.legal_max_amount or 0)
        return BotDecision.submit_bid(amount) if amount > 0 else BotDecision.pass_turn()

    @staticmethod
    def _ceil_fraction(value: Fraction) -> int:
        return (value.numerator + value.denominator - 1) // value.denominator


SURPLUS_V1_POLICY = SurplusPolicy()
SURPLUS_V2_POLICY = SurplusPolicy(use_posterior_values=True)
SURPLUS_V3_POLICY = SurplusPolicy(
    use_posterior_values=True,
    use_objective_values=True,
)
SURPLUS_V4_POLICY = SurplusPolicy(
    use_posterior_values=True,
    bid_investments=True,
)
SURPLUS_V5_POLICY = SurplusPolicy(
    use_posterior_values=True,
    bid_investments=True,
    use_market_prices=True,
)
SURPLUS_V6_POLICY = SurplusPolicy(
    use_posterior_values=True,
    use_objective_values=True,
    bid_investments=True,
    use_market_prices=True,
)
SURPLUS_V7_POLICY = SurplusPolicy(
    use_posterior_values=True,
    use_objective_values=True,
    bid_investments=True,
    use_market_prices=True,
    resource_value_numerator=13,
    resource_value_denominator=16,
    objective_value_numerator=3,
    objective_value_denominator=8,
)
SURPLUS_V8_POLICY = SurplusPolicy(
    use_posterior_values=True,
    use_objective_values=True,
    use_opponent_objective_threat=True,
    bid_investments=True,
    use_market_prices=True,
    resource_value_numerator=3,
    resource_value_denominator=4,
    objective_value_numerator=3,
    objective_value_denominator=8,
    opponent_objective_numerator=1,
    opponent_objective_denominator=32,
)
SURPLUS_V9_POLICY = SurplusPolicy(
    use_posterior_values=True,
    use_objective_values=True,
    use_opponent_objective_threat=True,
    use_objective_progress=True,
    bid_investments=True,
    bid_liquidity_loans=True,
    manage_liquidity=True,
    use_market_prices=True,
    resource_value_numerator=3,
    resource_value_denominator=4,
    objective_value_numerator=3,
    objective_value_denominator=8,
    opponent_objective_numerator=1,
    opponent_objective_denominator=32,
    objective_progress_numerator=1,
    objective_progress_denominator=8,
    resource_reserve_numerator=3,
    resource_reserve_denominator=4,
    investment_reserve_numerator=3,
    investment_reserve_denominator=2,
    objective_reserve_release_numerator=1,
    objective_reserve_release_denominator=2,
    loan_trigger_numerator=5,
    loan_trigger_denominator=1,
    loan_fee_numerator=3,
    loan_fee_denominator=10,
)
SURPLUS_V10_POLICY = SurplusPolicy(
    use_posterior_values=True,
    use_objective_values=True,
    use_opponent_objective_threat=True,
    use_objective_progress=True,
    bid_investments=True,
    bid_liquidity_loans=True,
    manage_liquidity=True,
    use_action_liquidity_demand=True,
    use_market_prices=True,
    resource_value_numerator=3,
    resource_value_denominator=4,
    objective_value_numerator=3,
    objective_value_denominator=8,
    opponent_objective_numerator=1,
    opponent_objective_denominator=32,
    objective_progress_numerator=1,
    objective_progress_denominator=8,
    resource_reserve_numerator=1,
    resource_reserve_denominator=8,
    investment_reserve_numerator=3,
    investment_reserve_denominator=10,
    objective_reserve_release_numerator=1,
    objective_reserve_release_denominator=2,
    loan_trigger_numerator=3,
    loan_trigger_denominator=4,
    loan_fee_numerator=2,
    loan_fee_denominator=5,
    loan_opening_fee_numerator=7,
    loan_opening_fee_denominator=20,
    auction1_fallback_price=5,
    auction2_fallback_price=10,
)


class SurplusV1Brain(SurplusBrain):
    def __init__(self) -> None:
        super().__init__(SURPLUS_V1_POLICY)


class SurplusV2Brain(SurplusBrain):
    def __init__(self) -> None:
        super().__init__(SURPLUS_V2_POLICY)


class SurplusV3Brain(SurplusBrain):
    def __init__(self) -> None:
        super().__init__(SURPLUS_V3_POLICY)


class SurplusV4Brain(SurplusBrain):
    def __init__(self) -> None:
        super().__init__(SURPLUS_V4_POLICY)


class SurplusV5Brain(SurplusBrain):
    def __init__(self) -> None:
        super().__init__(SURPLUS_V5_POLICY)


class SurplusV6Brain(SurplusBrain):
    def __init__(self) -> None:
        super().__init__(SURPLUS_V6_POLICY)


class SurplusV7Brain(SurplusBrain):
    def __init__(self) -> None:
        super().__init__(SURPLUS_V7_POLICY)


class SurplusV8Brain(SurplusBrain):
    def __init__(self) -> None:
        super().__init__(SURPLUS_V8_POLICY)


class SurplusV9Brain(SurplusBrain):
    def __init__(self) -> None:
        super().__init__(SURPLUS_V9_POLICY)


class SurplusV10Brain(SurplusBrain):
    def __init__(self) -> None:
        super().__init__(SURPLUS_V10_POLICY)


def _surplus_v1_factory(seed: int | None) -> SurplusV1Brain:
    del seed
    return SurplusV1Brain()


def _surplus_v2_factory(seed: int | None) -> SurplusV2Brain:
    del seed
    return SurplusV2Brain()


def _surplus_v3_factory(seed: int | None) -> SurplusV3Brain:
    del seed
    return SurplusV3Brain()


def _surplus_v4_factory(seed: int | None) -> SurplusV4Brain:
    del seed
    return SurplusV4Brain()


def _surplus_v5_factory(seed: int | None) -> SurplusV5Brain:
    del seed
    return SurplusV5Brain()


def _surplus_v6_factory(seed: int | None) -> SurplusV6Brain:
    del seed
    return SurplusV6Brain()


def _surplus_v7_factory(seed: int | None) -> SurplusV7Brain:
    del seed
    return SurplusV7Brain()


def _surplus_v8_factory(seed: int | None) -> SurplusV8Brain:
    del seed
    return SurplusV8Brain()


def _surplus_v9_factory(seed: int | None) -> SurplusV9Brain:
    del seed
    return SurplusV9Brain()


def _surplus_v10_factory(seed: int | None) -> SurplusV10Brain:
    del seed
    return SurplusV10Brain()


SURPLUS_V1_BOT_SPEC = BotSpec.for_simulation("surplus-v1", _surplus_v1_factory)
SURPLUS_V2_BOT_SPEC = BotSpec.for_simulation("surplus-v2", _surplus_v2_factory)
SURPLUS_V3_BOT_SPEC = BotSpec.for_simulation("surplus-v3", _surplus_v3_factory)
SURPLUS_V4_BOT_SPEC = BotSpec.for_simulation("surplus-v4", _surplus_v4_factory)
SURPLUS_V5_BOT_SPEC = BotSpec.for_simulation("surplus-v5", _surplus_v5_factory)
SURPLUS_V6_BOT_SPEC = BotSpec.for_simulation("surplus-v6", _surplus_v6_factory)
SURPLUS_V7_BOT_SPEC = BotSpec.for_simulation("surplus-v7", _surplus_v7_factory)
SURPLUS_V8_BOT_SPEC = BotSpec.for_simulation("surplus-v8", _surplus_v8_factory)
SURPLUS_V9_BOT_SPEC = BotSpec.for_simulation("surplus-v9", _surplus_v9_factory)
SURPLUS_V10_BOT_SPEC = BotSpec.for_simulation("surplus-v10", _surplus_v10_factory)
SURPLUS_BOT_SPEC = BotSpec.for_simulation("surplus", _surplus_v10_factory)
