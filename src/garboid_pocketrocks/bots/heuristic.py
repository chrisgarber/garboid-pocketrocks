from __future__ import annotations

from dataclasses import dataclass

from pocketrocks import BotDecision, DecisionContext

from garboid_pocketrocks.adapters.public_history import PublicAuctionResolved, PublicHistory
from garboid_pocketrocks.bots.base import BotSpec, PocketRocksFastBot
from garboid_pocketrocks.diagnostics.trace import (
    ExplainedBotDecision,
    HeuristicBidExplanation,
    OpponentAwareHeuristicBidExplanation,
)
from garboid_pocketrocks.heuristics.errors import HeuristicInputError
from garboid_pocketrocks.heuristics.game_phase import GamePhase, game_phase_for_turn_index
from garboid_pocketrocks.heuristics.opponent_bids import (
    DEFAULT_OPPONENT_BID_MODEL_CONFIG,
    OPPONENT_BID_MODEL_NAME,
    CompetitiveBidPoint,
    OpponentBidDistribution,
    OpponentBidModelConfig,
    PublicOpponentBidContext,
    forecast_opponent_bids,
    opponent_bid_model_config_digest,
)
from garboid_pocketrocks.heuristics.profiles import (
    HEURISTIC_V1,
    HEURISTIC_V2,
    HEURISTIC_V3,
    HeuristicProfile,
)
from garboid_pocketrocks.heuristics.valuation import BidEvaluation, HeuristicValuator
from garboid_pocketrocks.knowledge import RulesetKnowledge


@dataclass(frozen=True, slots=True)
class _HeuristicChoice:
    decision: BotDecision
    bid_evaluation: BidEvaluation | None


@dataclass(frozen=True, slots=True)
class _OpponentAwareHeuristicChoice:
    decision: BotDecision
    bid_evaluation: BidEvaluation | None
    public_game_phase: GamePhase | None
    opponent_distributions: tuple[OpponentBidDistribution, ...]
    competitive_bid_points: tuple[CompetitiveBidPoint, ...]


class HeuristicBotBrain:
    """Synchronous adapter from a heuristic profile to SDK decisions."""

    def __init__(self, profile: HeuristicProfile) -> None:
        self.valuator = HeuristicValuator(profile)

    def choose_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
        history: PublicHistory = (),
    ) -> BotDecision:
        del history
        return self._choose_raw(context, ruleset).decision

    def choose_explained_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
        history: PublicHistory,
    ) -> ExplainedBotDecision:
        """Choose once and retain the valuation that produced a heuristic bid."""

        del history
        choice = self._choose_raw(context, ruleset)
        evaluation = choice.bid_evaluation
        if evaluation is None:
            return ExplainedBotDecision(decision=choice.decision)
        bid = evaluation.chosen_bid
        point = evaluation.points[bid]
        return ExplainedBotDecision(
            decision=choice.decision,
            explanation=HeuristicBidExplanation(
                resource_value=point.breakdown.resource,
                objective_completion_value=point.breakdown.objective_completion,
                objective_progress_value=point.breakdown.objective_progress,
                terminal_cash_value=point.breakdown.terminal_cash,
                liquidity_value=point.breakdown.liquidity,
                future_cash_value=point.breakdown.future_cash,
                total_value=point.breakdown.total,
                reservation_bid=evaluation.reservation_bid,
                chosen_bid=bid,
            ),
        )

    def _choose_raw(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
    ) -> _HeuristicChoice:
        """Choose once without constructing diagnostic records."""

        try:
            if context.decision_kind == "selectInfoToReveal":
                return _HeuristicChoice(
                    BotDecision.select_info_to_reveal(
                        self.valuator.choose_reveal(context, ruleset),
                    ),
                    None,
                )
            evaluation = self.valuator.evaluate_bid(context, ruleset)
            bid = evaluation.chosen_bid
            return _HeuristicChoice(
                BotDecision.pass_turn() if bid == 0 else BotDecision.submit_bid(bid),
                evaluation,
            )
        except HeuristicInputError:
            return _HeuristicChoice(BotDecision.pass_turn(), None)


class OpponentAwareHeuristicBotBrain:
    """Choose bids by combining a frozen valuation with public bid forecasts."""

    def __init__(
        self,
        profile: HeuristicProfile,
        model_config: OpponentBidModelConfig = DEFAULT_OPPONENT_BID_MODEL_CONFIG,
    ) -> None:
        if not isinstance(profile, HeuristicProfile):
            raise HeuristicInputError("opponent-aware heuristic profile has the wrong type")
        if not isinstance(model_config, OpponentBidModelConfig):
            raise HeuristicInputError("opponent bid model config has the wrong type")
        self.profile = profile
        self.valuator = HeuristicValuator(profile)
        self.model_config = model_config
        self.model_config_digest = opponent_bid_model_config_digest(model_config)

    def choose_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
        history: PublicHistory = (),
    ) -> BotDecision:
        """Choose with the immutable public history supplied to every bot brain."""

        return self._choose_raw(context, ruleset, history).decision

    def choose_explained_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
        history: PublicHistory,
    ) -> ExplainedBotDecision:
        """Choose once and retain the public inputs behind an opponent-aware bid."""

        choice = self._choose_raw(context, ruleset, history)
        evaluation = choice.bid_evaluation
        phase = choice.public_game_phase
        if evaluation is None or phase is None:
            return ExplainedBotDecision(decision=choice.decision)

        chosen_bid = 0 if choice.decision.action_kind == "pass" else choice.decision.value
        if chosen_bid is None:
            raise HeuristicInputError("opponent-aware bid decision has no amount")
        selected_point = evaluation.points[chosen_bid]
        breakdown = selected_point.breakdown
        reservation_bid = max(
            (
                point.effective_bid
                for point in choice.competitive_bid_points
                if point.win_delta >= 0.0
            ),
            default=0,
        )
        return ExplainedBotDecision(
            decision=choice.decision,
            explanation=OpponentAwareHeuristicBidExplanation(
                resource_value=breakdown.resource,
                objective_completion_value=breakdown.objective_completion,
                objective_progress_value=breakdown.objective_progress,
                terminal_cash_value=breakdown.terminal_cash,
                liquidity_value=breakdown.liquidity,
                future_cash_value=breakdown.future_cash,
                total_value=breakdown.total,
                reservation_bid=reservation_bid,
                chosen_bid=chosen_bid,
                public_game_phase=phase,
                model_name=OPPONENT_BID_MODEL_NAME,
                model_config_digest=self.model_config_digest,
                opponent_distributions=choice.opponent_distributions,
                competitive_bid_points=choice.competitive_bid_points,
            ),
        )

    def _choose_raw(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
        history: PublicHistory,
    ) -> _OpponentAwareHeuristicChoice:
        """Value and forecast once, then maximize expected competitive surplus."""

        if context.decision_kind == "selectInfoToReveal":
            try:
                decision = BotDecision.select_info_to_reveal(
                    self.valuator.choose_reveal(context, ruleset),
                )
            except HeuristicInputError:
                decision = BotDecision.pass_turn()
            return _OpponentAwareHeuristicChoice(
                decision=decision,
                bid_evaluation=None,
                public_game_phase=None,
                opponent_distributions=(),
                competitive_bid_points=(),
            )
        if context.decision_kind != "submitBid":
            raise HeuristicInputError(
                f"unsupported opponent-aware decision kind {context.decision_kind!r}",
            )
        if context.current_action_id is None:
            raise HeuristicInputError("opponent-aware bid requires a current action")
        if context.legal_max_amount is None:
            raise HeuristicInputError("opponent-aware bid requires a legal maximum")
        if not history:
            raise HeuristicInputError("opponent-aware bids require public history")

        completed_round_count = sum(isinstance(event, PublicAuctionResolved) for event in history)
        phase = game_phase_for_turn_index(completed_round_count)
        try:
            model_context = PublicOpponentBidContext(
                player_count=context.player_count,
                starting_cash=context.starting_cash,
                value_chart=context.value_chart,
                current_action_id=context.current_action_id,
                cash_by_seat=context.cash_by_seat,
                tiebreak_seat=context.tiebreak_seat,
                bot_seat=context.bot_seat,
                legal_max_amount=context.legal_max_amount,
                game_phase=phase,
            )
        except ValueError as error:
            raise HeuristicInputError(str(error)) from error

        evaluation = self.valuator.evaluate_bid(context, ruleset)
        forecast = forecast_opponent_bids(history, model_context, self.model_config)
        legal_support = tuple(range(context.legal_max_amount + 1))
        valuation_support = tuple(point.bid for point in evaluation.points)
        forecast_support = tuple(item.effective_bid for item in forecast.legal_bid_forecasts)
        if valuation_support != legal_support:
            raise HeuristicInputError("heuristic valuation does not cover every legal bid")
        if forecast_support != legal_support:
            raise HeuristicInputError("opponent forecast does not cover every legal bid")

        try:
            competitive_points = tuple(
                CompetitiveBidPoint(
                    effective_bid=value_point.bid,
                    win_probability=forecast_point.win_probability,
                    win_delta=value_point.win_delta,
                    expected_surplus=(forecast_point.win_probability * value_point.win_delta),
                )
                for value_point, forecast_point in zip(
                    evaluation.points,
                    forecast.legal_bid_forecasts,
                    strict=True,
                )
            )
        except ValueError as error:
            raise HeuristicInputError(str(error)) from error
        selected = max(
            competitive_points,
            key=lambda point: (point.expected_surplus, -point.effective_bid),
        )
        decision = (
            BotDecision.pass_turn()
            if selected.effective_bid == 0
            else BotDecision.submit_bid(selected.effective_bid)
        )
        return _OpponentAwareHeuristicChoice(
            decision=decision,
            bid_evaluation=evaluation,
            public_game_phase=phase,
            opponent_distributions=forecast.opponent_distributions,
            competitive_bid_points=competitive_points,
        )


class AggressiveHeuristicV1Brain(HeuristicBotBrain):
    def __init__(self, seed: int | None = None) -> None:
        del seed
        super().__init__(HEURISTIC_V1.aggressive)


class BalancedHeuristicV1Brain(HeuristicBotBrain):
    def __init__(self, seed: int | None = None) -> None:
        del seed
        super().__init__(HEURISTIC_V1.balanced)


class PassiveHeuristicV1Brain(HeuristicBotBrain):
    def __init__(self, seed: int | None = None) -> None:
        del seed
        super().__init__(HEURISTIC_V1.passive)


class AggressiveHeuristicV2Brain(HeuristicBotBrain):
    def __init__(self, seed: int | None = None) -> None:
        del seed
        super().__init__(HEURISTIC_V2.aggressive)


class BalancedHeuristicV2Brain(HeuristicBotBrain):
    def __init__(self, seed: int | None = None) -> None:
        del seed
        super().__init__(HEURISTIC_V2.balanced)


class PassiveHeuristicV2Brain(HeuristicBotBrain):
    def __init__(self, seed: int | None = None) -> None:
        del seed
        super().__init__(HEURISTIC_V2.passive)


class AggressiveHeuristicV3Brain(HeuristicBotBrain):
    def __init__(self, seed: int | None = None) -> None:
        del seed
        super().__init__(HEURISTIC_V3.aggressive)


class BalancedHeuristicV3Brain(HeuristicBotBrain):
    def __init__(self, seed: int | None = None) -> None:
        del seed
        super().__init__(HEURISTIC_V3.balanced)


class PassiveHeuristicV3Brain(HeuristicBotBrain):
    def __init__(self, seed: int | None = None) -> None:
        del seed
        super().__init__(HEURISTIC_V3.passive)


BALANCED_HEURISTIC_V5_CANDIDATE_MODEL_CONFIG = OpponentBidModelConfig(
    prior_strength=4.0,
    minimum_history_rounds=2,
    same_action_phase_weight=4.0,
    partial_match_weight=2.0,
    fallback_weight=1.0,
)


class BalancedHeuristicV5CandidateBrain(OpponentAwareHeuristicBotBrain):
    """Development-only balanced candidate that learns from public bid history."""

    def __init__(self, seed: int | None = None) -> None:
        del seed
        super().__init__(
            HEURISTIC_V3.balanced,
            BALANCED_HEURISTIC_V5_CANDIDATE_MODEL_CONFIG,
        )


class AggressiveHeuristicBrain(AggressiveHeuristicV3Brain):
    """Latest aggressive heuristic brain."""


class BalancedHeuristicBrain(BalancedHeuristicV3Brain):
    """Latest balanced heuristic brain."""


class PassiveHeuristicBrain(PassiveHeuristicV3Brain):
    """Latest passive heuristic brain."""


class AggressiveHeuristicBot(PocketRocksFastBot):
    """Live wrapper for the aggressive heuristic."""

    BOT_ID = "bot_386b81bb-14df-477a-8d4c-0231cf1b3b1a"
    BOT_NAME = "aggressive"

    @classmethod
    def build_brain(cls, seed: int | None) -> AggressiveHeuristicBrain:
        del seed
        return AggressiveHeuristicBrain()


class BalancedHeuristicBot(PocketRocksFastBot):
    """Live wrapper for the balanced heuristic."""

    BOT_ID = "bot_265c84aa-f28e-4a35-b4de-a4f4ee406415"
    BOT_NAME = "balanced"

    @classmethod
    def build_brain(cls, seed: int | None) -> BalancedHeuristicBrain:
        del seed
        return BalancedHeuristicBrain()


class PassiveHeuristicBot(PocketRocksFastBot):
    """Live wrapper for the passive heuristic."""

    BOT_ID = "bot_9d33c9de-4d90-4608-9a58-d2c77d93e0bd"
    BOT_NAME = "passive"

    @classmethod
    def build_brain(cls, seed: int | None) -> PassiveHeuristicBrain:
        del seed
        return PassiveHeuristicBrain()


AGGRESSIVE_HEURISTIC_BOT_SPEC = BotSpec.from_bot_class(AggressiveHeuristicBot)
BALANCED_HEURISTIC_BOT_SPEC = BotSpec.from_bot_class(BalancedHeuristicBot)
PASSIVE_HEURISTIC_BOT_SPEC = BotSpec.from_bot_class(PassiveHeuristicBot)

AGGRESSIVE_HEURISTIC_V1_BOT_SPEC = BotSpec.for_simulation(
    "aggressive-v1",
    AggressiveHeuristicV1Brain,
)
BALANCED_HEURISTIC_V1_BOT_SPEC = BotSpec.for_simulation(
    "balanced-v1",
    BalancedHeuristicV1Brain,
)
PASSIVE_HEURISTIC_V1_BOT_SPEC = BotSpec.for_simulation(
    "passive-v1",
    PassiveHeuristicV1Brain,
)

AGGRESSIVE_HEURISTIC_V2_BOT_SPEC = BotSpec.for_simulation(
    "aggressive-v2",
    AggressiveHeuristicV2Brain,
)
BALANCED_HEURISTIC_V2_BOT_SPEC = BotSpec.for_simulation(
    "balanced-v2",
    BalancedHeuristicV2Brain,
)
PASSIVE_HEURISTIC_V2_BOT_SPEC = BotSpec.for_simulation(
    "passive-v2",
    PassiveHeuristicV2Brain,
)

AGGRESSIVE_HEURISTIC_V3_BOT_SPEC = BotSpec.for_simulation(
    "aggressive-v3",
    AggressiveHeuristicV3Brain,
)
BALANCED_HEURISTIC_V3_BOT_SPEC = BotSpec.for_simulation(
    "balanced-v3",
    BalancedHeuristicV3Brain,
)
PASSIVE_HEURISTIC_V3_BOT_SPEC = BotSpec.for_simulation(
    "passive-v3",
    PassiveHeuristicV3Brain,
)

BALANCED_HEURISTIC_V5_CANDIDATE_BOT_SPEC = BotSpec.for_simulation(
    (
        "balanced-v5-candidate-opponent-aware-"
        f"{opponent_bid_model_config_digest(BALANCED_HEURISTIC_V5_CANDIDATE_MODEL_CONFIG)[:12]}"
    ),
    BalancedHeuristicV5CandidateBrain,
)
