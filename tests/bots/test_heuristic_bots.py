from __future__ import annotations

import pickle
from collections.abc import Callable
from dataclasses import replace
from typing import cast

import pytest
from pocketrocks import OBJECTIVES, ActionId, BotDecision, DecisionContext, Suit
from pocketrocks.sim.constants import VALUE_CHARTS

import garboid_pocketrocks.bots.heuristic as heuristic_module
from garboid_pocketrocks.adapters.public_history import (
    PublicAuctionResolved,
    PublicEventKind,
    PublicGameSetup,
    PublicHistory,
    PublicTurnOpened,
)
from garboid_pocketrocks.bots import (
    AggressiveHeuristicBot,
    AggressiveHeuristicBrain,
    AggressiveHeuristicV1Brain,
    AggressiveHeuristicV2Brain,
    AggressiveHeuristicV3Brain,
    BalancedHeuristicBot,
    BalancedHeuristicBrain,
    BalancedHeuristicV1Brain,
    BalancedHeuristicV2Brain,
    BalancedHeuristicV3Brain,
    BotSpec,
    PassiveHeuristicBot,
    PassiveHeuristicBrain,
    PassiveHeuristicV1Brain,
    PassiveHeuristicV2Brain,
    PassiveHeuristicV3Brain,
    PocketRocksFastBot,
)
from garboid_pocketrocks.bots.heuristic import (
    AGGRESSIVE_HEURISTIC_BOT_SPEC,
    AGGRESSIVE_HEURISTIC_V1_BOT_SPEC,
    AGGRESSIVE_HEURISTIC_V2_BOT_SPEC,
    AGGRESSIVE_HEURISTIC_V3_BOT_SPEC,
    BALANCED_HEURISTIC_BOT_SPEC,
    BALANCED_HEURISTIC_V1_BOT_SPEC,
    BALANCED_HEURISTIC_V2_BOT_SPEC,
    BALANCED_HEURISTIC_V3_BOT_SPEC,
    BALANCED_HEURISTIC_V5_CANDIDATE_BOT_SPEC,
    BALANCED_HEURISTIC_V5_CANDIDATE_MODEL_CONFIG,
    PASSIVE_HEURISTIC_BOT_SPEC,
    PASSIVE_HEURISTIC_V1_BOT_SPEC,
    PASSIVE_HEURISTIC_V2_BOT_SPEC,
    PASSIVE_HEURISTIC_V3_BOT_SPEC,
    BalancedHeuristicV5CandidateBrain,
    HeuristicBotBrain,
    OpponentAwareHeuristicBotBrain,
)
from garboid_pocketrocks.bots.registry import BOT_SPECS, DEFAULT_TOURNAMENT_BOT_SPECS
from garboid_pocketrocks.diagnostics.trace import (
    HeuristicBidExplanation,
    OpponentAwareHeuristicBidExplanation,
)
from garboid_pocketrocks.heuristics.errors import HeuristicInputError
from garboid_pocketrocks.heuristics.opponent_bids import (
    OPPONENT_BID_MODEL_NAME,
    LegalBidWinningForecast,
    OpponentBidForecast,
    OpponentBidModelConfig,
    forecast_opponent_bids,
    opponent_bid_model_config_digest,
)
from garboid_pocketrocks.heuristics.profiles import (
    BALANCED_PROFILE,
    HEURISTIC_V1,
    HEURISTIC_V2,
    HEURISTIC_V3,
)
from garboid_pocketrocks.heuristics.valuation import BidEvaluation, HeuristicValuator
from garboid_pocketrocks.knowledge import (
    RulesetKnowledge,
    canonical_knowledge,
    knowledge_for_context,
)
from garboid_pocketrocks.simulator.monte_carlo import (
    MonteCarloConfig,
    MonteCarloRunner,
)


def make_knowledge(
    *,
    private_cards: int = 0,
    resource_counts: tuple[int, ...] = (2, 2, 2, 2, 2),
) -> RulesetKnowledge:
    return RulesetKnowledge(
        name="heuristic-bot-test",
        player_count=3,
        starting_cash=30,
        private_cards_per_player=private_cards,
        resource_counts=resource_counts,
        action_counts=(12, 8, 3, 2, 3, 2),
        value_chart=(0, 4, 8, 12, 16, 20),
        objective_pool=tuple(sorted(OBJECTIVES)),
        active_objective_count=0,
        objectives_enabled=False,
    )


def make_context(
    *,
    decision_kind: str = "submitBid",
    action_id: ActionId = ActionId.AUCTION1,
    current_resources: tuple[int, int] = (1, 0),
    hand: tuple[int, ...] = (),
    legal_max: int | None = 30,
) -> DecisionContext:
    return DecisionContext(
        request_id="heuristic-bot-test",
        deadline_at=2**63 - 1,
        received_at=0,
        decision_kind=decision_kind,  # type: ignore[arg-type]
        player_count=3,
        starting_cash=30,
        value_chart=(0, 4, 8, 12, 16, 20),
        objective_ids=(),
        current_action_id=int(action_id),
        current_resource_ids=current_resources,
        cash_by_seat=(30, 30, 30),
        tiebreak_seat=2,
        won_resource_counts_by_seat=((0, 0, 0, 0, 0),) * 3,
        revealed_info_counts_by_seat=((0, 0, 0, 0, 0),) * 3,
        owned_objective_ids_by_seat=((), (), ()),
        bot_seat=0,
        current_hand_suit_ids=hand,
        legal_max_amount=legal_max,
        revealable_count=len(hand),
    )


def make_public_history(
    context: DecisionContext,
    *,
    completed_round_count: int = 0,
) -> PublicHistory:
    events: list[object] = [
        PublicGameSetup(
            kind=PublicEventKind.GAME_SETUP,
            player_count=context.player_count,
            starting_cash=context.starting_cash,
            value_chart=context.value_chart,
            initial_tiebreak_seat=context.tiebreak_seat,
            objective_ids=context.objective_ids,
        )
    ]
    for _ in range(completed_round_count):
        events.extend(
            (
                PublicTurnOpened(
                    kind=PublicEventKind.TURN_OPENED,
                    action_id=int(ActionId.AUCTION1),
                    resource_ids=(int(Suit.BRICK), 0),
                ),
                PublicAuctionResolved(
                    kind=PublicEventKind.AUCTION_RESOLVED,
                    bids_by_seat=(0, 0, 1),
                ),
            )
        )
    if context.current_action_id is not None:
        events.append(
            PublicTurnOpened(
                kind=PublicEventKind.TURN_OPENED,
                action_id=context.current_action_id,
                resource_ids=context.current_resource_ids,
            )
        )
    return cast(PublicHistory, tuple(events))


def test_heuristic_bots_have_distinct_static_public_identities() -> None:
    assert issubclass(AggressiveHeuristicBot, PocketRocksFastBot)
    assert issubclass(BalancedHeuristicBot, PocketRocksFastBot)
    assert issubclass(PassiveHeuristicBot, PocketRocksFastBot)
    assert AggressiveHeuristicBot.BOT_ID == "bot_386b81bb-14df-477a-8d4c-0231cf1b3b1a"
    assert BalancedHeuristicBot.BOT_ID == "bot_265c84aa-f28e-4a35-b4de-a4f4ee406415"
    assert PassiveHeuristicBot.BOT_ID == "bot_9d33c9de-4d90-4608-9a58-d2c77d93e0bd"
    assert {
        AggressiveHeuristicBot.BOT_NAME,
        BalancedHeuristicBot.BOT_NAME,
        PassiveHeuristicBot.BOT_NAME,
    } == {"aggressive", "balanced", "passive"}


def test_versioned_heuristic_specs_use_names_as_private_simulation_ids() -> None:
    specs = (
        AGGRESSIVE_HEURISTIC_V1_BOT_SPEC,
        BALANCED_HEURISTIC_V1_BOT_SPEC,
        PASSIVE_HEURISTIC_V1_BOT_SPEC,
        AGGRESSIVE_HEURISTIC_V2_BOT_SPEC,
        BALANCED_HEURISTIC_V2_BOT_SPEC,
        PASSIVE_HEURISTIC_V2_BOT_SPEC,
        AGGRESSIVE_HEURISTIC_V3_BOT_SPEC,
        BALANCED_HEURISTIC_V3_BOT_SPEC,
        PASSIVE_HEURISTIC_V3_BOT_SPEC,
    )

    assert tuple(spec.name for spec in specs) == (
        "aggressive-v1",
        "balanced-v1",
        "passive-v1",
        "aggressive-v2",
        "balanced-v2",
        "passive-v2",
        "aggressive-v3",
        "balanced-v3",
        "passive-v3",
    )
    assert all(spec.bot_id == spec.name for spec in specs)


def test_balanced_v5_candidate_has_a_frozen_local_only_identity() -> None:
    brain = BALANCED_HEURISTIC_V5_CANDIDATE_BOT_SPEC.make_brain(seed=42)
    digest = opponent_bid_model_config_digest(
        BALANCED_HEURISTIC_V5_CANDIDATE_MODEL_CONFIG,
    )

    assert isinstance(brain, BalancedHeuristicV5CandidateBrain)
    assert isinstance(brain, OpponentAwareHeuristicBotBrain)
    assert brain.valuator.profile is HEURISTIC_V3.balanced
    assert brain.model_config is BALANCED_HEURISTIC_V5_CANDIDATE_MODEL_CONFIG
    assert BALANCED_HEURISTIC_V5_CANDIDATE_MODEL_CONFIG == OpponentBidModelConfig(
        prior_strength=4.0,
        minimum_history_rounds=2,
        same_action_phase_weight=4.0,
        partial_match_weight=2.0,
        fallback_weight=1.0,
    )
    assert BALANCED_HEURISTIC_V5_CANDIDATE_BOT_SPEC.name == (
        f"balanced-v5-candidate-opponent-aware-{digest[:12]}"
    )
    assert (
        BALANCED_HEURISTIC_V5_CANDIDATE_BOT_SPEC.bot_id
        == BALANCED_HEURISTIC_V5_CANDIDATE_BOT_SPEC.name
    )
    assert BALANCED_HEURISTIC_V5_CANDIDATE_BOT_SPEC not in BOT_SPECS
    assert BALANCED_HEURISTIC_V5_CANDIDATE_BOT_SPEC not in DEFAULT_TOURNAMENT_BOT_SPECS
    assert pickle.loads(pickle.dumps(BALANCED_HEURISTIC_V5_CANDIDATE_BOT_SPEC)) == (
        BALANCED_HEURISTIC_V5_CANDIDATE_BOT_SPEC
    )


@pytest.mark.parametrize(
    ("spec", "brain_class", "profile"),
    (
        (
            AGGRESSIVE_HEURISTIC_V1_BOT_SPEC,
            AggressiveHeuristicV1Brain,
            HEURISTIC_V1.aggressive,
        ),
        (
            BALANCED_HEURISTIC_V1_BOT_SPEC,
            BalancedHeuristicV1Brain,
            HEURISTIC_V1.balanced,
        ),
        (PASSIVE_HEURISTIC_V1_BOT_SPEC, PassiveHeuristicV1Brain, HEURISTIC_V1.passive),
        (
            AGGRESSIVE_HEURISTIC_V2_BOT_SPEC,
            AggressiveHeuristicV2Brain,
            HEURISTIC_V2.aggressive,
        ),
        (
            BALANCED_HEURISTIC_V2_BOT_SPEC,
            BalancedHeuristicV2Brain,
            HEURISTIC_V2.balanced,
        ),
        (PASSIVE_HEURISTIC_V2_BOT_SPEC, PassiveHeuristicV2Brain, HEURISTIC_V2.passive),
        (
            AGGRESSIVE_HEURISTIC_V3_BOT_SPEC,
            AggressiveHeuristicV3Brain,
            HEURISTIC_V3.aggressive,
        ),
        (
            BALANCED_HEURISTIC_V3_BOT_SPEC,
            BalancedHeuristicV3Brain,
            HEURISTIC_V3.balanced,
        ),
        (PASSIVE_HEURISTIC_V3_BOT_SPEC, PassiveHeuristicV3Brain, HEURISTIC_V3.passive),
    ),
)
def test_versioned_spec_factories_use_pinned_profiles(
    spec: BotSpec,
    brain_class: type[HeuristicBotBrain],
    profile: object,
) -> None:
    brain = spec.make_brain(seed=42)

    assert isinstance(brain, brain_class)
    assert brain.valuator.profile is profile


@pytest.mark.parametrize(
    ("latest_brain", "v3_brain"),
    (
        (AggressiveHeuristicBrain, AggressiveHeuristicV3Brain),
        (BalancedHeuristicBrain, BalancedHeuristicV3Brain),
        (PassiveHeuristicBrain, PassiveHeuristicV3Brain),
    ),
)
def test_unversioned_brains_match_v3_decisions(
    latest_brain: Callable[[], HeuristicBotBrain],
    v3_brain: Callable[[], HeuristicBotBrain],
) -> None:
    context = make_context(action_id=ActionId.AUCTION2, legal_max=17)
    knowledge = make_knowledge()

    assert latest_brain().choose_decision(context, knowledge) == v3_brain().choose_decision(
        context,
        knowledge,
    )


def _representative_legacy_explanation(
    *,
    liquidity_value: float,
    total_value: float,
    reservation_bid: int,
    chosen_bid: int,
) -> HeuristicBidExplanation:
    return HeuristicBidExplanation(
        resource_value=5.8181818181818175,
        objective_completion_value=0,
        objective_progress_value=0,
        terminal_cash_value=-float(chosen_bid),
        liquidity_value=liquidity_value,
        future_cash_value=0.0,
        total_value=total_value,
        reservation_bid=reservation_bid,
        chosen_bid=chosen_bid,
    )


def test_v1_through_v3_representative_decisions_and_explanations_are_unchanged() -> None:
    context = make_context(
        action_id=ActionId.AUCTION2,
        current_resources=(int(Suit.BRICK), int(Suit.WOOD)),
        hand=(int(Suit.ORE), int(Suit.SHEEP)),
        legal_max=9,
    )
    knowledge = make_knowledge(private_cards=2, resource_counts=(3, 3, 3, 3, 3))
    brain_classes = (
        AggressiveHeuristicV1Brain,
        BalancedHeuristicV1Brain,
        PassiveHeuristicV1Brain,
        AggressiveHeuristicV2Brain,
        BalancedHeuristicV2Brain,
        PassiveHeuristicV2Brain,
        AggressiveHeuristicV3Brain,
        BalancedHeuristicV3Brain,
        PassiveHeuristicV3Brain,
    )

    actual = tuple(
        brain_class().choose_explained_decision(context, knowledge, ())
        for brain_class in brain_classes
    )

    expected = (
        (
            BotDecision.submit_bid(3),
            _representative_legacy_explanation(
                liquidity_value=-0.6036876255108243,
                total_value=2.214494192670993,
                reservation_bid=4,
                chosen_bid=3,
            ),
        ),
        (
            BotDecision.submit_bid(3),
            _representative_legacy_explanation(
                liquidity_value=-0.3219667336057732,
                total_value=2.4962150845760442,
                reservation_bid=5,
                chosen_bid=3,
            ),
        ),
        (
            BotDecision.submit_bid(2),
            _representative_legacy_explanation(
                liquidity_value=-0.0795591546343255,
                total_value=3.738622663547492,
                reservation_bid=5,
                chosen_bid=2,
            ),
        ),
        (
            BotDecision.submit_bid(3),
            _representative_legacy_explanation(
                liquidity_value=-0.6036876255108243,
                total_value=2.214494192670993,
                reservation_bid=4,
                chosen_bid=3,
            ),
        ),
        (
            BotDecision.submit_bid(3),
            _representative_legacy_explanation(
                liquidity_value=-0.3219667336057732,
                total_value=2.4962150845760442,
                reservation_bid=5,
                chosen_bid=3,
            ),
        ),
        (
            BotDecision.submit_bid(3),
            _representative_legacy_explanation(
                liquidity_value=-0.12073752510216496,
                total_value=2.6974442930796525,
                reservation_bid=5,
                chosen_bid=3,
            ),
        ),
        (
            BotDecision.submit_bid(3),
            _representative_legacy_explanation(
                liquidity_value=-1.2073752510216487,
                total_value=1.6108065671601688,
                reservation_bid=4,
                chosen_bid=3,
            ),
        ),
        (
            BotDecision.submit_bid(2),
            _representative_legacy_explanation(
                liquidity_value=-0.7425521099203714,
                total_value=3.075629708261446,
                reservation_bid=4,
                chosen_bid=2,
            ),
        ),
        (
            BotDecision.submit_bid(2),
            _representative_legacy_explanation(
                liquidity_value=-0.1325985910572096,
                total_value=3.685583227124608,
                reservation_bid=5,
                chosen_bid=2,
            ),
        ),
    )

    assert tuple(item.decision for item in actual) == tuple(
        decision for decision, _explanation in expected
    )
    for item, (_decision, expected_explanation) in zip(actual, expected, strict=True):
        assert isinstance(item.explanation, HeuristicBidExplanation)
        assert item.explanation.reservation_bid == expected_explanation.reservation_bid
        assert item.explanation.chosen_bid == expected_explanation.chosen_bid
        assert (
            item.explanation.resource_value,
            item.explanation.objective_completion_value,
            item.explanation.objective_progress_value,
            item.explanation.terminal_cash_value,
            item.explanation.liquidity_value,
            item.explanation.future_cash_value,
            item.explanation.total_value,
        ) == pytest.approx(
            (
                expected_explanation.resource_value,
                expected_explanation.objective_completion_value,
                expected_explanation.objective_progress_value,
                expected_explanation.terminal_cash_value,
                expected_explanation.liquidity_value,
                expected_explanation.future_cash_value,
                expected_explanation.total_value,
            ),
            rel=1e-14,
            abs=1e-14,
        )


@pytest.mark.parametrize(
    ("brain_class", "bot_class"),
    (
        (AggressiveHeuristicBrain, AggressiveHeuristicBot),
        (BalancedHeuristicBrain, BalancedHeuristicBot),
        (PassiveHeuristicBrain, PassiveHeuristicBot),
    ),
)
def test_profile_brain_factories_are_deterministic_and_fresh(
    brain_class: type[HeuristicBotBrain],
    bot_class: type[PocketRocksFastBot],
) -> None:
    context = make_context(
        action_id=ActionId.AUCTION1,
        current_resources=(int(Suit.BRICK), 0),
        legal_max=8,
    )
    knowledge = make_knowledge()
    first = bot_class.build_brain(11)
    second = bot_class.build_brain(999)

    assert isinstance(first, brain_class)
    assert first is not second
    assert first.choose_decision(context, knowledge, ()) == second.choose_decision(
        context,
        knowledge,
        (),
    )


@pytest.mark.parametrize(
    "brain_class",
    (AggressiveHeuristicBrain, BalancedHeuristicBrain, PassiveHeuristicBrain),
)
def test_profile_brains_return_legal_bid_and_reveal_decisions(
    brain_class: Callable[[], HeuristicBotBrain],
) -> None:
    knowledge = make_knowledge(
        private_cards=2,
        resource_counts=(3, 3, 3, 3, 3),
    )
    bid_context = make_context(
        action_id=ActionId.AUCTION2,
        current_resources=(int(Suit.BRICK), int(Suit.WOOD)),
        hand=(int(Suit.ORE), int(Suit.SHEEP)),
        legal_max=9,
    )
    reveal_context = make_context(
        decision_kind="selectInfoToReveal",
        action_id=ActionId.AUCTION1,
        current_resources=(0, 0),
        hand=(int(Suit.ORE), int(Suit.SHEEP)),
        legal_max=None,
    )
    brain = brain_class()

    bid = brain.choose_decision(bid_context, knowledge)
    reveal = brain.choose_decision(reveal_context, knowledge)

    assert bid_context.is_legal(bid)
    assert reveal_context.is_legal(reveal)
    assert reveal.action_kind == "selectInfoToReveal"


def test_heuristic_explanation_reuses_the_single_bid_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = make_context(
        action_id=ActionId.AUCTION2,
        current_resources=(int(Suit.BRICK), int(Suit.WOOD)),
        legal_max=9,
    )
    knowledge = make_knowledge()
    brain = HeuristicBotBrain(BALANCED_PROFILE)
    original_evaluate = HeuristicValuator.evaluate_bid
    evaluations = []

    def record_evaluation(
        valuator: HeuristicValuator,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
    ) -> BidEvaluation:
        evaluation = original_evaluate(valuator, context, ruleset)
        evaluations.append(evaluation)
        return evaluation

    monkeypatch.setattr(HeuristicValuator, "evaluate_bid", record_evaluation)

    explained = brain.choose_explained_decision(context, knowledge, ())

    assert len(evaluations) == 1
    evaluation = evaluations[0]
    point = evaluation.points[evaluation.chosen_bid]
    assert explained.decision == (
        BotDecision.pass_turn()
        if evaluation.chosen_bid == 0
        else BotDecision.submit_bid(evaluation.chosen_bid)
    )
    assert explained.explanation == HeuristicBidExplanation(
        resource_value=point.breakdown.resource,
        objective_completion_value=point.breakdown.objective_completion,
        objective_progress_value=point.breakdown.objective_progress,
        terminal_cash_value=point.breakdown.terminal_cash,
        liquidity_value=point.breakdown.liquidity,
        future_cash_value=point.breakdown.future_cash,
        total_value=point.breakdown.total,
        reservation_bid=evaluation.reservation_bid,
        chosen_bid=evaluation.chosen_bid,
    )


def test_opponent_aware_candidate_requires_history_for_bids_but_not_reveals() -> None:
    bid_context = make_context(legal_max=30)
    reveal_context = make_context(
        decision_kind="selectInfoToReveal",
        current_resources=(0, 0),
        hand=(int(Suit.ORE), int(Suit.SHEEP)),
        legal_max=None,
    )
    reveal_knowledge = make_knowledge(private_cards=2, resource_counts=(3, 3, 3, 3, 3))
    candidate = BalancedHeuristicV5CandidateBrain()

    with pytest.raises(HeuristicInputError, match="public history"):
        candidate.choose_decision(bid_context, make_knowledge())
    assert candidate.choose_decision(reveal_context, reveal_knowledge) == (
        BalancedHeuristicV3Brain().choose_decision(reveal_context, reveal_knowledge)
    )


def test_opponent_aware_candidate_values_and_forecasts_once_for_its_explanation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = make_context(
        action_id=ActionId.AUCTION2,
        current_resources=(int(Suit.BRICK), int(Suit.WOOD)),
        legal_max=30,
    )
    history = make_public_history(context)
    knowledge = make_knowledge()
    candidate = BalancedHeuristicV5CandidateBrain()
    original_evaluate = HeuristicValuator.evaluate_bid
    original_forecast = forecast_opponent_bids
    evaluations: list[BidEvaluation] = []
    forecasts: list[OpponentBidForecast] = []

    def record_evaluation(
        valuator: HeuristicValuator,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
    ) -> BidEvaluation:
        evaluation = original_evaluate(valuator, context, ruleset)
        evaluations.append(evaluation)
        return evaluation

    def record_forecast(*args: object, **kwargs: object) -> OpponentBidForecast:
        forecast = original_forecast(*args, **kwargs)  # type: ignore[arg-type]
        forecasts.append(forecast)
        return forecast

    monkeypatch.setattr(HeuristicValuator, "evaluate_bid", record_evaluation)
    monkeypatch.setattr(heuristic_module, "forecast_opponent_bids", record_forecast)

    explained = candidate.choose_explained_decision(context, knowledge, history)

    assert len(evaluations) == 1
    assert len(forecasts) == 1
    assert isinstance(explained.explanation, OpponentAwareHeuristicBidExplanation)
    explanation = explained.explanation
    chosen_point = evaluations[0].points[explanation.chosen_bid]
    assert explanation.model_name == OPPONENT_BID_MODEL_NAME
    assert explanation.model_config_digest == opponent_bid_model_config_digest(
        BALANCED_HEURISTIC_V5_CANDIDATE_MODEL_CONFIG,
    )
    assert explanation.opponent_distributions == forecasts[0].opponent_distributions
    assert explanation.total_value == chosen_point.breakdown.total
    assert explanation.terminal_cash_value == chosen_point.breakdown.terminal_cash
    assert tuple(point.effective_bid for point in explanation.competitive_bid_points) == (
        tuple(range(31))
    )
    assert explained.decision == candidate.choose_decision(
        context,
        knowledge,
        history,
    )


def test_opponent_aware_candidate_uses_the_lower_bid_when_surplus_ties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = make_context(legal_max=30)
    history = make_public_history(context)
    original_forecast = forecast_opponent_bids

    def zero_win_forecast(*args: object, **kwargs: object) -> OpponentBidForecast:
        forecast = original_forecast(*args, **kwargs)  # type: ignore[arg-type]
        return OpponentBidForecast(
            opponent_distributions=forecast.opponent_distributions,
            legal_bid_forecasts=tuple(
                LegalBidWinningForecast(
                    effective_bid=point.effective_bid,
                    win_probability=0.0,
                )
                for point in forecast.legal_bid_forecasts
            ),
        )

    monkeypatch.setattr(heuristic_module, "forecast_opponent_bids", zero_win_forecast)

    assert (
        BalancedHeuristicV5CandidateBrain().choose_decision(
            context,
            make_knowledge(),
            history,
        )
        == BotDecision.pass_turn()
    )


def test_opponent_aware_candidate_rejects_an_incomplete_forecast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = make_context(legal_max=30)
    history = make_public_history(context)
    complete_forecast = forecast_opponent_bids

    def omit_highest_bid(*args: object, **kwargs: object) -> OpponentBidForecast:
        forecast = complete_forecast(*args, **kwargs)  # type: ignore[arg-type]
        return OpponentBidForecast(
            opponent_distributions=forecast.opponent_distributions,
            legal_bid_forecasts=forecast.legal_bid_forecasts[:-1],
        )

    monkeypatch.setattr(heuristic_module, "forecast_opponent_bids", omit_highest_bid)

    with pytest.raises(HeuristicInputError, match="every legal bid"):
        BalancedHeuristicV5CandidateBrain().choose_decision(
            context,
            make_knowledge(),
            history,
        )


def test_opponent_aware_candidate_is_deterministic_and_responds_to_public_history() -> None:
    context = make_context(legal_max=30)
    sparse_history = make_public_history(context)
    learned_history = make_public_history(context, completed_round_count=2)
    candidate = BalancedHeuristicV5CandidateBrain()
    knowledge = make_knowledge()

    first = candidate.choose_explained_decision(context, knowledge, sparse_history)
    second = candidate.choose_explained_decision(context, knowledge, sparse_history)
    learned = candidate.choose_explained_decision(context, knowledge, learned_history)

    assert first == second
    assert isinstance(first.explanation, OpponentAwareHeuristicBidExplanation)
    assert isinstance(learned.explanation, OpponentAwareHeuristicBidExplanation)
    assert first.explanation.opponent_distributions != learned.explanation.opponent_distributions


def test_opponent_aware_candidate_derives_phase_only_from_completed_public_rounds() -> None:
    context = make_context(legal_max=30)
    history = make_public_history(context, completed_round_count=5)

    explained = BalancedHeuristicV5CandidateBrain().choose_explained_decision(
        context,
        make_knowledge(),
        history,
    )

    assert isinstance(explained.explanation, OpponentAwareHeuristicBidExplanation)
    assert explained.explanation.public_game_phase == "middle"


def test_opponent_aware_candidate_rejects_malformed_public_history() -> None:
    with pytest.raises(HeuristicInputError, match="public history"):
        BalancedHeuristicV5CandidateBrain().choose_decision(
            make_context(legal_max=30),
            make_knowledge(),
            (),
        )


def test_ordinary_heuristic_choice_does_not_construct_an_explanation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = make_context(
        action_id=ActionId.AUCTION2,
        current_resources=(int(Suit.BRICK), int(Suit.WOOD)),
        legal_max=9,
    )
    knowledge = make_knowledge()
    brain = HeuristicBotBrain(BALANCED_PROFILE)
    expected = brain.choose_decision(context, knowledge)

    def reject_explanation(**_values: object) -> None:
        raise RuntimeError("explanation construction is disabled")

    monkeypatch.setattr(
        heuristic_module,
        "HeuristicBidExplanation",
        reject_explanation,
    )

    assert brain.choose_decision(context, knowledge) == expected
    with pytest.raises(RuntimeError, match="explanation construction"):
        brain.choose_explained_decision(context, knowledge, ())


def test_heuristic_reveal_decision_has_no_bid_explanation() -> None:
    context = make_context(
        decision_kind="selectInfoToReveal",
        action_id=ActionId.AUCTION1,
        current_resources=(0, 0),
        hand=(int(Suit.ORE), int(Suit.SHEEP)),
        legal_max=None,
    )
    knowledge = make_knowledge(private_cards=2, resource_counts=(3, 3, 3, 3, 3))

    explained = HeuristicBotBrain(BALANCED_PROFILE).choose_explained_decision(
        context,
        knowledge,
        (),
    )

    assert context.is_legal(explained.decision)
    assert explained.explanation is None


@pytest.mark.parametrize("chart_name", ("B", "C", "D", "E"))
@pytest.mark.parametrize(
    ("brain_class", "bot_class"),
    (
        (AggressiveHeuristicBrain, AggressiveHeuristicBot),
        (BalancedHeuristicBrain, BalancedHeuristicBot),
        (PassiveHeuristicBrain, PassiveHeuristicBot),
    ),
)
def test_live_wrapper_reconciles_contextual_public_rules(
    chart_name: str,
    brain_class: Callable[[], HeuristicBotBrain],
    bot_class: type[PocketRocksFastBot],
) -> None:
    chart = VALUE_CHARTS[chart_name]
    context = DecisionContext(
        request_id="contextual-rules",
        deadline_at=2**63 - 1,
        received_at=0,
        decision_kind="submitBid",
        player_count=3,
        starting_cash=30,
        value_chart=chart,
        objective_ids=(1, 2, 3, 4),
        current_action_id=int(ActionId.AUCTION2),
        current_resource_ids=(int(Suit.BRICK), int(Suit.WOOD)),
        cash_by_seat=(30, 30, 30),
        tiebreak_seat=2,
        won_resource_counts_by_seat=((0, 0, 0, 0, 0),) * 3,
        revealed_info_counts_by_seat=((0, 0, 0, 0, 0),) * 3,
        owned_objective_ids_by_seat=((), (), ()),
        bot_seat=0,
        current_hand_suit_ids=tuple(int(suit) for suit in Suit),
        legal_max_amount=30,
        revealable_count=5,
    )
    contextual_knowledge = canonical_knowledge(
        context.player_count,
        value_chart=chart_name,
    )
    expected = brain_class().choose_decision(context, contextual_knowledge)
    bot = bot_class(
        api_key="test-key",
        server_url="ws://example.test",
        reconnect=False,
    )

    actual = bot.choose_decision_sync(context)

    assert expected.action_kind == "submitBid"
    assert context.is_legal(expected)
    assert actual == expected


def test_contextual_knowledge_uses_sdk_canonical_hidden_and_deck_priors() -> None:
    context = replace(
        make_context(
            action_id=ActionId.AUCTION2,
            current_resources=(int(Suit.BRICK), int(Suit.WOOD)),
            hand=(int(Suit.ORE), int(Suit.SHEEP)),
            legal_max=27,
        ),
        player_count=4,
        starting_cash=27,
        value_chart=VALUE_CHARTS["E"],
        objective_ids=(1, 99),
        cash_by_seat=(27, 27, 27, 27),
        tiebreak_seat=3,
        won_resource_counts_by_seat=((0, 0, 0, 0, 0),) * 4,
        revealed_info_counts_by_seat=(
            (0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0),
            (1, 1, 0, 0, 0),
            (0, 0, 0, 0, 0),
        ),
        owned_objective_ids_by_seat=((), (), (), ()),
        bot_seat=2,
    )
    bot = BalancedHeuristicBot(
        api_key="test-key",
        server_url="ws://example.test",
        reconnect=False,
    )

    knowledge = bot._knowledge_for_context(context)

    assert knowledge == knowledge_for_context(context)
    assert knowledge.name == "live-E"
    assert knowledge.player_count == 4
    assert knowledge.starting_cash == 27
    assert knowledge.private_cards_per_player == 4
    assert knowledge.value_chart == VALUE_CHARTS["E"]
    assert knowledge.active_objective_count == 2
    assert knowledge.objectives_enabled
    assert knowledge.resource_counts == (6, 6, 6, 6, 6)
    assert knowledge.action_counts == (12, 8, 3, 2, 3, 2)
    assert knowledge.objective_pool == tuple(sorted(OBJECTIVES))

    disabled_knowledge = bot._knowledge_for_context(replace(context, objective_ids=()))
    assert disabled_knowledge.active_objective_count == 0
    assert not disabled_knowledge.objectives_enabled


def test_brain_safely_passes_only_on_heuristic_input_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = make_context()
    knowledge = make_knowledge()
    brain = HeuristicBotBrain(BALANCED_PROFILE)

    def invalid_input(
        valuator: HeuristicValuator,
        context: object,
        ruleset: object,
    ) -> object:
        del valuator, context, ruleset
        raise HeuristicInputError("bad public context")

    monkeypatch.setattr(HeuristicValuator, "evaluate_bid", invalid_input)
    assert brain.choose_decision(context, knowledge) == BotDecision.pass_turn()

    def programming_error(
        valuator: HeuristicValuator,
        context: object,
        ruleset: object,
    ) -> object:
        del valuator, context, ruleset
        raise RuntimeError("bug")

    monkeypatch.setattr(HeuristicValuator, "evaluate_bid", programming_error)
    with pytest.raises(RuntimeError, match="bug"):
        brain.choose_decision(context, knowledge)


def test_exported_heuristic_specs_are_picklable_and_build_fresh_brains() -> None:
    specs = (
        AGGRESSIVE_HEURISTIC_BOT_SPEC,
        BALANCED_HEURISTIC_BOT_SPEC,
        PASSIVE_HEURISTIC_BOT_SPEC,
        AGGRESSIVE_HEURISTIC_V1_BOT_SPEC,
        BALANCED_HEURISTIC_V1_BOT_SPEC,
        PASSIVE_HEURISTIC_V1_BOT_SPEC,
        AGGRESSIVE_HEURISTIC_V2_BOT_SPEC,
        BALANCED_HEURISTIC_V2_BOT_SPEC,
        PASSIVE_HEURISTIC_V2_BOT_SPEC,
        AGGRESSIVE_HEURISTIC_V3_BOT_SPEC,
        BALANCED_HEURISTIC_V3_BOT_SPEC,
        PASSIVE_HEURISTIC_V3_BOT_SPEC,
    )

    for spec in specs:
        restored = pickle.loads(pickle.dumps(spec))
        assert isinstance(restored, BotSpec)
        assert restored == spec
        assert restored.make_brain(seed=1) is not restored.make_brain(seed=1)


def test_two_worker_monte_carlo_smoke_uses_all_three_heuristic_specs() -> None:
    specs = (
        AGGRESSIVE_HEURISTIC_BOT_SPEC,
        BALANCED_HEURISTIC_BOT_SPEC,
        PASSIVE_HEURISTIC_BOT_SPEC,
    )
    result = MonteCarloRunner.run(
        MonteCarloConfig(
            bot_specs=specs,
            games=3,
            player_counts=(3,),
            value_charts=("A",),
            root_seed=1234,
        ),
        workers=2,
    )

    assert {summary.bot_ids for summary in result.game_summaries}
    assert {statistics.bot_id for statistics in result.bot_statistics} == {
        spec.bot_id for spec in specs
    }
    assert all(statistics.faults == 0 for statistics in result.bot_statistics)
