from __future__ import annotations

import pickle
from collections.abc import Callable
from dataclasses import replace

import pytest
from pocketrocks import OBJECTIVES, ActionId, BotDecision, DecisionContext, Suit
from pocketrocks.sim.constants import VALUE_CHARTS

import garboid_pocketrocks.bots.heuristic as heuristic_module
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
    PASSIVE_HEURISTIC_BOT_SPEC,
    PASSIVE_HEURISTIC_V1_BOT_SPEC,
    PASSIVE_HEURISTIC_V2_BOT_SPEC,
    PASSIVE_HEURISTIC_V3_BOT_SPEC,
    HeuristicBotBrain,
)
from garboid_pocketrocks.diagnostics.trace import HeuristicBidExplanation
from garboid_pocketrocks.heuristics.errors import HeuristicInputError
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
    assert first.choose_decision(context, knowledge) == second.choose_decision(context, knowledge)


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
