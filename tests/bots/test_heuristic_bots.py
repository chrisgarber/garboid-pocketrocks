from __future__ import annotations

import pickle
from collections.abc import Callable

import pytest
from pocketrocks import OBJECTIVES, ActionId, BotDecision, DecisionContext, Suit

from garboid_pocketrocks.bots import (
    AGGRESSIVE_HEURISTIC_BOT_SPEC,
    BALANCED_HEURISTIC_BOT_SPEC,
    PASSIVE_HEURISTIC_BOT_SPEC,
    AggressiveHeuristicBot,
    AggressiveHeuristicBrain,
    BalancedHeuristicBot,
    BalancedHeuristicBrain,
    BotSpec,
    PassiveHeuristicBot,
    PassiveHeuristicBrain,
    PocketRocksFastBot,
)
from garboid_pocketrocks.bots.heuristic import HeuristicBotBrain
from garboid_pocketrocks.heuristics.errors import HeuristicInputError
from garboid_pocketrocks.heuristics.profiles import BALANCED_PROFILE
from garboid_pocketrocks.heuristics.valuation import HeuristicValuator
from garboid_pocketrocks.rules import LIVE_RULESET, RulesetKnowledge
from garboid_pocketrocks.simulator.monte_carlo import (
    MonteCarloConfig,
    MonteCarloRunner,
)
from garboid_pocketrocks.simulator.sampling import FixedRulesetSampler


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
    assert AggressiveHeuristicBot.BOT_ID == "bot_00000000-0000-4000-8000-00000000000a"
    assert BalancedHeuristicBot.BOT_ID == "bot_00000000-0000-4000-8000-00000000000b"
    assert PassiveHeuristicBot.BOT_ID == "bot_00000000-0000-4000-8000-00000000000c"
    assert {
        AggressiveHeuristicBot.BOT_NAME,
        BalancedHeuristicBot.BOT_NAME,
        PassiveHeuristicBot.BOT_NAME,
    } == {"aggressive", "balanced", "passive"}


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
    assert first.choose_decision(context, knowledge) == second.choose_decision(
        context, knowledge
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
            ruleset_sampler=FixedRulesetSampler(LIVE_RULESET),
            root_seed=1234,
        ),
        workers=2,
    )

    assert {summary.bot_ids for summary in result.game_summaries}
    assert {statistics.bot_id for statistics in result.bot_statistics} == {
        spec.bot_id for spec in specs
    }
    assert all(statistics.faults == 0 for statistics in result.bot_statistics)
