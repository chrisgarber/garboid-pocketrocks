from __future__ import annotations

import asyncio
import inspect
import pickle
from importlib import import_module
from pathlib import Path

import pytest
from pocketrocks import ActionId, BotDecision, DecisionContext, Suit
from pocketrocks.sim.sample_bots import GreedyValueBot

from garboid_pocketrocks.bots.sdk_samples import (
    SDK_GREEDY_VALUE_V1_BOT_SPEC,
    SdkGreedyValueV1Brain,
    _run_immediate_decision,
)
from garboid_pocketrocks.knowledge import canonical_knowledge

SDK_REVISION = "51cad378ee1e70a78e39ebbb25957ea003444873"


def _bid_context(
    *,
    resources: tuple[int, int],
    hand: tuple[int, ...],
    legal_max: int,
    revealed_brick: int = 0,
) -> DecisionContext:
    return DecisionContext(
        request_id="sdk-greedy-value-test",
        deadline_at=2**63 - 1,
        received_at=0,
        decision_kind="submitBid",
        player_count=3,
        starting_cash=30,
        value_chart=(0, 4, 8, 12, 16, 20),
        objective_ids=(),
        current_action_id=int(ActionId.AUCTION1),
        current_resource_ids=resources,
        cash_by_seat=(30, 30, 30),
        tiebreak_seat=2,
        won_resource_counts_by_seat=((0, 0, 0, 0, 0),) * 3,
        revealed_info_counts_by_seat=(
            (revealed_brick, 0, 0, 0, 0),
            (0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0),
        ),
        owned_objective_ids_by_seat=((), (), ()),
        bot_seat=0,
        current_hand_suit_ids=hand,
        legal_max_amount=legal_max,
        revealable_count=len(hand),
    )


def _reveal_context() -> DecisionContext:
    return DecisionContext(
        request_id="sdk-greedy-value-test",
        deadline_at=2**63 - 1,
        received_at=0,
        decision_kind="selectInfoToReveal",
        player_count=3,
        starting_cash=30,
        value_chart=(0, 4, 8, 12, 16, 20),
        objective_ids=(),
        current_action_id=int(ActionId.AUCTION1),
        current_resource_ids=(int(Suit.BRICK), 0),
        cash_by_seat=(30, 30, 30),
        tiebreak_seat=2,
        won_resource_counts_by_seat=((0, 0, 0, 0, 0),) * 3,
        revealed_info_counts_by_seat=((0, 0, 0, 0, 0),) * 3,
        owned_objective_ids_by_seat=((), (), ()),
        bot_seat=0,
        current_hand_suit_ids=(int(Suit.BRICK), int(Suit.WOOD)),
        legal_max_amount=None,
        revealable_count=2,
    )


def _sdk_decision(context: DecisionContext) -> BotDecision:
    bot = GreedyValueBot(api_key="test-key", bot_id="test-bot", reconnect=False)
    return asyncio.run(bot.choose_decision(context))


def test_sdk_greedy_value_adapter_has_frozen_local_identity() -> None:
    module = import_module("garboid_pocketrocks.bots.sdk_samples")

    spec = module.SDK_GREEDY_VALUE_V1_BOT_SPEC
    assert spec.name == "sdk-greedy-value-v1"
    assert spec.bot_id == "sdk-greedy-value-v1"
    assert not hasattr(module.SdkGreedyValueV1Brain, "BOT_ID")


def test_sdk_dependency_revision_is_pinned_for_v1_reproducibility() -> None:
    dependency = (
        "pocketrocks-python-sdk @ "
        f"git+https://github.com/chrisgarber/pocketrocks-python-sdk.git@{SDK_REVISION}"
    )

    assert f'"{dependency}",' in Path("pyproject.toml").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("context", "expected"),
    (
        (
            _bid_context(
                resources=(int(Suit.BRICK), 0),
                hand=(int(Suit.BRICK), int(Suit.BRICK)),
                revealed_brick=1,
                legal_max=30,
            ),
            BotDecision.submit_bid(12),
        ),
        (
            _bid_context(
                resources=(int(Suit.WOOD), 0),
                hand=(int(Suit.BRICK),),
                legal_max=30,
            ),
            BotDecision.submit_bid(0),
        ),
        (
            _bid_context(
                resources=(int(Suit.BRICK), 0),
                hand=(int(Suit.BRICK), int(Suit.BRICK)),
                revealed_brick=1,
                legal_max=7,
            ),
            BotDecision.submit_bid(7),
        ),
        (_reveal_context(), BotDecision.select_info_to_reveal(0)),
    ),
)
def test_sdk_greedy_value_v1_pins_and_matches_sdk_decisions(
    context: DecisionContext,
    expected: BotDecision,
) -> None:
    actual = SdkGreedyValueV1Brain().choose_decision(context, canonical_knowledge(3))

    assert actual == expected
    assert actual == _sdk_decision(context)
    assert context.is_legal(actual)


def test_sdk_greedy_value_v1_spec_is_pickle_safe_and_deterministic() -> None:
    restored = pickle.loads(pickle.dumps(SDK_GREEDY_VALUE_V1_BOT_SPEC))
    context = _bid_context(
        resources=(int(Suit.BRICK), 0),
        hand=(int(Suit.BRICK),),
        legal_max=30,
    )
    knowledge = canonical_knowledge(3)

    assert restored == SDK_GREEDY_VALUE_V1_BOT_SPEC
    assert restored.make_brain(seed=1).choose_decision(context, knowledge) == restored.make_brain(
        seed=999
    ).choose_decision(context, knowledge)


async def _suspending_decision() -> BotDecision:
    await asyncio.sleep(0)
    return BotDecision.pass_turn()


def test_immediate_bridge_rejects_and_closes_a_suspending_coroutine() -> None:
    coroutine = _suspending_decision()

    with pytest.raises(RuntimeError, match="must complete synchronously"):
        _run_immediate_decision(coroutine)

    assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED
