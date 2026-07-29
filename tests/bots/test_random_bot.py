from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest
from pocketrocks import ActionId, BotDecision, DecisionContext, Suit
from pocketrocks.testing import FakeTransport, decode_frames, scenario

from garboid_pocketrocks.bots.base import BotSpec, PocketRocksFastBot
from garboid_pocketrocks.bots.random_bot import RandomBot, RandomBotBrain
from garboid_pocketrocks.rules import LIVE_RULESET

RANDOM_BOT_ID = "bot_e0e2c541-1615-4f47-983c-224e7d888d89"


def _bot(seed: int) -> RandomBot:
    return RandomBot(
        seed=seed,
        api_key="test-key",
        bot_id="test-bot",
        server_url="ws://example.test",
        reconnect=False,
    )


def _bid_context(max_amount: int | None) -> DecisionContext:
    return (
        scenario(players=3, starting_cash=20)
        .turn(ActionId.AUCTION1, resources=(Suit.BRICK, Suit.WOOD))
        .deciding(seat=0, hand=[Suit.BRICK], kind="submitBid")
        .override(legal_max_amount=max_amount)
        .to_context()
    )


def _reveal_context(count: int) -> DecisionContext:
    hand = [Suit.BRICK] * max(0, count)
    return (
        scenario(players=3, starting_cash=20)
        .deciding(seat=0, hand=hand, kind="selectInfoToReveal")
        .override(revealable_count=count)
        .to_context()
    )


def _choose(bot: RandomBot, context: DecisionContext) -> BotDecision:
    return asyncio.run(bot.choose_decision(context))


async def _choose_all(
    bot: RandomBot,
    contexts: Sequence[DecisionContext],
) -> list[BotDecision]:
    return [await bot.choose_decision(context) for context in contexts]


def test_random_bot_has_static_public_identity() -> None:
    assert issubclass(RandomBot, PocketRocksFastBot)
    assert RandomBot.BOT_ID == RANDOM_BOT_ID
    assert RandomBot.BOT_NAME == "random"


def test_random_brain_and_async_bridge_return_same_decision() -> None:
    context = _bid_context(7)
    brain = RandomBotBrain(seed=42)
    expected = brain.choose_decision(context, LIVE_RULESET.knowledge(3))
    bot = _bot(seed=42)

    assert _choose(bot, context) == expected
    assert _bot(seed=42).choose_decision_sync(context) == RandomBotBrain(
        seed=42
    ).choose_decision(context, LIVE_RULESET.knowledge(3))


def test_bot_spec_builds_fresh_brains() -> None:
    spec = BotSpec.from_bot_class(RandomBot)
    left = spec.make_brain(seed=11)
    right = spec.make_brain(seed=11)

    assert left is not right
    assert spec.bot_id == RANDOM_BOT_ID


@pytest.mark.parametrize("max_amount", [None, 0, -1])
def test_nonpositive_or_missing_bid_limit_passes(max_amount: int | None) -> None:
    context = _bid_context(max_amount)

    decision = _choose(_bot(seed=1), context)

    assert decision == BotDecision.pass_turn()
    assert context.is_legal(decision)


def test_positive_bid_limit_produces_every_legal_amount_across_seeds() -> None:
    context = _bid_context(7)

    decisions = [_choose(_bot(seed=seed), context) for seed in range(100)]
    amounts = {0 if decision.action_kind == "pass" else decision.value for decision in decisions}

    assert amounts == set(range(8))
    assert all(context.is_legal(decision) for decision in decisions)


def test_empty_reveal_range_passes() -> None:
    context = _reveal_context(0)

    decision = _choose(_bot(seed=1), context)

    assert decision == BotDecision.pass_turn()
    assert context.is_legal(decision)


def test_reveal_choices_cover_valid_indices_across_seeds() -> None:
    context = _reveal_context(3)

    decisions = [_choose(_bot(seed=seed), context) for seed in range(50)]

    assert {decision.value for decision in decisions} == {0, 1, 2}
    assert all(decision.action_kind == "selectInfoToReveal" for decision in decisions)
    assert all(context.is_legal(decision) for decision in decisions)


def test_equal_seeds_produce_equal_decision_sequences() -> None:
    contexts = [_bid_context(11), _reveal_context(4)] * 10

    left = asyncio.run(_choose_all(_bot(seed=42), contexts))
    right = asyncio.run(_choose_all(_bot(seed=42), contexts))

    assert left == right


def test_fake_transport_drives_random_bot_runtime() -> None:
    request_id = "11111111-1111-1111-1111-111111111111"
    game = (
        scenario(players=3, starting_cash=20)
        .turn(ActionId.AUCTION1, resources=(Suit.BRICK, Suit.WOOD))
        .deciding(
            seat=0,
            hand=[Suit.BRICK, Suit.WOOD],
            kind="submitBid",
            request_id=request_id,
        )
    )
    context = game.to_context()
    transport = FakeTransport([game.to_bytes()])
    bot = RandomBot(
        seed=7,
        api_key="test-key",
        bot_id="test-bot",
        server_url="ws://example.test",
        reconnect=False,
        transport=transport,
    )

    asyncio.run(bot.run_async())

    sent = decode_frames(transport.sent_messages)
    assert len(sent) == 1
    response = sent[0]
    assert response.kind == "decisionResponse"
    assert response.request_id == request_id
    decision = BotDecision(action_kind=response.action_kind, value=response.value)
    assert context.is_legal(decision)
    assert transport.disconnected
