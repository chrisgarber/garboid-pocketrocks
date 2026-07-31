from __future__ import annotations

import asyncio

import pytest
from pocketrocks import ActionId, BotDecision, Suit
from pocketrocks.testing import FakeTransport, decode_frames, scenario

pytest.importorskip("torch")

from garboid_pocketrocks.bots.base import PocketRocksFastBot  # noqa: E402
from garboid_pocketrocks.neural.live_bot import PpoLargeTeenBot  # noqa: E402
from garboid_pocketrocks.neural.tournament_bot import (  # noqa: E402
    VectorPpoLargeV1G350kBrain,
)


def test_ppo_large_teen_is_public_alias_for_frozen_large_policy() -> None:
    assert issubclass(PpoLargeTeenBot, PocketRocksFastBot)
    assert PpoLargeTeenBot.BOT_NAME == "ppo-large-teen"
    assert PpoLargeTeenBot.BOT_ID == "bot_dd7807c1-93bc-4f70-80c8-a2f2d7d26429"
    assert isinstance(PpoLargeTeenBot.build_brain(seed=7), VectorPpoLargeV1G350kBrain)


def test_live_bot_uses_raw_sdk_history_to_return_a_legal_decision() -> None:
    request_id = "22222222-2222-2222-2222-222222222222"
    game = (
        scenario(players=3, starting_cash=30)
        .turn(ActionId.AUCTION1, resources=(Suit.BRICK, Suit.WOOD))
        .deciding(
            seat=0,
            hand=(Suit.BRICK, Suit.WOOD, Suit.ORE, Suit.SHEEP, Suit.WHEAT),
            kind="submitBid",
            request_id=request_id,
        )
    )
    context = game.to_context()
    transport = FakeTransport([game.to_bytes()])
    bot = PpoLargeTeenBot(
        api_key="test-key",
        server_url="ws://example.test",
        reconnect=False,
        transport=transport,
    )

    assert bot.uses_raw_decision()
    asyncio.run(bot.run_async())

    sent = decode_frames(transport.sent_messages)
    assert len(sent) == 1
    response = sent[0]
    assert response.kind == "decisionResponse"
    assert response.request_id == request_id
    decision = BotDecision(action_kind=response.action_kind, value=response.value)
    assert context.is_legal(decision)
    assert transport.disconnected
