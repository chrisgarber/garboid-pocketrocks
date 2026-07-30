from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest
from pocketrocks import ActionId, BotDecision, DecisionContext, Suit
from pocketrocks.testing import scenario

from garboid_pocketrocks.bots import CodexBot
from garboid_pocketrocks.bots.llm import CODEX_BOT_SPEC
from garboid_pocketrocks.bots.llm.brain import StatelessLLMBrain
from garboid_pocketrocks.bots.llm.codex_bot import _brain_from_args, _parser
from garboid_pocketrocks.bots.llm.codex_cli import CodexCLIBackend
from garboid_pocketrocks.rules import LIVE_RULESET
from garboid_pocketrocks.simulator.cli import _BOT_REGISTRY, _bot_names

CODEX_DEVELOPMENT_BOT_ID = "bot_00000000-0000-4000-8000-00000000000d"


class ConstantBackend:
    def __init__(self, response: str) -> None:
        self.response = response

    def complete(self, prompt: str, *, timeout_seconds: float) -> str:
        del prompt, timeout_seconds
        return self.response


def _context() -> DecisionContext:
    return (
        scenario(players=3, starting_cash=30)
        .turn(ActionId.AUCTION1, resources=(Suit.BRICK, Suit.WOOD))
        .deciding(seat=0, hand=[Suit.ORE], kind="submitBid")
        .override(legal_max_amount=9, deadline_at=2**63 - 1)
        .to_context()
    )


def _bot(brain: StatelessLLMBrain) -> CodexBot:
    return CodexBot(
        brain=brain,
        api_key="test-key",
        bot_id="test-bot",
        server_url="ws://example.test",
        reconnect=False,
    )


def test_codex_bot_has_development_identity_and_fresh_default_brain() -> None:
    left = CodexBot.build_brain(None)
    right = CodexBot.build_brain(42)

    assert CodexBot.BOT_ID == CODEX_DEVELOPMENT_BOT_ID
    assert CodexBot.BOT_NAME == "codex"
    assert isinstance(left, StatelessLLMBrain)
    assert isinstance(left.backend, CodexCLIBackend)
    assert left is not right
    assert CODEX_BOT_SPEC.name == "codex"
    assert isinstance(CODEX_BOT_SPEC.make_brain(), StatelessLLMBrain)


def test_injected_backend_drives_legal_sync_decision() -> None:
    brain = StatelessLLMBrain(ConstantBackend("6"))
    bot = _bot(brain)
    context = _context()

    decision = bot.choose_decision_sync(context)

    assert decision == BotDecision.submit_bid(6)
    assert context.is_legal(decision)


def test_live_async_bridge_offloads_sync_brain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    brain = StatelessLLMBrain(ConstantBackend("4"))
    bot = _bot(brain)
    context = _context()
    calls: list[tuple[Callable[..., BotDecision], tuple[object, ...]]] = []

    async def fake_to_thread(
        function: Callable[..., BotDecision],
        *args: object,
    ) -> BotDecision:
        calls.append((function, args))
        return function(*args)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    decision = asyncio.run(bot.choose_decision(context))

    assert decision == BotDecision.submit_bid(4)
    assert len(calls) == 1
    assert calls[0][0] == bot.choose_decision_sync
    assert calls[0][1] == (context,)


def test_live_cli_arguments_build_configured_brain() -> None:
    args = _parser().parse_args(
        [
            "--model",
            "gpt-test",
            "--timeout-seconds",
            "12.5",
            "--codex-executable",
            "/opt/codex",
        ]
    )

    brain = _brain_from_args(args)

    assert brain.timeout_seconds == 12.5
    assert isinstance(brain.backend, CodexCLIBackend)
    assert brain.backend.model == "gpt-test"
    assert brain.backend.executable == "/opt/codex"


def test_live_cli_rejects_nonpositive_timeout() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(["--timeout-seconds", "0"])


def test_simulator_registry_accepts_codex() -> None:
    assert _bot_names("codex,random") == ("codex", "random")
    assert _BOT_REGISTRY["codex"] == CODEX_BOT_SPEC
    assert isinstance(_BOT_REGISTRY["codex"].make_brain(), StatelessLLMBrain)


def test_codex_brain_reconciles_live_context_rules() -> None:
    brain = StatelessLLMBrain(ConstantBackend("2"))
    bot = _bot(brain)
    context = _context()

    decision = bot.choose_decision_sync(context)

    assert decision == BotDecision.submit_bid(2)
    assert LIVE_RULESET.knowledge(context.player_count).player_count == context.player_count
