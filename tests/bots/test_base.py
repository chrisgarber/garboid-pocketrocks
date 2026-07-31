from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from pocketrocks import ActionId, BotDecision, DecisionContext, PocketRocksBot, Suit
from pocketrocks.testing import scenario

import garboid_pocketrocks.bots.base as base_module
from garboid_pocketrocks.adapters.public_history import (
    PublicHistory,
    PublicHistoryCompatibilityError,
    public_history_from_sdk_frame,
)
from garboid_pocketrocks.bots.base import BotBrain, PocketRocksFastBot
from garboid_pocketrocks.knowledge import RulesetKnowledge, knowledge_for_context


class _TestBot(PocketRocksFastBot):
    BOT_ID = "test-bot"
    BOT_NAME = "test"

    @classmethod
    def build_brain(cls, seed: int | None) -> BotBrain:
        del seed
        raise AssertionError("tests inject a brain")


class _RecordingHistoryBrain:
    def __init__(self, decision: BotDecision) -> None:
        self.decision = decision
        self.legacy_calls = 0
        self.history_calls = 0
        self.received_context: DecisionContext | None = None
        self.received_ruleset: RulesetKnowledge | None = None
        self.received_history: PublicHistory | None = None

    def choose_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
    ) -> BotDecision:
        del context, ruleset
        self.legacy_calls += 1
        return BotDecision.pass_turn()

    def choose_decision_with_history(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
        history: PublicHistory,
    ) -> BotDecision:
        self.history_calls += 1
        self.received_context = context
        self.received_ruleset = ruleset
        self.received_history = history
        return self.decision


class _RecordingLegacyBrain:
    def __init__(self, decision: BotDecision) -> None:
        self.decision = decision
        self.calls = 0
        self.received_context: DecisionContext | None = None
        self.received_ruleset: RulesetKnowledge | None = None

    def choose_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
    ) -> BotDecision:
        self.calls += 1
        self.received_context = context
        self.received_ruleset = ruleset
        return self.decision


class _PublicEventsOnlyFrame:
    def __init__(self, common_events: tuple[object, ...]) -> None:
        self.common_events = common_events

    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"live bridge touched non-public frame attribute {name!r}")


def _context() -> DecisionContext:
    return (
        scenario(players=3, starting_cash=30, objective_ids=(1, 2, 3, 4))
        .turn(ActionId.AUCTION1, resources=(Suit.BRICK, 0))
        .deciding(seat=0, hand=(Suit.WOOD,), kind="submitBid")
        .to_context()
    )


def _frame() -> object:
    return SimpleNamespace(
        common_events=(
            SimpleNamespace(
                kind="gameSetup",
                player_count=3,
                starting_cash=30,
                value_chart=(0, 4, 8, 12, 16, 20),
                initial_tiebreak_seat=0,
                objective_ids=(1, 2, 3, 4),
            ),
            SimpleNamespace(
                kind="turnOpened",
                action_id=int(ActionId.AUCTION1),
                resource_ids=(int(Suit.BRICK), 0),
            ),
        )
    )


def _bot(brain: BotBrain) -> _TestBot:
    return _TestBot(
        brain=brain,
        api_key="test-key",
        server_url="ws://example.test",
        reconnect=False,
    )


def test_raw_bridge_passes_exact_parsed_history_to_history_aware_brain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    frame = _frame()
    history = public_history_from_sdk_frame(frame)
    parsed_frames: list[object] = []

    def parse_frame(actual_frame: object) -> PublicHistory:
        parsed_frames.append(actual_frame)
        return history

    monkeypatch.setattr(base_module, "public_history_from_sdk_frame", parse_frame)
    decision = BotDecision.submit_bid(4)
    brain = _RecordingHistoryBrain(decision)

    actual = asyncio.run(_bot(brain).choose_raw_decision(frame, context))

    assert actual == decision
    assert len(parsed_frames) == 1
    assert parsed_frames[0] is frame
    assert brain.history_calls == 1
    assert brain.legacy_calls == 0
    assert brain.received_context is context
    assert brain.received_ruleset == knowledge_for_context(context)
    assert brain.received_history is history


def test_raw_bridge_invokes_ordinary_brain_once_with_legacy_result() -> None:
    context = _context()
    decision = BotDecision.submit_bid(3)
    brain = _RecordingLegacyBrain(decision)

    actual = asyncio.run(_bot(brain).choose_raw_decision(_frame(), context))

    assert actual == decision
    assert brain.calls == 1
    assert brain.received_context is context
    assert brain.received_ruleset == knowledge_for_context(context)


def test_context_only_entry_points_preserve_legacy_history_aware_brain_behavior() -> None:
    context = _context()
    brain = _RecordingHistoryBrain(BotDecision.submit_bid(4))
    bot = _bot(brain)

    sync_decision = bot.choose_decision_sync(context)
    async_decision = asyncio.run(bot.choose_decision(context))

    assert sync_decision == BotDecision.pass_turn()
    assert async_decision == BotDecision.pass_turn()
    assert brain.legacy_calls == 2
    assert brain.history_calls == 0


def test_fast_bot_reports_raw_decision_support() -> None:
    assert _bot(_RecordingLegacyBrain(BotDecision.pass_turn())).uses_raw_decision()
    assert _TestBot.choose_raw_decision is not PocketRocksBot.choose_raw_decision


def test_raw_bridge_rejects_malformed_history_before_invoking_brain() -> None:
    brain = _RecordingLegacyBrain(BotDecision.pass_turn())

    with pytest.raises(PublicHistoryCompatibilityError):
        asyncio.run(_bot(brain).choose_raw_decision(object(), _context()))

    assert brain.calls == 0


def test_raw_bridge_does_not_touch_non_public_frame_attributes() -> None:
    frame = _frame()
    assert isinstance(frame, SimpleNamespace)
    poisoned_frame = _PublicEventsOnlyFrame(frame.common_events)
    brain = _RecordingLegacyBrain(BotDecision.pass_turn())

    actual = asyncio.run(_bot(brain).choose_raw_decision(poisoned_frame, _context()))

    assert actual == BotDecision.pass_turn()
    assert brain.calls == 1
