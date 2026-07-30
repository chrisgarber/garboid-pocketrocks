from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import pytest
from pocketrocks import ActionId, BotDecision, DecisionContext, Suit

from garboid_pocketrocks.bots.llm.brain import StatelessLLMBrain
from garboid_pocketrocks.rules import LIVE_RULESET, RulesetKnowledge


@dataclass
class ScriptedBackend:
    outcomes: list[str | Exception]
    calls: list[tuple[str, float]] = field(default_factory=list)

    def complete(self, prompt: str, *, timeout_seconds: float) -> str:
        self.calls.append((prompt, timeout_seconds))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@dataclass
class RecordingSkill:
    corrections: list[str | None] = field(default_factory=list)

    def render(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
        *,
        correction: str | None = None,
    ) -> str:
        self.corrections.append(correction)
        return f"request={context.request_id}; ruleset={ruleset.name}; correction={correction}"


def _context(
    *,
    decision_kind: str = "submitBid",
    legal_max: int | None = 12,
    hand: tuple[int, ...] = (int(Suit.BRICK), int(Suit.WOOD)),
    deadline_at: int = 2**63 - 1,
) -> DecisionContext:
    return DecisionContext(
        request_id="llm-brain-test",
        deadline_at=deadline_at,
        received_at=0,
        decision_kind=decision_kind,  # type: ignore[arg-type]
        player_count=3,
        starting_cash=30,
        value_chart=(0, 4, 8, 12, 16, 20),
        objective_ids=(),
        current_action_id=int(ActionId.AUCTION1),
        current_resource_ids=(int(Suit.ORE), 0),
        cash_by_seat=(30, 22, 17),
        tiebreak_seat=2,
        won_resource_counts_by_seat=((0, 0, 0, 0, 0),) * 3,
        revealed_info_counts_by_seat=((0, 0, 0, 0, 0),) * 3,
        owned_objective_ids_by_seat=((), (), ()),
        bot_seat=0,
        current_hand_suit_ids=hand,
        legal_max_amount=legal_max,
        revealable_count=len(hand),
    )


def _brain(
    outcomes: list[str | Exception],
) -> tuple[StatelessLLMBrain, ScriptedBackend, RecordingSkill]:
    backend = ScriptedBackend(outcomes)
    skill = RecordingSkill()
    return (
        StatelessLLMBrain(backend, prompt_skill=skill),
        backend,
        skill,
    )


@pytest.mark.parametrize(
    ("response", "expected"),
    (
        (" 7\n", BotDecision.submit_bid(7)),
        ("0", BotDecision.pass_turn()),
    ),
)
def test_valid_bid_integer_becomes_sdk_decision(
    response: str,
    expected: BotDecision,
) -> None:
    brain, backend, skill = _brain([response])

    decision = brain.choose_decision(_context(), LIVE_RULESET.knowledge(3))

    assert decision == expected
    assert len(backend.calls) == 1
    assert skill.corrections == [None]


def test_valid_reveal_integer_becomes_sdk_decision() -> None:
    brain, _, _ = _brain(["1"])
    context = _context(decision_kind="selectInfoToReveal", legal_max=None)

    decision = brain.choose_decision(context, LIVE_RULESET.knowledge(3))

    assert decision == BotDecision.select_info_to_reveal(1)
    assert context.is_legal(decision)


@pytest.mark.parametrize("invalid", ("bid 3", "+3", "-1", "3.0", '{"bid": 3}', "", "13"))
def test_invalid_response_retries_with_correction(invalid: str) -> None:
    brain, backend, skill = _brain([invalid, "3"])

    decision = brain.choose_decision(_context(), LIVE_RULESET.knowledge(3))

    assert decision == BotDecision.submit_bid(3)
    assert len(backend.calls) == 2
    assert skill.corrections[0] is None
    assert skill.corrections[1] is not None
    assert "invalid" in skill.corrections[1].lower()


def test_backend_exception_retries_once() -> None:
    brain, backend, skill = _brain([RuntimeError("provider unavailable"), "4"])

    decision = brain.choose_decision(_context(), LIVE_RULESET.knowledge(3))

    assert decision == BotDecision.submit_bid(4)
    assert len(backend.calls) == 2
    assert skill.corrections[1] is not None
    assert "provider unavailable" in skill.corrections[1]


@pytest.mark.parametrize(
    ("context", "expected"),
    (
        (_context(), BotDecision.pass_turn()),
        (
            _context(decision_kind="selectInfoToReveal", legal_max=None),
            BotDecision.select_info_to_reveal(0),
        ),
    ),
)
def test_second_failure_logs_and_uses_deterministic_fallback(
    context: DecisionContext,
    expected: BotDecision,
    caplog: pytest.LogCaptureFixture,
) -> None:
    brain, backend, _ = _brain(["not an integer", RuntimeError("still unavailable")])

    with caplog.at_level(logging.WARNING):
        decision = brain.choose_decision(context, LIVE_RULESET.knowledge(3))

    assert decision == expected
    assert context.is_legal(decision)
    assert len(backend.calls) == 2
    assert "llm-brain-test" in caplog.text
    assert context.decision_kind in caplog.text
    assert "attempt 2/2" in caplog.text
    assert "fallback" in caplog.text


@pytest.mark.parametrize(
    "context",
    (
        _context(legal_max=0),
        _context(legal_max=None),
        _context(decision_kind="selectInfoToReveal", legal_max=None, hand=()),
    ),
)
def test_no_meaningful_choice_skips_backend(context: DecisionContext) -> None:
    brain, backend, skill = _brain([])

    decision = brain.choose_decision(context, LIVE_RULESET.knowledge(3))

    assert decision == BotDecision.pass_turn()
    assert backend.calls == []
    assert skill.corrections == []


def test_timeout_is_bounded_and_reserves_deadline_for_retry() -> None:
    deadline_at = int(time.time() * 1000) + 10_000
    context = _context(deadline_at=deadline_at)
    backend = ScriptedBackend(["invalid", "5"])
    brain = StatelessLLMBrain(
        backend,
        prompt_skill=RecordingSkill(),
        timeout_seconds=30.0,
        deadline_margin_seconds=0.5,
    )

    decision = brain.choose_decision(context, LIVE_RULESET.knowledge(3))

    assert decision == BotDecision.submit_bid(5)
    assert len(backend.calls) == 2
    first_timeout = backend.calls[0][1]
    second_timeout = backend.calls[1][1]
    assert 0 < first_timeout <= 4.75
    assert 0 < second_timeout <= 9.5
    assert first_timeout <= second_timeout


def test_expired_deadline_falls_back_without_calling_backend() -> None:
    context = _context(deadline_at=int(time.time() * 1000) - 1)
    brain, backend, _ = _brain(["7"])

    decision = brain.choose_decision(context, LIVE_RULESET.knowledge(3))

    assert decision == BotDecision.pass_turn()
    assert backend.calls == []


def test_config_rejects_nonpositive_timeouts() -> None:
    backend = ScriptedBackend([])

    with pytest.raises(ValueError, match="timeout_seconds must be positive"):
        StatelessLLMBrain(backend, timeout_seconds=0)

    with pytest.raises(ValueError, match="deadline_margin_seconds must be nonnegative"):
        StatelessLLMBrain(backend, deadline_margin_seconds=-0.1)
