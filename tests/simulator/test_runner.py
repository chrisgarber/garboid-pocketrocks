from __future__ import annotations

import pytest
from pocketrocks import BotDecision, DecisionContext

from garboid_pocketrocks.adapters.public_history import (
    PublicGameSetup,
    PublicHistory,
    public_history_from_sdk_events,
)
from garboid_pocketrocks.bots import BotSpec, RandomBot
from garboid_pocketrocks.knowledge import RulesetKnowledge
from garboid_pocketrocks.simulator.runner import FaultMode, MatchRunner


class RaisingBrain:
    def choose_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
    ) -> BotDecision:
        del context, ruleset
        raise RuntimeError("brain exploded")


def _raising_brain(seed: int | None) -> RaisingBrain:
    del seed
    return RaisingBrain()


_RECORDED_HISTORIES: list[PublicHistory] = []


class HistoryRecordingBrain:
    def choose_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
    ) -> BotDecision:
        del context, ruleset
        raise RuntimeError("runner omitted public history")

    def choose_decision_with_history(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
        history: PublicHistory,
    ) -> BotDecision:
        del ruleset
        _RECORDED_HISTORIES.append(history)
        if context.decision_kind == "selectInfoToReveal":
            return BotDecision.select_info_to_reveal(0)
        return BotDecision.submit_bid(0)


def _history_recording_brain(seed: int | None) -> HistoryRecordingBrain:
    del seed
    return HistoryRecordingBrain()


def _random_lineup() -> tuple[BotSpec, ...]:
    return tuple(BotSpec.from_bot_class(RandomBot) for _ in range(3))


def test_match_runner_is_reproducible_and_uses_fresh_brains() -> None:
    lineup = _random_lineup()

    left = MatchRunner.run(
        lineup,
        player_count=3,
        seed=91,
        value_chart="E",
    )
    right = MatchRunner.run(
        lineup,
        player_count=3,
        seed=91,
        value_chart="E",
    )

    assert left.result == right.result
    assert left.events == right.events
    assert left.replay == right.replay


def test_record_and_pass_records_brain_failure() -> None:
    lineup = (
        BotSpec("raising", "test-raising", _raising_brain),
        *_random_lineup()[:2],
    )

    match = MatchRunner.run(
        lineup,
        player_count=3,
        seed=5,
        fault_mode=FaultMode.RECORD_AND_PASS,
    )

    assert match.faults
    assert match.faults[0].seat == 0
    assert match.faults[0].bot_name == "raising"
    assert match.faults[0].error_type == "RuntimeError"
    assert match.result.scores


def test_raise_mode_propagates_original_brain_exception() -> None:
    lineup = (
        BotSpec("raising", "test-raising", _raising_brain),
        *_random_lineup()[:2],
    )

    with pytest.raises(RuntimeError, match="brain exploded"):
        MatchRunner.run(
            lineup,
            player_count=3,
            seed=5,
            fault_mode=FaultMode.RAISE,
        )


def test_match_runner_supplies_exact_public_history() -> None:
    _RECORDED_HISTORIES.clear()
    lineup = (
        BotSpec.for_simulation("history-recording", _history_recording_brain),
        *_random_lineup()[:2],
    )

    match = MatchRunner.run(
        lineup,
        player_count=3,
        seed=17,
        value_chart="C",
    )

    assert _RECORDED_HISTORIES
    assert all(isinstance(history[0], PublicGameSetup) for history in _RECORDED_HISTORIES)
    for history in _RECORDED_HISTORIES:
        assert history == public_history_from_sdk_events(match.events[: len(history)])


def test_automatic_reveals_are_not_recorded_as_bot_decisions() -> None:
    match = MatchRunner.run(
        _random_lineup(),
        player_count=3,
        seed=31,
    )

    automatic_reveals = tuple(
        turn.reveal for turn in match.turns if turn.reveal is not None and turn.reveal.auto
    )
    choice_reveal_decisions = tuple(
        decisions for _step, decisions in match.replay.decisions if len(decisions) == 1
    )
    nonautomatic_reveals = tuple(
        turn.reveal for turn in match.turns if turn.reveal is not None and not turn.reveal.auto
    )

    assert automatic_reveals
    assert len(choice_reveal_decisions) == len(nonautomatic_reveals)
