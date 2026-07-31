from __future__ import annotations

import random

import pytest
from pocketrocks import BotDecision, DecisionContext
from pocketrocks.exceptions import InvalidBotDecision

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


def _construction_failure(seed: int | None) -> RaisingBrain:
    del seed
    raise LookupError("factory exploded")


_RECORDED_BRAIN_SEEDS: list[int | None] = []


def _seed_recording_brain(seed: int | None) -> RaisingBrain:
    _RECORDED_BRAIN_SEEDS.append(seed)
    return RaisingBrain()


def _seed_recording_failure(seed: int | None) -> RaisingBrain:
    _RECORDED_BRAIN_SEEDS.append(seed)
    raise LookupError("seeded factory exploded")


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


class IllegalDecisionBrain:
    def choose_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
    ) -> BotDecision:
        del ruleset
        if context.decision_kind == "selectInfoToReveal":
            return BotDecision.select_info_to_reveal(context.revealable_count)
        assert context.legal_max_amount is not None
        return BotDecision.submit_bid(context.legal_max_amount + 1)


def _illegal_decision_brain(seed: int | None) -> IllegalDecisionBrain:
    del seed
    return IllegalDecisionBrain()


def _random_lineup() -> tuple[BotSpec, ...]:
    return tuple(BotSpec.from_bot_class(RandomBot) for _ in range(3))


def test_initialize_brains_preserves_per_seat_rng_order() -> None:
    from garboid_pocketrocks.simulator.bot_execution import initialize_brains

    seed = 73
    lineup = tuple(
        BotSpec(f"recording-{seat}", f"recording-{seat}", _seed_recording_brain)
        for seat in range(3)
    )
    expected_rng = random.Random(seed)
    expected_seeds = [expected_rng.randrange(2**63) for _seat in lineup]
    _RECORDED_BRAIN_SEEDS.clear()

    brains, faults = initialize_brains(
        lineup,
        seed=seed,
        fault_mode=FaultMode.RAISE,
    )

    assert len(brains) == len(lineup)
    assert faults == ()
    assert _RECORDED_BRAIN_SEEDS == expected_seeds


def test_initialize_brains_consumes_failed_factory_seed_before_later_seats() -> None:
    from garboid_pocketrocks.simulator.bot_execution import initialize_brains

    seed = 73
    lineup = (
        BotSpec("recording-before", "recording-before", _seed_recording_brain),
        BotSpec("recording-failure", "recording-failure", _seed_recording_failure),
        BotSpec("recording-after", "recording-after", _seed_recording_brain),
    )
    expected_rng = random.Random(seed)
    expected_seeds = [expected_rng.randrange(2**63) for _seat in lineup]
    _RECORDED_BRAIN_SEEDS.clear()

    brains, faults = initialize_brains(
        lineup,
        seed=seed,
        fault_mode=FaultMode.RECORD_AND_PASS,
    )

    assert _RECORDED_BRAIN_SEEDS == expected_seeds
    assert brains[0] is not None
    assert brains[1] is None
    assert brains[2] is not None
    assert len(faults) == 1
    assert faults[0].turn_index == 0
    assert faults[0].seat == 1
    assert faults[0].bot_name == "recording-failure"
    assert faults[0].error_type == "LookupError"
    assert faults[0].message == "seeded factory exploded"


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


def test_illegal_decisions_follow_fault_mode_and_use_phase_fallbacks() -> None:
    lineup = (
        BotSpec("illegal", "test-illegal", _illegal_decision_brain),
        *_random_lineup()[:2],
    )

    recorded = MatchRunner.run(
        lineup,
        player_count=3,
        seed=5,
        fault_mode=FaultMode.RECORD_AND_PASS,
    )
    fallback_decisions = tuple(
        decision
        for _step, decisions in recorded.replay.decisions
        for seat, decision in decisions
        if seat == 0
    )

    assert {decision.action_kind for decision in fallback_decisions} == {
        "selectInfoToReveal",
        "submitBid",
    }
    assert all(decision.value == 0 for decision in fallback_decisions)
    assert len(recorded.faults) == len(fallback_decisions)
    assert all(fault.seat == 0 for fault in recorded.faults)
    assert all(fault.bot_name == "illegal" for fault in recorded.faults)
    assert all(fault.error_type == "InvalidBotDecision" for fault in recorded.faults)

    with pytest.raises(InvalidBotDecision, match="bid exceeds legal maximum"):
        MatchRunner.run(
            lineup,
            player_count=3,
            seed=5,
            fault_mode=FaultMode.RAISE,
        )


def test_record_and_pass_records_one_construction_failure() -> None:
    lineup = (
        BotSpec("broken-factory", "test-broken-factory", _construction_failure),
        *_random_lineup()[:2],
    )

    match = MatchRunner.run(
        lineup,
        player_count=3,
        seed=5,
        fault_mode=FaultMode.RECORD_AND_PASS,
    )

    assert len(match.faults) == 1
    assert match.faults[0].turn_index == 0
    assert match.faults[0].seat == 0
    assert match.faults[0].error_type == "LookupError"
    assert match.faults[0].message == "factory exploded"


def test_raise_mode_propagates_original_construction_exception() -> None:
    lineup = (
        BotSpec("broken-factory", "test-broken-factory", _construction_failure),
        *_random_lineup()[:2],
    )

    with pytest.raises(LookupError, match="factory exploded"):
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
