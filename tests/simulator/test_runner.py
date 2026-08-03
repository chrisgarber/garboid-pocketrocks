from __future__ import annotations

import random
from dataclasses import replace

import pytest
from pocketrocks import BotDecision, DecisionContext
from pocketrocks.exceptions import InvalidBotDecision

from garboid_pocketrocks.adapters.public_history import (
    PublicGameSetup,
    PublicHistory,
    public_history_from_sdk_events,
)
from garboid_pocketrocks.bots import BotSpec, RandomBot
from garboid_pocketrocks.diagnostics.trace import (
    PublicDecisionOutcome,
    RecordedAction,
)
from garboid_pocketrocks.knowledge import RulesetKnowledge
from garboid_pocketrocks.simulator.runner import FaultMode, MatchRunner


class RaisingBrain:
    def choose_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
        history: PublicHistory,
    ) -> BotDecision:
        del context, ruleset, history
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
        history: PublicHistory,
    ) -> BotDecision:
        del ruleset, history
        if context.decision_kind == "selectInfoToReveal":
            return BotDecision.select_info_to_reveal(context.revealable_count)
        assert context.legal_max_amount is not None
        return BotDecision.submit_bid(context.legal_max_amount + 1)


def _illegal_decision_brain(seed: int | None) -> IllegalDecisionBrain:
    del seed
    return IllegalDecisionBrain()


_FAILING_EXPLANATION_CALLS = 0


class AlwaysPassBrain:
    def choose_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
        history: PublicHistory,
    ) -> BotDecision:
        del ruleset, history
        if context.decision_kind == "selectInfoToReveal":
            return BotDecision.select_info_to_reveal(0)
        return BotDecision.submit_bid(0)


class FailingExplanationBrain(AlwaysPassBrain):
    def choose_explained_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
        history: PublicHistory,
    ) -> object:
        del context, ruleset, history
        global _FAILING_EXPLANATION_CALLS
        _FAILING_EXPLANATION_CALLS += 1
        raise RuntimeError("explanation path must remain opt-in")


def _always_pass_brain(seed: int | None) -> AlwaysPassBrain:
    del seed
    return AlwaysPassBrain()


def _failing_explanation_brain(seed: int | None) -> FailingExplanationBrain:
    del seed
    return FailingExplanationBrain()


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


def test_decision_traces_are_opt_in_finalized_and_behavior_preserving() -> None:
    lineup = _random_lineup()
    omitted = MatchRunner.run(
        lineup,
        player_count=3,
        seed=31,
        value_chart="C",
        game_index=7,
    )
    captured = MatchRunner.run(
        lineup,
        player_count=3,
        seed=31,
        value_chart="C",
        game_index=7,
        capture_decision_traces=True,
    )

    assert omitted.decision_traces == ()
    assert replace(captured, decision_traces=()) == omitted
    replayed_decisions = {
        (step_index, seat): RecordedAction.from_decision(decision)
        for step_index, decisions in captured.replay.decisions
        for seat, decision in decisions
    }
    assert len(captured.decision_traces) == len(replayed_decisions)
    assert tuple(
        (trace.game_index, trace.step_index, trace.seat) for trace in captured.decision_traces
    ) == tuple(
        sorted(
            (trace.game_index, trace.step_index, trace.seat) for trace in captured.decision_traces
        )
    )
    scores_by_seat = {score.seat: score for score in captured.result.scores}
    for trace in captured.decision_traces:
        score = scores_by_seat[trace.seat]
        assert not hasattr(trace, "root_seed")
        assert not hasattr(trace, "engine_seed")
        assert trace.chart == "C"
        assert trace.selected_action == replayed_decisions[(trace.step_index, trace.seat)]
        assert trace.outcome == PublicDecisionOutcome(
            rank=score.rank,
            final_money=score.final_money,
            first_place_tied=(
                score.rank == 1 and sum(other.rank == 1 for other in captured.result.scores) > 1
            ),
        )


def test_captured_fault_fallbacks_keep_the_ordinary_decision_and_source() -> None:
    lineup = (
        BotSpec("raising", "test-raising", _raising_brain),
        *_random_lineup()[:2],
    )

    match = MatchRunner.run(
        lineup,
        player_count=3,
        seed=5,
        fault_mode=FaultMode.RECORD_AND_PASS,
        capture_decision_traces=True,
    )

    fallback_traces = tuple(
        trace for trace in match.decision_traces if trace.bot_id == "test-raising"
    )
    assert fallback_traces
    assert all(trace.selection_source == "fault_fallback" for trace in fallback_traces)
    assert all(trace.explanation is None for trace in fallback_traces)


def test_tracing_off_never_invokes_the_explanation_protocol() -> None:
    global _FAILING_EXPLANATION_CALLS
    _FAILING_EXPLANATION_CALLS = 0
    ordinary_spec = BotSpec("opt-in", "opt-in", _always_pass_brain)
    explanation_spec = BotSpec("opt-in", "opt-in", _failing_explanation_brain)
    expected = MatchRunner.run(
        (ordinary_spec, *_random_lineup()[1:]),
        player_count=3,
        seed=31,
        value_chart="C",
        fault_mode=FaultMode.RAISE,
    )
    actual = MatchRunner.run(
        (explanation_spec, *_random_lineup()[1:]),
        player_count=3,
        seed=31,
        value_chart="C",
        fault_mode=FaultMode.RAISE,
    )

    assert _FAILING_EXPLANATION_CALLS == 0
    assert actual.result == expected.result
    assert actual.events == expected.events
    assert actual.turns == expected.turns
    assert actual.faults == expected.faults
    assert actual.replay == expected.replay
