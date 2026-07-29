from __future__ import annotations

import pytest
from pocketrocks import BotDecision, DecisionContext

from garboid_pocketrocks.bots import BotSpec, RandomBot
from garboid_pocketrocks.rules import LIVE_RULESET, RulesetKnowledge
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


def _random_lineup() -> tuple[BotSpec, ...]:
    return tuple(BotSpec.from_bot_class(RandomBot) for _ in range(3))


def test_match_runner_is_reproducible_and_uses_fresh_brains() -> None:
    lineup = _random_lineup()

    left = MatchRunner.run(
        lineup,
        ruleset=LIVE_RULESET,
        player_count=3,
        seed=91,
    )
    right = MatchRunner.run(
        lineup,
        ruleset=LIVE_RULESET,
        player_count=3,
        seed=91,
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
        ruleset=LIVE_RULESET,
        player_count=3,
        seed=5,
        fault_mode=FaultMode.RECORD_AND_PASS,
    )

    assert match.faults
    assert match.faults[0].seat == 0
    assert match.faults[0].bot_name == "raising"
    assert match.faults[0].error_type == "RuntimeError"
    assert any(event.kind.value == "bot_fault" for event in match.events)


def test_raise_mode_propagates_original_brain_exception() -> None:
    lineup = (
        BotSpec("raising", "test-raising", _raising_brain),
        *_random_lineup()[:2],
    )

    with pytest.raises(RuntimeError, match="brain exploded"):
        MatchRunner.run(
            lineup,
            ruleset=LIVE_RULESET,
            player_count=3,
            seed=5,
            fault_mode=FaultMode.RAISE,
        )
