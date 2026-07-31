from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from pocketrocks import BotDecision, DecisionContext

from garboid_pocketrocks.adapters.public_history import PublicHistory
from garboid_pocketrocks.bots import BotSpec, RandomBot
from garboid_pocketrocks.knowledge import RulesetKnowledge
from garboid_pocketrocks.simulator.batch_match import run_batch_matches
from garboid_pocketrocks.simulator.monte_carlo import GameJob
from garboid_pocketrocks.simulator.replay import replay_match, save_replay
from garboid_pocketrocks.simulator.runner import FaultMode, MatchResult, MatchRunner


def _random_lineup(player_count: int) -> tuple[BotSpec, ...]:
    return tuple(
        BotSpec(
            name=f"random-{seat}",
            bot_id=f"random-{seat}",
            brain_factory=RandomBot.build_brain,
        )
        for seat in range(player_count)
    )


def _job(
    *,
    game_index: int = 0,
    seed: int = 17,
    player_count: int = 3,
    value_chart: str = "A",
    objectives_enabled: bool = True,
    lineup: tuple[BotSpec, ...] | None = None,
    fault_mode: FaultMode = FaultMode.RAISE,
) -> GameJob:
    return GameJob(
        game_index=game_index,
        root_seed=20260730,
        seed=seed,
        player_count=player_count,
        value_chart=value_chart,
        objectives_enabled=objectives_enabled,
        lineup=lineup if lineup is not None else _random_lineup(player_count),
        fault_mode=fault_mode,
    )


def _run_scalar(job: GameJob) -> MatchResult:
    return MatchRunner.run(
        job.lineup,
        player_count=job.player_count,
        seed=job.seed,
        value_chart=job.value_chart,
        objectives_enabled=job.objectives_enabled,
        fault_mode=job.fault_mode,
    )


@pytest.mark.parametrize("player_count", (3, 4, 5))
@pytest.mark.parametrize("value_chart", ("A", "C", "E"))
@pytest.mark.parametrize("objectives_enabled", (True, False))
def test_batch_match_matches_scalar_replay_bytes(
    tmp_path: Path,
    player_count: int,
    value_chart: str,
    objectives_enabled: bool,
) -> None:
    job = _job(
        seed=104_729,
        player_count=player_count,
        value_chart=value_chart,
        objectives_enabled=objectives_enabled,
    )

    scalar = _run_scalar(job)
    (batch,) = run_batch_matches((job,))
    scalar_path = tmp_path / "scalar.json"
    batch_path = tmp_path / "batch.json"
    save_replay(scalar.replay, scalar_path)
    save_replay(batch.replay, batch_path)

    assert batch.result == scalar.result
    assert batch.turns == scalar.turns
    assert batch.faults == scalar.faults
    assert batch.replay == scalar.replay
    assert scalar_path.read_bytes() == batch_path.read_bytes()
    assert batch.events == ()
    replayed = replay_match(batch.replay)
    assert replayed.turns == batch.turns
    assert replayed.result == batch.result


def test_singleton_and_multirow_batching_return_identical_rows_in_input_order() -> None:
    jobs = tuple(_job(game_index=index, seed=seed) for index, seed in enumerate((17, 31, 91)))

    combined = run_batch_matches(jobs)
    singletons = tuple(run_batch_matches((job,))[0] for job in jobs)

    assert combined == singletons
    assert tuple(match.replay.seed for match in combined) == tuple(job.seed for job in jobs)


_RECORDED_HISTORIES: list[PublicHistory] = []


class _HistoryRecordingBrain:
    def choose_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
    ) -> BotDecision:
        del context, ruleset
        raise AssertionError("history-aware brain used the history-free entry point")

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


def _history_recording_brain(seed: int | None) -> _HistoryRecordingBrain:
    del seed
    return _HistoryRecordingBrain()


def test_history_aware_brains_receive_the_same_scalar_and_batch_history() -> None:
    lineup = (
        BotSpec.for_simulation("history-recording", _history_recording_brain),
        *_random_lineup(3)[1:],
    )
    job = _job(seed=17, value_chart="C", lineup=lineup)

    _RECORDED_HISTORIES.clear()
    _run_scalar(job)
    scalar_histories = tuple(_RECORDED_HISTORIES)
    _RECORDED_HISTORIES.clear()
    run_batch_matches((job,))
    batch_histories = tuple(_RECORDED_HISTORIES)

    assert batch_histories == scalar_histories
    assert batch_histories


class _RaisingBrain:
    def choose_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
    ) -> BotDecision:
        del context, ruleset
        raise RuntimeError("brain exploded")


def _raising_brain(seed: int | None) -> _RaisingBrain:
    del seed
    return _RaisingBrain()


def _construction_failure(seed: int | None) -> _RaisingBrain:
    del seed
    raise LookupError("factory exploded")


@pytest.mark.parametrize(
    "factory",
    (_raising_brain, _construction_failure),
    ids=("runtime", "construction"),
)
def test_record_and_pass_faults_match_scalar(
    factory: Callable[[int | None], _RaisingBrain],
) -> None:
    lineup = (
        BotSpec.for_simulation("broken", factory),
        *_random_lineup(3)[1:],
    )
    job = _job(seed=5, lineup=lineup, fault_mode=FaultMode.RECORD_AND_PASS)

    scalar = _run_scalar(job)
    (batch,) = run_batch_matches((job,))

    assert batch.faults == scalar.faults
    assert batch.result == scalar.result
    assert batch.replay == scalar.replay
    assert batch.faults
    if factory is _construction_failure:
        assert len(batch.faults) == 1
        assert batch.faults[0].turn_index == 0


@pytest.mark.parametrize(
    ("factory", "error_type", "message"),
    (
        (_raising_brain, RuntimeError, "brain exploded"),
        (_construction_failure, LookupError, "factory exploded"),
    ),
)
def test_raise_mode_propagates_original_batch_error(
    factory: Callable[[int | None], _RaisingBrain],
    error_type: type[Exception],
    message: str,
) -> None:
    lineup = (
        BotSpec.for_simulation("broken", factory),
        *_random_lineup(3)[1:],
    )
    job = _job(seed=5, lineup=lineup)

    with pytest.raises(error_type, match=message):
        run_batch_matches((job,))


def test_batch_replay_decision_steps_preserve_phase_shape() -> None:
    job = _job(seed=31)

    (match,) = run_batch_matches((job,))

    assert tuple(step for step, _decisions in match.replay.decisions) == tuple(
        range(len(match.replay.decisions))
    )
    bid_steps = tuple(
        decisions
        for _step, decisions in match.replay.decisions
        if len(decisions) == job.player_count
    )
    reveal_steps = tuple(
        decisions for _step, decisions in match.replay.decisions if len(decisions) == 1
    )
    automatic_reveals = tuple(
        turn.reveal for turn in match.turns if turn.reveal is not None and turn.reveal.auto
    )
    choice_reveals = tuple(
        turn.reveal for turn in match.turns if turn.reveal is not None and not turn.reveal.auto
    )

    assert all(
        tuple(seat for seat, _decision in decisions) == tuple(range(job.player_count))
        for decisions in bid_steps
    )
    assert tuple(decisions[0][0] for decisions in reveal_steps) == tuple(
        reveal.seat for reveal in choice_reveals
    )
    assert all(decisions[0][1].action_kind == "selectInfoToReveal" for decisions in reveal_steps)
    assert automatic_reveals
    assert len(reveal_steps) == len(choice_reveals)


def test_batch_job_validation_is_unchanged() -> None:
    assert run_batch_matches(()) == ()

    with pytest.raises(ValueError, match="share one player count"):
        run_batch_matches((_job(player_count=3), _job(player_count=4)))

    invalid_lineup = _random_lineup(3)[:2]
    with pytest.raises(ValueError, match="lineup length"):
        run_batch_matches((_job(player_count=3, lineup=invalid_lineup),))
