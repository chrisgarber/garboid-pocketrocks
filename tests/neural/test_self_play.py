from __future__ import annotations

import math

import pytest

pytest.importorskip("torch")

from garboid_pocketrocks.neural.config import training_encoder_config  # noqa: E402
from garboid_pocketrocks.neural.planning import (  # noqa: E402
    SelfPlayEpisodePlan,
    plan_mirror_episodes,
)
from garboid_pocketrocks.neural.self_play import (  # noqa: E402
    PendingPolicyRequest,
    PolicyResponse,
    SelfPlayError,
    SelfPlayGame,
)
from garboid_pocketrocks.training.rewards import RewardConfig  # noqa: E402


def _plan_for(ruleset_name: str, player_count: int) -> SelfPlayEpisodePlan:
    return next(
        plan
        for plan in plan_mirror_episodes(
            root_seed=42,
            update_index=0,
            games_per_cell=1,
            policy_identity="current",
        )
        if plan.ruleset_name == ruleset_name and plan.player_count == player_count
    )


def _started_game(ruleset_name: str, player_count: int) -> SelfPlayGame:
    return SelfPlayGame.start(
        _plan_for(ruleset_name, player_count),
        encoder_config=training_encoder_config(),
        reward_config=RewardConfig(),
    )


def _pass_response(request: PendingPolicyRequest) -> PolicyResponse:
    return PolicyResponse(
        episode_index=request.episode_index,
        seat=request.seat,
        decision_index=request.decision_index,
        action=0,
        old_log_probability=-0.5,
        old_value=0.0,
    )


def test_all_pending_bids_are_requested_before_resolution() -> None:
    game = _started_game("live-A", 3)

    requests = game.pending_requests()

    assert tuple(request.seat for request in requests) == (0, 1, 2)
    assert all(request.policy_identity == "current" for request in requests)
    assert all(request.trainable for request in requests)
    with pytest.raises(SelfPlayError, match="every pending seat"):
        game.apply((_pass_response(requests[0]),))


def test_every_seat_yields_a_terminated_trainable_trajectory() -> None:
    game = _started_game("live-E", 5)

    while not game.terminated:
        requests = game.pending_requests()
        assert requests
        game.apply(tuple(_pass_response(request) for request in requests))

    episode = game.episode()
    assert len(episode.trajectories) == 5
    assert tuple(trajectory.seat for trajectory in episode.trajectories) == (
        0,
        1,
        2,
        3,
        4,
    )
    assert all(
        trajectory.trainable and trajectory.transitions for trajectory in episode.trajectories
    )
    assert all(trajectory.transitions[-1].terminated for trajectory in episode.trajectories)
    assert all(
        not transition.truncated
        for trajectory in episode.trajectories
        for transition in trajectory.transitions
    )
    assert all(
        transition.observation.action_mask[transition.action]
        for trajectory in episode.trajectories
        for transition in trajectory.transitions
    )
    assert all(
        math.isfinite(transition.reward)
        for trajectory in episode.trajectories
        for transition in trajectory.transitions
    )


def test_unresolved_bids_never_enter_another_seats_observation() -> None:
    game = _started_game("live-A", 3)

    before = game.pending_requests()

    assert all(int(request.observation.history_valid.sum()) == 2 for request in before)


def test_pending_requests_are_idempotent_until_responses_arrive() -> None:
    game = _started_game("live-C", 4)

    first = game.pending_requests()
    second = game.pending_requests()

    assert first == second


def test_responses_must_match_request_identity_and_be_finite() -> None:
    game = _started_game("live-D", 3)
    requests = game.pending_requests()
    wrong = PolicyResponse(
        episode_index=requests[0].episode_index,
        seat=requests[0].seat,
        decision_index=requests[0].decision_index + 1,
        action=0,
        old_log_probability=-0.5,
        old_value=0.0,
    )

    with pytest.raises(SelfPlayError, match="identity"):
        game.apply((wrong, *tuple(_pass_response(item) for item in requests[1:])))

    nonfinite = PolicyResponse(
        episode_index=requests[0].episode_index,
        seat=requests[0].seat,
        decision_index=requests[0].decision_index,
        action=0,
        old_log_probability=math.nan,
        old_value=0.0,
    )
    with pytest.raises(SelfPlayError, match="finite"):
        game.apply((nonfinite, *tuple(_pass_response(item) for item in requests[1:])))


def test_episode_is_unavailable_before_termination() -> None:
    game = _started_game("live-B", 3)

    with pytest.raises(SelfPlayError, match="terminated"):
        game.episode()
