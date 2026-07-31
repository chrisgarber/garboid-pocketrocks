"""One deterministic all-seat neural self-play game."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from pocketrocks import BotDecision, DecisionContext
from pocketrocks.exceptions import InvalidBotDecision

from garboid_pocketrocks.adapters.public_history import (
    public_history_from_sdk_events,
)
from garboid_pocketrocks.knowledge import (
    RulesetKnowledge,
    canonical_knowledge,
    value_chart_from_ruleset_name,
)
from garboid_pocketrocks.neural.config import NeuralEncoderConfig
from garboid_pocketrocks.neural.encoding import (
    NeuralObservation,
    NeuralObservationEncoder,
)
from garboid_pocketrocks.neural.planning import (
    SelfPlayEpisodePlan,
    decision_seed,
)
from garboid_pocketrocks.neural.rollout import (
    MultiSeatEpisode,
    RolloutMetadata,
    RolloutTransition,
    SeatTrajectory,
    _immutable_observation,
)
from garboid_pocketrocks.simulator.session import SdkGameSession
from garboid_pocketrocks.training.actions import ActionCodec
from garboid_pocketrocks.training.bounds import EnvironmentBounds
from garboid_pocketrocks.training.rewards import (
    RewardBreakdown,
    RewardConfig,
    RewardTracker,
)


class SelfPlayError(ValueError):
    """Raised when a self-play game cannot preserve its training contract."""


@dataclass(frozen=True, slots=True)
class PendingPolicyRequest:
    """One immutable inference request for a pending game seat."""

    episode_index: int
    seat: int
    decision_index: int
    policy_identity: str
    trainable: bool
    sampling_seed: int
    observation: NeuralObservation
    context: DecisionContext
    ruleset: RulesetKnowledge


@dataclass(frozen=True, slots=True)
class PolicyResponse:
    """The old-policy quantities selected for one pending request."""

    episode_index: int
    seat: int
    decision_index: int
    action: int
    old_log_probability: float
    old_value: float

    def __post_init__(self) -> None:
        for name in ("episode_index", "seat", "decision_index", "action"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise SelfPlayError(f"{name} must be an integer")


@dataclass(frozen=True, slots=True)
class _OpenTransition:
    request: PendingPolicyRequest
    context: DecisionContext
    response: PolicyResponse


class SelfPlayGame:
    """Advance one engine while retaining independent trajectories for all seats."""

    def __init__(
        self,
        plan: SelfPlayEpisodePlan,
        *,
        encoder_config: NeuralEncoderConfig,
        reward_config: RewardConfig,
        session: SdkGameSession,
        reward_tracker: RewardTracker,
    ) -> None:
        del reward_config
        self.plan = plan
        self.encoder_config = encoder_config
        self.bounds = EnvironmentBounds(
            max_bid=encoder_config.max_bid,
            max_hand_size=encoder_config.max_hand_size,
        )
        self.codec = ActionCodec(self.bounds)
        self.encoder = NeuralObservationEncoder(encoder_config, self.bounds)
        self.knowledge = canonical_knowledge(
            plan.player_count,
            value_chart=value_chart_from_ruleset_name(plan.ruleset_name),
        )
        self.session = session
        self.reward_tracker = reward_tracker
        self._pending_contexts: dict[int, DecisionContext] = {}
        self._pending_requests: tuple[PendingPolicyRequest, ...] = ()
        self._open_by_seat: dict[int, _OpenTransition] = {}
        self._pending_rewards = {seat: RewardBreakdown() for seat in range(plan.player_count)}
        self._decision_counts = {seat: 0 for seat in range(plan.player_count)}
        self._completed: dict[int, list[RolloutTransition]] = {
            seat: [] for seat in range(plan.player_count)
        }
        self._prepare_pending_requests()

    @classmethod
    def start(
        cls,
        plan: SelfPlayEpisodePlan,
        *,
        encoder_config: NeuralEncoderConfig,
        reward_config: RewardConfig,
    ) -> SelfPlayGame:
        """Start one supported game and prepare its first simultaneous requests."""

        if plan.ruleset_name not in encoder_config.supported_ruleset_names:
            raise SelfPlayError("episode ruleset is outside the encoder support")
        if plan.player_count not in encoder_config.supported_player_counts:
            raise SelfPlayError("episode player count is outside encoder support")
        chart = value_chart_from_ruleset_name(plan.ruleset_name)
        session = SdkGameSession.start(
            player_count=plan.player_count,
            seed=plan.engine_seed,
            value_chart=chart,
        )
        reward_tracker = RewardTracker(reward_config)
        reward_tracker.reset(session.snapshot)
        return cls(
            plan,
            encoder_config=encoder_config,
            reward_config=reward_config,
            session=session,
            reward_tracker=reward_tracker,
        )

    @property
    def terminated(self) -> bool:
        return self.session.terminated

    def pending_requests(self) -> tuple[PendingPolicyRequest, ...]:
        """Return the cached complete request batch for the current engine phase."""

        return self._pending_requests

    def apply(self, responses: Sequence[PolicyResponse]) -> None:
        """Apply exactly one response for every currently pending seat."""

        if self.terminated:
            raise SelfPlayError("cannot apply responses to a terminated game")
        supplied = tuple(responses)
        expected_by_seat = {request.seat: request for request in self._pending_requests}
        response_by_seat = {response.seat: response for response in supplied}
        if len(response_by_seat) != len(supplied) or set(response_by_seat) != set(expected_by_seat):
            raise SelfPlayError("responses must cover every pending seat exactly once")

        decisions: dict[int, BotDecision] = {}
        for seat in sorted(expected_by_seat):
            request = expected_by_seat[seat]
            response = response_by_seat[seat]
            if (
                response.episode_index,
                response.seat,
                response.decision_index,
            ) != (
                request.episode_index,
                request.seat,
                request.decision_index,
            ):
                raise SelfPlayError("policy response identity does not match request")
            if not math.isfinite(response.old_log_probability) or not math.isfinite(
                response.old_value
            ):
                raise SelfPlayError("policy response values must be finite")
            if not 0 <= response.action < self.codec.size:
                raise SelfPlayError("policy response action is outside action space")
            if not bool(request.observation.action_mask[response.action]):
                raise SelfPlayError("policy response action is illegal under stored mask")
            context = self._pending_contexts[seat]
            decision = self.codec.decode(response.action)
            try:
                context.validate(decision)
            except InvalidBotDecision as error:
                raise SelfPlayError("policy response failed SDK validation") from error
            decisions[seat] = decision
            if seat in self._open_by_seat:
                raise SelfPlayError("seat already has an open transition")
            self._open_by_seat[seat] = _OpenTransition(
                request=request,
                context=context,
                response=response,
            )
            self._decision_counts[seat] += 1

        transition = self.session.step(decisions)
        rewards = self.reward_tracker.update(transition)
        for seat, reward in rewards.items():
            self._pending_rewards[seat] = _add_breakdowns(
                self._pending_rewards[seat],
                reward,
            )

        self._pending_contexts = {}
        self._pending_requests = ()
        if self.terminated:
            for seat in range(self.plan.player_count):
                self._finalize_open(seat, terminated=True)
            return
        self._prepare_pending_requests()

    def episode(self) -> MultiSeatEpisode:
        """Return the complete immutable game after termination."""

        if not self.terminated or self.session.result is None:
            raise SelfPlayError("episode is available only after the game terminated")
        trajectories = tuple(
            SeatTrajectory(
                seat=seat,
                policy_identity=self.plan.seat_policies[seat].identity,
                trainable=self.plan.seat_policies[seat].trainable,
                transitions=tuple(self._completed[seat]),
            )
            for seat in range(self.plan.player_count)
        )
        if any(not trajectory.transitions for trajectory in trajectories):
            raise SelfPlayError("terminated game omitted a seat trajectory")
        return MultiSeatEpisode(
            plan=self.plan,
            trajectories=trajectories,
            result=self.session.result,
        )

    def _prepare_pending_requests(self) -> None:
        if self.session.terminated:
            raise SelfPlayError("active game has no pending decision batch")
        contexts = dict(self.session.pending.contexts)
        history = public_history_from_sdk_events(self.session.events)
        requests: list[PendingPolicyRequest] = []
        for seat in sorted(contexts):
            if seat in self._open_by_seat:
                self._finalize_open(seat, terminated=False)
            context = contexts[seat]
            observation = self.encoder.encode(
                context,
                self.knowledge,
                history,
            )
            policy = self.plan.seat_policies[seat]
            decision_index = self._decision_counts[seat]
            requests.append(
                PendingPolicyRequest(
                    episode_index=self.plan.episode_index,
                    seat=seat,
                    decision_index=decision_index,
                    policy_identity=policy.identity,
                    trainable=policy.trainable,
                    sampling_seed=decision_seed(
                        self.plan,
                        seat,
                        decision_index,
                    ),
                    observation=_immutable_observation(observation),
                    context=context,
                    ruleset=self.knowledge,
                )
            )
        self._pending_contexts = contexts
        self._pending_requests = tuple(requests)

    def _finalize_open(self, seat: int, *, terminated: bool) -> None:
        try:
            opened = self._open_by_seat.pop(seat)
        except KeyError as error:
            raise SelfPlayError("seat has no open transition to finalize") from error
        breakdown = self._pending_rewards[seat]
        self._pending_rewards[seat] = RewardBreakdown()
        opponent_names = tuple(
            assignment.identity
            for opponent_seat, assignment in enumerate(self.plan.seat_policies)
            if opponent_seat != seat
        )
        self._completed[seat].append(
            RolloutTransition(
                observation=opened.request.observation,
                context=opened.context,
                action=opened.response.action,
                old_log_probability=opened.response.old_log_probability,
                old_value=opened.response.old_value,
                reward=breakdown.total,
                reward_breakdown=breakdown,
                terminated=terminated,
                truncated=False,
                bid_logits=(),
                reveal_logits=(),
                masked_logits=(),
                illegal_probability=0.0,
                metadata=RolloutMetadata(
                    ruleset_name=self.plan.ruleset_name,
                    player_count=self.plan.player_count,
                    learner_seat=seat,
                    opponent_names=opponent_names,
                    environment_seed=self.plan.engine_seed,
                    opponent_seed=self.plan.engine_seed,
                    policy_seed=self.plan.seat_sampling_seeds[seat],
                ),
            )
        )


def _add_breakdowns(
    left: RewardBreakdown,
    right: RewardBreakdown,
) -> RewardBreakdown:
    return RewardBreakdown(
        accounting=left.accounting + right.accounting,
        terminal_resource=left.terminal_resource + right.terminal_resource,
        placement=left.placement + right.placement,
        shaping=left.shaping + right.shaping,
        penalty=left.penalty + right.penalty,
    )
