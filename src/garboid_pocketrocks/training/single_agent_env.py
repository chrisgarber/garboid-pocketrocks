from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import asdict
from enum import StrEnum
from typing import Any, cast

import gymnasium as gym
import numpy as np
from gymnasium.spaces.space import MaskNDArray
from pocketrocks import BotDecision, DecisionContext

from garboid_pocketrocks.adapters.public_history import (
    PublicHistory,
    public_history_from_sdk_events,
)
from garboid_pocketrocks.bots.base import BotBrain, BotSpec
from garboid_pocketrocks.knowledge import RulesetKnowledge, canonical_knowledge
from garboid_pocketrocks.simulator.seeding import derive_seed
from garboid_pocketrocks.simulator.session import SdkGameSession, SessionTransition
from garboid_pocketrocks.training.actions import ActionCodec
from garboid_pocketrocks.training.bounds import EnvironmentBounds
from garboid_pocketrocks.training.observations import ObservationEncoder
from garboid_pocketrocks.training.rewards import RewardBreakdown, RewardConfig, RewardTracker

Seat = int


class InvalidActionMode(StrEnum):
    RAISE = "raise"
    PENALIZE_AND_PASS = "penalize_and_pass"


class _MaskedDiscrete(gym.spaces.Discrete[np.int32]):
    """A fixed action space whose unmasked samples use the universal pass action."""

    def sample(
        self,
        mask: MaskNDArray | None = None,
        probability: MaskNDArray | None = None,
    ) -> np.int32:
        if mask is None and probability is None:
            return np.int32(0)
        return super().sample(mask=mask, probability=probability)


class PocketRocksEnv(gym.Env[dict[str, Any], int]):
    """Gymnasium environment that controls one SDK-simulated PocketRocks seat."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        opponent_specs: Sequence[BotSpec],
        value_charts: tuple[str, ...],
        player_count: int,
        bounds: EnvironmentBounds,
        objectives_enabled: tuple[bool, ...] = (True,),
        reward_config: RewardConfig = RewardConfig(),  # noqa: B008
        learner_seat: int | None = None,
        invalid_action_mode: InvalidActionMode = InvalidActionMode.RAISE,
    ) -> None:
        if not 3 <= player_count <= 5:
            raise ValueError("player_count must be between 3 and 5")
        if len(opponent_specs) != player_count - 1:
            raise ValueError("opponent_specs must contain one entry for every non-learner seat")
        if learner_seat is not None and not 0 <= learner_seat < player_count:
            raise ValueError("learner_seat is outside player count")
        if not value_charts:
            raise ValueError("value_charts must be nonempty")
        if not objectives_enabled:
            raise ValueError("objectives_enabled must be nonempty")
        self.opponent_specs = tuple(opponent_specs)
        self.value_charts = tuple(chart.upper() for chart in value_charts)
        self.objectives_enabled = objectives_enabled
        self.player_count = player_count
        self.bounds = bounds
        self.reward_config = reward_config
        self.fixed_learner_seat = learner_seat
        self.invalid_action_mode = invalid_action_mode
        self.action_codec = ActionCodec(bounds)
        self.observation_encoder = ObservationEncoder(bounds)
        self.action_space = cast(
            gym.spaces.Space[int],
            _MaskedDiscrete(self.action_codec.size, dtype=np.int32),
        )
        self.observation_space = self.observation_encoder.observation_space
        self._validate_variants()

        self.learner_seat = learner_seat if learner_seat is not None else 0
        self.session: SdkGameSession | None = None
        self.transition: SessionTransition | None = None
        self._knowledge: RulesetKnowledge | None = None
        self._brains: dict[Seat, BotBrain] = {}
        self._learner_context: DecisionContext | None = None
        self._reward_tracker = RewardTracker(reward_config)
        self._root_seed = 0
        self._step_breakdown = RewardBreakdown()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        reset_options = {} if options is None else options
        unknown_options = set(reset_options) - {"opponent_seed"}
        if unknown_options:
            raise ValueError(f"unknown reset options: {sorted(unknown_options)!r}")
        if "opponent_seed" in reset_options:
            opponent_seed_value = reset_options["opponent_seed"]
            if not isinstance(opponent_seed_value, int) or isinstance(opponent_seed_value, bool):
                raise ValueError("opponent_seed must be an integer")
        else:
            opponent_seed_value = None
        super().reset(seed=seed)
        if seed is not None:
            self._root_seed = seed
        else:
            self._root_seed = int(self.np_random.integers(0, 2**63 - 1))
        seat_rng = random.Random(self._root_seed)
        self.learner_seat = (
            self.fixed_learner_seat
            if self.fixed_learner_seat is not None
            else seat_rng.randrange(self.player_count)
        )
        value_chart = random.Random(derive_seed(self._root_seed, "value_chart", 0)).choice(
            self.value_charts
        )
        objectives_enabled = random.Random(
            derive_seed(self._root_seed, "objectives_enabled", 0)
        ).choice(self.objectives_enabled)
        self.session = SdkGameSession.start(
            player_count=self.player_count,
            seed=self._root_seed,
            value_chart=value_chart,
            objectives_enabled=objectives_enabled,
        )
        self._knowledge = canonical_knowledge(
            self.player_count,
            value_chart=value_chart,
            objectives_enabled=objectives_enabled,
        )
        initial = self.session.snapshot
        self.transition = SessionTransition(
            before=initial,
            snapshot=initial,
            pending=self.session.pending,
            result=None,
            decisions=(),
            events=self.session.events,
            turn_records=(),
        )
        self._reward_tracker.reset(initial)
        opponent_seed = self._root_seed if opponent_seed_value is None else opponent_seed_value
        self._brains = self._make_opponent_brains(random.Random(opponent_seed))
        self._learner_context = self._context_for_learner()
        return self._observation(), {"learner_seat": self.learner_seat}

    @property
    def learner_context(self) -> DecisionContext:
        if self._learner_context is None:
            raise RuntimeError("environment must be reset before observing")
        return self._learner_context

    @property
    def ruleset_knowledge(self) -> RulesetKnowledge:
        if self._knowledge is None:
            raise RuntimeError("environment must be reset before observing")
        return self._knowledge

    @property
    def public_history(self) -> PublicHistory:
        if self.session is None:
            raise RuntimeError("environment must be reset before observing")
        return public_history_from_sdk_events(self.session.events)

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self.session is None or self.transition is None or self._learner_context is None:
            raise RuntimeError("environment must be reset before stepping")
        if self.session.terminated:
            raise RuntimeError("cannot step a terminated environment")
        breakdown = RewardBreakdown()
        decision = self._decision_for_learner(action)
        if decision is None:
            breakdown = self._reward_tracker.invalid_action(self.learner_seat)
            decision = BotDecision.pass_turn()
        self._resolve_batch({self.learner_seat: decision}, breakdown)
        return self._return_step()

    def render(self) -> None:
        return None

    def _decision_for_learner(self, action: int) -> BotDecision | None:
        if self._learner_context is None:
            raise RuntimeError("environment must be reset before stepping")
        if not self.action_space.contains(action):
            if self.invalid_action_mode is InvalidActionMode.RAISE:
                raise ValueError("action is outside the fixed action space")
            return None
        if not self.action_codec.mask(self._learner_context)[int(action)]:
            if self.invalid_action_mode is InvalidActionMode.RAISE:
                raise ValueError("action is not legal in the current context")
            return None
        return self.action_codec.decode(int(action))

    def _resolve_batch(
        self,
        learner_decisions: dict[Seat, BotDecision],
        accumulated: RewardBreakdown,
    ) -> None:
        assert self.session is not None
        while not self.session.terminated:
            decisions = dict(learner_decisions)
            for seat, context in self.session.pending.contexts:
                if seat == self.learner_seat:
                    continue
                decisions[seat] = self._brains[seat].choose_decision(
                    context,
                    self._knowledge_for_game(),
                )
            self.transition = self.session.step(decisions)
            accumulated = _add_breakdowns(
                accumulated,
                self._reward_tracker.update(self.transition)[self.learner_seat],
            )
            learner_decisions = {}
            if self.session.terminated:
                self._step_breakdown = accumulated
                return
            contexts = self.session.pending.contexts_by_seat
            if self.learner_seat in contexts:
                self._learner_context = contexts[self.learner_seat]
                self._step_breakdown = accumulated
                return

    def _return_step(self) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        assert self.session is not None
        breakdown = self._step_breakdown
        return (
            self._observation(),
            breakdown.total,
            self.session.terminated,
            False,
            {"reward_breakdown": asdict(breakdown)},
        )

    def _observation(self) -> dict[str, Any]:
        if self._learner_context is None or self._knowledge is None:
            raise RuntimeError("environment must be reset before observing")
        return self.observation_encoder.encode(self._learner_context, self._knowledge)

    def _context_for_learner(self) -> DecisionContext:
        assert self.session is not None
        return self.session.pending.contexts_by_seat[self.learner_seat]

    def _knowledge_for_game(self) -> RulesetKnowledge:
        assert self._knowledge is not None
        return self._knowledge

    def _make_opponent_brains(self, rng: random.Random) -> dict[Seat, BotBrain]:
        brains: dict[Seat, BotBrain] = {}
        specs = iter(self.opponent_specs)
        for seat in range(self.player_count):
            if seat != self.learner_seat:
                brains[seat] = next(specs).make_brain(seed=rng.randrange(2**63))
        return brains

    def _validate_variants(self) -> None:
        for value_chart in self.value_charts:
            for objectives_enabled in self.objectives_enabled:
                session = SdkGameSession.start(
                    player_count=self.player_count,
                    seed=0,
                    value_chart=value_chart,
                    objectives_enabled=objectives_enabled,
                )
                knowledge = canonical_knowledge(
                    self.player_count,
                    value_chart=value_chart,
                    objectives_enabled=objectives_enabled,
                )
                for _seat, context in session.pending.contexts:
                    self.observation_encoder.encode(context, knowledge)


def _add_breakdowns(left: RewardBreakdown, right: RewardBreakdown) -> RewardBreakdown:
    return RewardBreakdown(
        accounting=left.accounting + right.accounting,
        terminal_resource=left.terminal_resource + right.terminal_resource,
        placement=left.placement + right.placement,
        shaping=left.shaping + right.shaping,
        penalty=left.penalty + right.penalty,
    )
