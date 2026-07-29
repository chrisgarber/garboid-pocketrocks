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

from garboid_pocketrocks.adapters.public_history import PublicHistory
from garboid_pocketrocks.adapters.simulator_history import (
    SimulatorPublicHistoryAdapter,
)
from garboid_pocketrocks.bots.base import BotBrain, BotSpec
from garboid_pocketrocks.rules import Ruleset, RulesetKnowledge
from garboid_pocketrocks.simulator.engine import EngineTransition, GameEngine
from garboid_pocketrocks.simulator.model import Seat
from garboid_pocketrocks.simulator.sampling import RulesetSampler
from garboid_pocketrocks.training.actions import ActionCodec
from garboid_pocketrocks.training.bounds import EnvironmentBounds
from garboid_pocketrocks.training.observations import ObservationEncoder
from garboid_pocketrocks.training.rewards import RewardBreakdown, RewardConfig, RewardTracker


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
    """Gymnasium environment that controls one PocketRocks seat."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        opponent_specs: Sequence[BotSpec],
        ruleset_sampler: RulesetSampler,
        player_count: int,
        bounds: EnvironmentBounds,
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
        self.opponent_specs = tuple(opponent_specs)
        self.ruleset_sampler = ruleset_sampler
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
        self._validate_sampler_support()

        self.learner_seat = learner_seat if learner_seat is not None else 0
        self.transition: EngineTransition | None = None
        self._knowledge: RulesetKnowledge | None = None
        self._brains: dict[Seat, BotBrain] = {}
        self._learner_context: DecisionContext | None = None
        self._history_adapter: SimulatorPublicHistoryAdapter | None = None
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
        ruleset = self.ruleset_sampler.sample(root_seed=self._root_seed, game_index=0)
        self.transition = GameEngine.start(
            ruleset,
            player_count=self.player_count,
            seed=self._root_seed,
        )
        self._knowledge = ruleset.knowledge(self.player_count)
        self._reward_tracker.reset(self.transition.state)
        opponent_seed = self._root_seed if opponent_seed_value is None else opponent_seed_value
        self._brains = self._make_opponent_brains(random.Random(opponent_seed))
        self._history_adapter = SimulatorPublicHistoryAdapter.from_initial_transition(
            self.transition
        )
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
        if self._history_adapter is None:
            raise RuntimeError("environment must be reset before observing")
        return self._history_adapter.history

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self.transition is None or self._learner_context is None:
            raise RuntimeError("environment must be reset before stepping")
        if self.transition.terminated:
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
        assert self.transition is not None
        while not self.transition.terminated:
            assert self.transition.pending is not None
            decisions = dict(learner_decisions)
            for seat, context in self.transition.pending.contexts:
                if seat == self.learner_seat:
                    continue
                decisions[seat] = self._brains[seat].choose_decision(
                    context,
                    self._knowledge_for_game(),
                )
            self.transition = GameEngine.step(self.transition.state, decisions)
            self._history_for_game().append(self.transition.events)
            accumulated = _add_breakdowns(
                accumulated,
                self._reward_tracker.update(self.transition)[self.learner_seat],
            )
            learner_decisions = {}
            if self.transition.terminated:
                self._learner_context = self._learner_context
                self._step_breakdown = accumulated
                return
            assert self.transition.pending is not None
            contexts = self.transition.pending.contexts_by_seat
            if self.learner_seat in contexts:
                self._learner_context = contexts[self.learner_seat]
                self._step_breakdown = accumulated
                return

    def _return_step(self) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        assert self.transition is not None
        breakdown = self._step_breakdown
        return (
            self._observation(),
            breakdown.total,
            self.transition.terminated,
            False,
            {"reward_breakdown": asdict(breakdown)},
        )

    def _observation(self) -> dict[str, Any]:
        if self._learner_context is None or self._knowledge is None:
            raise RuntimeError("environment must be reset before observing")
        return self.observation_encoder.encode(self._learner_context, self._knowledge)

    def _context_for_learner(self) -> DecisionContext:
        assert self.transition is not None
        assert self.transition.pending is not None
        return self.transition.pending.contexts_by_seat[self.learner_seat]

    def _knowledge_for_game(self) -> RulesetKnowledge:
        assert self._knowledge is not None
        return self._knowledge

    def _history_for_game(self) -> SimulatorPublicHistoryAdapter:
        assert self._history_adapter is not None
        return self._history_adapter

    def _make_opponent_brains(self, rng: random.Random) -> dict[Seat, BotBrain]:
        brains: dict[Seat, BotBrain] = {}
        specs = iter(self.opponent_specs)
        for seat in range(self.player_count):
            if seat != self.learner_seat:
                brains[seat] = next(specs).make_brain(seed=rng.randrange(2**63))
        return brains

    def _validate_sampler_support(self) -> None:
        support = self.ruleset_sampler.support()
        if not support:
            raise ValueError("ruleset sampler support must not be empty")
        for ruleset in support:
            self._validate_ruleset(ruleset)

    def _validate_ruleset(self, ruleset: Ruleset) -> None:
        ruleset.setup_for(self.player_count)
        transition = GameEngine.start(ruleset, player_count=self.player_count, seed=0)
        knowledge = ruleset.knowledge(self.player_count)
        assert transition.pending is not None
        for _, context in transition.pending.contexts:
            self.observation_encoder.encode(context, knowledge)


def _add_breakdowns(left: RewardBreakdown, right: RewardBreakdown) -> RewardBreakdown:
    return RewardBreakdown(
        accounting=left.accounting + right.accounting,
        terminal_resource=left.terminal_resource + right.terminal_resource,
        placement=left.placement + right.placement,
        shaping=left.shaping + right.shaping,
        penalty=left.penalty + right.penalty,
    )
