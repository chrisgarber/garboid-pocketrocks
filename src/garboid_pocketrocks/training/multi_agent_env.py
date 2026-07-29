from __future__ import annotations

from dataclasses import asdict
from typing import Any

import gymnasium as gym
import numpy as np
from pettingzoo import AECEnv  # type: ignore[import-untyped]
from pettingzoo.utils import AgentSelector  # type: ignore[import-untyped]
from pocketrocks import BotDecision, DecisionContext

from garboid_pocketrocks.rules import Ruleset, RulesetKnowledge
from garboid_pocketrocks.simulator.engine import EngineTransition, GameEngine
from garboid_pocketrocks.simulator.model import Seat
from garboid_pocketrocks.simulator.sampling import RulesetSampler
from garboid_pocketrocks.training.actions import ActionCodec
from garboid_pocketrocks.training.bounds import EnvironmentBounds
from garboid_pocketrocks.training.observations import ObservationEncoder
from garboid_pocketrocks.training.rewards import RewardConfig, RewardTracker


class PocketRocksAECEnv(AECEnv[str, dict[str, Any], int]):  # type: ignore[misc]
    """AEC adapter that preserves PocketRocks' sealed-bid decision batches."""

    metadata = {"name": "pocketrocks_aec_v0", "render_modes": []}

    def __init__(
        self,
        *,
        ruleset_sampler: RulesetSampler,
        player_count: int,
        bounds: EnvironmentBounds,
        reward_config: RewardConfig = RewardConfig(),  # noqa: B008
    ) -> None:
        super().__init__()
        if not 3 <= player_count <= 5:
            raise ValueError("player_count must be between 3 and 5")
        self.ruleset_sampler = ruleset_sampler
        self.player_count = player_count
        self.bounds = bounds
        self.action_codec = ActionCodec(bounds)
        self.observation_encoder = ObservationEncoder(bounds)
        self.reward_tracker = RewardTracker(reward_config)
        self._validate_sampler_support()

        self.possible_agents = [self._agent_name(seat) for seat in range(player_count)]
        self.agents: list[str] = []
        self.action_spaces = {
            agent: self.action_codec.action_space for agent in self.possible_agents
        }
        self.observation_spaces = {
            agent: self.observation_encoder.observation_space for agent in self.possible_agents
        }
        self._agent_selector = AgentSelector(self.possible_agents)
        self._contexts: dict[Seat, DecisionContext] = {}
        self._pending_decisions: dict[Seat, BotDecision] = {}
        self._knowledge: RulesetKnowledge | None = None
        self.transition: EngineTransition | None = None

    def observation_space(self, agent: str) -> gym.spaces.Dict:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> gym.spaces.Discrete[np.int32]:
        return self.action_spaces[agent]

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        del options
        root_seed = 0 if seed is None else seed
        ruleset = self.ruleset_sampler.sample(root_seed=root_seed, game_index=0)
        self.transition = GameEngine.start(ruleset, player_count=self.player_count, seed=root_seed)
        self._knowledge = ruleset.knowledge(self.player_count)
        self.reward_tracker.reset(self.transition.state)
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos: dict[str, dict[str, Any]] = {agent: {} for agent in self.agents}
        self._pending_decisions = {}
        self._cache_pending_contexts()
        self.agent_selection = self._agent_selector.reset()

    def observe(self, agent: str) -> dict[str, Any]:
        seat = self._seat_for(agent)
        if self._knowledge is None or seat not in self._contexts:
            raise RuntimeError("environment must be reset before observing")
        return self.observation_encoder.encode(self._contexts[seat], self._knowledge)

    def step(self, action: int | None) -> None:
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return
        if action is None:
            raise ValueError("live agents require an integer action")
        current_agent = self.agent_selection
        seat = self._seat_for(current_agent)
        context = self._contexts[seat]
        if not self.action_codec.action_space.contains(action):
            raise ValueError("action is outside the fixed action space")
        if not self.action_codec.mask(context)[int(action)]:
            raise ValueError("action is not legal in the current context")

        self._clear_rewards()
        self._cumulative_rewards[current_agent] = 0.0
        self._pending_decisions[seat] = self.action_codec.decode(int(action))
        assert self.transition is not None
        assert self.transition.pending is not None
        if len(self._pending_decisions) < len(self.transition.pending.acting_seats):
            self.agent_selection = self._agent_selector.next()
            return

        self._resolve_pending_decisions()

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None

    def _resolve_pending_decisions(self) -> None:
        assert self.transition is not None
        transition = GameEngine.step(self.transition.state, self._pending_decisions)
        self.transition = transition
        self._pending_decisions = {}
        self._publish_transition_rewards(transition)
        if transition.terminated:
            for agent in self.agents:
                self.terminations[agent] = True
            self.agent_selection = self._deads_step_first()
            return
        self._cache_pending_contexts()
        assert transition.pending is not None
        if len(transition.pending.acting_seats) == 1:
            self._agent_selector.reinit([self._agent_name(transition.pending.acting_seats[0])])
        else:
            self._agent_selector.reinit(self.possible_agents)
        self.agent_selection = self._agent_selector.reset()

    def _publish_transition_rewards(self, transition: EngineTransition) -> None:
        reward_breakdowns = self.reward_tracker.update(transition)
        for seat, breakdown in reward_breakdowns.items():
            agent = self._agent_name(seat)
            self.rewards[agent] = breakdown.total
            self.infos[agent] = {"reward_breakdown": asdict(breakdown)}
        if transition.result is not None:
            for score in transition.result.scores:
                self.infos[self._agent_name(score.seat)]["score"] = asdict(score)
        self._accumulate_rewards()

    def _cache_pending_contexts(self) -> None:
        assert self.transition is not None
        assert self.transition.pending is not None
        self._contexts.update(self.transition.pending.contexts)

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

    @staticmethod
    def _agent_name(seat: Seat) -> str:
        return f"seat_{seat}"

    @staticmethod
    def _seat_for(agent: str) -> Seat:
        try:
            prefix, seat = agent.split("_", maxsplit=1)
        except ValueError as error:
            raise ValueError(f"unknown agent {agent!r}") from error
        if prefix != "seat" or not seat.isdecimal():
            raise ValueError(f"unknown agent {agent!r}")
        return int(seat)
