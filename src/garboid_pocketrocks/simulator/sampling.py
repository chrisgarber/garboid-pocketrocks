from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from itertools import product
from typing import Protocol, runtime_checkable

from garboid_pocketrocks.rules import PlayerSetup, Ruleset

ObjectiveOption = tuple[tuple[int, ...], int, bool]


def derive_seed(root_seed: int, namespace: str, index: int) -> int:
    payload = f"{root_seed}:{namespace}:{index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


@runtime_checkable
class RulesetSampler(Protocol):
    def support(self) -> tuple[Ruleset, ...]:
        """Return every ruleset this sampler can produce."""

    def sample(self, *, root_seed: int, game_index: int) -> Ruleset:
        """Resolve one valid ruleset without global random state."""


@dataclass(frozen=True, slots=True)
class FixedRulesetSampler:
    ruleset: Ruleset

    def support(self) -> tuple[Ruleset, ...]:
        return (self.ruleset,)

    def sample(self, *, root_seed: int, game_index: int) -> Ruleset:
        del root_seed, game_index
        return self.ruleset


@dataclass(frozen=True, slots=True)
class WeightedRulesetSampler:
    weighted_rulesets: tuple[tuple[Ruleset, int], ...]
    _support: tuple[Ruleset, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.weighted_rulesets:
            raise ValueError("weighted ruleset sampler requires at least one ruleset")
        if any(
            isinstance(weight, bool) or not isinstance(weight, int) or weight <= 0
            for _, weight in self.weighted_rulesets
        ):
            raise ValueError("ruleset weights must be positive integers")
        object.__setattr__(
            self,
            "_support",
            tuple(dict.fromkeys(ruleset for ruleset, _ in self.weighted_rulesets)),
        )

    def support(self) -> tuple[Ruleset, ...]:
        return self._support

    def sample(self, *, root_seed: int, game_index: int) -> Ruleset:
        rng = random.Random(derive_seed(root_seed, "ruleset", game_index))
        rulesets = tuple(ruleset for ruleset, _ in self.weighted_rulesets)
        weights = tuple(weight for _, weight in self.weighted_rulesets)
        return rng.choices(rulesets, weights=weights, k=1)[0]


@dataclass(frozen=True, slots=True)
class RulesetVariationSampler:
    base: Ruleset
    resource_count_options: tuple[tuple[int, ...], ...]
    action_count_options: tuple[tuple[int, ...], ...]
    setup_options: tuple[tuple[PlayerSetup, ...], ...]
    value_chart_options: tuple[tuple[int, ...], ...]
    objective_options: tuple[ObjectiveOption, ...]
    _support: tuple[Ruleset, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        option_groups = (
            ("resource_count_options", self.resource_count_options),
            ("action_count_options", self.action_count_options),
            ("setup_options", self.setup_options),
            ("value_chart_options", self.value_chart_options),
            ("objective_options", self.objective_options),
        )
        for name, options in option_groups:
            if not options:
                raise ValueError(f"{name} must be nonempty")

        supported = tuple(
            self._build_ruleset(resources, actions, setups, chart, objectives)
            for resources, actions, setups, chart, objectives in product(
                self.resource_count_options,
                self.action_count_options,
                self.setup_options,
                self.value_chart_options,
                self.objective_options,
            )
        )
        object.__setattr__(self, "_support", supported)

    def support(self) -> tuple[Ruleset, ...]:
        return self._support

    def sample(self, *, root_seed: int, game_index: int) -> Ruleset:
        rng = random.Random(derive_seed(root_seed, "ruleset", game_index))
        return self._build_ruleset(
            rng.choice(self.resource_count_options),
            rng.choice(self.action_count_options),
            rng.choice(self.setup_options),
            rng.choice(self.value_chart_options),
            rng.choice(self.objective_options),
        )

    def _build_ruleset(
        self,
        resource_counts: tuple[int, ...],
        action_counts: tuple[int, ...],
        player_setups: tuple[PlayerSetup, ...],
        value_chart: tuple[int, ...],
        objective_option: ObjectiveOption,
    ) -> Ruleset:
        objective_pool, active_objective_count, objectives_enabled = objective_option
        public_fields = {
            "resource_counts": resource_counts,
            "action_counts": action_counts,
            "player_setups": tuple(
                (
                    setup.player_count,
                    setup.starting_cash,
                    setup.private_cards_per_player,
                )
                for setup in player_setups
            ),
            "value_chart": value_chart,
            "objective_pool": objective_pool,
            "active_objective_count": active_objective_count,
            "objectives_enabled": objectives_enabled,
        }
        canonical = json.dumps(
            public_fields,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        digest = hashlib.sha256(canonical).hexdigest()[:12]
        return Ruleset(
            name=f"{self.base.name}-{digest}",
            resource_counts=resource_counts,
            action_counts=action_counts,
            player_setups=player_setups,
            value_chart=value_chart,
            objective_pool=objective_pool,
            active_objective_count=active_objective_count,
            objectives_enabled=objectives_enabled,
        )
