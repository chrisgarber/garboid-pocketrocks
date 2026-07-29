from __future__ import annotations

from dataclasses import dataclass, replace

from pocketrocks import OBJECTIVES, ActionId


class RulesetValidationError(ValueError):
    """Raised before setup when a ruleset cannot produce a valid game."""


@dataclass(frozen=True, slots=True)
class PlayerSetup:
    player_count: int
    starting_cash: int
    private_cards_per_player: int


@dataclass(frozen=True, slots=True)
class RulesetKnowledge:
    name: str
    player_count: int
    starting_cash: int
    private_cards_per_player: int
    resource_counts: tuple[int, ...]
    action_counts: tuple[int, ...]
    value_chart: tuple[int, ...]
    objective_pool: tuple[int, ...]
    active_objective_count: int
    objectives_enabled: bool


@dataclass(frozen=True, slots=True)
class Ruleset:
    name: str
    resource_counts: tuple[int, ...]
    action_counts: tuple[int, ...]
    player_setups: tuple[PlayerSetup, ...]
    value_chart: tuple[int, ...]
    objective_pool: tuple[int, ...]
    active_objective_count: int = 4
    objectives_enabled: bool = True

    def __post_init__(self) -> None:
        if len(self.resource_counts) != 5:
            raise RulesetValidationError("ruleset requires five resource counts")
        if any(count < 0 for count in self.resource_counts):
            raise RulesetValidationError("resource counts must be nonnegative")
        if len(self.action_counts) != 6:
            raise RulesetValidationError("ruleset requires six action counts")
        if any(count < 0 for count in self.action_counts):
            raise RulesetValidationError("action counts must be nonnegative")
        if len(self.value_chart) != 6:
            raise RulesetValidationError("ruleset requires six value-chart buckets")
        if len(set(self.objective_pool)) != len(self.objective_pool):
            raise RulesetValidationError("objective IDs must be unique")
        if any(objective_id not in OBJECTIVES for objective_id in self.objective_pool):
            raise RulesetValidationError("objective pool contains an unknown ID")
        if not 0 <= self.active_objective_count <= len(self.objective_pool):
            raise RulesetValidationError("active objective count exceeds objective pool")
        if not self.objectives_enabled and self.active_objective_count != 0:
            raise RulesetValidationError(
                "disabled objectives require active objective count zero"
            )
        for setup in self.player_setups:
            self._validate_setup(setup)

    def _validate_setup(self, setup: PlayerSetup) -> None:
        if not 3 <= setup.player_count <= 5:
            raise RulesetValidationError("player count must be between 3 and 5")
        if setup.starting_cash <= 0:
            raise RulesetValidationError("starting cash must be positive")
        if setup.private_cards_per_player < 0:
            raise RulesetValidationError("private-card count must be nonnegative")
        biddable = sum(self.resource_counts) - (
            setup.player_count * setup.private_cards_per_player
        )
        if biddable <= 0:
            raise RulesetValidationError("setup must leave a biddable resource")
        auction_capacity = self.action_count(ActionId.AUCTION1) + (
            2 * self.action_count(ActionId.AUCTION2)
        )
        if auction_capacity < biddable:
            raise RulesetValidationError("action deck has insufficient auction capacity")

    def setup_for(self, player_count: int) -> PlayerSetup:
        for setup in self.player_setups:
            if setup.player_count == player_count:
                return setup
        raise RulesetValidationError(
            f"ruleset {self.name!r} does not support {player_count} players"
        )

    def action_count(self, action_id: ActionId) -> int:
        return self.action_counts[int(action_id) - 1]

    def knowledge(self, player_count: int) -> RulesetKnowledge:
        setup = self.setup_for(player_count)
        return RulesetKnowledge(
            name=self.name,
            player_count=player_count,
            starting_cash=setup.starting_cash,
            private_cards_per_player=setup.private_cards_per_player,
            resource_counts=self.resource_counts,
            action_counts=self.action_counts,
            value_chart=self.value_chart,
            objective_pool=self.objective_pool,
            active_objective_count=self.active_objective_count,
            objectives_enabled=self.objectives_enabled,
        )


VALUE_CHARTS: dict[str, tuple[int, ...]] = {
    "A": (0, 4, 8, 12, 16, 20),
    "B": (20, 16, 12, 8, 4, 0),
    "C": (0, 2, 5, 9, 14, 20),
    "D": (20, 18, 15, 11, 6, 0),
    "E": (0, 4, 10, 18, 6, 0),
}

LIVE_RULESET = Ruleset(
    name="live-A",
    resource_counts=(6, 6, 6, 6, 6),
    action_counts=(12, 8, 3, 2, 3, 2),
    player_setups=(
        PlayerSetup(3, 30, 5),
        PlayerSetup(4, 25, 4),
        PlayerSetup(5, 20, 3),
    ),
    value_chart=VALUE_CHARTS["A"],
    objective_pool=tuple(sorted(OBJECTIVES)),
    active_objective_count=4,
)


def live_ruleset(
    chart: str = "A",
    objectives_enabled: bool = True,
) -> Ruleset:
    try:
        value_chart = VALUE_CHARTS[chart.upper()]
    except KeyError as error:
        raise RulesetValidationError(f"unknown value chart {chart!r}") from error
    return replace(
        LIVE_RULESET,
        name=f"live-{chart.upper()}",
        value_chart=value_chart,
        objectives_enabled=objectives_enabled,
        active_objective_count=4 if objectives_enabled else 0,
    )
