from __future__ import annotations

from dataclasses import dataclass, field

from pocketrocks import OBJECTIVES

from garboid_pocketrocks.simulator.engine import EngineTransition
from garboid_pocketrocks.simulator.events import EventKind
from garboid_pocketrocks.simulator.model import GameState, PlayerState, Seat


@dataclass(frozen=True, slots=True)
class RewardConfig:
    accounting_weight: float = 1.0
    win_bonus: float = 1.0
    placement_bonuses: tuple[float, ...] = ()
    invalid_action_penalty: float = 0.0
    event_bonuses: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        event_kinds = tuple(kind for kind, _ in self.event_bonuses)
        if len(set(event_kinds)) != len(event_kinds):
            raise ValueError("event bonuses must not contain duplicate event kinds")
        unknown = set(event_kinds) - {kind.value for kind in EventKind}
        if unknown:
            raise ValueError(f"event bonuses contain unknown event kinds: {sorted(unknown)!r}")


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    accounting: float = 0.0
    terminal_resource: float = 0.0
    placement: float = 0.0
    shaping: float = 0.0
    penalty: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.accounting + self.terminal_resource + self.placement + self.shaping + self.penalty
        )


@dataclass(slots=True)
class RewardTracker:
    """Tracks per-seat public potential and decomposes every reward source."""

    config: RewardConfig = RewardConfig()
    _previous_potential: dict[Seat, int] = field(default_factory=dict, init=False)
    _starting_cash: dict[Seat, int] = field(default_factory=dict, init=False)
    _terminal: bool = field(default=False, init=False)

    def reset(self, state: GameState) -> None:
        starting_cash = state.ruleset.setup_for(state.player_count).starting_cash
        self._previous_potential = {
            player.seat: _public_potential(player) for player in state.players
        }
        self._starting_cash = {player.seat: starting_cash for player in state.players}
        self._terminal = False

    def update(self, transition: EngineTransition) -> dict[Seat, RewardBreakdown]:
        if not self._previous_potential:
            raise ValueError("reward tracker must be reset before update")
        if self._terminal:
            return {seat: RewardBreakdown() for seat in self._previous_potential}

        rewards = {seat: RewardBreakdown() for seat in self._previous_potential}
        for player in transition.state.players:
            seat = player.seat
            current = _public_potential(player)
            accounting = self.config.accounting_weight * (
                (current - self._previous_potential[seat]) / self._starting_cash[seat]
            )
            rewards[seat] = RewardBreakdown(accounting=accounting)
            self._previous_potential[seat] = current

        rewards = self._apply_event_shaping(rewards, transition)
        if transition.result is not None:
            rewards = self._apply_terminal_rewards(rewards, transition)
            self._terminal = True
        return rewards

    def invalid_action(self, seat: Seat) -> RewardBreakdown:
        if seat not in self._previous_potential:
            raise ValueError(f"unknown reward seat {seat}")
        return RewardBreakdown(penalty=-self.config.invalid_action_penalty)

    def _apply_event_shaping(
        self,
        rewards: dict[Seat, RewardBreakdown],
        transition: EngineTransition,
    ) -> dict[Seat, RewardBreakdown]:
        event_bonuses = dict(self.config.event_bonuses)
        shaping_by_seat = {seat: 0.0 for seat in rewards}
        for event in transition.events:
            if event.seat is not None:
                shaping_by_seat[event.seat] += event_bonuses.get(event.kind.value, 0.0)
        return {
            seat: _replace_breakdown(reward, shaping=shaping_by_seat[seat])
            for seat, reward in rewards.items()
        }

    def _apply_terminal_rewards(
        self,
        rewards: dict[Seat, RewardBreakdown],
        transition: EngineTransition,
    ) -> dict[Seat, RewardBreakdown]:
        assert transition.result is not None
        winners = tuple(score.seat for score in transition.result.scores if score.rank == 1)
        win_share = self.config.win_bonus / len(winners)
        scores = {score.seat: score for score in transition.result.scores}
        final_rewards: dict[Seat, RewardBreakdown] = {}
        for seat, reward in rewards.items():
            score = scores[seat]
            terminal_resource = self.config.accounting_weight * (
                (score.final_money - self._previous_potential[seat]) / self._starting_cash[seat]
            )
            placement = win_share if score.rank == 1 else 0.0
            if score.rank <= len(self.config.placement_bonuses):
                placement += self.config.placement_bonuses[score.rank - 1]
            final_rewards[seat] = _replace_breakdown(
                reward,
                terminal_resource=terminal_resource,
                placement=placement,
            )
        return final_rewards


def _public_potential(player: PlayerState) -> int:
    return (
        player.cash
        + sum(investment.locked + investment.payout for investment in player.investments)
        - sum(loan.principal for loan in player.loans)
        + sum(OBJECTIVES[objective_id].payout for objective_id in player.owned_objective_ids)
    )


def _replace_breakdown(
    reward: RewardBreakdown,
    *,
    terminal_resource: float | None = None,
    placement: float | None = None,
    shaping: float | None = None,
) -> RewardBreakdown:
    return RewardBreakdown(
        accounting=reward.accounting,
        terminal_resource=reward.terminal_resource
        if terminal_resource is None
        else terminal_resource,
        placement=reward.placement if placement is None else placement,
        shaping=reward.shaping if shaping is None else shaping,
        penalty=reward.penalty,
    )
