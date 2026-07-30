from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

from pocketrocks import OBJECTIVES

from garboid_pocketrocks.simulator.session import (
    PlayerSnapshot,
    SessionSnapshot,
    SessionTransition,
)

type Seat = int


class RewardEventKind(StrEnum):
    AUCTION_RESOLVED = "auction_resolved"
    RESOURCES_AWARDED = "resources_awarded"
    LOAN_ACQUIRED = "loan_acquired"
    INVESTMENT_ACQUIRED = "investment_acquired"
    OBJECTIVE_CLAIMED = "objective_claimed"
    INFORMATION_REVEALED = "information_revealed"


@dataclass(frozen=True, slots=True)
class RewardConfig:
    accounting_weight: float = 1.0
    win_bonus: float = 1.0
    placement_bonuses: tuple[float, ...] = ()
    invalid_action_penalty: float = 0.0
    event_bonuses: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        coefficients = (
            self.accounting_weight,
            self.win_bonus,
            self.invalid_action_penalty,
            *self.placement_bonuses,
            *(bonus for _, bonus in self.event_bonuses),
        )
        if not all(math.isfinite(coefficient) for coefficient in coefficients):
            raise ValueError("reward coefficients must be finite")
        if self.invalid_action_penalty < 0:
            raise ValueError("invalid action penalty must be nonnegative")
        event_kinds = tuple(kind for kind, _bonus in self.event_bonuses)
        if len(set(event_kinds)) != len(event_kinds):
            raise ValueError("event bonuses must not contain duplicate event kinds")
        unknown = set(event_kinds) - {kind.value for kind in RewardEventKind}
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
    """Tracks public SDK-state potential and decomposes every reward source."""

    config: RewardConfig = RewardConfig()
    _previous_potential: dict[Seat, int] = field(default_factory=dict, init=False)
    _starting_cash: dict[Seat, int] = field(default_factory=dict, init=False)
    _terminal: bool = field(default=False, init=False)

    def reset(self, snapshot: SessionSnapshot) -> None:
        self._previous_potential = {
            player.seat: _public_potential(player) for player in snapshot.players
        }
        self._starting_cash = {player.seat: player.cash for player in snapshot.players}
        self._terminal = False

    def update(self, transition: SessionTransition) -> dict[Seat, RewardBreakdown]:
        if not self._previous_potential:
            raise ValueError("reward tracker must be reset before update")
        if self._terminal:
            return {seat: RewardBreakdown() for seat in self._previous_potential}

        rewards = {seat: RewardBreakdown() for seat in self._previous_potential}
        for player in transition.snapshot.players:
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
        transition: SessionTransition,
    ) -> dict[Seat, RewardBreakdown]:
        bonuses = dict(self.config.event_bonuses)
        shaping = {seat: 0.0 for seat in rewards}
        reveal_only = (
            bool(transition.decisions)
            and len(transition.decisions) == 1
            and transition.before.current_action is None
        )
        for turn in transition.turn_records:
            seat = turn.winner_seat
            if reveal_only:
                if turn.reveal is not None:
                    shaping[seat] += bonuses.get(
                        RewardEventKind.INFORMATION_REVEALED.value,
                        0.0,
                    )
                continue
            shaping[seat] += bonuses.get(RewardEventKind.AUCTION_RESOLVED.value, 0.0)
            if turn.bundle_suits:
                shaping[seat] += bonuses.get(
                    RewardEventKind.RESOURCES_AWARDED.value,
                    0.0,
                )
            elif turn.action.startswith("Loan"):
                shaping[seat] += bonuses.get(RewardEventKind.LOAN_ACQUIRED.value, 0.0)
            elif turn.action.startswith("Invest"):
                shaping[seat] += bonuses.get(
                    RewardEventKind.INVESTMENT_ACQUIRED.value,
                    0.0,
                )
            if turn.claimed_objective_wire_ids:
                shaping[seat] += bonuses.get(
                    RewardEventKind.OBJECTIVE_CLAIMED.value,
                    0.0,
                )
            if turn.reveal is not None:
                shaping[seat] += bonuses.get(
                    RewardEventKind.INFORMATION_REVEALED.value,
                    0.0,
                )
        return {
            seat: _replace_breakdown(reward, shaping=shaping[seat])
            for seat, reward in rewards.items()
        }

    def _apply_terminal_rewards(
        self,
        rewards: dict[Seat, RewardBreakdown],
        transition: SessionTransition,
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


def _public_potential(player: PlayerSnapshot) -> int:
    return (
        player.cash
        + sum(locked + payout for locked, payout in player.investments)
        - sum(player.loans)
        + sum(OBJECTIVES[objective_id].payout for objective_id in player.objective_ids)
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
        terminal_resource=(
            reward.terminal_resource if terminal_resource is None else terminal_resource
        ),
        placement=reward.placement if placement is None else placement,
        shaping=reward.shaping if shaping is None else shaping,
        penalty=reward.penalty,
    )
