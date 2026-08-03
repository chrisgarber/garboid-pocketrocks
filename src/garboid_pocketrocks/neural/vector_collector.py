"""Neural self-play directly on the SDK's vectorized game engine."""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from pocketrocks import DecisionContext
from pocketrocks.sim import BatchSimEngine, ScoreRow
from pocketrocks.sim.constants import (
    ACTION_WIRE_IDS,
    OBJECTIVE_PAYOUTS,
    STARTING_CASH,
)

from garboid_pocketrocks.adapters.public_history import (
    PublicAuctionResolved,
    PublicEvent,
    PublicEventKind,
    PublicGameSetup,
    PublicInformationRevealed,
    PublicTurnOpened,
)
from garboid_pocketrocks.bots.base import BotBrain
from garboid_pocketrocks.knowledge import (
    RulesetKnowledge,
    canonical_knowledge,
    value_chart_from_ruleset_name,
)
from garboid_pocketrocks.neural.collector import (
    CollectorMetrics,
    _freeze_policies,
    _infer_policy_requests,
    _percentile,
    _restore_policy_modes,
    _validate_collection,
)
from garboid_pocketrocks.neural.config import NeuralEncoderConfig
from garboid_pocketrocks.neural.encoding import (
    NeuralObservation,
    NeuralObservationEncoder,
)
from garboid_pocketrocks.neural.model import NeuralPolicy
from garboid_pocketrocks.neural.planning import (
    SelfPlayEpisodePlan,
    decision_seed,
)
from garboid_pocketrocks.neural.rollout import (
    MultiSeatEpisode,
    RolloutBatch,
    RolloutMetadata,
    RolloutTransition,
    SeatTrajectory,
    _immutable_observation,
)
from garboid_pocketrocks.neural.self_play import (
    PendingPolicyRequest,
    PolicyResponse,
)
from garboid_pocketrocks.simulator.batch_context import build_batch_context
from garboid_pocketrocks.simulator.session import (
    SessionResult,
    SessionScore,
)
from garboid_pocketrocks.training.bounds import EnvironmentBounds
from garboid_pocketrocks.training.rewards import RewardBreakdown, RewardConfig

_AUCTION_ACTION_IDS = {
    ACTION_WIRE_IDS["Auction1"],
    ACTION_WIRE_IDS["Auction2"],
}
_LOAN_ACTION_IDS = {
    ACTION_WIRE_IDS["Loan10"],
    ACTION_WIRE_IDS["Loan20"],
}
_INVESTMENT_ACTION_IDS = {
    ACTION_WIRE_IDS["Invest5"],
    ACTION_WIRE_IDS["Invest10"],
}


@dataclass(frozen=True, slots=True)
class _OpenTransition:
    request: PendingPolicyRequest
    context: DecisionContext
    response: PolicyResponse


def vector_plan_batches(
    plans: Sequence[SelfPlayEpisodePlan],
    *,
    batch_size: int = 64,
) -> tuple[tuple[SelfPlayEpisodePlan, ...], ...]:
    """Partition plans into stable, homogeneous SDK engine batches."""

    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    by_players: dict[int, list[SelfPlayEpisodePlan]] = defaultdict(list)
    for plan in plans:
        by_players[plan.player_count].append(plan)
    return tuple(
        tuple(group[offset : offset + batch_size])
        for player_count in sorted(by_players)
        for group in (by_players[player_count],)
        for offset in range(0, len(group), batch_size)
    )


def collect_self_play_vectorized(
    policies: Mapping[str, NeuralPolicy],
    plans: Sequence[SelfPlayEpisodePlan],
    *,
    encoder_config: NeuralEncoderConfig,
    reward_config: RewardConfig,
    device: torch.device,
    engine_batch_size: int = 64,
    max_inference_batch: int = 512,
) -> tuple[RolloutBatch, CollectorMetrics]:
    """Collect all-seat trajectories with the SDK's NumPy batch engine."""

    collected_plans = tuple(plans)
    _validate_collection(
        policies,
        collected_plans,
        encoder_config=encoder_config,
        device=device,
        active_games=engine_batch_size,
        max_inference_batch=max_inference_batch,
    )
    batches = vector_plan_batches(
        collected_plans,
        batch_size=engine_batch_size,
    )
    started = time.perf_counter()
    prior_modes = _freeze_policies(policies)
    completed: list[MultiSeatEpisode] = []
    inference_sizes: list[int] = []
    inference_seconds = 0.0
    decisions = 0
    try:
        for batch_plans in batches:
            episodes, batch_decisions, batch_inference = _collect_engine_batch(
                policies,
                batch_plans,
                encoder_config=encoder_config,
                reward_config=reward_config,
                device=device,
                max_inference_batch=max_inference_batch,
                inference_sizes=inference_sizes,
            )
            completed.extend(episodes)
            decisions += batch_decisions
            inference_seconds += batch_inference
    finally:
        _restore_policy_modes(prior_modes)

    elapsed = time.perf_counter() - started
    completed.sort(key=lambda episode: episode.plan.episode_index)
    cells = Counter((episode.plan.ruleset_name, episode.plan.player_count) for episode in completed)
    metrics = CollectorMetrics(
        games=len(completed),
        decisions=decisions,
        elapsed_seconds=elapsed,
        inference_seconds=inference_seconds,
        inference_batches=len(inference_sizes),
        inference_batch_sizes=tuple(inference_sizes),
        cell_games=tuple(
            (ruleset, players, games) for (ruleset, players), games in sorted(cells.items())
        ),
        worker_busy_seconds=elapsed,
        inference_batch_p50=_percentile(inference_sizes, 0.50),
        inference_batch_p95=_percentile(inference_sizes, 0.95),
    )
    return RolloutBatch.from_multi_seat(completed), metrics


def _collect_engine_batch(
    policies: Mapping[str, NeuralPolicy],
    plans: tuple[SelfPlayEpisodePlan, ...],
    *,
    encoder_config: NeuralEncoderConfig,
    reward_config: RewardConfig,
    device: torch.device,
    max_inference_batch: int,
    inference_sizes: list[int],
) -> tuple[tuple[MultiSeatEpisode, ...], int, float]:
    player_count = plans[0].player_count
    engine = BatchSimEngine.start(
        player_count=player_count,
        seeds=tuple(plan.engine_seed for plan in plans),
        value_charts=tuple(value_chart_from_ruleset_name(plan.ruleset_name) for plan in plans),
    )
    bounds = EnvironmentBounds(
        max_bid=encoder_config.max_bid,
        max_hand_size=encoder_config.max_hand_size,
    )
    encoder = NeuralObservationEncoder(encoder_config, bounds)
    histories, knowledge = _initialize_public_histories(engine, plans)

    pending_rewards = [[RewardBreakdown() for _ in range(player_count)] for _ in plans]
    decision_counts = [[0 for _ in range(player_count)] for _ in plans]
    opened: dict[tuple[int, int], _OpenTransition] = {}
    completed: list[list[list[RolloutTransition]]] = [
        [[] for _ in range(player_count)] for _ in plans
    ]
    previous_potential = _potential(engine)
    decisions = 0
    inference_seconds = 0.0
    fixed_brains: dict[tuple[int, int], BotBrain] = {}

    while True:
        action_ids = engine.flip_actions()
        active_rows = np.flatnonzero(action_ids > 0)
        if not len(active_rows):
            break
        resources = engine.upcoming.copy()
        legal = engine.legal_max_bids()
        _append_turn_opened_events(
            histories,
            active_rows,
            action_ids,
            resources,
        )
        bid_requests, bid_contexts = _prepare_bid_requests(
            engine,
            plans,
            active_rows,
            action_ids,
            resources,
            legal,
            encoder,
            knowledge,
            histories,
            decision_counts,
            opened,
            pending_rewards,
            completed,
        )

        bid_responses, elapsed = _infer_policy_requests(
            policies,
            tuple(bid_requests),
            device=device,
            max_inference_batch=max_inference_batch,
            inference_batch_sizes=inference_sizes,
            fixed_brains=fixed_brains,
        )
        inference_seconds += elapsed
        decisions += len(bid_responses)
        bids = _record_bid_responses(
            plans,
            active_rows,
            bid_requests,
            bid_contexts,
            bid_responses,
            opened,
            decision_counts,
        )

        objectives_before = engine.owned_objectives.copy()
        outcome = engine.resolve_bids(bids)
        for row_value in active_rows:
            row = int(row_value)
            histories[row].append(
                PublicAuctionResolved(
                    kind=PublicEventKind.AUCTION_RESOLVED,
                    bids_by_seat=tuple(int(value) for value in outcome.effective_bids[row]),
                )
            )
        current_potential = _potential(engine)
        _apply_resolve_rewards(
            engine,
            active_rows,
            action_ids,
            outcome.winner_seats,
            objectives_before,
            previous_potential,
            current_potential,
            pending_rewards,
            reward_config,
        )
        previous_potential = current_potential

        choice_rows = np.flatnonzero(outcome.reveal_modes == 2)
        reveal_requests, reveal_contexts = _prepare_reveal_requests(
            engine,
            plans,
            choice_rows,
            outcome.winner_seats,
            action_ids,
            resources,
            encoder,
            knowledge,
            histories,
            decision_counts,
            opened,
            pending_rewards,
            completed,
        )

        reveal_responses, elapsed = _infer_policy_requests(
            policies,
            tuple(reveal_requests),
            device=device,
            max_inference_batch=max_inference_batch,
            inference_batch_sizes=inference_sizes,
            fixed_brains=fixed_brains,
        )
        inference_seconds += elapsed
        decisions += len(reveal_responses)
        reveal_indices = _record_reveal_responses(
            plans,
            choice_rows,
            outcome.winner_seats,
            outcome.reveal_modes,
            reveal_requests,
            reveal_contexts,
            reveal_responses,
            encoder_config.max_bid,
            opened,
            decision_counts,
        )
        _apply_and_record_reveals(
            engine,
            outcome.winner_seats,
            outcome.reveal_modes,
            reveal_indices,
            histories,
            pending_rewards,
            reward_config,
        )

    episodes = _finalize_batch_episodes(
        engine,
        plans,
        previous_potential,
        pending_rewards,
        opened,
        completed,
        reward_config,
    )
    return episodes, decisions, inference_seconds


def _initialize_public_histories(
    engine: BatchSimEngine,
    plans: tuple[SelfPlayEpisodePlan, ...],
) -> tuple[list[list[PublicEvent]], list[RulesetKnowledge]]:
    histories: list[list[PublicEvent]] = []
    knowledge: list[RulesetKnowledge] = []
    for row, plan in enumerate(plans):
        game_knowledge = canonical_knowledge(
            engine.player_count,
            value_chart=value_chart_from_ruleset_name(plan.ruleset_name),
        )
        knowledge.append(game_knowledge)
        histories.append(
            [
                PublicGameSetup(
                    kind=PublicEventKind.GAME_SETUP,
                    player_count=engine.player_count,
                    starting_cash=game_knowledge.starting_cash,
                    value_chart=game_knowledge.value_chart,
                    initial_tiebreak_seat=int(engine.tiebreak_seats[row]),
                    objective_ids=tuple(
                        int(value) for value in engine.objective_ids[row] if value > 0
                    ),
                )
            ]
        )
    return histories, knowledge


def _append_turn_opened_events(
    histories: list[list[PublicEvent]],
    active_rows: np.ndarray,
    action_ids: np.ndarray,
    resources: np.ndarray,
) -> None:
    for row_value in active_rows:
        row = int(row_value)
        histories[row].append(
            PublicTurnOpened(
                kind=PublicEventKind.TURN_OPENED,
                action_id=int(action_ids[row]),
                resource_ids=(
                    int(resources[row, 0]),
                    int(resources[row, 1]),
                ),
            )
        )


def _prepare_bid_requests(
    engine: BatchSimEngine,
    plans: tuple[SelfPlayEpisodePlan, ...],
    active_rows: np.ndarray,
    action_ids: np.ndarray,
    resources: np.ndarray,
    legal: np.ndarray,
    encoder: NeuralObservationEncoder,
    knowledge: Sequence[RulesetKnowledge],
    histories: list[list[PublicEvent]],
    decision_counts: list[list[int]],
    opened: dict[tuple[int, int], _OpenTransition],
    pending_rewards: list[list[RewardBreakdown]],
    completed: list[list[list[RolloutTransition]]],
) -> tuple[list[PendingPolicyRequest], dict[tuple[int, int], DecisionContext]]:
    requests: list[PendingPolicyRequest] = []
    contexts: dict[tuple[int, int], DecisionContext] = {}
    for row_value in active_rows:
        row = int(row_value)
        plan = plans[row]
        for seat in range(engine.player_count):
            _finalize_if_open(
                row,
                seat,
                plans=plans,
                opened=opened,
                pending_rewards=pending_rewards,
                completed=completed,
                terminated=False,
            )
            context = build_batch_context(
                engine,
                row=row,
                seat=seat,
                decision_kind="submitBid",
                action_id=int(action_ids[row]),
                resource_ids=(
                    int(resources[row, 0]),
                    int(resources[row, 1]),
                ),
                turn_index=int(engine.turn_indices[row]),
                legal_max_amount=int(legal[row, seat]),
            )
            requests.append(
                _request(
                    plan,
                    seat,
                    decision_counts[row][seat],
                    encoder._encode_trusted(
                        context,
                        knowledge[row],
                        tuple(histories[row]),
                    ),
                    context=context,
                    ruleset=knowledge[row],
                    history=tuple(histories[row]),
                )
            )
            contexts[(row, seat)] = context
    return requests, contexts


def _record_bid_responses(
    plans: tuple[SelfPlayEpisodePlan, ...],
    active_rows: np.ndarray,
    requests: Sequence[PendingPolicyRequest],
    contexts: Mapping[tuple[int, int], DecisionContext],
    responses: Sequence[PolicyResponse],
    opened: dict[tuple[int, int], _OpenTransition],
    decision_counts: list[list[int]],
) -> np.ndarray:
    response_by_key = {(response.episode_index, response.seat): response for response in responses}
    request_by_key = {(request.episode_index, request.seat): request for request in requests}
    player_count = plans[0].player_count
    bids = np.zeros((len(plans), player_count), dtype=np.int16)
    for row_value in active_rows:
        row = int(row_value)
        plan = plans[row]
        for seat in range(player_count):
            response = response_by_key[(plan.episode_index, seat)]
            opened[(row, seat)] = _OpenTransition(
                request=request_by_key[(plan.episode_index, seat)],
                context=contexts[(row, seat)],
                response=response,
            )
            decision_counts[row][seat] += 1
            bids[row, seat] = response.action
    return bids


def _prepare_reveal_requests(
    engine: BatchSimEngine,
    plans: tuple[SelfPlayEpisodePlan, ...],
    choice_rows: np.ndarray,
    winner_seats: np.ndarray,
    action_ids: np.ndarray,
    resources: np.ndarray,
    encoder: NeuralObservationEncoder,
    knowledge: Sequence[RulesetKnowledge],
    histories: list[list[PublicEvent]],
    decision_counts: list[list[int]],
    opened: dict[tuple[int, int], _OpenTransition],
    pending_rewards: list[list[RewardBreakdown]],
    completed: list[list[list[RolloutTransition]]],
) -> tuple[list[PendingPolicyRequest], dict[tuple[int, int], DecisionContext]]:
    requests: list[PendingPolicyRequest] = []
    contexts: dict[tuple[int, int], DecisionContext] = {}
    for row_value in choice_rows:
        row = int(row_value)
        seat = int(winner_seats[row])
        _finalize_if_open(
            row,
            seat,
            plans=plans,
            opened=opened,
            pending_rewards=pending_rewards,
            completed=completed,
            terminated=False,
        )
        context = build_batch_context(
            engine,
            row=row,
            seat=seat,
            decision_kind="selectInfoToReveal",
            action_id=int(action_ids[row]),
            resource_ids=(
                int(resources[row, 0]),
                int(resources[row, 1]),
            ),
            turn_index=int(engine.turn_indices[row]) - 1,
            legal_max_amount=None,
        )
        requests.append(
            _request(
                plans[row],
                seat,
                decision_counts[row][seat],
                encoder._encode_trusted(
                    context,
                    knowledge[row],
                    tuple(histories[row]),
                ),
                context=context,
                ruleset=knowledge[row],
                history=tuple(histories[row]),
            )
        )
        contexts[(row, seat)] = context
    return requests, contexts


def _record_reveal_responses(
    plans: tuple[SelfPlayEpisodePlan, ...],
    choice_rows: np.ndarray,
    winner_seats: np.ndarray,
    reveal_modes: np.ndarray,
    requests: Sequence[PendingPolicyRequest],
    contexts: Mapping[tuple[int, int], DecisionContext],
    responses: Sequence[PolicyResponse],
    max_bid: int,
    opened: dict[tuple[int, int], _OpenTransition],
    decision_counts: list[list[int]],
) -> np.ndarray:
    response_by_key = {(response.episode_index, response.seat): response for response in responses}
    request_by_key = {(request.episode_index, request.seat): request for request in requests}
    reveal_indices = np.full(len(plans), -1, dtype=np.int16)
    reveal_indices[reveal_modes == 1] = 0
    for row_value in choice_rows:
        row = int(row_value)
        seat = int(winner_seats[row])
        response = response_by_key[(plans[row].episode_index, seat)]
        opened[(row, seat)] = _OpenTransition(
            request=request_by_key[(plans[row].episode_index, seat)],
            context=contexts[(row, seat)],
            response=response,
        )
        decision_counts[row][seat] += 1
        reveal_indices[row] = 0 if response.action == 0 else response.action - max_bid - 1
    return reveal_indices


def _apply_and_record_reveals(
    engine: BatchSimEngine,
    winner_seats: np.ndarray,
    reveal_modes: np.ndarray,
    reveal_indices: np.ndarray,
    histories: list[list[PublicEvent]],
    pending_rewards: list[list[RewardBreakdown]],
    reward_config: RewardConfig,
) -> None:
    revealed = tuple(
        (
            int(row_value),
            int(winner_seats[int(row_value)]),
            int(
                engine.hand_cards[
                    int(row_value),
                    int(winner_seats[int(row_value)]),
                    int(reveal_indices[int(row_value)]),
                ]
            ),
        )
        for row_value in np.flatnonzero(reveal_modes > 0)
    )
    engine.apply_reveals(reveal_indices)
    information_bonus = dict(reward_config.event_bonuses).get(
        "information_revealed",
        0.0,
    )
    for row, seat, suit in revealed:
        histories[row].append(
            PublicInformationRevealed(
                kind=PublicEventKind.INFORMATION_REVEALED,
                seat=seat,
                suit_id=suit,
            )
        )
        pending_rewards[row][seat] = _add_breakdowns(
            pending_rewards[row][seat],
            RewardBreakdown(shaping=information_bonus),
        )


def _finalize_batch_episodes(
    engine: BatchSimEngine,
    plans: tuple[SelfPlayEpisodePlan, ...],
    previous_potential: np.ndarray,
    pending_rewards: list[list[RewardBreakdown]],
    opened: dict[tuple[int, int], _OpenTransition],
    completed: list[list[list[RolloutTransition]]],
    reward_config: RewardConfig,
) -> tuple[MultiSeatEpisode, ...]:
    results = _results(engine)
    _apply_terminal_rewards(
        engine,
        results,
        previous_potential,
        pending_rewards,
        reward_config,
    )
    episodes: list[MultiSeatEpisode] = []
    for row, plan in enumerate(plans):
        for seat in range(engine.player_count):
            _finalize_if_open(
                row,
                seat,
                plans=plans,
                opened=opened,
                pending_rewards=pending_rewards,
                completed=completed,
                terminated=True,
            )
        episodes.append(
            MultiSeatEpisode(
                plan=plan,
                trajectories=tuple(
                    SeatTrajectory(
                        seat=seat,
                        policy_identity=plan.seat_policies[seat].identity,
                        trainable=plan.seat_policies[seat].trainable,
                        transitions=tuple(completed[row][seat]),
                    )
                    for seat in range(engine.player_count)
                ),
                result=results[row],
            )
        )
    return tuple(episodes)


def _request(
    plan: SelfPlayEpisodePlan,
    seat: int,
    decision_index: int,
    observation: NeuralObservation,
    *,
    context: DecisionContext,
    ruleset: RulesetKnowledge,
    history: tuple[PublicEvent, ...],
) -> PendingPolicyRequest:
    assignment = plan.seat_policies[seat]
    return PendingPolicyRequest(
        episode_index=plan.episode_index,
        seat=seat,
        decision_index=decision_index,
        policy_identity=assignment.identity,
        trainable=assignment.trainable,
        sampling_seed=decision_seed(plan, seat, decision_index),
        observation=_immutable_observation(observation),
        context=context,
        ruleset=ruleset,
        history=history,
    )


def _potential(engine: BatchSimEngine) -> np.ndarray:
    objective_payouts = np.asarray(
        [OBJECTIVE_PAYOUTS.get(objective_id, 0) for objective_id in range(1, 31)],
        dtype=np.int16,
    )
    objectives = np.sum(
        engine.owned_objectives.astype(np.int16) * objective_payouts[None, None, :],
        axis=2,
    )
    return (engine.cash + engine.investment_values - engine.loan_principal + objectives).astype(
        np.int32
    )


def _apply_resolve_rewards(
    engine: BatchSimEngine,
    active_rows: np.ndarray,
    action_ids: np.ndarray,
    winners: np.ndarray,
    objectives_before: np.ndarray,
    previous_potential: np.ndarray,
    current_potential: np.ndarray,
    pending_rewards: list[list[RewardBreakdown]],
    config: RewardConfig,
) -> None:
    bonuses = dict(config.event_bonuses)
    starting_cash = STARTING_CASH[engine.player_count]
    for row_value in active_rows:
        row = int(row_value)
        winner = int(winners[row])
        action = int(action_ids[row])
        objective_claimed = bool(
            np.any(engine.owned_objectives[row, winner] & ~objectives_before[row, winner])
        )
        shaping = bonuses.get("auction_resolved", 0.0)
        if action in _AUCTION_ACTION_IDS:
            shaping += bonuses.get("resources_awarded", 0.0)
        elif action in _LOAN_ACTION_IDS:
            shaping += bonuses.get("loan_acquired", 0.0)
        elif action in _INVESTMENT_ACTION_IDS:
            shaping += bonuses.get("investment_acquired", 0.0)
        if objective_claimed:
            shaping += bonuses.get("objective_claimed", 0.0)
        for seat in range(engine.player_count):
            accounting = config.accounting_weight * (
                (int(current_potential[row, seat]) - int(previous_potential[row, seat]))
                / starting_cash
            )
            pending_rewards[row][seat] = _add_breakdowns(
                pending_rewards[row][seat],
                RewardBreakdown(
                    accounting=accounting,
                    shaping=shaping if seat == winner else 0.0,
                ),
            )


def _apply_terminal_rewards(
    engine: BatchSimEngine,
    results: tuple[SessionResult, ...],
    previous_potential: np.ndarray,
    pending_rewards: list[list[RewardBreakdown]],
    config: RewardConfig,
) -> None:
    starting_cash = STARTING_CASH[engine.player_count]
    for row, result in enumerate(results):
        first_place = tuple(score for score in result.scores if score.rank == 1)
        for score in result.scores:
            placement = config.win_bonus / len(first_place) if score.rank == 1 else 0.0
            if score.rank <= len(config.placement_bonuses):
                placement += config.placement_bonuses[score.rank - 1]
            terminal_resource = config.accounting_weight * (
                (score.final_money - int(previous_potential[row, score.seat])) / starting_cash
            )
            pending_rewards[row][score.seat] = _add_breakdowns(
                pending_rewards[row][score.seat],
                RewardBreakdown(
                    terminal_resource=terminal_resource,
                    placement=placement,
                ),
            )


def _results(engine: BatchSimEngine) -> tuple[SessionResult, ...]:
    scores = engine.scores()
    rankings = engine.rankings()
    output = []
    for row in range(engine.batch_size):
        totals = tuple(int(value) for value in scores.total[row])
        rows = tuple(
            ScoreRow(
                seat=seat,
                name=f"seat-{seat}",
                cash=int(scores.cash[row, seat]),
                items_value=int(scores.items[row, seat]),
                objectives_value=int(scores.objectives[row, seat]),
                investments_value=int(scores.investments[row, seat]),
                loans_value=int(scores.loans[row, seat]),
                total=totals[seat],
            )
            for seat in range(engine.player_count)
        )
        output.append(
            SessionResult(
                scores=tuple(
                    SessionScore(
                        seat=seat,
                        final_money=totals[seat],
                        rank=1 + sum(other > totals[seat] for other in totals),
                    )
                    for seat in range(engine.player_count)
                ),
                rows=rows,
                ranking=tuple(int(seat) for seat in rankings[row]),
            )
        )
    return tuple(output)


def _finalize_if_open(
    row: int,
    seat: int,
    *,
    plans: tuple[SelfPlayEpisodePlan, ...],
    opened: dict[tuple[int, int], _OpenTransition],
    pending_rewards: list[list[RewardBreakdown]],
    completed: list[list[list[RolloutTransition]]],
    terminated: bool,
) -> None:
    item = opened.pop((row, seat), None)
    if item is None:
        return
    plan = plans[row]
    assignment = plan.seat_policies[seat]
    breakdown = pending_rewards[row][seat]
    pending_rewards[row][seat] = RewardBreakdown()
    completed[row][seat].append(
        RolloutTransition(
            observation=item.request.observation,
            context=item.context,
            action=item.response.action,
            old_log_probability=item.response.old_log_probability,
            old_value=item.response.old_value,
            reward=breakdown.total,
            reward_breakdown=breakdown,
            terminated=terminated,
            truncated=False,
            bid_logits=(),
            reveal_logits=(),
            masked_logits=(),
            illegal_probability=0.0,
            metadata=RolloutMetadata(
                ruleset_name=plan.ruleset_name,
                player_count=plan.player_count,
                learner_seat=seat,
                opponent_names=tuple(
                    opponent.identity
                    for opponent_seat, opponent in enumerate(plan.seat_policies)
                    if opponent_seat != seat
                ),
                environment_seed=plan.engine_seed,
                opponent_seed=plan.engine_seed,
                policy_seed=plan.seat_sampling_seeds[seat],
            ),
        )
    )
    if assignment.identity != item.request.policy_identity:
        raise RuntimeError("policy assignment changed during vector collection")


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
