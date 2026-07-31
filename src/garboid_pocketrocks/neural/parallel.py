"""Spawned simulation workers with one central batched neural policy."""

from __future__ import annotations

import multiprocessing
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from contextlib import suppress
from multiprocessing.connection import Connection, wait
from multiprocessing.process import BaseProcess
from typing import cast

import torch

from garboid_pocketrocks.neural.collector import (
    CollectorMetrics,
    _freeze_policies,
    _infer_policy_requests,
    _percentile,
    _restore_policy_modes,
    _validate_collection,
)
from garboid_pocketrocks.neural.config import NeuralEncoderConfig
from garboid_pocketrocks.neural.model import NeuralPolicy
from garboid_pocketrocks.neural.planning import SelfPlayEpisodePlan
from garboid_pocketrocks.neural.rollout import (
    MultiSeatEpisode,
    RolloutBatch,
)
from garboid_pocketrocks.neural.self_play import (
    PolicyResponse,
)
from garboid_pocketrocks.neural.worker import (
    WorkerEpisodes,
    WorkerFailure,
    WorkerInferenceBatch,
    WorkerResponseBatch,
    run_plan_shard,
)
from garboid_pocketrocks.training.rewards import RewardConfig


class ParallelCollectionError(RuntimeError):
    """Raised when a worker pool cannot return a complete rollout."""


def collect_self_play_parallel(
    policies: Mapping[str, NeuralPolicy],
    plans: Sequence[SelfPlayEpisodePlan],
    *,
    encoder_config: NeuralEncoderConfig,
    reward_config: RewardConfig,
    device: torch.device,
    workers: int,
    active_games_per_worker: int,
    max_inference_batch: int,
    max_queue_delay_ms: float,
) -> tuple[RolloutBatch, CollectorMetrics]:
    """Collect games in spawned processes while inference stays central."""

    collected_plans = tuple(plans)
    if not isinstance(workers, int) or isinstance(workers, bool) or workers <= 0:
        raise ParallelCollectionError("workers must be a positive integer")
    if max_queue_delay_ms < 0:
        raise ParallelCollectionError("max_queue_delay_ms must be nonnegative")
    _validate_collection(
        policies,
        collected_plans,
        encoder_config=encoder_config,
        device=device,
        active_games=active_games_per_worker,
        max_inference_batch=max_inference_batch,
    )
    worker_count = min(workers, len(collected_plans))
    shards = tuple(tuple(collected_plans[index::worker_count]) for index in range(worker_count))
    parents, processes = _spawn_plan_workers(
        shards,
        encoder_config=encoder_config,
        reward_config=reward_config,
        active_games_per_worker=active_games_per_worker,
    )

    start = time.perf_counter()
    prior_modes = _freeze_policies(policies)
    open_connections = set(parents)
    completed: list[MultiSeatEpisode] = []
    inference_sizes: list[int] = []
    decisions = 0
    inference_seconds = 0.0
    ipc_seconds = 0.0
    queue_wait_seconds = 0.0
    worker_busy_seconds = 0.0
    failed = True
    try:
        while open_connections:
            messages, ready_wait_seconds, receive_ipc_seconds = _receive_ready_messages(
                open_connections,
                max_queue_delay_ms=max_queue_delay_ms,
            )
            queue_wait_seconds += ready_wait_seconds
            ipc_seconds += receive_ipc_seconds

            inference_messages: list[tuple[Connection, WorkerInferenceBatch]] = []
            for connection, message in messages:
                if isinstance(message, WorkerFailure):
                    raise ParallelCollectionError(
                        f"worker {message.worker_id} failed: {message.message}\n"
                        f"{message.traceback_text}"
                    )
                if isinstance(message, WorkerEpisodes):
                    completed.extend(message.episodes)
                    worker_busy_seconds += message.busy_seconds
                    open_connections.remove(connection)
                    connection.close()
                elif isinstance(message, WorkerInferenceBatch):
                    inference_messages.append((connection, message))
                else:
                    raise ParallelCollectionError("worker sent an unknown message")

            if inference_messages:
                requests = tuple(
                    request for _, message in inference_messages for request in message.requests
                )
                decisions += len(requests)
                responses, request_inference_seconds = _infer_policy_requests(
                    policies,
                    requests,
                    device=device,
                    max_inference_batch=max_inference_batch,
                    inference_batch_sizes=inference_sizes,
                )
                inference_seconds += request_inference_seconds
                ipc_seconds += _send_worker_responses(
                    inference_messages,
                    responses,
                )
        failed = False
    finally:
        _restore_policy_modes(prior_modes)
        _shutdown_plan_workers(
            parents,
            processes,
            terminate=failed,
        )

    if any(child_process.exitcode != 0 for child_process in processes):
        raise ParallelCollectionError("one or more workers exited unsuccessfully")
    return _build_parallel_metrics(
        completed,
        decisions=decisions,
        elapsed_seconds=time.perf_counter() - start,
        inference_seconds=inference_seconds,
        inference_sizes=inference_sizes,
        queue_wait_seconds=queue_wait_seconds,
        ipc_seconds=ipc_seconds,
        worker_busy_seconds=worker_busy_seconds,
    )


def _spawn_plan_workers(
    shards: Sequence[tuple[SelfPlayEpisodePlan, ...]],
    *,
    encoder_config: NeuralEncoderConfig,
    reward_config: RewardConfig,
    active_games_per_worker: int,
) -> tuple[tuple[Connection, ...], tuple[BaseProcess, ...]]:
    context = multiprocessing.get_context("spawn")
    parents: list[Connection] = []
    processes: list[BaseProcess] = []
    for worker_id, shard in enumerate(shards):
        parent: Connection | None = None
        child: Connection | None = None
        try:
            parent, child = context.Pipe()
            created_process = context.Process(
                target=run_plan_shard,
                args=(
                    child,
                    worker_id,
                    shard,
                    encoder_config,
                    reward_config,
                    active_games_per_worker,
                ),
                name=f"self-play-{worker_id}",
            )
            created_process.start()
        except BaseException:
            if child is not None:
                with suppress(BaseException):
                    child.close()
            if parent is not None:
                with suppress(BaseException):
                    parent.close()
            with suppress(BaseException):
                _shutdown_plan_workers(parents, processes, terminate=True)
            raise
        child.close()
        parents.append(parent)
        processes.append(created_process)
    return tuple(parents), tuple(processes)


def _receive_ready_messages(
    open_connections: set[Connection],
    *,
    max_queue_delay_ms: float,
) -> tuple[tuple[tuple[Connection, object], ...], float, float]:
    wait_start = time.perf_counter()
    ready = cast(set[Connection], set(wait(open_connections)))
    queue_wait_seconds = time.perf_counter() - wait_start
    deadline = time.perf_counter() + max_queue_delay_ms / 1_000.0
    while len(ready) < len(open_connections):
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break
        extra = cast(
            set[Connection],
            set(wait(open_connections - ready, timeout=remaining)),
        )
        if not extra:
            break
        ready.update(extra)

    messages: list[tuple[Connection, object]] = []
    ipc_seconds = 0.0
    for connection in ready:
        ipc_start = time.perf_counter()
        try:
            message = connection.recv()
        except EOFError as error:
            raise ParallelCollectionError("worker exited without a completion message") from error
        ipc_seconds += time.perf_counter() - ipc_start
        messages.append((connection, message))
    return tuple(messages), queue_wait_seconds, ipc_seconds


def _send_worker_responses(
    inference_messages: Sequence[tuple[Connection, WorkerInferenceBatch]],
    responses: Sequence[PolicyResponse],
) -> float:
    by_worker: dict[int, list[PolicyResponse]] = defaultdict(list)
    request_workers = {
        (
            request.episode_index,
            request.seat,
            request.decision_index,
        ): message.worker_id
        for _, message in inference_messages
        for request in message.requests
    }
    for response in responses:
        key = (
            response.episode_index,
            response.seat,
            response.decision_index,
        )
        by_worker[request_workers[key]].append(response)

    ipc_seconds = 0.0
    for connection, message in inference_messages:
        ipc_start = time.perf_counter()
        connection.send(
            WorkerResponseBatch(
                message.worker_id,
                message.sequence,
                tuple(by_worker[message.worker_id]),
            )
        )
        ipc_seconds += time.perf_counter() - ipc_start
    return ipc_seconds


def _shutdown_plan_workers(
    parents: Sequence[Connection],
    processes: Sequence[BaseProcess],
    *,
    terminate: bool,
) -> None:
    if terminate:
        for child_process in processes:
            if child_process.is_alive():
                child_process.terminate()
    for child_process in processes:
        child_process.join(timeout=5.0)
        if child_process.is_alive():
            child_process.kill()
            child_process.join()
    for connection in parents:
        connection.close()


def _build_parallel_metrics(
    completed: list[MultiSeatEpisode],
    *,
    decisions: int,
    elapsed_seconds: float,
    inference_seconds: float,
    inference_sizes: Sequence[int],
    queue_wait_seconds: float,
    ipc_seconds: float,
    worker_busy_seconds: float,
) -> tuple[RolloutBatch, CollectorMetrics]:
    completed.sort(key=lambda item: item.plan.episode_index)
    cells = Counter((item.plan.ruleset_name, item.plan.player_count) for item in completed)
    metrics = CollectorMetrics(
        games=len(completed),
        decisions=decisions,
        elapsed_seconds=elapsed_seconds,
        inference_seconds=inference_seconds,
        inference_batches=len(inference_sizes),
        inference_batch_sizes=tuple(inference_sizes),
        cell_games=tuple(
            (ruleset, count, games) for (ruleset, count), games in sorted(cells.items())
        ),
        queue_wait_seconds=queue_wait_seconds,
        ipc_seconds=ipc_seconds,
        worker_busy_seconds=worker_busy_seconds,
        inference_batch_p50=_percentile(inference_sizes, 0.50),
        inference_batch_p95=_percentile(inference_sizes, 0.95),
    )
    return RolloutBatch.from_multi_seat(completed), metrics
