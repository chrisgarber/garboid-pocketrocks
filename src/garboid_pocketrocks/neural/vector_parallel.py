"""Spawned 64+ game SDK vector actors with local frozen inference."""

from __future__ import annotations

import multiprocessing
import time
import traceback
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from multiprocessing.process import BaseProcess
from typing import cast

import torch

from garboid_pocketrocks.neural.collector import (
    CollectorMetrics,
    _percentile,
    _validate_collection,
)
from garboid_pocketrocks.neural.config import (
    NeuralEncoderConfig,
)
from garboid_pocketrocks.neural.model import NeuralPolicy
from garboid_pocketrocks.neural.planning import SelfPlayEpisodePlan
from garboid_pocketrocks.neural.policy_snapshot import (
    PolicySnapshot,
    load_policy_snapshots,
    snapshot_policies,
)
from garboid_pocketrocks.neural.rollout import MultiSeatEpisode, RolloutBatch
from garboid_pocketrocks.neural.vector_collector import (
    collect_self_play_vectorized,
    vector_plan_batches,
)
from garboid_pocketrocks.training.rewards import RewardConfig


class VectorParallelError(RuntimeError):
    """Raised when a local vector actor cannot return a complete shard."""


@dataclass(frozen=True, slots=True)
class _VectorWorkerResult:
    worker_id: int
    episodes: tuple[MultiSeatEpisode, ...]
    metrics: CollectorMetrics


@dataclass(frozen=True, slots=True)
class _VectorWorkerFailure:
    worker_id: int
    message: str
    traceback_text: str


def collect_self_play_vectorized_parallel(
    policies: Mapping[str, NeuralPolicy],
    plans: Sequence[SelfPlayEpisodePlan],
    *,
    encoder_config: NeuralEncoderConfig,
    reward_config: RewardConfig,
    workers: int,
    engine_batch_size: int = 128,
    max_inference_batch: int = 1024,
) -> tuple[RolloutBatch, CollectorMetrics]:
    """Collect vector-engine shards concurrently on CPU actor processes."""

    collected_plans = tuple(plans)
    if not isinstance(workers, int) or isinstance(workers, bool) or workers <= 0:
        raise VectorParallelError("workers must be a positive integer")
    _validate_collection(
        policies,
        collected_plans,
        encoder_config=encoder_config,
        device=torch.device("cpu"),
        active_games=engine_batch_size,
        max_inference_batch=max_inference_batch,
    )
    shards = _shard_vector_plans(
        collected_plans,
        workers=workers,
        engine_batch_size=engine_batch_size,
    )
    snapshots = snapshot_policies(policies)
    started = time.perf_counter()
    parents, processes = _spawn_vector_workers(
        shards,
        snapshots,
        encoder_config=encoder_config,
        reward_config=reward_config,
        engine_batch_size=engine_batch_size,
        max_inference_batch=max_inference_batch,
    )
    try:
        results, queue_wait_seconds, ipc_seconds = _receive_vector_results(parents)
    except BaseException:
        _shutdown_vector_workers(
            parents,
            processes,
            terminate=True,
        )
        raise
    _shutdown_vector_workers(
        parents,
        processes,
        terminate=False,
    )

    if any(process.exitcode != 0 for process in processes):
        raise VectorParallelError("one or more vector actors exited unsuccessfully")
    return _aggregate_vector_results(
        results,
        elapsed_seconds=time.perf_counter() - started,
        queue_wait_seconds=queue_wait_seconds,
        ipc_seconds=ipc_seconds,
    )


def _shard_vector_plans(
    plans: tuple[SelfPlayEpisodePlan, ...],
    *,
    workers: int,
    engine_batch_size: int,
) -> tuple[tuple[SelfPlayEpisodePlan, ...], ...]:
    units = vector_plan_batches(
        plans,
        batch_size=engine_batch_size,
    )
    worker_count = min(workers, len(units))
    return tuple(
        tuple(
            plan
            for unit_index, unit in enumerate(units)
            if unit_index % worker_count == worker_id
            for plan in unit
        )
        for worker_id in range(worker_count)
    )


def _spawn_vector_workers(
    shards: Sequence[tuple[SelfPlayEpisodePlan, ...]],
    snapshots: tuple[PolicySnapshot, ...],
    *,
    encoder_config: NeuralEncoderConfig,
    reward_config: RewardConfig,
    engine_batch_size: int,
    max_inference_batch: int,
) -> tuple[tuple[Connection, ...], tuple[BaseProcess, ...]]:
    context = multiprocessing.get_context("spawn")
    parents: list[Connection] = []
    processes: list[BaseProcess] = []
    for worker_id, shard in enumerate(shards):
        parent: Connection | None = None
        child: Connection | None = None
        try:
            parent, child = context.Pipe()
            process: BaseProcess = context.Process(
                target=_run_vector_shard,
                args=(
                    child,
                    worker_id,
                    shard,
                    snapshots,
                    encoder_config,
                    reward_config,
                    engine_batch_size,
                    max_inference_batch,
                ),
                name=f"vector-self-play-{worker_id}",
            )
            process.start()
        except BaseException:
            if child is not None:
                with suppress(BaseException):
                    child.close()
            if parent is not None:
                with suppress(BaseException):
                    parent.close()
            with suppress(BaseException):
                _shutdown_vector_workers(parents, processes, terminate=True)
            raise
        child.close()
        parents.append(parent)
        processes.append(process)
    return tuple(parents), tuple(processes)


def _receive_vector_results(
    parents: Sequence[Connection],
) -> tuple[tuple[_VectorWorkerResult, ...], float, float]:
    open_connections = set(parents)
    results: list[_VectorWorkerResult] = []
    queue_wait_seconds = 0.0
    ipc_seconds = 0.0
    while open_connections:
        wait_started = time.perf_counter()
        ready = [cast(Connection, connection) for connection in wait(open_connections)]
        queue_wait_seconds += time.perf_counter() - wait_started
        for connection in ready:
            ipc_started = time.perf_counter()
            try:
                message = connection.recv()
            except EOFError as error:
                raise VectorParallelError("vector actor exited without a result") from error
            ipc_seconds += time.perf_counter() - ipc_started
            open_connections.remove(connection)
            connection.close()
            if isinstance(message, _VectorWorkerFailure):
                raise VectorParallelError(
                    f"vector actor {message.worker_id} failed: "
                    f"{message.message}\n{message.traceback_text}"
                )
            if not isinstance(message, _VectorWorkerResult):
                raise VectorParallelError("vector actor returned an unknown message")
            results.append(message)
    return tuple(results), queue_wait_seconds, ipc_seconds


def _shutdown_vector_workers(
    parents: Sequence[Connection],
    processes: Sequence[BaseProcess],
    *,
    terminate: bool,
) -> None:
    if terminate:
        for process in processes:
            if process.is_alive():
                process.terminate()
    for process in processes:
        process.join(timeout=5.0)
        if process.is_alive():
            process.kill()
            process.join()
    for connection in parents:
        connection.close()


def _aggregate_vector_results(
    results: Sequence[_VectorWorkerResult],
    *,
    elapsed_seconds: float,
    queue_wait_seconds: float,
    ipc_seconds: float,
) -> tuple[RolloutBatch, CollectorMetrics]:
    ordered_results = tuple(sorted(results, key=lambda result: result.worker_id))
    episodes = [episode for result in ordered_results for episode in result.episodes]
    episodes.sort(key=lambda episode: episode.plan.episode_index)
    inference_sizes = tuple(
        size for result in ordered_results for size in result.metrics.inference_batch_sizes
    )
    cells = Counter((episode.plan.ruleset_name, episode.plan.player_count) for episode in episodes)
    decisions = sum(result.metrics.decisions for result in ordered_results)
    metrics = CollectorMetrics(
        games=len(episodes),
        decisions=decisions,
        elapsed_seconds=elapsed_seconds,
        inference_seconds=sum(result.metrics.inference_seconds for result in ordered_results),
        inference_batches=len(inference_sizes),
        inference_batch_sizes=inference_sizes,
        cell_games=tuple(
            (ruleset, players, games) for (ruleset, players), games in sorted(cells.items())
        ),
        queue_wait_seconds=queue_wait_seconds,
        ipc_seconds=ipc_seconds,
        worker_busy_seconds=sum(result.metrics.elapsed_seconds for result in ordered_results),
        inference_batch_p50=_percentile(inference_sizes, 0.50),
        inference_batch_p95=_percentile(inference_sizes, 0.95),
    )
    return RolloutBatch.from_multi_seat(episodes), metrics


def _run_vector_shard(
    connection: Connection,
    worker_id: int,
    plans: tuple[SelfPlayEpisodePlan, ...],
    snapshots: tuple[PolicySnapshot, ...],
    encoder_config: NeuralEncoderConfig,
    reward_config: RewardConfig,
    engine_batch_size: int,
    max_inference_batch: int,
) -> None:
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        policies = load_policy_snapshots(snapshots)
        rollout, metrics = collect_self_play_vectorized(
            policies,
            plans,
            encoder_config=encoder_config,
            reward_config=reward_config,
            device=torch.device("cpu"),
            engine_batch_size=engine_batch_size,
            max_inference_batch=max_inference_batch,
        )
        connection.send(
            _VectorWorkerResult(
                worker_id=worker_id,
                episodes=rollout.episodes,
                metrics=metrics,
            )
        )
    except BaseException as error:
        connection.send(
            _VectorWorkerFailure(
                worker_id=worker_id,
                message=str(error),
                traceback_text=traceback.format_exc(),
            )
        )
    finally:
        connection.close()
