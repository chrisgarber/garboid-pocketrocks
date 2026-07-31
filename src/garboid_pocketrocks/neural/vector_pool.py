"""Persistent spawned CPU vector actors for update-boundary self-play."""

from __future__ import annotations

import multiprocessing
import subprocess
import threading
import time
import traceback
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from multiprocessing.process import BaseProcess
from types import TracebackType
from typing import Self, cast

import torch

from garboid_pocketrocks.neural.collector import (
    CollectorMetrics,
    _percentile,
    _validate_collection,
)
from garboid_pocketrocks.neural.config import NeuralEncoderConfig
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


class VectorPoolError(RuntimeError):
    """Raised when a persistent vector actor pool cannot complete a request."""


@dataclass(frozen=True, slots=True)
class _CollectCommand:
    request_id: int
    plans: tuple[SelfPlayEpisodePlan, ...]
    snapshots: tuple[PolicySnapshot, ...]


@dataclass(frozen=True, slots=True)
class _CloseCommand:
    pass


@dataclass(frozen=True, slots=True)
class _WorkerResult:
    request_id: int
    worker_id: int
    episodes: tuple[MultiSeatEpisode, ...]
    metrics: CollectorMetrics


@dataclass(frozen=True, slots=True)
class _WorkerFailure:
    request_id: int
    worker_id: int
    message: str
    traceback_text: str


@dataclass(frozen=True, slots=True)
class _WorkerReady:
    worker_id: int


class VectorActorPool:
    """Own reusable CPU actors and collect one frozen-policy update at a time."""

    def __init__(
        self,
        *,
        encoder_config: NeuralEncoderConfig,
        reward_config: RewardConfig,
        workers: int,
        engine_batch_size: int = 128,
        max_inference_batch: int = 1024,
        expected_repository_commit: str | None = None,
    ) -> None:
        self._require_positive_integer("workers", workers)
        self._require_positive_integer("engine_batch_size", engine_batch_size)
        self._require_positive_integer("max_inference_batch", max_inference_batch)
        self._encoder_config = encoder_config
        self._reward_config = reward_config
        self._worker_count = workers
        self._engine_batch_size = engine_batch_size
        self._max_inference_batch = max_inference_batch
        self._connections: list[Connection] = []
        self._processes: list[BaseProcess] = []
        self._collect_lock = threading.Lock()
        self._next_request_id = 0
        self._closed = False

        context = multiprocessing.get_context("spawn")
        try:
            for worker_id in range(workers):
                parent, child = context.Pipe()
                process: BaseProcess = context.Process(
                    target=_run_vector_actor,
                    args=(
                        child,
                        worker_id,
                        encoder_config,
                        reward_config,
                        engine_batch_size,
                        max_inference_batch,
                        expected_repository_commit,
                    ),
                    name=f"persistent-vector-self-play-{worker_id}",
                )
                try:
                    process.start()
                except BaseException:
                    parent.close()
                    child.close()
                    raise
                child.close()
                self._connections.append(parent)
                self._processes.append(process)
            self._await_worker_startup()
        except BaseException:
            self._shutdown()
            raise

    @property
    def closed(self) -> bool:
        """Whether the actor pool has permanently stopped."""

        return self._closed

    @property
    def worker_pids(self) -> tuple[int, ...]:
        """Operating-system process IDs, stable for the pool's lifetime."""

        return tuple(cast(int, process.pid) for process in self._processes)

    def __enter__(self) -> Self:
        self._ensure_open()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        exception_traceback: TracebackType | None,
    ) -> None:
        del exception_type, exception, exception_traceback
        self.close()

    def collect(
        self,
        policies: Mapping[str, NeuralPolicy],
        plans: Sequence[SelfPlayEpisodePlan],
    ) -> tuple[RolloutBatch, CollectorMetrics]:
        """Collect one complete update using a new exact policy snapshot."""

        self._ensure_open()
        if not self._collect_lock.acquire(blocking=False):
            raise VectorPoolError("vector actor pool is already collecting")
        dispatched = False
        try:
            self._ensure_open()
            request_id, shards, snapshots = self._prepare_collect_command(
                policies,
                plans,
            )
            started = time.perf_counter()
            dispatched = True
            pending, dispatch_ipc_seconds = self._dispatch_collect_commands(
                request_id=request_id,
                shards=shards,
                snapshots=snapshots,
            )
            results, queue_wait_seconds, receive_ipc_seconds = _receive_collect_results(
                pending,
                request_id=request_id,
            )

            return self._combine_results(
                results,
                elapsed_seconds=time.perf_counter() - started,
                queue_wait_seconds=queue_wait_seconds,
                ipc_seconds=dispatch_ipc_seconds + receive_ipc_seconds,
            )
        except BaseException:
            if dispatched:
                self._shutdown()
            raise
        finally:
            self._collect_lock.release()

    def close(self) -> None:
        """Stop every actor and permanently close the pool."""

        with self._collect_lock:
            self._shutdown()

    def _shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        for connection, process in zip(
            self._connections,
            self._processes,
            strict=True,
        ):
            if process.is_alive():
                try:
                    connection.send(_CloseCommand())
                except BrokenPipeError, EOFError, OSError:
                    pass
            connection.close()
        for process in self._processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
            if process.is_alive():
                process.kill()
                process.join()

    def _ensure_open(self) -> None:
        if self._closed:
            raise VectorPoolError("vector actor pool is closed")

    def _prepare_collect_command(
        self,
        policies: Mapping[str, NeuralPolicy],
        plans: Sequence[SelfPlayEpisodePlan],
    ) -> tuple[
        int,
        tuple[tuple[SelfPlayEpisodePlan, ...], ...],
        tuple[PolicySnapshot, ...],
    ]:
        collected_plans = tuple(plans)
        _validate_collection(
            policies,
            collected_plans,
            encoder_config=self._encoder_config,
            device=torch.device("cpu"),
            active_games=self._engine_batch_size,
            max_inference_batch=self._max_inference_batch,
        )
        units = vector_plan_batches(
            collected_plans,
            batch_size=self._engine_batch_size,
        )
        active_workers = min(self._worker_count, len(units))
        shards = tuple(
            tuple(
                plan
                for unit_index, unit in enumerate(units)
                if unit_index % active_workers == worker_id
                for plan in unit
            )
            for worker_id in range(active_workers)
        )
        snapshots = snapshot_policies(policies)
        request_id = self._next_request_id
        self._next_request_id += 1
        return request_id, shards, snapshots

    def _dispatch_collect_commands(
        self,
        *,
        request_id: int,
        shards: Sequence[tuple[SelfPlayEpisodePlan, ...]],
        snapshots: tuple[PolicySnapshot, ...],
    ) -> tuple[dict[Connection, int], float]:
        pending: dict[Connection, int] = {}
        ipc_seconds = 0.0
        for worker_id, shard in enumerate(shards):
            process = self._processes[worker_id]
            if not process.is_alive():
                raise VectorPoolError(
                    f"vector actor {worker_id} exited before request {request_id}"
                )
            connection = self._connections[worker_id]
            ipc_started = time.perf_counter()
            connection.send(
                _CollectCommand(
                    request_id=request_id,
                    plans=shard,
                    snapshots=snapshots,
                )
            )
            ipc_seconds += time.perf_counter() - ipc_started
            pending[connection] = worker_id
        return pending, ipc_seconds

    def _await_worker_startup(self) -> None:
        pending = {connection: worker_id for worker_id, connection in enumerate(self._connections)}
        while pending:
            ready = wait(pending, timeout=30.0)
            if not ready:
                waiting_for = ", ".join(str(worker_id) for worker_id in pending.values())
                raise VectorPoolError(
                    f"vector actors {waiting_for} did not attest their source at startup"
                )
            for connection in ready:
                worker_id = pending.pop(cast(Connection, connection))
                try:
                    message = cast(Connection, connection).recv()
                except EOFError as error:
                    raise VectorPoolError(
                        f"vector actor {worker_id} exited before source attestation"
                    ) from error
                if isinstance(message, _WorkerFailure):
                    raise VectorPoolError(
                        f"vector actor {message.worker_id} failed source attestation: "
                        f"{message.message}\n{message.traceback_text}"
                    )
                if not isinstance(message, _WorkerReady) or message.worker_id != worker_id:
                    raise VectorPoolError(
                        f"vector actor {worker_id} returned invalid source attestation"
                    )

    @staticmethod
    def _require_positive_integer(name: str, value: object) -> None:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise VectorPoolError(f"{name} must be a positive integer")

    @staticmethod
    def _combine_results(
        results: Sequence[_WorkerResult],
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
        cells = Counter(
            (episode.plan.ruleset_name, episode.plan.player_count) for episode in episodes
        )
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


def _receive_collect_results(
    pending: dict[Connection, int],
    *,
    request_id: int,
) -> tuple[tuple[_WorkerResult, ...], float, float]:
    results: list[_WorkerResult] = []
    queue_wait_seconds = 0.0
    ipc_seconds = 0.0
    while pending:
        wait_started = time.perf_counter()
        ready = [cast(Connection, connection) for connection in wait(pending)]
        queue_wait_seconds += time.perf_counter() - wait_started
        for connection in ready:
            worker_id = pending.pop(connection)
            ipc_started = time.perf_counter()
            try:
                message = connection.recv()
            except EOFError as error:
                raise VectorPoolError(
                    f"vector actor {worker_id} exited without a result for request {request_id}"
                ) from error
            ipc_seconds += time.perf_counter() - ipc_started
            results.append(
                _validate_worker_result(
                    message,
                    worker_id=worker_id,
                    request_id=request_id,
                )
            )
    return tuple(results), queue_wait_seconds, ipc_seconds


def _validate_worker_result(
    message: object,
    *,
    worker_id: int,
    request_id: int,
) -> _WorkerResult:
    if isinstance(message, _WorkerFailure):
        raise VectorPoolError(
            f"vector actor {message.worker_id} failed: {message.message}\n{message.traceback_text}"
        )
    if not isinstance(message, _WorkerResult):
        raise VectorPoolError(f"vector actor {worker_id} returned an unknown message")
    if message.request_id != request_id or message.worker_id != worker_id:
        raise VectorPoolError(f"vector actor {worker_id} returned a stale result")
    return message


def _run_vector_actor(
    connection: Connection,
    worker_id: int,
    encoder_config: NeuralEncoderConfig,
    reward_config: RewardConfig,
    engine_batch_size: int,
    max_inference_batch: int,
    expected_repository_commit: str | None,
) -> None:
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        try:
            _require_expected_repository_source(expected_repository_commit)
        except BaseException as error:
            connection.send(
                _WorkerFailure(
                    request_id=-1,
                    worker_id=worker_id,
                    message=str(error),
                    traceback_text=traceback.format_exc(),
                )
            )
            return
        connection.send(_WorkerReady(worker_id=worker_id))
        while True:
            try:
                command = connection.recv()
            except EOFError:
                return
            if isinstance(command, _CloseCommand):
                return
            if not isinstance(command, _CollectCommand):
                connection.send(
                    _WorkerFailure(
                        request_id=-1,
                        worker_id=worker_id,
                        message="vector actor received an unknown command",
                        traceback_text="",
                    )
                )
                continue
            try:
                policies = load_policy_snapshots(command.snapshots)
                rollout, metrics = collect_self_play_vectorized(
                    policies,
                    command.plans,
                    encoder_config=encoder_config,
                    reward_config=reward_config,
                    device=torch.device("cpu"),
                    engine_batch_size=engine_batch_size,
                    max_inference_batch=max_inference_batch,
                )
                connection.send(
                    _WorkerResult(
                        request_id=command.request_id,
                        worker_id=worker_id,
                        episodes=rollout.episodes,
                        metrics=metrics,
                    )
                )
            except BaseException as error:
                connection.send(
                    _WorkerFailure(
                        request_id=command.request_id,
                        worker_id=worker_id,
                        message=str(error),
                        traceback_text=traceback.format_exc(),
                    )
                )
    finally:
        connection.close()


def _require_expected_repository_source(expected_repository_commit: str | None) -> None:
    """Require an official worker to start from one clean, stable Git commit."""

    if expected_repository_commit is None:
        return
    repository_commit_before_status = _repository_commit()
    dirty = _repository_status()
    repository_commit_after_status = _repository_commit()
    if (
        repository_commit_before_status != expected_repository_commit
        or repository_commit_after_status != expected_repository_commit
        or dirty
    ):
        raise VectorPoolError(
            "official vector actor source does not match the expected clean repository commit"
        )


def _repository_commit() -> str:
    try:
        return subprocess.run(
            ("git", "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise VectorPoolError("vector actor requires readable Git provenance") from error


def _repository_status() -> str:
    try:
        return subprocess.run(
            ("git", "status", "--porcelain"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise VectorPoolError("vector actor requires readable Git provenance") from error
