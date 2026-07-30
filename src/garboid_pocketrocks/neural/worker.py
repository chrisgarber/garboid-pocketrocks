"""Spawn-safe game-only workers for central neural inference."""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from multiprocessing.connection import Connection

from garboid_pocketrocks.neural.config import NeuralEncoderConfig
from garboid_pocketrocks.neural.planning import SelfPlayEpisodePlan
from garboid_pocketrocks.neural.rollout import MultiSeatEpisode
from garboid_pocketrocks.neural.self_play import (
    PendingPolicyRequest,
    PolicyResponse,
    SelfPlayGame,
)
from garboid_pocketrocks.training.rewards import RewardConfig


@dataclass(frozen=True, slots=True)
class WorkerInferenceBatch:
    worker_id: int
    sequence: int
    requests: tuple[PendingPolicyRequest, ...]


@dataclass(frozen=True, slots=True)
class WorkerResponseBatch:
    worker_id: int
    sequence: int
    responses: tuple[PolicyResponse, ...]


@dataclass(frozen=True, slots=True)
class WorkerEpisodes:
    worker_id: int
    episodes: tuple[MultiSeatEpisode, ...]
    busy_seconds: float


@dataclass(frozen=True, slots=True)
class WorkerFailure:
    worker_id: int
    message: str
    traceback_text: str


def run_plan_shard(
    connection: Connection,
    worker_id: int,
    plans: tuple[SelfPlayEpisodePlan, ...],
    encoder_config: NeuralEncoderConfig,
    reward_config: RewardConfig,
    active_games: int,
) -> None:
    """Advance a shard while sending every neural decision to the parent."""

    started = time.perf_counter()
    try:
        remaining = iter(plans)
        active: dict[int, SelfPlayGame] = {}
        completed: list[MultiSeatEpisode] = []

        def fill() -> None:
            while len(active) < active_games:
                try:
                    plan = next(remaining)
                except StopIteration:
                    break
                active[plan.episode_index] = SelfPlayGame.start(
                    plan,
                    encoder_config=encoder_config,
                    reward_config=reward_config,
                )

        fill()
        sequence = 0
        while active:
            requests = tuple(
                sorted(
                    (request for game in active.values() for request in game.pending_requests()),
                    key=lambda item: (
                        item.episode_index,
                        item.seat,
                        item.decision_index,
                    ),
                )
            )
            connection.send(WorkerInferenceBatch(worker_id, sequence, requests))
            response = connection.recv()
            if (
                not isinstance(response, WorkerResponseBatch)
                or response.worker_id != worker_id
                or response.sequence != sequence
            ):
                raise RuntimeError("central inference response is invalid")
            responses_by_episode: dict[int, list[PolicyResponse]] = {}
            for item in response.responses:
                responses_by_episode.setdefault(item.episode_index, []).append(item)
            for episode_index in tuple(sorted(active)):
                game = active[episode_index]
                game.apply(responses_by_episode[episode_index])
                if game.terminated:
                    completed.append(game.episode())
                    del active[episode_index]
            fill()
            sequence += 1
        connection.send(
            WorkerEpisodes(
                worker_id,
                tuple(completed),
                time.perf_counter() - started,
            )
        )
    except BaseException as error:
        connection.send(
            WorkerFailure(
                worker_id,
                str(error),
                traceback.format_exc(),
            )
        )
    finally:
        connection.close()
