"""JSON-safe configuration for durable neural training runs."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, cast

from garboid_pocketrocks.neural.config import ModelProfile
from garboid_pocketrocks.neural.ppo import PPOConfig
from garboid_pocketrocks.training.rewards import RewardConfig

DeviceName = Literal["auto", "cpu", "cuda", "mps"]
WorkerSetting = int | Literal["auto"]


@dataclass(frozen=True, slots=True)
class ParallelConfig:
    """Resolved or calibratable parallel collection settings."""

    workers: WorkerSetting = "auto"
    active_games_per_worker: int = 128
    max_inference_batch: int = 1024
    max_queue_delay_ms: float = 1.0

    def __post_init__(self) -> None:
        if self.workers != "auto":
            _require_positive_int("workers", self.workers)
        _require_positive_int(
            "active_games_per_worker",
            self.active_games_per_worker,
        )
        _require_positive_int("max_inference_batch", self.max_inference_batch)
        _require_positive_number("max_queue_delay_ms", self.max_queue_delay_ms)


@dataclass(frozen=True, slots=True)
class TrainingRunConfig:
    """Complete immutable configuration for one training lineage."""

    root_seed: int = 42
    device: DeviceName = "auto"
    deterministic_algorithms: bool = True
    model_profile: ModelProfile = "small"
    learner_threads: int = 1
    games_per_cell: int | None = 100
    max_updates: int | None = None
    max_wall_seconds: float | None = None
    target_decisions_per_update: int | None = None
    checkpoint_interval_seconds: float | None = None
    evaluation_interval_seconds: float | None = None
    evaluation_games_per_seat_cell: int = 2
    evaluate_at_start: bool = False
    evaluate_at_end: bool = False
    league_fraction: float = 0.0
    keep_periodic_checkpoints: int = 4
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self) -> None:
        _require_int("root_seed", self.root_seed)
        if self.device not in ("auto", "cpu", "cuda", "mps"):
            raise ValueError("device must be auto, cpu, cuda, or mps")
        if not isinstance(self.deterministic_algorithms, bool):
            raise ValueError("deterministic_algorithms must be a boolean")
        if self.model_profile not in ("small", "medium", "large"):
            raise ValueError("model_profile must be small, medium, or large")
        _require_positive_int("learner_threads", self.learner_threads)
        if (self.games_per_cell is None) == (self.target_decisions_per_update is None):
            raise ValueError(
                "exactly one of games_per_cell or target_decisions_per_update is required"
            )
        if self.games_per_cell is not None:
            _require_positive_int("games_per_cell", self.games_per_cell)
        if self.target_decisions_per_update is not None:
            _require_positive_int(
                "target_decisions_per_update",
                self.target_decisions_per_update,
            )
        for name in ("max_updates",):
            value = getattr(self, name)
            if value is not None:
                _require_positive_int(name, value)
        for name in (
            "max_wall_seconds",
            "checkpoint_interval_seconds",
            "evaluation_interval_seconds",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_positive_number(name, value)
        _require_positive_int(
            "evaluation_games_per_seat_cell",
            self.evaluation_games_per_seat_cell,
        )
        if not isinstance(self.evaluate_at_start, bool) or not isinstance(
            self.evaluate_at_end,
            bool,
        ):
            raise ValueError("evaluation flags must be booleans")
        if (
            not isinstance(self.league_fraction, (int, float))
            or isinstance(self.league_fraction, bool)
            or not math.isfinite(float(self.league_fraction))
            or not 0.0 <= float(self.league_fraction) < 1.0
        ):
            raise ValueError("league_fraction must be finite from zero to one")
        _require_positive_int(
            "keep_periodic_checkpoints",
            self.keep_periodic_checkpoints,
        )
        if not isinstance(self.parallel, ParallelConfig):
            raise ValueError("parallel must be a ParallelConfig")
        if not isinstance(self.ppo, PPOConfig):
            raise ValueError("ppo must be a PPOConfig")
        if not isinstance(self.reward, RewardConfig):
            raise ValueError("reward must be a RewardConfig")

    @classmethod
    def from_json(cls, path: Path) -> TrainingRunConfig:
        """Load an exact-key configuration object from JSON."""

        try:
            parsed: object = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("training configuration JSON could not be read") from error
        payload = _object_dict(parsed, "training configuration")
        _reject_unknown(
            payload,
            {
                "root_seed",
                "device",
                "deterministic_algorithms",
                "model_profile",
                "learner_threads",
                "games_per_cell",
                "max_updates",
                "max_wall_seconds",
                "target_decisions_per_update",
                "checkpoint_interval_seconds",
                "evaluation_interval_seconds",
                "evaluation_games_per_seat_cell",
                "evaluate_at_start",
                "evaluate_at_end",
                "league_fraction",
                "keep_periodic_checkpoints",
                "parallel",
                "ppo",
                "reward",
            },
            "training configuration",
        )
        defaults = cls()
        return cls(
            root_seed=_int_value(
                payload.get("root_seed", defaults.root_seed),
                "root_seed",
            ),
            device=_device_value(payload.get("device", defaults.device)),
            deterministic_algorithms=_bool_value(
                payload.get(
                    "deterministic_algorithms",
                    defaults.deterministic_algorithms,
                ),
                "deterministic_algorithms",
            ),
            model_profile=_model_profile_value(
                payload.get("model_profile", defaults.model_profile)
            ),
            learner_threads=_int_value(
                payload.get("learner_threads", defaults.learner_threads),
                "learner_threads",
            ),
            games_per_cell=_optional_int_value(
                payload.get("games_per_cell", defaults.games_per_cell),
                "games_per_cell",
            ),
            max_updates=_optional_int_value(
                payload.get("max_updates", defaults.max_updates),
                "max_updates",
            ),
            max_wall_seconds=_optional_float_value(
                payload.get("max_wall_seconds", defaults.max_wall_seconds),
                "max_wall_seconds",
            ),
            target_decisions_per_update=_optional_int_value(
                payload.get(
                    "target_decisions_per_update",
                    defaults.target_decisions_per_update,
                ),
                "target_decisions_per_update",
            ),
            checkpoint_interval_seconds=_optional_float_value(
                payload.get(
                    "checkpoint_interval_seconds",
                    defaults.checkpoint_interval_seconds,
                ),
                "checkpoint_interval_seconds",
            ),
            evaluation_interval_seconds=_optional_float_value(
                payload.get(
                    "evaluation_interval_seconds",
                    defaults.evaluation_interval_seconds,
                ),
                "evaluation_interval_seconds",
            ),
            evaluation_games_per_seat_cell=_int_value(
                payload.get(
                    "evaluation_games_per_seat_cell",
                    defaults.evaluation_games_per_seat_cell,
                ),
                "evaluation_games_per_seat_cell",
            ),
            evaluate_at_start=_bool_value(
                payload.get("evaluate_at_start", defaults.evaluate_at_start),
                "evaluate_at_start",
            ),
            evaluate_at_end=_bool_value(
                payload.get("evaluate_at_end", defaults.evaluate_at_end),
                "evaluate_at_end",
            ),
            league_fraction=_float_value(
                payload.get("league_fraction", defaults.league_fraction),
                "league_fraction",
            ),
            keep_periodic_checkpoints=_int_value(
                payload.get(
                    "keep_periodic_checkpoints",
                    defaults.keep_periodic_checkpoints,
                ),
                "keep_periodic_checkpoints",
            ),
            parallel=_read_parallel(payload.get("parallel", defaults.parallel)),
            ppo=_read_ppo(payload.get("ppo", defaults.ppo)),
            reward=_read_reward(payload.get("reward", defaults.reward)),
        )

    def to_json_dict(self) -> dict[str, object]:
        """Return the complete configuration as JSON-compatible values."""

        serialized = json.loads(json.dumps(asdict(self), allow_nan=False, sort_keys=True))
        if not isinstance(serialized, dict):
            raise AssertionError("dataclass configuration must serialize to an object")
        return cast(dict[str, object], serialized)


def _reject_unknown(
    payload: dict[str, object],
    expected: set[str],
    name: str,
) -> None:
    unknown = set(payload) - expected
    if unknown:
        raise ValueError(f"{name} contains unknown fields: {sorted(unknown)!r}")


def _object_dict(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object")
    return cast(dict[str, object], dict(value))


def _object_sequence(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    return cast(list[object], value)


def _read_parallel(value: object) -> ParallelConfig:
    if isinstance(value, ParallelConfig):
        return value
    payload = _object_dict(value, "parallel")
    _reject_unknown(
        payload,
        {
            "workers",
            "active_games_per_worker",
            "max_inference_batch",
            "max_queue_delay_ms",
        },
        "parallel",
    )
    defaults = ParallelConfig()
    workers_value = payload.get("workers", defaults.workers)
    workers: WorkerSetting
    if workers_value == "auto":
        workers = "auto"
    else:
        workers = _int_value(workers_value, "workers")
    return ParallelConfig(
        workers=workers,
        active_games_per_worker=_int_value(
            payload.get(
                "active_games_per_worker",
                defaults.active_games_per_worker,
            ),
            "active_games_per_worker",
        ),
        max_inference_batch=_int_value(
            payload.get("max_inference_batch", defaults.max_inference_batch),
            "max_inference_batch",
        ),
        max_queue_delay_ms=_float_value(
            payload.get("max_queue_delay_ms", defaults.max_queue_delay_ms),
            "max_queue_delay_ms",
        ),
    )


def _read_ppo(value: object) -> PPOConfig:
    if isinstance(value, PPOConfig):
        return value
    payload = _object_dict(value, "ppo")
    _reject_unknown(
        payload,
        {
            "gamma",
            "gae_lambda",
            "clip_ratio",
            "value_loss_coefficient",
            "entropy_coefficient",
            "max_gradient_norm",
            "learning_rate",
            "epochs",
            "minibatch_size",
        },
        "ppo",
    )
    defaults = PPOConfig()
    return PPOConfig(
        gamma=_float_value(payload.get("gamma", defaults.gamma), "gamma"),
        gae_lambda=_float_value(
            payload.get("gae_lambda", defaults.gae_lambda),
            "gae_lambda",
        ),
        clip_ratio=_float_value(
            payload.get("clip_ratio", defaults.clip_ratio),
            "clip_ratio",
        ),
        value_loss_coefficient=_float_value(
            payload.get(
                "value_loss_coefficient",
                defaults.value_loss_coefficient,
            ),
            "value_loss_coefficient",
        ),
        entropy_coefficient=_float_value(
            payload.get("entropy_coefficient", defaults.entropy_coefficient),
            "entropy_coefficient",
        ),
        max_gradient_norm=_float_value(
            payload.get("max_gradient_norm", defaults.max_gradient_norm),
            "max_gradient_norm",
        ),
        learning_rate=_float_value(
            payload.get("learning_rate", defaults.learning_rate),
            "learning_rate",
        ),
        epochs=_int_value(payload.get("epochs", defaults.epochs), "epochs"),
        minibatch_size=_int_value(
            payload.get("minibatch_size", defaults.minibatch_size),
            "minibatch_size",
        ),
    )


def _read_reward(value: object) -> RewardConfig:
    if isinstance(value, RewardConfig):
        return value
    payload = _object_dict(value, "reward")
    _reject_unknown(
        payload,
        {
            "accounting_weight",
            "win_bonus",
            "placement_bonuses",
            "invalid_action_penalty",
            "event_bonuses",
        },
        "reward",
    )
    defaults = RewardConfig()
    placement_raw = payload.get(
        "placement_bonuses",
        list(defaults.placement_bonuses),
    )
    placement = tuple(
        _float_value(item, "placement bonus")
        for item in _object_sequence(placement_raw, "placement_bonuses")
    )
    events_raw = payload.get(
        "event_bonuses",
        [list(item) for item in defaults.event_bonuses],
    )
    bonuses: list[tuple[str, float]] = []
    for raw_bonus in _object_sequence(events_raw, "event_bonuses"):
        pair = _object_sequence(raw_bonus, "event bonus")
        if len(pair) != 2 or not isinstance(pair[0], str):
            raise ValueError("event bonuses must contain string/number pairs")
        bonuses.append((pair[0], _float_value(pair[1], "event bonus")))
    return RewardConfig(
        accounting_weight=_float_value(
            payload.get("accounting_weight", defaults.accounting_weight),
            "accounting_weight",
        ),
        win_bonus=_float_value(
            payload.get("win_bonus", defaults.win_bonus),
            "win_bonus",
        ),
        placement_bonuses=placement,
        invalid_action_penalty=_float_value(
            payload.get(
                "invalid_action_penalty",
                defaults.invalid_action_penalty,
            ),
            "invalid_action_penalty",
        ),
        event_bonuses=tuple(bonuses),
    )


def _device_value(value: object) -> DeviceName:
    if value not in ("auto", "cpu", "cuda", "mps"):
        raise ValueError("device must be auto, cpu, cuda, or mps")
    return value


def _model_profile_value(value: object) -> ModelProfile:
    if value not in ("small", "medium", "large"):
        raise ValueError("model_profile must be small, medium, or large")
    return value


def _int_value(value: object, name: str) -> int:
    _require_int(name, value)
    return cast(int, value)


def _optional_int_value(value: object, name: str) -> int | None:
    if value is None:
        return None
    return _int_value(value, name)


def _float_value(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    return float(value)


def _optional_float_value(value: object, name: str) -> float | None:
    if value is None:
        return None
    return _float_value(value, name)


def _bool_value(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _require_int(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")


def _require_positive_int(name: str, value: object) -> None:
    _require_int(name, value)
    assert isinstance(value, int)
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_positive_number(name: str, value: object) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive finite number")
