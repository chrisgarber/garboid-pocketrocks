"""Deterministic heuristic demonstrations and policy-only pretraining."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
from pocketrocks import BotDecision

from garboid_pocketrocks.adapters.public_history import public_history_from_sdk_events
from garboid_pocketrocks.knowledge import (
    canonical_knowledge,
    ruleset_name,
    value_chart_from_ruleset_name,
)
from garboid_pocketrocks.neural.config import NeuralEncoderConfig
from garboid_pocketrocks.neural.encoding import (
    NeuralObservation,
    NeuralObservationEncoder,
    batch_observations,
)
from garboid_pocketrocks.neural.heuristic_teachers import (
    BALANCED_V3_PROFILE_DIGEST as BALANCED_V3_PROFILE_DIGEST,
)
from garboid_pocketrocks.neural.heuristic_teachers import (
    BALANCED_V3_TEACHER_IDENTITY as BALANCED_V3_TEACHER_IDENTITY,
)
from garboid_pocketrocks.neural.heuristic_teachers import (
    build_released_v3_brain,
    released_v3_profile_digest,
)
from garboid_pocketrocks.neural.model import NeuralPolicy
from garboid_pocketrocks.neural.policy import evaluate_masked_policy
from garboid_pocketrocks.neural.seeding import derive_seed
from garboid_pocketrocks.simulator.session import SdkGameSession
from garboid_pocketrocks.training.actions import ActionCodec
from garboid_pocketrocks.training.bounds import EnvironmentBounds

_BEHAVIOR_CLONING_SEED_TAG = 14
BEHAVIOR_CLONING_OPTIMIZATION_ORDER = "sequential-shard-major-epochs-v1"


class BehaviorCloningError(ValueError):
    """Raised when demonstrations cannot satisfy the cloning contract."""


@dataclass(frozen=True, slots=True)
class BehaviorCloningConfig:
    """Fixed planning and optimization budget for one cloning stage.

    The optimizer is intentionally kept across sequential shards, so changing
    ``games_per_shard`` changes the update order and therefore the learned weights.
    """

    root_seed: int
    rounds: int
    games_per_cell: int
    epochs: int = 3
    minibatch_size: int = 512
    games_per_shard: int = 64
    optimization_order: str = BEHAVIOR_CLONING_OPTIMIZATION_ORDER
    learning_rate: float = 3e-4
    max_gradient_norm: float = 0.5
    teacher_identity: str = BALANCED_V3_TEACHER_IDENTITY
    teacher_profile_digest: str = BALANCED_V3_PROFILE_DIGEST

    def __post_init__(self) -> None:
        _require_int("root_seed", self.root_seed)
        for name in (
            "rounds",
            "games_per_cell",
            "epochs",
            "minibatch_size",
            "games_per_shard",
        ):
            _require_positive_int(name, getattr(self, name))
        for name in ("learning_rate", "max_gradient_norm"):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or value <= 0.0
            ):
                raise BehaviorCloningError(f"{name} must be finite and positive")
        if self.teacher_identity != BALANCED_V3_TEACHER_IDENTITY:
            raise BehaviorCloningError("behavior cloning teacher must be balanced-v3")
        if self.optimization_order != BEHAVIOR_CLONING_OPTIMIZATION_ORDER:
            raise BehaviorCloningError("behavior cloning optimization order is not pinned")
        if self.teacher_profile_digest != BALANCED_V3_PROFILE_DIGEST:
            raise BehaviorCloningError("balanced-v3 profile digest does not match the pin")

    @property
    def config_digest(self) -> str:
        """Return a stable identity for the complete cloning configuration."""

        return _json_digest(asdict(self))

    def to_json_dict(self) -> dict[str, object]:
        """Return the complete exact-key JSON representation."""

        return {
            "root_seed": self.root_seed,
            "rounds": self.rounds,
            "games_per_cell": self.games_per_cell,
            "epochs": self.epochs,
            "minibatch_size": self.minibatch_size,
            "games_per_shard": self.games_per_shard,
            "optimization_order": self.optimization_order,
            "learning_rate": float(self.learning_rate),
            "max_gradient_norm": float(self.max_gradient_norm),
            "teacher_identity": self.teacher_identity,
            "teacher_profile_digest": self.teacher_profile_digest,
        }

    @classmethod
    def from_json_dict(cls, value: object) -> BehaviorCloningConfig:
        """Read a complete nested config while rejecting schema drift."""

        payload = _json_object(value, "behavior cloning config")
        expected = {
            "root_seed",
            "rounds",
            "games_per_cell",
            "epochs",
            "minibatch_size",
            "games_per_shard",
            "optimization_order",
            "learning_rate",
            "max_gradient_norm",
            "teacher_identity",
            "teacher_profile_digest",
        }
        if set(payload) != expected:
            missing = sorted(expected - set(payload))
            extra = sorted(set(payload) - expected)
            raise BehaviorCloningError(
                "behavior cloning config keys do not match the schema: "
                f"missing={missing!r}, extra={extra!r}"
            )
        return cls(
            root_seed=_json_int(payload["root_seed"], "root_seed"),
            rounds=_json_int(payload["rounds"], "rounds"),
            games_per_cell=_json_int(payload["games_per_cell"], "games_per_cell"),
            epochs=_json_int(payload["epochs"], "epochs"),
            minibatch_size=_json_int(payload["minibatch_size"], "minibatch_size"),
            games_per_shard=_json_int(payload["games_per_shard"], "games_per_shard"),
            optimization_order=_json_string(payload["optimization_order"], "optimization_order"),
            learning_rate=_json_number(payload["learning_rate"], "learning_rate"),
            max_gradient_norm=_json_number(
                payload["max_gradient_norm"],
                "max_gradient_norm",
            ),
            teacher_identity=_json_string(
                payload["teacher_identity"],
                "teacher_identity",
            ),
            teacher_profile_digest=_json_string(
                payload["teacher_profile_digest"],
                "teacher_profile_digest",
            ),
        )


@dataclass(frozen=True, slots=True)
class BehaviorCloningGamePlan:
    """One deterministic, public-rules demonstration game."""

    round_index: int
    game_index: int
    ruleset_name: str
    player_count: int
    engine_seed: int

    def __post_init__(self) -> None:
        _require_nonnegative_int("round_index", self.round_index)
        _require_nonnegative_int("game_index", self.game_index)
        if self.ruleset_name not in {ruleset_name(chart) for chart in "ABCDE"}:
            raise BehaviorCloningError("plan ruleset must be a supported live chart")
        if self.player_count not in (3, 4, 5):
            raise BehaviorCloningError("plan player count must be three, four, or five")
        if not 0 <= self.engine_seed < 2**63:
            raise BehaviorCloningError("plan engine seed must be an unsigned 63-bit integer")


@dataclass(frozen=True, slots=True)
class BehaviorCloningExample:
    """One deployable neural observation paired with a legal teacher action."""

    observation: NeuralObservation
    action: int

    def __post_init__(self) -> None:
        if not isinstance(self.action, int) or isinstance(self.action, bool):
            raise BehaviorCloningError("teacher action must be an integer")
        if not 0 <= self.action < len(self.observation.action_mask):
            raise BehaviorCloningError("teacher action is outside the action space")
        if not bool(self.observation.action_mask[self.action]):
            raise BehaviorCloningError("teacher action is illegal under the public mask")


@dataclass(frozen=True, slots=True)
class BehaviorCloningDataset:
    """Immutable examples plus public provenance and integrity metadata."""

    examples: tuple[BehaviorCloningExample, ...]
    game_count: int
    cell_game_counts: tuple[tuple[str, int, int], ...]
    teacher_identity: str
    teacher_profile_digest: str
    dataset_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "examples", tuple(self.examples))
        object.__setattr__(self, "cell_game_counts", tuple(self.cell_game_counts))
        if not self.examples:
            raise BehaviorCloningError("behavior cloning dataset must not be empty")
        if self.game_count <= 0 or sum(row[2] for row in self.cell_game_counts) != self.game_count:
            raise BehaviorCloningError("dataset game counts are inconsistent")
        if self.teacher_identity != BALANCED_V3_TEACHER_IDENTITY:
            raise BehaviorCloningError("dataset teacher identity is not pinned")
        if self.teacher_profile_digest != BALANCED_V3_PROFILE_DIGEST:
            raise BehaviorCloningError("dataset teacher profile is not pinned")
        if self.dataset_digest != _dataset_digest(self.examples):
            raise BehaviorCloningError("dataset digest does not match its examples")


@dataclass(frozen=True, slots=True)
class BehaviorCloningUpdateMetrics:
    """Finite diagnostics for one policy-only optimizer step."""

    shard_index: int
    epoch_index: int
    minibatch_index: int
    example_count: int
    negative_log_likelihood: float
    teacher_agreement: float
    entropy: float
    pre_clip_gradient_norm: float
    post_clip_gradient_norm: float


@dataclass(frozen=True, slots=True)
class BehaviorCloningMetrics:
    """Complete deterministic report for one cloning stage."""

    config_digest: str
    dataset_digest: str
    example_count: int
    epochs: int
    optimizer_steps: int
    updates: tuple[BehaviorCloningUpdateMetrics, ...]


@dataclass(frozen=True, slots=True)
class BehaviorCloningProvenance:
    """Concise JSON-safe identity and budget summary for a completed stage."""

    config_digest: str
    teacher_identity: str
    teacher_profile_digest: str
    dataset_digest: str
    demonstration_games: int
    demonstration_examples: int
    cell_game_counts: tuple[tuple[str, int, int], ...]
    epochs: int
    optimizer_steps: int

    @classmethod
    def create(
        cls,
        config: BehaviorCloningConfig,
        dataset: BehaviorCloningDataset,
        metrics: BehaviorCloningMetrics,
    ) -> BehaviorCloningProvenance:
        """Bind consistent configuration, demonstration, and training records."""

        if metrics.config_digest != config.config_digest:
            raise BehaviorCloningError("training metrics do not match cloning config")
        if metrics.dataset_digest != dataset.dataset_digest:
            raise BehaviorCloningError("training metrics do not match demonstration dataset")
        if metrics.example_count != len(dataset.examples):
            raise BehaviorCloningError("training metrics example count is inconsistent")
        if metrics.epochs != config.epochs:
            raise BehaviorCloningError("training metrics epoch count is inconsistent")
        if dataset.teacher_identity != config.teacher_identity or (
            dataset.teacher_profile_digest != config.teacher_profile_digest
        ):
            raise BehaviorCloningError("demonstration teacher does not match cloning config")
        return cls(
            config_digest=config.config_digest,
            teacher_identity=dataset.teacher_identity,
            teacher_profile_digest=dataset.teacher_profile_digest,
            dataset_digest=dataset.dataset_digest,
            demonstration_games=dataset.game_count,
            demonstration_examples=len(dataset.examples),
            cell_game_counts=dataset.cell_game_counts,
            epochs=metrics.epochs,
            optimizer_steps=metrics.optimizer_steps,
        )

    def to_json_dict(self) -> dict[str, object]:
        """Return a concise nested payload without observations or model state."""

        return {
            "schema_version": 1,
            "method": "behavior_cloning",
            "config_digest": self.config_digest,
            "teacher": {
                "identity": self.teacher_identity,
                "profile_digest": self.teacher_profile_digest,
            },
            "demonstrations": {
                "dataset_digest": self.dataset_digest,
                "games": self.demonstration_games,
                "examples": self.demonstration_examples,
                "cell_game_counts": [list(row) for row in self.cell_game_counts],
            },
            "training": {
                "epochs": self.epochs,
                "optimizer_steps": self.optimizer_steps,
            },
        }


def balanced_v3_profile_digest() -> str:
    """Recompute the frozen promotion profile digest for balanced-v3."""

    return released_v3_profile_digest(BALANCED_V3_TEACHER_IDENTITY)


def plan_behavior_cloning_games(
    config: BehaviorCloningConfig,
) -> tuple[BehaviorCloningGamePlan, ...]:
    """Plan a balanced A-E, three-to-five-player demonstration matrix."""

    _verify_teacher_pin()
    plans: list[BehaviorCloningGamePlan] = []
    for round_index in range(config.rounds):
        for _repetition in range(config.games_per_cell):
            for chart in "ABCDE":
                for player_count in (3, 4, 5):
                    game_index = len(plans)
                    plans.append(
                        BehaviorCloningGamePlan(
                            round_index=round_index,
                            game_index=game_index,
                            ruleset_name=ruleset_name(chart),
                            player_count=player_count,
                            engine_seed=derive_seed(
                                config.root_seed,
                                "environment",
                                _BEHAVIOR_CLONING_SEED_TAG,
                                round_index,
                                game_index,
                            ),
                        )
                    )
    return tuple(plans)


def behavior_cloning_game_shards(
    plans: tuple[BehaviorCloningGamePlan, ...],
    *,
    games_per_shard: int,
) -> tuple[tuple[BehaviorCloningGamePlan, ...], ...]:
    """Split ordered demonstrations so full experiments stay memory-bounded."""

    _require_positive_int("games_per_shard", games_per_shard)
    if not plans:
        raise BehaviorCloningError("behavior cloning sharding requires game plans")
    return tuple(
        plans[offset : offset + games_per_shard] for offset in range(0, len(plans), games_per_shard)
    )


def collect_behavior_cloning_dataset(
    plans: tuple[BehaviorCloningGamePlan, ...],
    *,
    encoder_config: NeuralEncoderConfig,
) -> BehaviorCloningDataset:
    """Collect public observations and legal balanced-v3 decisions only."""

    _verify_teacher_pin()
    if not plans:
        raise BehaviorCloningError("at least one demonstration plan is required")
    if tuple(plan.game_index for plan in plans) != tuple(
        range(plans[0].game_index, plans[0].game_index + len(plans))
    ):
        raise BehaviorCloningError("demonstration plans must use contiguous game indices")

    bounds = EnvironmentBounds(encoder_config.max_bid, encoder_config.max_hand_size)
    codec = ActionCodec(bounds)
    encoder = NeuralObservationEncoder(encoder_config, bounds, action_codec=codec)
    teacher = build_released_v3_brain(BALANCED_V3_TEACHER_IDENTITY)
    examples: list[BehaviorCloningExample] = []
    cell_counts: dict[tuple[str, int], int] = {}

    for plan in plans:
        chart = value_chart_from_ruleset_name(plan.ruleset_name)
        knowledge = canonical_knowledge(plan.player_count, value_chart=chart)
        session = SdkGameSession.start(
            player_count=plan.player_count,
            seed=plan.engine_seed,
            value_chart=chart,
        )
        cell = (plan.ruleset_name, plan.player_count)
        cell_counts[cell] = cell_counts.get(cell, 0) + 1

        while not session.terminated:
            history = public_history_from_sdk_events(session.events)
            decisions: dict[int, BotDecision] = {}
            for seat, context in session.pending.contexts:
                decision = teacher.choose_decision(context, knowledge)
                context.validate(decision)
                action = codec.encode(decision)
                observation = _immutable_observation(encoder.encode(context, knowledge, history))
                examples.append(BehaviorCloningExample(observation, action))
                decisions[seat] = decision
            session.step(decisions)

    frozen_examples = tuple(examples)
    return BehaviorCloningDataset(
        examples=frozen_examples,
        game_count=len(plans),
        cell_game_counts=tuple(
            (ruleset, player_count, count)
            for (ruleset, player_count), count in sorted(cell_counts.items())
        ),
        teacher_identity=BALANCED_V3_TEACHER_IDENTITY,
        teacher_profile_digest=BALANCED_V3_PROFILE_DIGEST,
        dataset_digest=_dataset_digest(frozen_examples),
    )


class BehaviorCloningTrainer:
    """Apply deterministic masked cross-entropy without a value target."""

    def __init__(self, model: NeuralPolicy, config: BehaviorCloningConfig) -> None:
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            foreach=False,
        )

    def train(
        self,
        dataset: BehaviorCloningDataset,
        *,
        shard_index: int = 0,
    ) -> BehaviorCloningMetrics:
        """Train only toward legal teacher actions and return every update metric."""

        _verify_teacher_pin()
        _require_nonnegative_int("shard_index", shard_index)
        if (
            dataset.teacher_identity != self.config.teacher_identity
            or dataset.teacher_profile_digest != self.config.teacher_profile_digest
        ):
            raise BehaviorCloningError("dataset teacher does not match training config")
        examples = dataset.examples
        prior_mode = self.model.training
        updates: list[BehaviorCloningUpdateMetrics] = []
        try:
            self.model.train()
            for epoch_index in range(self.config.epochs):
                minibatch_index = 0
                generator = torch.Generator(device="cpu").manual_seed(
                    derive_seed(
                        self.config.root_seed,
                        "minibatch",
                        _BEHAVIOR_CLONING_SEED_TAG,
                        shard_index,
                        epoch_index,
                    )
                )
                order = torch.randperm(len(examples), generator=generator).tolist()
                for start in range(0, len(order), self.config.minibatch_size):
                    indices = order[start : start + self.config.minibatch_size]
                    chosen = tuple(examples[index] for index in indices)
                    batch = batch_observations(
                        tuple(example.observation for example in chosen),
                        self.device,
                    )
                    output = self.model(batch)
                    selection = evaluate_masked_policy(
                        output,
                        batch,
                        generator=None,
                        deterministic=True,
                    )
                    teacher_actions = torch.tensor(
                        tuple(example.action for example in chosen),
                        dtype=torch.int64,
                        device=self.device,
                    )
                    log_probabilities = torch.log_softmax(selection.masked_logits, dim=-1)
                    loss = -log_probabilities.gather(
                        1,
                        teacher_actions.unsqueeze(1),
                    ).mean()
                    if not torch.isfinite(loss).item():
                        raise BehaviorCloningError("behavior cloning loss is not finite")

                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()  # type: ignore[no-untyped-call]
                    pre_clip = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_gradient_norm,
                        error_if_nonfinite=True,
                    )
                    post_clip = _gradient_norm(self.model)
                    self.optimizer.step()
                    if any(
                        not torch.isfinite(parameter).all().item()
                        for parameter in self.model.parameters()
                    ):
                        raise BehaviorCloningError("behavior cloning produced nonfinite weights")

                    updates.append(
                        BehaviorCloningUpdateMetrics(
                            shard_index=shard_index,
                            epoch_index=epoch_index,
                            minibatch_index=minibatch_index,
                            example_count=len(chosen),
                            negative_log_likelihood=float(loss.detach().item()),
                            teacher_agreement=float(
                                (selection.actions == teacher_actions).float().mean().item()
                            ),
                            entropy=float(selection.entropy.detach().mean().item()),
                            pre_clip_gradient_norm=float(pre_clip.detach().item()),
                            post_clip_gradient_norm=post_clip,
                        )
                    )
                    minibatch_index += 1
        finally:
            self.model.train(prior_mode)

        return BehaviorCloningMetrics(
            config_digest=self.config.config_digest,
            dataset_digest=dataset.dataset_digest,
            example_count=len(examples),
            epochs=self.config.epochs,
            optimizer_steps=len(updates),
            updates=tuple(updates),
        )


def _verify_teacher_pin() -> None:
    if balanced_v3_profile_digest() != BALANCED_V3_PROFILE_DIGEST:
        raise BehaviorCloningError("released balanced-v3 profile no longer matches its digest")


def _dataset_digest(examples: tuple[BehaviorCloningExample, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(BALANCED_V3_TEACHER_IDENTITY.encode("utf-8"))
    digest.update(BALANCED_V3_PROFILE_DIGEST.encode("ascii"))
    for example in examples:
        digest.update(example.action.to_bytes(4, "big", signed=False))
        for name in (
            "global_ids",
            "global_numeric",
            "objective_bits",
            "seat_numeric",
            "seat_valid",
            "private_hand_ids",
            "hand_valid",
            "history_ids",
            "history_numeric",
            "history_valid",
            "action_mask",
        ):
            array = np.ascontiguousarray(getattr(example.observation, name))
            digest.update(name.encode("ascii"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
            digest.update(array.tobytes())
    return digest.hexdigest()


def _json_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise BehaviorCloningError(f"{name} must be a JSON object")
    return dict(value)


def _json_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise BehaviorCloningError(f"{name} must be an integer")
    return value


def _json_number(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise BehaviorCloningError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise BehaviorCloningError(f"{name} must be finite")
    return result


def _json_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise BehaviorCloningError(f"{name} must be a string")
    return value


def _immutable_observation(observation: NeuralObservation) -> NeuralObservation:
    values: dict[str, np.ndarray] = {}
    for name in (
        "global_ids",
        "global_numeric",
        "objective_bits",
        "seat_numeric",
        "seat_valid",
        "private_hand_ids",
        "hand_valid",
        "history_ids",
        "history_numeric",
        "history_valid",
        "action_mask",
    ):
        copied = getattr(observation, name).copy()
        copied.flags.writeable = False
        values[name] = copied
    return NeuralObservation(**values)


def _gradient_norm(model: NeuralPolicy) -> float:
    squared = torch.zeros((), device=next(model.parameters()).device)
    for parameter in model.parameters():
        if parameter.grad is not None:
            squared += torch.sum(parameter.grad.detach() ** 2)
    value = float(torch.sqrt(squared).item())
    if not math.isfinite(value):
        raise BehaviorCloningError("behavior cloning gradient norm is not finite")
    return value


def _require_int(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise BehaviorCloningError(f"{name} must be an integer")


def _require_nonnegative_int(name: str, value: object) -> None:
    _require_int(name, value)
    assert isinstance(value, int)
    if value < 0:
        raise BehaviorCloningError(f"{name} must be nonnegative")


def _require_positive_int(name: str, value: object) -> None:
    _require_int(name, value)
    assert isinstance(value, int)
    if value <= 0:
        raise BehaviorCloningError(f"{name} must be positive")
