"""Precommitted fixed-compute contract for heuristic-to-neural ablations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from garboid_pocketrocks.neural.behavior_cloning import BehaviorCloningConfig
from garboid_pocketrocks.neural.heuristic_auxiliary import HeuristicAuxiliaryValueConfig
from garboid_pocketrocks.neural.heuristic_curriculum import (
    FOCAL_SEAT_CONTROL_V1,
    HEURISTIC_OPPONENT_CURRICULUM_V1,
)
from garboid_pocketrocks.neural.heuristic_teachers import (
    RELEASED_HEURISTIC_V3_IDENTITIES,
    released_v3_profile_digest,
)
from garboid_pocketrocks.neural.ppo import PPOConfig
from garboid_pocketrocks.neural.run_config import (
    HeuristicExperimentArm,
    ParallelConfig,
    TrainingRunConfig,
)
from garboid_pocketrocks.training.rewards import RewardConfig

REFERENCE_NEURAL_IDENTITY = "vector_ppo_large_v1_g350k"
REFERENCE_PARAMETER_DIGEST = "088160ad4006b2bac3691980d7f3e9dc56635fd57e6ad2b94068497e199f0e5c"
REFERENCE_TRAINING_GAMES = 349_860
REFERENCE_UPDATES = 196
EXPERIMENT_ROOT_SEED = 42
EXPERIMENT_GAMES_PER_CELL = 119
TRAINING_CELL_COUNT = 15

HeuristicBootstrapStrategy = HeuristicExperimentArm


@dataclass(frozen=True, slots=True)
class HeuristicBootstrapArm:
    """One immutable development arm with an exact complete-game budget."""

    strategy: HeuristicBootstrapStrategy
    behavior_cloning_rounds: int
    ppo_updates: int
    opponent_training: str
    auxiliary_target: str
    schedule_digest: str

    @property
    def demonstration_games(self) -> int:
        return self.behavior_cloning_rounds * EXPERIMENT_GAMES_PER_CELL * TRAINING_CELL_COUNT

    @property
    def ppo_games(self) -> int:
        return self.ppo_updates * EXPERIMENT_GAMES_PER_CELL * TRAINING_CELL_COUNT

    @property
    def total_training_games(self) -> int:
        return self.demonstration_games + self.ppo_games

    @property
    def digest(self) -> str:
        payload = {
            "auxiliary_target": self.auxiliary_target,
            "behavior_cloning_rounds": self.behavior_cloning_rounds,
            "games_per_cell": EXPERIMENT_GAMES_PER_CELL,
            "opponent_training": self.opponent_training,
            "ppo_updates": self.ppo_updates,
            "root_seed": EXPERIMENT_ROOT_SEED,
            "strategy": self.strategy,
            "schedule_digest": self.schedule_digest,
            "teacher_profile_digests": {
                identity: released_v3_profile_digest(identity)
                for identity in RELEASED_HEURISTIC_V3_IDENTITIES
            },
        }
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()


HEURISTIC_BOOTSTRAP_ARMS = (
    HeuristicBootstrapArm(
        strategy="fixed-compute-control-v1",
        behavior_cloning_rounds=0,
        ppo_updates=196,
        opponent_training="mirror-self-play",
        auxiliary_target="disabled",
        schedule_digest="mirror-self-play-v1",
    ),
    HeuristicBootstrapArm(
        strategy="focal-seat-control-v1",
        behavior_cloning_rounds=0,
        ppo_updates=196,
        opponent_training="focal-seat-control-v1",
        auxiliary_target="disabled",
        schedule_digest=FOCAL_SEAT_CONTROL_V1.digest,
    ),
    HeuristicBootstrapArm(
        strategy="behavior-cloning-balanced-v3-v1",
        behavior_cloning_rounds=4,
        ppo_updates=192,
        opponent_training="mirror-self-play",
        auxiliary_target="disabled",
        schedule_digest="mirror-self-play-v1",
    ),
    HeuristicBootstrapArm(
        strategy="auxiliary-value-balanced-v3-v1",
        behavior_cloning_rounds=0,
        ppo_updates=196,
        opponent_training="mirror-self-play",
        auxiliary_target="balanced-v3-bid-win-delta-v1",
        schedule_digest="mirror-self-play-v1",
    ),
    HeuristicBootstrapArm(
        strategy="heuristic-opponent-curriculum-v1",
        behavior_cloning_rounds=0,
        ppo_updates=196,
        opponent_training="heuristic-opponent-curriculum-v1",
        auxiliary_target="disabled",
        schedule_digest=HEURISTIC_OPPONENT_CURRICULUM_V1.digest,
    ),
)


def bootstrap_strategy(config: TrainingRunConfig) -> HeuristicBootstrapStrategy:
    """Name the one active strategy, with ordinary self-play as the control."""

    if config.behavior_cloning is not None:
        return "behavior-cloning-balanced-v3-v1"
    if config.heuristic_auxiliary.target != "disabled":
        return "auxiliary-value-balanced-v3-v1"
    if config.opponent_training == "heuristic-opponent-curriculum-v1":
        return "heuristic-opponent-curriculum-v1"
    if config.opponent_training == "focal-seat-control-v1":
        return "focal-seat-control-v1"
    return "fixed-compute-control-v1"


def validate_fixed_compute_arm(config: TrainingRunConfig) -> HeuristicBootstrapArm:
    """Reject a development config that drifts from its precommitted arm."""

    strategy = bootstrap_strategy(config)
    arm = next(item for item in HEURISTIC_BOOTSTRAP_ARMS if item.strategy == strategy)
    cloning_rounds = config.behavior_cloning.rounds if config.behavior_cloning else 0
    expected_cloning = (
        BehaviorCloningConfig(
            root_seed=42,
            rounds=4,
            games_per_cell=119,
            epochs=3,
            minibatch_size=512,
            games_per_shard=64,
            learning_rate=0.0003,
            max_gradient_norm=0.5,
        )
        if strategy == "behavior-cloning-balanced-v3-v1"
        else None
    )
    expected_auxiliary = (
        HeuristicAuxiliaryValueConfig.balanced_v3()
        if strategy == "auxiliary-value-balanced-v3-v1"
        else HeuristicAuxiliaryValueConfig()
    )
    if (
        config.experiment_arm != strategy
        or config.root_seed != EXPERIMENT_ROOT_SEED
        or config.device != "cpu"
        or config.deterministic_algorithms is not True
        or config.model_profile != "large"
        or config.learner_threads != 4
        or config.games_per_cell != EXPERIMENT_GAMES_PER_CELL
        or config.max_updates != arm.ppo_updates
        or cloning_rounds != arm.behavior_cloning_rounds
        or config.behavior_cloning != expected_cloning
        or config.opponent_training != arm.opponent_training
        or config.heuristic_auxiliary != expected_auxiliary
        or config.max_wall_seconds is not None
        or config.target_decisions_per_update is not None
        or config.parallel
        != ParallelConfig(
            workers=8,
            active_games_per_worker=64,
            max_inference_batch=1024,
            max_queue_delay_ms=1.0,
        )
        or config.ppo != PPOConfig(epochs=3)
        or config.reward != RewardConfig()
        or config.checkpoint_interval_seconds is not None
        or config.evaluation_interval_seconds is not None
        or config.evaluate_at_start
        or config.evaluate_at_end
        or config.league_fraction != 0.0
    ):
        raise ValueError(f"training config does not match fixed arm {strategy}")
    if arm.total_training_games != REFERENCE_TRAINING_GAMES:
        raise AssertionError("precommitted arm does not match the reference game budget")
    return arm
