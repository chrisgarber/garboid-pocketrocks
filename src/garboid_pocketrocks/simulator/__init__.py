"""SDK-backed PocketRocks simulation and evaluation."""

from garboid_pocketrocks.simulator.errors import (
    ActingSeatsError,
    IllegalDecisionError,
    InvalidPhaseError,
    SimulationError,
)
from garboid_pocketrocks.simulator.monte_carlo import (
    BehaviorStatistics,
    BotStatistics,
    GameJob,
    GameSummary,
    MonteCarloConfig,
    MonteCarloResult,
    MonteCarloRunner,
    RulesetStatistics,
    SeatStatistics,
)
from garboid_pocketrocks.simulator.replay import (
    MatchReplay,
    ReplayDivergence,
    ReplayedMatch,
    load_replay,
    replay_match,
    save_replay,
)
from garboid_pocketrocks.simulator.runner import (
    BotFault,
    FaultMode,
    MatchResult,
    MatchRunner,
)
from garboid_pocketrocks.simulator.seeding import derive_seed
from garboid_pocketrocks.simulator.session import (
    PendingDecisions,
    PlayerSnapshot,
    SdkGameSession,
    SessionResult,
    SessionScore,
    SessionSnapshot,
    SessionTransition,
)

__all__ = [
    "ActingSeatsError",
    "BehaviorStatistics",
    "BotFault",
    "BotStatistics",
    "FaultMode",
    "GameJob",
    "GameSummary",
    "IllegalDecisionError",
    "InvalidPhaseError",
    "MatchReplay",
    "MatchResult",
    "MatchRunner",
    "MonteCarloConfig",
    "MonteCarloResult",
    "MonteCarloRunner",
    "PendingDecisions",
    "PlayerSnapshot",
    "ReplayDivergence",
    "ReplayedMatch",
    "RulesetStatistics",
    "SdkGameSession",
    "SeatStatistics",
    "SessionResult",
    "SessionScore",
    "SessionSnapshot",
    "SessionTransition",
    "SimulationError",
    "derive_seed",
    "load_replay",
    "replay_match",
    "save_replay",
]
