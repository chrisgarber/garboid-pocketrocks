class SimulationError(RuntimeError):
    """Base error for deterministic simulator failures."""


class InvalidPhaseError(SimulationError):
    """Raised when a transition is requested in the wrong phase."""


class ActingSeatsError(SimulationError):
    """Raised when a decision batch does not contain the required seats."""


class IllegalDecisionError(SimulationError):
    """Raised when an SDK decision is not legal for its simulated context."""


class ActionDeckExhaustedError(SimulationError):
    """Raised when a valid game still needs an action after its deck is empty."""
