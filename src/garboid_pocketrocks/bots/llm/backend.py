from __future__ import annotations

from typing import Protocol


class LLMBackend(Protocol):
    def complete(self, prompt: str, *, timeout_seconds: float) -> str:
        """Return the backend's raw final response for one independent prompt."""


class LLMResponseError(ValueError):
    """Raised when an LLM response is not one legal integer."""
