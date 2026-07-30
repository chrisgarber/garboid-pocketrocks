from __future__ import annotations

import logging
import re

from pocketrocks import BotDecision, DecisionContext

from garboid_pocketrocks.bots.llm.backend import LLMBackend, LLMResponseError
from garboid_pocketrocks.bots.llm.prompting import PocketRocksPromptSkill, PromptSkill
from garboid_pocketrocks.rules import RulesetKnowledge

logger = logging.getLogger(__name__)

_ATTEMPT_COUNT = 2
_MAX_ERROR_LENGTH = 500


class StatelessLLMBrain:
    """Synchronous, provider-neutral LLM adapter for independent decisions."""

    def __init__(
        self,
        backend: LLMBackend,
        *,
        prompt_skill: PromptSkill | None = None,
        timeout_seconds: float = 30.0,
        deadline_margin_seconds: float = 0.25,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if deadline_margin_seconds < 0:
            raise ValueError("deadline_margin_seconds must be nonnegative")
        self.backend = backend
        self.prompt_skill = prompt_skill or PocketRocksPromptSkill()
        self.timeout_seconds = timeout_seconds
        self.deadline_margin_seconds = deadline_margin_seconds

    def choose_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
    ) -> BotDecision:
        legal_range = _legal_range(context)
        if legal_range is None:
            return BotDecision.pass_turn()

        correction: str | None = None
        for attempt_index in range(_ATTEMPT_COUNT):
            attempts_remaining = _ATTEMPT_COUNT - attempt_index
            timeout = self._timeout_for(context, attempts_remaining)
            if timeout is None:
                logger.warning(
                    "LLM request %s (%s) has no deadline remaining; using fallback",
                    context.request_id,
                    context.decision_kind,
                )
                return _fallback(context)
            try:
                prompt = self.prompt_skill.render(
                    context,
                    ruleset,
                    correction=correction,
                )
                response = self.backend.complete(prompt, timeout_seconds=timeout)
                value = _parse_integer(response, legal_range)
                decision = _decision(context, value)
                context.validate(decision)
                return decision
            except Exception as error:
                concise_error = _concise_error(error)
                attempt_number = attempt_index + 1
                if attempt_number < _ATTEMPT_COUNT:
                    logger.warning(
                        "LLM request %s (%s) attempt %d/%d failed: %s; retrying",
                        context.request_id,
                        context.decision_kind,
                        attempt_number,
                        _ATTEMPT_COUNT,
                        concise_error,
                    )
                    correction = f"The previous response was invalid: {concise_error}"
                    continue
                logger.warning(
                    "LLM request %s (%s) attempt %d/%d failed: %s; using fallback",
                    context.request_id,
                    context.decision_kind,
                    attempt_number,
                    _ATTEMPT_COUNT,
                    concise_error,
                )
                return _fallback(context)
        raise AssertionError("the fixed attempt loop must return")

    def _timeout_for(
        self,
        context: DecisionContext,
        attempts_remaining: int,
    ) -> float | None:
        usable_seconds = (context.remaining_deadline_ms / 1000.0) - self.deadline_margin_seconds
        if usable_seconds <= 0:
            return None
        return min(self.timeout_seconds, usable_seconds / attempts_remaining)


def _legal_range(context: DecisionContext) -> tuple[int, int] | None:
    if context.decision_kind == "submitBid":
        maximum = context.legal_max_amount
        return None if maximum is None or maximum <= 0 else (0, maximum)
    return None if context.revealable_count <= 0 else (0, context.revealable_count - 1)


def _parse_integer(response: str, legal_range: tuple[int, int]) -> int:
    stripped = response.strip()
    if re.fullmatch(r"[0-9]+", stripped) is None:
        raise LLMResponseError("invalid response: expected only ASCII digits")
    try:
        value = int(stripped)
    except ValueError as error:
        raise LLMResponseError("invalid response: integer is too long") from error
    minimum, maximum = legal_range
    if not minimum <= value <= maximum:
        raise LLMResponseError(
            f"invalid response: integer {value} is outside legal range {minimum} through {maximum}"
        )
    return value


def _decision(context: DecisionContext, value: int) -> BotDecision:
    if context.decision_kind == "submitBid":
        return BotDecision.pass_turn() if value == 0 else BotDecision.submit_bid(value)
    return BotDecision.select_info_to_reveal(value)


def _fallback(context: DecisionContext) -> BotDecision:
    if context.decision_kind == "submitBid":
        return BotDecision.pass_turn()
    return BotDecision.select_info_to_reveal(0)


def _concise_error(error: Exception) -> str:
    message = str(error).strip() or type(error).__name__
    return message[:_MAX_ERROR_LENGTH]
