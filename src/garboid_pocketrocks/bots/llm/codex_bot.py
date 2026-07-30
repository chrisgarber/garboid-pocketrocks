from __future__ import annotations

import argparse
import asyncio

from pocketrocks import BotDecision, DecisionContext

from garboid_pocketrocks.bots.base import BotSpec, PocketRocksFastBot
from garboid_pocketrocks.bots.llm.brain import StatelessLLMBrain
from garboid_pocketrocks.bots.llm.codex_cli import CodexCLIBackend


class CodexBot(PocketRocksFastBot):
    """Stateless local-Codex bot with a development-only public identity."""

    BOT_ID = "bot_00000000-0000-4000-8000-00000000000d"
    BOT_NAME = "codex"

    @classmethod
    def build_brain(cls, seed: int | None) -> StatelessLLMBrain:
        del seed
        return StatelessLLMBrain(CodexCLIBackend())

    async def choose_decision(self, context: DecisionContext) -> BotDecision:
        return await asyncio.to_thread(self.choose_decision_sync, context)


CODEX_BOT_SPEC = BotSpec.from_bot_class(CodexBot)


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the stateless local-Codex PocketRocks bot")
    parser.add_argument("--model", help="optional Codex model override")
    parser.add_argument(
        "--timeout-seconds",
        type=_positive_float,
        default=30.0,
        help="maximum seconds for each Codex attempt (default: 30)",
    )
    parser.add_argument(
        "--codex-executable",
        default="codex",
        help="Codex CLI executable or path (default: codex)",
    )
    return parser


def _brain_from_args(args: argparse.Namespace) -> StatelessLLMBrain:
    return StatelessLLMBrain(
        CodexCLIBackend(
            executable=args.codex_executable,
            model=args.model,
        ),
        timeout_seconds=args.timeout_seconds,
    )


def main() -> None:
    args = _parser().parse_args()
    CodexBot(brain=_brain_from_args(args)).run()
