from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

_MAX_DIAGNOSTIC_LENGTH = 2_000
_MAX_RESPONSE_LENGTH = 10_000


class CodexCLIError(RuntimeError):
    """Raised when a stateless Codex CLI invocation does not return text."""


@dataclass(frozen=True, slots=True)
class CodexCLIBackend:
    executable: str = "codex"
    model: str | None = None

    def __post_init__(self) -> None:
        if not self.executable.strip():
            raise ValueError("executable must be nonempty")
        if self.model is not None and not self.model.strip():
            raise ValueError("model must be nonempty when provided")

    def complete(self, prompt: str, *, timeout_seconds: float) -> str:
        with tempfile.TemporaryDirectory(prefix="garboid-codex-") as directory:
            output_path = Path(directory, "final-response.txt")
            args = [
                self.executable,
                "exec",
                "--ephemeral",
                "--skip-git-repo-check",
                "--ignore-user-config",
                "--ignore-rules",
                "--sandbox",
                "read-only",
                "--color",
                "never",
                "--cd",
                directory,
                "--output-last-message",
                str(output_path),
            ]
            if self.model is not None:
                args.extend(("--model", self.model))
            args.append("-")
            try:
                completed = subprocess.run(
                    args,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                    shell=False,
                    cwd=directory,
                )
            except subprocess.TimeoutExpired as error:
                raise CodexCLIError(
                    f"Codex CLI timed out after {timeout_seconds:g} seconds"
                ) from error
            except OSError as error:
                raise CodexCLIError(f"Codex CLI could not start: {error}") from error

            if completed.returncode != 0:
                diagnostic = (completed.stderr or completed.stdout).strip()
                diagnostic = diagnostic[:_MAX_DIAGNOSTIC_LENGTH] or "no diagnostic output"
                raise CodexCLIError(
                    f"Codex CLI exited with status {completed.returncode}: {diagnostic}"
                )
            try:
                response = output_path.read_text(encoding="utf-8")
            except OSError as error:
                raise CodexCLIError("Codex CLI did not produce a final response") from error
            if not response.strip():
                raise CodexCLIError("Codex CLI produced an empty final response")
            if len(response) > _MAX_RESPONSE_LENGTH:
                raise CodexCLIError(
                    f"Codex CLI final response exceeded {_MAX_RESPONSE_LENGTH} characters"
                )
            return response
