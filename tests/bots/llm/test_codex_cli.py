from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from garboid_pocketrocks.bots.llm.codex_cli import CodexCLIBackend, CodexCLIError


class RecordingRunner:
    def __init__(
        self,
        *,
        response: str | None = "7\n",
        returncode: int = 0,
        stderr: str = "",
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.returncode = returncode
        self.stderr = stderr
        self.error = error
        self.args: list[str] | None = None
        self.kwargs: dict[str, Any] | None = None

    def __call__(self, args: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.args = list(args)
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        if self.response is not None:
            output_index = self.args.index("--output-last-message") + 1
            Path(self.args[output_index]).write_text(self.response, encoding="utf-8")
        return subprocess.CompletedProcess(
            self.args,
            self.returncode,
            stdout="event output",
            stderr=self.stderr,
        )


def test_complete_invokes_isolated_stateless_codex_without_a_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = RecordingRunner(response=" 7\n")
    monkeypatch.setattr(subprocess, "run", runner)
    backend = CodexCLIBackend()

    response = backend.complete("choose a bid", timeout_seconds=4.5)

    assert response == " 7\n"
    assert runner.args is not None
    assert runner.args[:2] == ["codex", "exec"]
    assert "--ephemeral" in runner.args
    assert "--skip-git-repo-check" in runner.args
    assert "--ignore-user-config" in runner.args
    assert "--ignore-rules" in runner.args
    assert runner.args[runner.args.index("--sandbox") + 1] == "read-only"
    assert runner.args[runner.args.index("--color") + 1] == "never"
    assert runner.args[-1] == "-"
    assert runner.kwargs is not None
    assert runner.kwargs["input"] == "choose a bid"
    assert runner.kwargs["shell"] is False
    assert runner.kwargs["capture_output"] is True
    assert runner.kwargs["text"] is True
    assert runner.kwargs["check"] is False
    assert runner.kwargs["timeout"] == 4.5
    assert runner.kwargs["cwd"] == runner.args[runner.args.index("--cd") + 1]


def test_complete_supports_custom_executable_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = RecordingRunner(response="3")
    monkeypatch.setattr(subprocess, "run", runner)
    backend = CodexCLIBackend(executable="/opt/codex", model="gpt-test")

    assert backend.complete("prompt", timeout_seconds=2) == "3"

    assert runner.args is not None
    assert runner.args[0] == "/opt/codex"
    model_index = runner.args.index("--model")
    assert runner.args[model_index + 1] == "gpt-test"


def test_nonzero_exit_raises_bounded_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = RecordingRunner(response=None, returncode=9, stderr="x" * 5_000)
    monkeypatch.setattr(subprocess, "run", runner)

    with pytest.raises(CodexCLIError) as raised:
        CodexCLIBackend().complete("prompt", timeout_seconds=2)

    message = str(raised.value)
    assert "status 9" in message
    assert "x" * 2_000 in message
    assert "x" * 2_001 not in message


def test_timeout_becomes_backend_error(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = RecordingRunner(error=subprocess.TimeoutExpired(cmd=["codex", "exec"], timeout=1.25))
    monkeypatch.setattr(subprocess, "run", runner)

    with pytest.raises(CodexCLIError, match=r"timed out after 1\.25 seconds"):
        CodexCLIBackend().complete("prompt", timeout_seconds=1.25)


def test_missing_executable_becomes_backend_error(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = RecordingRunner(error=FileNotFoundError("codex missing"))
    monkeypatch.setattr(subprocess, "run", runner)

    with pytest.raises(CodexCLIError, match="could not start"):
        CodexCLIBackend().complete("prompt", timeout_seconds=2)


@pytest.mark.parametrize("response", (None, "", " \n"))
def test_missing_or_empty_final_message_is_an_error(
    response: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = RecordingRunner(response=response)
    monkeypatch.setattr(subprocess, "run", runner)

    with pytest.raises(CodexCLIError, match="final response"):
        CodexCLIBackend().complete("prompt", timeout_seconds=2)


def test_oversized_final_message_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = RecordingRunner(response="1" * 10_001)
    monkeypatch.setattr(subprocess, "run", runner)

    with pytest.raises(CodexCLIError, match="exceeded 10000 characters"):
        CodexCLIBackend().complete("prompt", timeout_seconds=2)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"executable": ""}, "executable must be nonempty"),
        ({"model": ""}, "model must be nonempty"),
    ),
)
def test_configuration_rejects_empty_values(
    kwargs: dict[str, str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        CodexCLIBackend(**kwargs)
