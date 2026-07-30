from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

from garboid_pocketrocks.neural.cli import main  # noqa: E402


def test_cli_exposes_all_training_commands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--help"]) == 0

    output = capsys.readouterr().out
    for command in ("smoke", "train", "resume", "evaluate", "inspect"):
        assert command in output


def test_resume_help_accepts_a_long_run_config(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["resume", "--help"]) == 0

    assert "--config" in capsys.readouterr().out


def test_low_volume_smoke_cli_writes_reloadable_checkpoint(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"

    exit_code = main(
        [
            "smoke",
            "--output-dir",
            str(output_dir),
            "--seed",
            "42",
            "--updates",
            "1",
            "--games-per-update",
            "1",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "checkpoint/manifest.json").is_file()
    assert (output_dir / "checkpoint/model.pt").is_file()
    assert (output_dir / "smoke-result.json").is_file()


def test_smoke_help_documents_exact_defaults(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["smoke", "--help"]) == 0

    output = capsys.readouterr().out
    assert "--seed" in output and "42" in output
    assert "--updates" in output and "2" in output
    assert "--games-per-update" in output and "16" in output
    assert "--device" in output and "cpu" in output


@pytest.mark.parametrize(
    "arguments",
    (
        ("--device", "cuda"),
        ("--updates", "0"),
        ("--games-per-update", "0"),
    ),
)
def test_smoke_cli_rejects_out_of_scope_values(
    tmp_path: Path,
    arguments: tuple[str, str],
) -> None:
    assert (
        main(
            [
                "smoke",
                "--output-dir",
                str(tmp_path / f"invalid-{arguments[0]}"),
                *arguments,
            ]
        )
        == 2
    )


def test_smoke_cli_rejects_nonempty_output_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    marker = output_dir / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    assert main(["smoke", "--output-dir", str(output_dir)]) == 2
    assert marker.read_text(encoding="utf-8") == "keep"
