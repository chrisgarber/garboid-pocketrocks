from __future__ import annotations

import json
from pathlib import Path

import pytest

from garboid_pocketrocks.visualizer.cli import main

from .test_analysis import _summary


def test_cli_renders_a_summary_only_tournament(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "summary.json").write_text(json.dumps(_summary()), encoding="utf-8")

    assert main((str(tmp_path),)) == 0

    output = tmp_path / "insights.html"
    assert output.is_file()
    assert "Tournament insights" in output.read_text(encoding="utf-8")
    assert capsys.readouterr().out.strip() == str(output)


def test_cli_requires_overwrite_for_an_existing_report(tmp_path: Path) -> None:
    (tmp_path / "summary.json").write_text(json.dumps(_summary()), encoding="utf-8")
    (tmp_path / "insights.html").write_text("keep", encoding="utf-8")

    with pytest.raises(SystemExit) as raised:
        main((str(tmp_path),))

    assert raised.value.code == 2
    assert (tmp_path / "insights.html").read_text(encoding="utf-8") == "keep"
