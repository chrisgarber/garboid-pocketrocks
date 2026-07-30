from __future__ import annotations

import subprocess
import sys
import tomllib
from importlib import import_module
from pathlib import Path


def test_neural_extra_is_optional_and_version_bounded() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["optional-dependencies"]["neural"] == ["torch>=2.13,<2.14"]
    assert "torch" not in "\n".join(project["project"]["dependencies"]).lower()
    assert project["tool"]["uv"]["sources"]["torch"] == {"index": "pytorch-cpu"}
    assert project["tool"]["uv"]["index"] == [
        {
            "name": "pytorch-cpu",
            "url": "https://download.pytorch.org/whl/cpu",
            "explicit": True,
        }
    ]


def test_default_pytest_run_uses_capped_workstealing_parallelism() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert "pytest-xdist>=3.8.0" in project["dependency-groups"]["dev"]
    assert project["tool"]["pytest"]["ini_options"]["addopts"] == [
        "-n=auto",
        "--maxprocesses=8",
        "--dist=worksteal",
    ]


def test_neural_lock_does_not_pull_cuda_runtime() -> None:
    lock = Path("uv.lock").read_text(encoding="utf-8")

    assert 'name = "torch"' in lock
    assert 'name = "nvidia-cuda-runtime"' not in lock


def test_core_training_import_does_not_import_torch() -> None:
    code = "import sys; import garboid_pocketrocks.training; assert 'torch' not in sys.modules"

    subprocess.run([sys.executable, "-c", code], check=True)


def test_neural_namespace_import_is_lazy() -> None:
    code = "import sys; import garboid_pocketrocks.neural; assert 'torch' not in sys.modules"

    subprocess.run([sys.executable, "-c", code], check=True)
    assert import_module("garboid_pocketrocks.neural").__name__ == ("garboid_pocketrocks.neural")


def test_core_and_neural_mypy_configs_have_separate_boundaries() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert project["tool"]["mypy"]["exclude"] == [
        "^src/garboid_pocketrocks/neural/",
        "^tests/neural/",
    ]
    neural_config = Path("mypy.neural.ini").read_text(encoding="utf-8")
    assert "strict = True" in neural_config
    assert "files = src,tests" in neural_config
    assert "exclude" not in neural_config


def test_ci_keeps_core_install_and_splits_neural_unit_and_smoke_steps() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "run: uv sync --locked\n" in workflow
    assert "run: uv sync --locked --extra neural" in workflow
    assert "run: uv run --extra neural mypy --config-file mypy.neural.ini src tests" in workflow
    assert 'run: uv run --extra neural pytest tests/neural -m "not neural_smoke" -q' in workflow
    assert (
        "uv run --extra neural pytest -n 0\n"
        "          tests/neural/test_smoke.py::"
        "test_full_curriculum_smoke_contract_at_one_game_per_cell\n"
        "          -q"
    ) in workflow
