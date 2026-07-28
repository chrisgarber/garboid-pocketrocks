# Garboid PocketRocks Repository Scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use braze-superpowers:subagent-driven-development (recommended) or braze-superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce and publish a verified, MIT-licensed Python 3.14 repository scaffold for PocketRocks bot, simulator, and training development.

**Architecture:** Use a `src`-layout Python package with empty, documented namespaces for policies, live adapters, simulation, and training. Mise selects Python and uv; uv owns the environment and dependency lock. The PocketRocks SDK is pinned to a source commit, and CI executes the same format, lint, type-check, and test commands used locally.

**Tech Stack:** CPython 3.14, mise 2026.7, uv 0.11.26, Hatchling, PocketRocks Python SDK, pytest, Ruff, mypy, GitHub Actions

## Global Constraints

- The GitHub repository must be public at `chrisgarber/garboid-pocketrocks`.
- The repository must use the MIT license with copyright holder Christopher Garber.
- Python must be constrained to `>=3.14,<3.15`.
- The PocketRocks SDK must be pinned to commit `597857446d47ac0890609a4767cad561578a2519`.
- `jaiparera/pocketrockscompetition` is a reference only and must not be added as a dependency.
- This milestone must not implement bot, adapter, simulator, or training behavior.
- Secrets, local training data, and model checkpoints must remain untracked.
- The scaffold implementation and its plan must be delivered as one commit after the already-committed design spec.
- The repository default branch must be `main`.

---

### Task 1: Build and verify the repository scaffold

**Files:**
- Create: `.env.example`
- Create: `.github/workflows/ci.yml`
- Create: `.gitignore`
- Create: `LICENSE`
- Create: `README.md`
- Create: `mise.toml`
- Create: `pyproject.toml`
- Create: `src/garboid_pocketrocks/__init__.py`
- Create: `src/garboid_pocketrocks/adapters/__init__.py`
- Create: `src/garboid_pocketrocks/bots/__init__.py`
- Create: `src/garboid_pocketrocks/simulator/__init__.py`
- Create: `src/garboid_pocketrocks/training/__init__.py`
- Create: `tests/test_package.py`
- Generate: `uv.lock`

**Interfaces:**
- Consumes: PocketRocks SDK distribution at Git commit `597857446d47ac0890609a4767cad561578a2519`
- Produces: importable package `garboid_pocketrocks` with `__version__: str == "0.1.0"`
- Produces: importable namespaces `garboid_pocketrocks.adapters`, `.bots`, `.simulator`, and `.training`
- Produces: local and CI quality commands defined in the README and GitHub Actions workflow

- [ ] **Step 1: Write the failing package smoke test**

Create `tests/test_package.py`:

```python
from importlib import import_module

from garboid_pocketrocks import __version__

NAMESPACES = (
    "garboid_pocketrocks.adapters",
    "garboid_pocketrocks.bots",
    "garboid_pocketrocks.simulator",
    "garboid_pocketrocks.training",
)


def test_package_version() -> None:
    assert __version__ == "0.1.0"


def test_planned_namespaces_are_importable() -> None:
    for namespace in NAMESPACES:
        assert import_module(namespace).__name__ == namespace


if __name__ == "__main__":
    test_package_version()
    test_planned_namespaces_are_importable()
```

- [ ] **Step 2: Run the smoke test to verify it fails**

Run:

```bash
PYTHONPATH=src python3 tests/test_package.py
```

Expected: non-zero exit with `ModuleNotFoundError: No module named 'garboid_pocketrocks'`.

- [ ] **Step 3: Add the package namespaces**

Create `src/garboid_pocketrocks/__init__.py`:

```python
"""PocketRocks bot research, simulation, and training."""

__version__ = "0.1.0"

__all__ = ["__version__"]
```

Create `src/garboid_pocketrocks/adapters/__init__.py`:

```python
"""Adapters between bot policies and external PocketRocks interfaces."""
```

Create `src/garboid_pocketrocks/bots/__init__.py`:

```python
"""Reusable PocketRocks bot policies."""
```

Create `src/garboid_pocketrocks/simulator/__init__.py`:

```python
"""Local PocketRocks game simulation and evaluation."""
```

Create `src/garboid_pocketrocks/training/__init__.py`:

```python
"""Training tools for learned PocketRocks policies."""
```

- [ ] **Step 4: Run the smoke test to verify it passes**

Run:

```bash
PYTHONPATH=src python3 tests/test_package.py
```

Expected: exit code `0` with no output.

- [ ] **Step 5: Add project and tool configuration**

Create `mise.toml`:

```toml
[tools]
python = "3.14"
uv = "0.11.26"

[settings]
python.uv_venv_auto = "source"
```

Create `pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "garboid-pocketrocks"
version = "0.1.0"
description = "PocketRocks bots, simulation, evaluation, and local training"
readme = "README.md"
requires-python = ">=3.14,<3.15"
license = "MIT"
license-files = ["LICENSE"]
authors = [
  { name = "Christopher Garber" },
]
dependencies = [
  "pocketrocks-python-sdk @ git+https://github.com/jaiparera/pocketrocks-python-sdk.git@597857446d47ac0890609a4767cad561578a2519",
]

[tool.hatch.metadata]
allow-direct-references = true

[dependency-groups]
dev = [
  "mypy",
  "pytest",
  "ruff",
]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py314"
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]

[tool.mypy]
python_version = "3.14"
strict = true
files = ["src", "tests"]
warn_unused_configs = true
```

Create `.env.example`:

```dotenv
POCKETROCKS_API_KEY=
POCKETROCKS_BOT_ID=
POCKETROCKS_SERVER_URL=
```

Create `.gitignore`:

```gitignore
# Secrets
.env
.env.*
!.env.example

# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
dist/
build/

# Test and tool caches
.coverage
htmlcov/
.mypy_cache/
.pytest_cache/
.ruff_cache/

# Editors and operating systems
.DS_Store
.idea/
.vscode/

# Local simulation and training artifacts
/artifacts/
/checkpoints/
/data/
*.onnx
*.pt
*.pth
```

- [ ] **Step 6: Add the MIT license**

Create `LICENSE`:

```text
MIT License

Copyright (c) 2026 Christopher Garber

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 7: Document the project, architecture, setup, and roadmap**

Create `README.md`:

````markdown
# Garboid PocketRocks

Bots, simulation, evaluation, and local training for
[PocketRocks](https://pocketrocks.xyz/).

This project builds bots for
[jaiparera/pocketrockscompetition](https://github.com/jaiparera/pocketrockscompetition)
and connects them to the live service through the
[PocketRocks Python SDK](https://github.com/jaiparera/pocketrocks-python-sdk).

## Architecture

Bot strategies will implement one shared policy contract:

```text
Live server -> SDK adapter -> normalized game view -> policy -> action
Simulator -----------------> normalized game view -> policy -> action
```

The live adapter and simulator will share policy implementations, allowing the
same random, heuristic, and learned bots to run in both environments. The
competition repository's simulator is a rules reference rather than a runtime
dependency.

## Requirements

- [mise](https://mise.jdx.dev/)
- Git

Mise installs the Python 3.14 release line and uv version declared by this
repository. uv manages the virtual environment and locked dependencies.

## Setup

```bash
mise install
uv sync --locked
```

Live service credentials will eventually be read from `.env`. Start from the
committed variable names:

```bash
cp .env.example .env
```

Do not commit `.env`.

## Quality checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest
```

## Roadmap

1. Establish the repository scaffold and quality gates.
2. Build a random baseline bot and connect it through the Python SDK.
3. Implement a deterministic game engine and Monte Carlo match runner.
4. Design and implement value-heuristic bot strategies.
5. Run seeded round-robin evaluations and compare strategies.
6. Build and locally train a neural policy.

Each milestone will be designed and tested independently.

## License

Garboid PocketRocks is available under the [MIT License](LICENSE).
````

- [ ] **Step 8: Add continuous integration**

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  pull_request:
  push:

permissions:
  contents: read

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - name: Check out repository
        uses: actions/checkout@v6

      - name: Install uv and Python
        uses: astral-sh/setup-uv@v8
        with:
          version: "0.11.26"
          python-version: "3.14"
          enable-cache: true

      - name: Synchronize dependencies
        run: uv sync --locked

      - name: Check formatting
        run: uv run ruff format --check .

      - name: Lint
        run: uv run ruff check .

      - name: Type-check
        run: uv run mypy src tests

      - name: Test
        run: uv run pytest
```

- [ ] **Step 9: Generate and synchronize the dependency lock**

Run:

```bash
uv lock
uv sync --locked
```

Expected: uv resolves Python 3.14-compatible dependencies, checks out the SDK at
`597857446d47ac0890609a4767cad561578a2519`, writes `uv.lock`, and installs the
project and development group without errors.

- [ ] **Step 10: Run the complete local quality gate**

Run:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest
```

Expected:

- Ruff formatting: all files already formatted.
- Ruff lint: no diagnostics.
- mypy: `Success: no issues found`.
- pytest: two tests pass.

- [ ] **Step 11: Inspect the final scaffold and commit it**

Run:

```bash
git status --short
git diff --check
git diff --stat
git log -1 --oneline
```

Expected:

- only the planned scaffold and plan files are uncommitted;
- `git diff --check` emits no output;
- the latest existing commit is the approved design spec.

Then run:

```bash
git add \
  .env.example \
  .github/workflows/ci.yml \
  .gitignore \
  LICENSE \
  README.md \
  docs/superpowers/plans/2026-07-28-repository-scaffold.md \
  mise.toml \
  pyproject.toml \
  src \
  tests \
  uv.lock
git commit -m "chore: scaffold PocketRocks bot project"
```

Expected: one commit containing the plan and verified scaffold.

---

### Task 2: Create and verify the public GitHub repository

**Files:**
- No local file changes

**Interfaces:**
- Consumes: authenticated GitHub CLI session for personal account `chrisgarber`
- Produces: public repository `https://github.com/chrisgarber/garboid-pocketrocks`
- Produces: `origin` remote pointing at that repository with local `main` pushed

- [ ] **Step 1: Verify the local publication state**

Run:

```bash
git status --short --branch
git branch --show-current
git remote -v
gh auth status
```

Expected:

- clean worktree on `main`;
- no existing `origin` remote;
- GitHub CLI is authenticated as `chrisgarber`.

If GitHub reports an expired login, run `gh auth login -h github.com` and complete
the interactive browser/device flow before continuing.

- [ ] **Step 2: Create the public repository and push main**

Run:

```bash
gh repo create chrisgarber/garboid-pocketrocks \
  --public \
  --source=. \
  --remote=origin \
  --description="PocketRocks bots, simulation, evaluation, and local training" \
  --push
```

Expected: GitHub creates the public repository, configures `origin`, and pushes
local `main`.

- [ ] **Step 3: Verify repository metadata and pushed state**

Run:

```bash
gh repo view chrisgarber/garboid-pocketrocks \
  --json nameWithOwner,visibility,defaultBranchRef,licenseInfo,url
git status --short --branch
git remote -v
```

Expected:

- `nameWithOwner` is `chrisgarber/garboid-pocketrocks`;
- visibility is `PUBLIC`;
- default branch is `main`;
- the detected license is MIT;
- local `main` tracks `origin/main`;
- the worktree is clean.
