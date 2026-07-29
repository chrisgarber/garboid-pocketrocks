# Local Bot Launcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use braze-superpowers:subagent-driven-development (recommended) or braze-superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Assign the activated heuristic identities and add one local command that runs every defined PocketRocks bot concurrently.

**Architecture:** A focused `bots.launcher` module owns the ordered live-wrapper registry, CLI parsing, and async group lifecycle. Each selected wrapper keeps an independent SDK runtime and transport; `asyncio.TaskGroup` coordinates startup and cancellation without changing strategy or simulator code.

**Tech Stack:** Python 3.14, asyncio structured concurrency, argparse, PocketRocks Python SDK, pytest, Ruff, mypy, uv

## Global Constraints

- `garboid-bots` launches `random, aggressive, balanced, passive` by default.
- `--bots` accepts a validated comma-separated subset and rejects empty, duplicate, or unknown names before bot construction.
- All bots share SDK environment configuration but always use their wrapper class's committed `BOT_ID`.
- A runtime that returns cancels the remaining group and produces a nonzero failure identifying the stopped bot.
- Ctrl+C cancels and disconnects the complete group without being reported as a bot failure.
- Automated tests must not connect to the live PocketRocks service.
- `.env` contents must never be printed, committed, or copied into test output.

---

### Task 1: Publish the activated heuristic identities

**Files:**
- Modify: `tests/bots/test_heuristic_bots.py:72-85`
- Modify: `src/garboid_pocketrocks/bots/heuristic.py:54-82`

**Interfaces:**
- Consumes: existing `PocketRocksFastBot` class constants.
- Produces: activated `AggressiveHeuristicBot.BOT_ID`, `BalancedHeuristicBot.BOT_ID`, and `PassiveHeuristicBot.BOT_ID` values used by `BotSpec` and the live launcher.

- [ ] **Step 1: Replace the placeholder expectations with the activated IDs**

```python
def test_heuristic_bots_have_distinct_static_public_identities() -> None:
    assert issubclass(AggressiveHeuristicBot, PocketRocksFastBot)
    assert issubclass(BalancedHeuristicBot, PocketRocksFastBot)
    assert issubclass(PassiveHeuristicBot, PocketRocksFastBot)
    assert AggressiveHeuristicBot.BOT_ID == "bot_386b81bb-14df-477a-8d4c-0231cf1b3b1a"
    assert BalancedHeuristicBot.BOT_ID == "bot_265c84aa-f28e-4a35-b4de-a4f4ee406415"
    assert PassiveHeuristicBot.BOT_ID == "bot_9d33c9de-4d90-4608-9a58-d2c77d93e0bd"
    assert {
        AggressiveHeuristicBot.BOT_NAME,
        BalancedHeuristicBot.BOT_NAME,
        PassiveHeuristicBot.BOT_NAME,
    } == {"aggressive", "balanced", "passive"}
```

- [ ] **Step 2: Run the identity test and verify RED**

Run:

```bash
uv run pytest tests/bots/test_heuristic_bots.py::test_heuristic_bots_have_distinct_static_public_identities -q
```

Expected: FAIL because each wrapper still exposes a development-only placeholder ID.

- [ ] **Step 3: Assign the real class constants and update wrapper descriptions**

```python
class AggressiveHeuristicBot(PocketRocksFastBot):
    """Live wrapper for the aggressive heuristic."""

    BOT_ID = "bot_386b81bb-14df-477a-8d4c-0231cf1b3b1a"
    BOT_NAME = "aggressive"


class BalancedHeuristicBot(PocketRocksFastBot):
    """Live wrapper for the balanced heuristic."""

    BOT_ID = "bot_265c84aa-f28e-4a35-b4de-a4f4ee406415"
    BOT_NAME = "balanced"


class PassiveHeuristicBot(PocketRocksFastBot):
    """Live wrapper for the passive heuristic."""

    BOT_ID = "bot_9d33c9de-4d90-4608-9a58-d2c77d93e0bd"
    BOT_NAME = "passive"
```

Keep each existing `build_brain` method unchanged beneath its corresponding class constants.

- [ ] **Step 4: Verify GREEN and focused static checks**

Run:

```bash
uv run pytest tests/bots/test_heuristic_bots.py -q
uv run ruff check src/garboid_pocketrocks/bots/heuristic.py tests/bots/test_heuristic_bots.py
uv run mypy src/garboid_pocketrocks/bots/heuristic.py tests/bots/test_heuristic_bots.py
```

Expected: all commands PASS.

- [ ] **Step 5: Commit the activated identities**

```bash
git add src/garboid_pocketrocks/bots/heuristic.py tests/bots/test_heuristic_bots.py
git commit -m "feat: publish heuristic bot identities"
```

---

### Task 2: Define the live bot registry and CLI selection

**Files:**
- Create: `src/garboid_pocketrocks/bots/launcher.py`
- Create: `tests/bots/test_launcher.py`

**Interfaces:**
- Consumes: `RandomBot`, `AggressiveHeuristicBot`, `BalancedHeuristicBot`, `PassiveHeuristicBot`.
- Produces: `RunnableBot.run_async() -> None`, `BotFactory = Callable[[], RunnableBot]`, `BOT_REGISTRY: dict[str, BotFactory]`, `_bot_names(value: str) -> tuple[str, ...]`, and `_parser() -> argparse.ArgumentParser`.

- [ ] **Step 1: Write registry and argument-selection tests**

Create `tests/bots/test_launcher.py` with:

```python
from __future__ import annotations

import argparse

import pytest

from garboid_pocketrocks.bots import (
    AggressiveHeuristicBot,
    BalancedHeuristicBot,
    PassiveHeuristicBot,
    RandomBot,
)
from garboid_pocketrocks.bots.launcher import BOT_REGISTRY, _bot_names, _parser


def test_registry_contains_every_live_wrapper_in_stable_order() -> None:
    assert tuple(BOT_REGISTRY) == ("random", "aggressive", "balanced", "passive")
    assert tuple(BOT_REGISTRY.values()) == (
        RandomBot,
        AggressiveHeuristicBot,
        BalancedHeuristicBot,
        PassiveHeuristicBot,
    )


def test_parser_selects_all_registered_bots_by_default() -> None:
    assert _parser().parse_args([]).bots == tuple(BOT_REGISTRY)


def test_bot_names_accepts_a_trimmed_subset_in_requested_order() -> None:
    assert _bot_names(" passive, aggressive ") == ("passive", "aggressive")


@pytest.mark.parametrize(
    ("value", "message"),
    (
        ("", "at least one bot name is required"),
        ("  ", "at least one bot name is required"),
        ("random,random", "duplicate bot name"),
        ("random,missing", "unknown bot name"),
    ),
)
def test_bot_names_rejects_invalid_selections(value: str, message: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match=message):
        _bot_names(value)
```

- [ ] **Step 2: Run the launcher test and verify RED**

Run:

```bash
uv run pytest tests/bots/test_launcher.py -q
```

Expected: collection ERROR because `garboid_pocketrocks.bots.launcher` does not exist.

- [ ] **Step 3: Add the registry, runtime protocol, and parser**

Create `src/garboid_pocketrocks/bots/launcher.py` with:

```python
from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Protocol

from garboid_pocketrocks.bots.heuristic import (
    AggressiveHeuristicBot,
    BalancedHeuristicBot,
    PassiveHeuristicBot,
)
from garboid_pocketrocks.bots.random_bot import RandomBot


class RunnableBot(Protocol):
    async def run_async(self) -> None:
        """Run one bot until cancellation or a terminal SDK outcome."""


BotFactory = Callable[[], RunnableBot]

BOT_REGISTRY: dict[str, BotFactory] = {
    RandomBot.BOT_NAME: RandomBot,
    AggressiveHeuristicBot.BOT_NAME: AggressiveHeuristicBot,
    BalancedHeuristicBot.BOT_NAME: BalancedHeuristicBot,
    PassiveHeuristicBot.BOT_NAME: PassiveHeuristicBot,
}


def _bot_names(value: str) -> tuple[str, ...]:
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    if not names:
        raise argparse.ArgumentTypeError("at least one bot name is required")
    duplicates = tuple(name for index, name in enumerate(names) if name in names[:index])
    if duplicates:
        duplicate_names = ", ".join(dict.fromkeys(duplicates))
        raise argparse.ArgumentTypeError(f"duplicate bot name(s): {duplicate_names}")
    unknown = tuple(name for name in names if name not in BOT_REGISTRY)
    if unknown:
        unknown_names = ", ".join(dict.fromkeys(unknown))
        raise argparse.ArgumentTypeError(f"unknown bot name(s): {unknown_names}")
    return names


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Garboid PocketRocks bots against the live service"
    )
    parser.add_argument(
        "--bots",
        type=_bot_names,
        default=tuple(BOT_REGISTRY),
        help=(
            "comma-separated bot names; defaults to "
            f"{','.join(BOT_REGISTRY)}"
        ),
    )
    return parser
```

- [ ] **Step 4: Verify GREEN and focused static checks**

Run:

```bash
uv run pytest tests/bots/test_launcher.py -q
uv run ruff check src/garboid_pocketrocks/bots/launcher.py tests/bots/test_launcher.py
uv run mypy src/garboid_pocketrocks/bots/launcher.py tests/bots/test_launcher.py
```

Expected: all commands PASS.

- [ ] **Step 5: Commit registry and selection behavior**

```bash
git add src/garboid_pocketrocks/bots/launcher.py tests/bots/test_launcher.py
git commit -m "feat: define live bot launcher registry"
```

---

### Task 3: Coordinate concurrent bot runtimes

**Files:**
- Modify: `tests/bots/test_launcher.py`
- Modify: `src/garboid_pocketrocks/bots/launcher.py`
- Modify: `pyproject.toml:28-31`

**Interfaces:**
- Consumes: `RunnableBot`, `BotFactory`, `BOT_REGISTRY`, and each SDK wrapper's `run_async()`.
- Produces: `BotRuntimeStopped.bot_name`, `run_bots(names, registry=BOT_REGISTRY) -> None`, and `main(argv=None) -> None`; installs `garboid-bots`.

- [ ] **Step 1: Add controllable fake runtimes and lifecycle tests**

Append to `tests/bots/test_launcher.py`:

```python
import asyncio
from collections.abc import Callable, Mapping

from garboid_pocketrocks.bots.launcher import (
    BotFactory,
    BotRuntimeStopped,
    main,
    run_bots,
)


class _ControlledBot:
    def __init__(
        self,
        *,
        started: asyncio.Event,
        cancelled: asyncio.Event,
        release: asyncio.Event,
        returns: bool,
    ) -> None:
        self._started = started
        self._cancelled = cancelled
        self._release = release
        self._returns = returns

    async def run_async(self) -> None:
        self._started.set()
        try:
            await self._release.wait()
            if not self._returns:
                await asyncio.Future[None]()
        except asyncio.CancelledError:
            self._cancelled.set()
            raise


def _factory(
    *,
    started: asyncio.Event,
    cancelled: asyncio.Event,
    release: asyncio.Event,
    returns: bool = False,
    constructions: list[str] | None = None,
    name: str = "",
) -> BotFactory:
    def build() -> _ControlledBot:
        if constructions is not None:
            constructions.append(name)
        return _ControlledBot(
            started=started,
            cancelled=cancelled,
            release=release,
            returns=returns,
        )

    return build


def test_run_bots_constructs_once_starts_concurrently_and_cancels_together() -> None:
    async def exercise() -> None:
        started = (asyncio.Event(), asyncio.Event())
        cancelled = (asyncio.Event(), asyncio.Event())
        releases = (asyncio.Event(), asyncio.Event())
        constructions: list[str] = []
        registry: Mapping[str, BotFactory] = {
            "first": _factory(
                started=started[0],
                cancelled=cancelled[0],
                release=releases[0],
                constructions=constructions,
                name="first",
            ),
            "second": _factory(
                started=started[1],
                cancelled=cancelled[1],
                release=releases[1],
                constructions=constructions,
                name="second",
            ),
        }

        launcher = asyncio.create_task(run_bots(("first", "second"), registry=registry))
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in started)),
            timeout=1,
        )
        assert constructions == ["first", "second"]

        launcher.cancel()
        with pytest.raises(asyncio.CancelledError):
            await launcher
        assert all(event.is_set() for event in cancelled)

    asyncio.run(exercise())


def test_runtime_return_cancels_siblings_and_identifies_stopped_bot() -> None:
    async def exercise() -> None:
        returning_started = asyncio.Event()
        sibling_started = asyncio.Event()
        returning_release = asyncio.Event()
        sibling_release = asyncio.Event()
        sibling_cancelled = asyncio.Event()
        registry: Mapping[str, BotFactory] = {
            "returning": _factory(
                started=returning_started,
                cancelled=asyncio.Event(),
                release=returning_release,
                returns=True,
            ),
            "sibling": _factory(
                started=sibling_started,
                cancelled=sibling_cancelled,
                release=sibling_release,
            ),
        }

        launcher = asyncio.create_task(
            run_bots(("returning", "sibling"), registry=registry)
        )
        await asyncio.wait_for(
            asyncio.gather(returning_started.wait(), sibling_started.wait()),
            timeout=1,
        )
        returning_release.set()

        with pytest.raises(BotRuntimeStopped, match="returning") as raised:
            await launcher
        assert raised.value.bot_name == "returning"
        assert sibling_cancelled.is_set()

    asyncio.run(exercise())


def test_invalid_cli_input_does_not_construct_bots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed = False

    def fail_if_constructed() -> _ControlledBot:
        nonlocal constructed
        constructed = True
        raise AssertionError("invalid CLI input must not construct bots")

    monkeypatch.setitem(BOT_REGISTRY, "random", fail_if_constructed)

    with pytest.raises(SystemExit) as raised:
        main(["--bots", "missing"])

    assert raised.value.code == 2
    assert not constructed
```

Move all imports to the top of the file and remove the unused `Callable` import if Ruff reports it.

- [ ] **Step 2: Run the lifecycle tests and verify RED**

Run:

```bash
uv run pytest tests/bots/test_launcher.py -q
```

Expected: collection ERROR because `BotRuntimeStopped`, `run_bots`, and `main` do not exist.

- [ ] **Step 3: Implement structured concurrency and the command entry point**

Expand `src/garboid_pocketrocks/bots/launcher.py` with these imports:

```python
import asyncio
from collections.abc import Callable, Mapping, Sequence

from pocketrocks._logging import install_default_logging
```

Then add:

```python
class BotRuntimeStopped(RuntimeError):
    def __init__(self, bot_name: str) -> None:
        self.bot_name = bot_name
        super().__init__(f"bot runtime stopped: {bot_name}")


async def _run_bot(name: str, bot: RunnableBot) -> None:
    await bot.run_async()
    raise BotRuntimeStopped(name)


async def run_bots(
    names: Sequence[str],
    *,
    registry: Mapping[str, BotFactory] = BOT_REGISTRY,
) -> None:
    bots = tuple((name, registry[name]()) for name in names)
    try:
        async with asyncio.TaskGroup() as group:
            for name, bot in bots:
                group.create_task(_run_bot(name, bot), name=name)
    except BaseExceptionGroup as errors:
        if errors.exceptions and all(
            isinstance(error, BotRuntimeStopped) for error in errors.exceptions
        ):
            stopped = errors.exceptions[0]
            assert isinstance(stopped, BotRuntimeStopped)
            raise stopped from None
        raise


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    install_default_logging()
    try:
        asyncio.run(run_bots(args.bots))
    except KeyboardInterrupt:
        return
    except BotRuntimeStopped as error:
        parser.exit(1, f"{parser.prog}: error: {error}\n")
```

Add the console script to `pyproject.toml`:

```toml
[project.scripts]
garboid-bots = "garboid_pocketrocks.bots.launcher:main"
garboid-random-bot = "garboid_pocketrocks.bots.random_bot:main"
garboid-simulate = "garboid_pocketrocks.simulator.cli:main"
garboid-train = "garboid_pocketrocks.neural.cli:main"
```

- [ ] **Step 4: Verify GREEN, console metadata, and focused static checks**

Run:

```bash
uv run pytest tests/bots/test_launcher.py -q
uv run garboid-bots --help
uv run ruff check src/garboid_pocketrocks/bots/launcher.py tests/bots/test_launcher.py
uv run mypy src/garboid_pocketrocks/bots/launcher.py tests/bots/test_launcher.py
```

Expected: tests and static checks PASS; help output lists `--bots` and exits without constructing or connecting a bot.

- [ ] **Step 5: Commit the concurrent launcher**

```bash
git add pyproject.toml src/garboid_pocketrocks/bots/launcher.py tests/bots/test_launcher.py
git commit -m "feat: run live bots concurrently"
```

---

### Task 4: Document the local fleet and run the complete quality gate

**Files:**
- Modify: `README.md:47-70`
- Modify: `README.md:172-186`
- Modify: `src/garboid_pocketrocks/simulator/cli.py:49-55`

**Interfaces:**
- Consumes: installed `garboid-bots` command, live identity class constants, and existing `.env` SDK configuration.
- Produces: user-facing instructions for all-bot and subset runs; removes the stale simulator warning that heuristic IDs are placeholders.

- [ ] **Step 1: Update live-bot usage near the existing random-bot command**

Rename the section to `## Live bots`, retain the direct seeded random-bot
examples, and add:

````markdown
Run every defined bot concurrently against the live service:

```bash
uv run garboid-bots
```

The launcher starts random, aggressive, balanced, and passive by default. Run
a subset with:

```bash
uv run garboid-bots --bots aggressive,passive
```

All selected bots share the API key, server, capacity, logging, and reconnect
settings from `.env`; each connection uses its wrapper's committed public bot
ID. Ctrl+C shuts down the complete local group.
````

- [ ] **Step 2: Replace placeholder IDs and remove stale warnings**

Change the heuristic identity table to:

```markdown
| `aggressive` | `AggressiveHeuristicBrain` | `AggressiveHeuristicBot` | `AGGRESSIVE_HEURISTIC_BOT_SPEC` | `bot_386b81bb-14df-477a-8d4c-0231cf1b3b1a` |
| `balanced` | `BalancedHeuristicBrain` | `BalancedHeuristicBot` | `BALANCED_HEURISTIC_BOT_SPEC` | `bot_265c84aa-f28e-4a35-b4de-a4f4ee406415` |
| `passive` | `PassiveHeuristicBrain` | `PassiveHeuristicBot` | `PASSIVE_HEURISTIC_BOT_SPEC` | `bot_9d33c9de-4d90-4608-9a58-d2c77d93e0bd` |
```

Replace the development-placeholder paragraph with:

```markdown
These activated public IDs are committed class constants and are used by the
local live launcher. Fast bot wrappers reconcile the chart, starting cash,
private-card count, and objective state exposed by each SDK context with their
configured ruleset knowledge. Pass an explicit `Ruleset` when a game uses
different resource or action deck counts, because the SDK context does not
expose those initial counts.
```

In `src/garboid_pocketrocks/simulator/cli.py`, change the `--bots` help to:

```python
help="comma-separated registered bot names (random, aggressive, balanced, passive)",
```

- [ ] **Step 3: Run documentation consistency checks**

Run:

```bash
rg -n "bot_00000000|development-only|must be replaced" README.md src tests
```

Expected: no matches.

- [ ] **Step 4: Run the complete repository quality gate**

Run:

```bash
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest
```

Expected: every command PASS with no live network connection.

- [ ] **Step 5: Review the final diff and commit documentation**

Run:

```bash
git diff --check
git status --short
git diff -- README.md src/garboid_pocketrocks/simulator/cli.py
```

Confirm that only planned files changed and no `.env` content appears. Then:

```bash
git add README.md src/garboid_pocketrocks/simulator/cli.py
git commit -m "docs: explain local bot launcher"
```

- [ ] **Step 6: Perform optional brief live verification only with explicit approval**

Confirm `.env` exists without reading it:

```bash
test -f .env
```

If the user explicitly authorizes a live connection, run:

```bash
uv run garboid-bots
```

Expected: logs show four successful connections. Interrupt promptly with
Ctrl+C. If credentials or the server reject a connection, report that outcome
separately from the passing offline quality gate and do not expose
configuration values.
