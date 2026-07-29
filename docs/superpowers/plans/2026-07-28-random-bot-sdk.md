# Random Bot SDK Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use braze-superpowers:subagent-driven-development (recommended) or braze-superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a seeded random `PocketRocksBot`, verify it through the SDK's in-memory runtime, publish it, and briefly test a real credentialed connection.

**Architecture:** `RandomBot` directly subclasses the SDK's `PocketRocksBot` and implements its async `choose_decision` contract. It returns SDK `BotDecision` values without a translation layer. The SDK owns configuration, connection management, validation, concurrency, and wire I/O.

**Tech Stack:** Python 3.14, PocketRocks Python SDK commit `597857446d47ac0890609a4767cad561578a2519`, pytest, Ruff, mypy, uv, GitHub Actions

## Global Constraints

- `RandomBot` must directly subclass `PocketRocksBot`.
- The bot must return official SDK `BotDecision` values.
- Bid amounts must be sampled uniformly from the inclusive legal range, with zero represented as `pass`.
- Reveal indices must be sampled uniformly from the valid index range.
- An optional integer seed must make decision sequences reproducible.
- Empty or impossible action ranges must safely pass.
- SDK connection, authentication, heartbeat, reconnection, deadline, validation, and transport behavior must not be reimplemented.
- No generic policy abstraction or simulator model is introduced in this milestone.
- Tests must use the SDK's public `scenario`, `FakeTransport`, and `decode_frames` helpers.
- No real credential may be printed, read into source control, or committed.
- The live connection must not begin until local verification, push, and CI all succeed.
- No pull request is needed; this project is being developed directly on its initial `main` branch with user approval.

---

### Task 1: Implement and document the random SDK bot

**Files:**
- Create: `src/garboid_pocketrocks/bots/random_bot.py`
- Create: `tests/bots/test_random_bot.py`
- Modify: `src/garboid_pocketrocks/bots/__init__.py`
- Modify: `pyproject.toml`
- Modify: `.env.example`
- Modify: `README.md`
- Regenerate: `uv.lock`
- Create: `docs/superpowers/plans/2026-07-28-random-bot-sdk.md`

**Interfaces:**
- Consumes: `pocketrocks.DecisionContext`
- Produces: `pocketrocks.BotDecision`
- Produces: `RandomBot(PocketRocksBot)`
- Produces: console command `garboid-random-bot [--seed INTEGER]`

- [ ] **Step 1: Write all random-bot behavior and runtime tests**

Create `tests/bots/test_random_bot.py`:

```python
from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest
from pocketrocks import ActionId, BotDecision, DecisionContext, Suit
from pocketrocks.testing import FakeTransport, decode_frames, scenario

from garboid_pocketrocks.bots.random_bot import RandomBot


def _bot(seed: int) -> RandomBot:
    return RandomBot(
        seed=seed,
        api_key="test-key",
        bot_id="test-bot",
        server_url="ws://example.test",
        reconnect=False,
    )


def _bid_context(max_amount: int | None) -> DecisionContext:
    return (
        scenario(players=3, starting_cash=20)
        .turn(ActionId.AUCTION1, resources=(Suit.BRICK, Suit.WOOD))
        .deciding(seat=0, hand=[Suit.BRICK], kind="submitBid")
        .override(legal_max_amount=max_amount)
        .to_context()
    )


def _reveal_context(count: int) -> DecisionContext:
    hand = [Suit.BRICK] * max(0, count)
    return (
        scenario(players=3, starting_cash=20)
        .deciding(seat=0, hand=hand, kind="selectInfoToReveal")
        .override(revealable_count=count)
        .to_context()
    )


def _choose(bot: RandomBot, context: DecisionContext) -> BotDecision:
    return asyncio.run(bot.choose_decision(context))


async def _choose_all(
    bot: RandomBot,
    contexts: Sequence[DecisionContext],
) -> list[BotDecision]:
    return [await bot.choose_decision(context) for context in contexts]


@pytest.mark.parametrize("max_amount", [None, 0, -1])
def test_nonpositive_or_missing_bid_limit_passes(max_amount: int | None) -> None:
    context = _bid_context(max_amount)

    decision = _choose(_bot(seed=1), context)

    assert decision == BotDecision.pass_turn()
    assert context.is_legal(decision)


def test_positive_bid_limit_produces_every_legal_amount_across_seeds() -> None:
    context = _bid_context(7)

    decisions = [_choose(_bot(seed=seed), context) for seed in range(100)]
    amounts = {0 if decision.action_kind == "pass" else decision.value for decision in decisions}

    assert amounts == set(range(8))
    assert all(context.is_legal(decision) for decision in decisions)


def test_empty_reveal_range_passes() -> None:
    context = _reveal_context(0)

    decision = _choose(_bot(seed=1), context)

    assert decision == BotDecision.pass_turn()
    assert context.is_legal(decision)


def test_reveal_choices_cover_valid_indices_across_seeds() -> None:
    context = _reveal_context(3)

    decisions = [_choose(_bot(seed=seed), context) for seed in range(50)]

    assert {decision.value for decision in decisions} == {0, 1, 2}
    assert all(decision.action_kind == "selectInfoToReveal" for decision in decisions)
    assert all(context.is_legal(decision) for decision in decisions)


def test_equal_seeds_produce_equal_decision_sequences() -> None:
    contexts = [_bid_context(11), _reveal_context(4)] * 10

    left = asyncio.run(_choose_all(_bot(seed=42), contexts))
    right = asyncio.run(_choose_all(_bot(seed=42), contexts))

    assert left == right


def test_fake_transport_drives_random_bot_runtime() -> None:
    request_id = "11111111-1111-1111-1111-111111111111"
    game = (
        scenario(players=3, starting_cash=20)
        .turn(ActionId.AUCTION1, resources=(Suit.BRICK, Suit.WOOD))
        .deciding(
            seat=0,
            hand=[Suit.BRICK, Suit.WOOD],
            kind="submitBid",
            request_id=request_id,
        )
    )
    context = game.to_context()
    transport = FakeTransport([game.to_bytes()])
    bot = RandomBot(
        seed=7,
        api_key="test-key",
        bot_id="test-bot",
        server_url="ws://example.test",
        reconnect=False,
        transport=transport,
    )

    asyncio.run(bot.run_async())

    sent = decode_frames(transport.sent_messages)
    assert len(sent) == 1
    response = sent[0]
    assert response.kind == "decisionResponse"
    assert response.request_id == request_id
    decision = BotDecision(action_kind=response.action_kind, value=response.value)
    assert context.is_legal(decision)
    assert transport.disconnected
```

- [ ] **Step 2: Run the focused tests and verify the red state**

Run:

```bash
uv run pytest tests/bots/test_random_bot.py -q
```

Expected: collection fails with
`ModuleNotFoundError: No module named 'garboid_pocketrocks.bots.random_bot'`.

- [ ] **Step 3: Implement the minimal random bot and console entry function**

Create `src/garboid_pocketrocks/bots/random_bot.py`:

```python
from __future__ import annotations

import argparse
import random
from typing import Any

from pocketrocks import BotDecision, DecisionContext, PocketRocksBot


class RandomBot(PocketRocksBot):
    """PocketRocks bot that samples uniformly from the legal action range."""

    def __init__(self, *, seed: int | None = None, **sdk_options: Any) -> None:
        super().__init__(**sdk_options)
        self._random = random.Random(seed)

    async def choose_decision(self, context: DecisionContext) -> BotDecision:
        if context.decision_kind == "submitBid":
            max_amount = context.legal_max_amount
            if max_amount is None or max_amount <= 0:
                return BotDecision.pass_turn()
            amount = self._random.randint(0, max_amount)
            return BotDecision.pass_turn() if amount == 0 else BotDecision.submit_bid(amount)

        if context.revealable_count <= 0:
            return BotDecision.pass_turn()
        index = self._random.randrange(context.revealable_count)
        return BotDecision.select_info_to_reveal(index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Garboid random PocketRocks bot")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed the bot's random decisions for reproducibility",
    )
    args = parser.parse_args()
    RandomBot(seed=args.seed).run()
```

Replace `src/garboid_pocketrocks/bots/__init__.py` with:

```python
"""Reusable PocketRocks bot policies."""

from garboid_pocketrocks.bots.random_bot import RandomBot

__all__ = ["RandomBot"]
```

- [ ] **Step 4: Run the focused tests and verify the green state**

Run:

```bash
uv run pytest tests/bots/test_random_bot.py -q
```

Expected: eight test cases pass: three parameter cases and five standalone
tests.

- [ ] **Step 5: Register the console command**

Add this table to `pyproject.toml` after the project dependency list:

```toml
[project.scripts]
garboid-random-bot = "garboid_pocketrocks.bots.random_bot:main"
```

Synchronize the editable installation:

```bash
uv lock
uv sync --locked
```

Verify the command is registered without starting a connection:

```bash
uv run garboid-random-bot --help
```

Expected: help text includes `--seed` and exits with code `0`.

- [ ] **Step 6: Expand the environment template**

Replace `.env.example` with:

```dotenv
# Copy this file to `.env`, then replace the required secret placeholder.
# Never commit `.env`; it contains your secret API key.

# Required secret — obtain this from the PocketRocks dashboard.
POCKETROCKS_API_KEY=paste-your-api-key-here

# Public bot identity committed for the random-bot profile.
RANDOM_BOT_ID=bot_e0e2c541-1615-4f47-983c-224e7d888d89

# Hosted PocketRocks server. The SDK uses this value by default.
POCKETROCKS_SERVER_URL=wss://pocketrocks.xyz

# Optional SDK settings.
# POCKETROCKS_BOT_CAPACITY=1
# POCKETROCKS_LOG_LEVEL=INFO
```

- [ ] **Step 7: Document the random bot**

Insert this section in `README.md` between Setup and Quality checks:

````markdown
## Random bot

`RandomBot` follows the
[official SDK starter](https://github.com/jaiparera/pocketrocks-python-sdk/tree/develop/starter)
and directly extends `PocketRocksBot`.

Its baseline strategy is intentionally uninformed:

- sample uniformly from every legal integer bid, treating zero as pass;
- sample uniformly from every revealable card index;
- pass when no bid or reveal is available.

Run it after filling in `.env`:

```bash
uv run garboid-random-bot
```

For a reproducible decision sequence:

```bash
uv run garboid-random-bot --seed 42
```

The automated tests use the SDK's in-memory `FakeTransport`, so they exercise
the complete SDK request/response path without connecting to the live service.
Starting the command above creates a real connection and may play games if the
configured bot is active.
````

Replace the first two roadmap items with:

```markdown
1. ✅ Establish the repository scaffold and quality gates.
2. ✅ Build a random baseline bot and connect it through the Python SDK.
```

- [ ] **Step 8: Verify lock consistency and the complete quality gate**

Run:

```bash
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest
```

Expected:

- lockfile is current;
- Ruff reports all files formatted and no lint errors;
- mypy reports no issues;
- the full pytest suite passes.

If `ruff format --check` reports formatting changes, run
`uv run ruff format .`, inspect the formatter-only diff, and repeat the complete
quality gate.

- [ ] **Step 9: Commit the verified implementation**

Run:

```bash
git status --short
git diff --check
git diff --stat
```

Expected: only the random-bot implementation, tests, plan, project metadata,
environment template, and README are changed.

Then run:

```bash
git add \
  .env.example \
  README.md \
  docs/superpowers/plans/2026-07-28-random-bot-sdk.md \
  pyproject.toml \
  src/garboid_pocketrocks/bots \
  tests/bots \
  uv.lock
git commit -m "feat: add random PocketRocks bot"
```

Expected: one implementation commit after the design commit.

---

### Task 2: Publish, verify CI, and test the live connection

**Files:**
- No planned source changes

**Interfaces:**
- Consumes: public GitHub repository `chrisgarber/garboid-pocketrocks`
- Consumes: user-created ignored `.env` with a PocketRocks API key and the
  committed `RANDOM_BOT_ID`
- Produces: successful CI run for the random-bot implementation
- Produces: observed SDK connection to `wss://pocketrocks.xyz`

- [ ] **Step 1: Push the design and implementation commits**

Run:

```bash
git status --short --branch
git push origin main
```

Expected: clean `main` tracks `origin/main` after pushing the design and
implementation commits.

- [ ] **Step 2: Wait for GitHub Actions**

Find the workflow run for the pushed implementation commit:

```bash
gh run list \
  --repo chrisgarber/garboid-pocketrocks \
  --limit 5 \
  --json databaseId,status,conclusion,headSha,url
```

Then monitor the matching run:

```bash
gh run watch RUN_ID \
  --repo chrisgarber/garboid-pocketrocks \
  --exit-status
```

Expected: the CI `quality` job completes successfully.

If CI fails, inspect it with:

```bash
gh run view RUN_ID \
  --repo chrisgarber/garboid-pocketrocks \
  --log-failed
```

Find the root cause, add a failing local reproduction when possible, apply one
minimal fix, rerun the complete local quality gate, commit, push, and monitor
the replacement run.

- [ ] **Step 3: Ask the user to create the ignored credential file**

Ask the user to run:

```bash
cp .env.example .env
```

and replace the API key placeholder. The random bot ID is already configured.
Wait for confirmation.

Check only for file presence:

```bash
test -f .env
```

Expected: exit code `0`. Do not print or parse `.env`.

- [ ] **Step 4: Warn immediately before the live run**

Tell the user:

> The live SDK connection can receive and answer game decisions if this bot is
> active. I will stop it immediately after confirming the connection.

The user already approved a live connection in the design conversation, so no
additional scope expansion is required.

- [ ] **Step 5: Run and stop the live bot**

Start an interactive process:

```bash
uv run garboid-random-bot --seed 0
```

Expected SDK log:

```text
connected to wss://pocketrocks.xyz as bot ...; waiting for decision requests
```

After observing that line, send Ctrl+C promptly. Do not display credential
values.

- [ ] **Step 6: Perform final verification**

Run:

```bash
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest
git status --short --branch
git rev-parse HEAD
git rev-parse origin/main
```

Expected:

- all local checks pass;
- the worktree is clean;
- local and remote commit SHAs match;
- `.env` remains ignored and untracked.
