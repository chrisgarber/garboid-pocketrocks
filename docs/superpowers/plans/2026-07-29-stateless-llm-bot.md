# Stateless LLM Bot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use braze-superpowers:subagent-driven-development (recommended) or braze-superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a provider-neutral stateless LLM PocketRocks brain, a local Codex CLI backend, and runnable simulator/live Codex bot integrations.

**Architecture:** A `StatelessLLMBrain` composes an `LLMBackend` protocol with a `PromptSkill` protocol. `PocketRocksPromptSkill` embeds a packaged portable `SKILL.md` and renders the SDK-visible snapshot; `CodexCLIBackend` handles isolated subprocess execution without knowing game rules.

**Tech Stack:** Python 3.14, PocketRocks Python SDK, `subprocess`, `importlib.resources`, pytest, mypy, Ruff

## Global Constraints

- Version 1 is stateless and stores no Codex session ID or conversational history.
- Automated tests must not invoke a live LLM or require network access.
- LLM output accepts only surrounding whitespace plus ASCII digits and must be in the exact legal range.
- Retry exactly once, then fall back to pass for bids or reveal index `0`.
- Never invoke the Codex CLI through a shell.
- Keep the PocketRocks rules prompt in a separate packaged `SKILL.md`.
- Preserve the synchronous `BotBrain` interface used by the simulator.

---

### Task 1: Portable PocketRocks Prompt Skill

**Files:**
- Create: `src/garboid_pocketrocks/bots/llm/prompting.py`
- Create: `src/garboid_pocketrocks/bots/llm/skills/pocketrocks/SKILL.md`
- Create: `tests/bots/llm/test_prompting.py`

**Interfaces:**
- Produces: `PromptSkill.render(context: DecisionContext, ruleset: RulesetKnowledge, *, correction: str | None = None) -> str`
- Produces: `PocketRocksPromptSkill.render(...) -> str`
- Consumes: SDK `ActionId`, `Suit`, `OBJECTIVES`, `describe_action`, `describe_objective`, and `objective_payout`

- [ ] **Step 1: Write failing prompt tests**

Create contexts for bidding and revealing and assert that rendered prompts contain:

```python
assert "Auction for 1 resource card" in prompt
assert "Seat 0 (YOU): cash=$30" in prompt
assert "0: Brick" in reveal_prompt
assert "Return exactly one base-10 integer from 0 through 1" in reveal_prompt
```

Also assert the skill text contains tie breaking, action effects, reveal rules,
and final scoring; active objective IDs are paired with SDK descriptions and
payouts; and a correction appears immediately before the repeated output
contract.

- [ ] **Step 2: Run prompt tests and verify RED**

Run: `UV_CACHE_DIR=/tmp/garboid-uv-cache uv run pytest tests/bots/llm/test_prompting.py -q`

Expected: collection fails because `garboid_pocketrocks.bots.llm.prompting`
does not exist.

- [ ] **Step 3: Implement the prompt protocol, packaged skill, and renderer**

Define:

```python
class PromptSkill(Protocol):
    def render(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
        *,
        correction: str | None = None,
    ) -> str: ...


class PocketRocksPromptSkill:
    def render(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
        *,
        correction: str | None = None,
    ) -> str: ...
```

Use `importlib.resources.files` to load `skills/pocketrocks/SKILL.md`.
Render named action/suit/objective state in deterministic seat and suit order.
End with the exact legal range; bid `0` means pass and reveal indices are
zero-based.

- [ ] **Step 4: Run prompt tests and verify GREEN**

Run: `UV_CACHE_DIR=/tmp/garboid-uv-cache uv run pytest tests/bots/llm/test_prompting.py -q`

Expected: all prompt tests pass.

### Task 2: Generic Stateless LLM Brain

**Files:**
- Create: `src/garboid_pocketrocks/bots/llm/backend.py`
- Create: `src/garboid_pocketrocks/bots/llm/brain.py`
- Create: `tests/bots/llm/test_brain.py`

**Interfaces:**
- Consumes: `PromptSkill.render(...) -> str`
- Produces: `LLMBackend.complete(prompt: str, *, timeout_seconds: float) -> str`
- Produces: `LLMResponseError`
- Produces: `StatelessLLMBrain(backend, *, prompt_skill=..., timeout_seconds=30.0, deadline_margin_seconds=0.25)`
- Produces: `StatelessLLMBrain.choose_decision(context, ruleset) -> BotDecision`

- [ ] **Step 1: Write failing brain tests**

Use a scripted in-memory backend and recording prompt skill. Cover:

```python
assert brain_with(" 7\n").choose_decision(bid_context, knowledge) == BotDecision.submit_bid(7)
assert brain_with("0").choose_decision(bid_context, knowledge) == BotDecision.pass_turn()
assert brain_with("1").choose_decision(
    reveal_context, knowledge
) == BotDecision.select_info_to_reveal(1)
```

Add separate tests proving:

- prose, signs, JSON, decimals, empty text, and out-of-range integers trigger a
  retry;
- the retry prompt includes a concise correction;
- a backend exception is retried;
- a second failure falls back to pass/index `0` and logs request metadata;
- no legal positive bid and no revealable card skip the backend;
- per-attempt timeouts are positive, never exceed configuration, and reserve
  room for the remaining attempt before a live deadline.

- [ ] **Step 2: Run brain tests and verify RED**

Run: `UV_CACHE_DIR=/tmp/garboid-uv-cache uv run pytest tests/bots/llm/test_brain.py -q`

Expected: collection fails because the backend and brain modules do not exist.

- [ ] **Step 3: Implement minimal generic backend and brain behavior**

Define:

```python
class LLMBackend(Protocol):
    def complete(self, prompt: str, *, timeout_seconds: float) -> str: ...


class LLMResponseError(ValueError):
    pass
```

In `StatelessLLMBrain`, validate with `re.fullmatch(r"[0-9]+", text.strip())`,
check inclusive bounds, convert to the matching `BotDecision`, and call
`context.validate`. Catch `Exception` around each backend/parse attempt,
render one correction retry, then return the deterministic legal fallback.
Compute each timeout from `context.remaining_deadline_ms`, the configured
maximum, the safety margin, and attempts remaining.

- [ ] **Step 4: Run brain tests and verify GREEN**

Run: `UV_CACHE_DIR=/tmp/garboid-uv-cache uv run pytest tests/bots/llm/test_brain.py -q`

Expected: all brain tests pass.

### Task 3: Isolated Codex CLI Backend

**Files:**
- Create: `src/garboid_pocketrocks/bots/llm/codex_cli.py`
- Create: `tests/bots/llm/test_codex_cli.py`

**Interfaces:**
- Consumes: `LLMBackend.complete(...)`
- Produces: `CodexCLIBackend(executable="codex", model: str | None = None)`
- Produces: `CodexCLIError`

- [ ] **Step 1: Write failing backend tests**

Patch `subprocess.run` with a side effect that writes to the path following
`--output-last-message`. Assert the call:

```python
assert args[:2] == ["codex", "exec"]
assert "--ephemeral" in args
assert "--ignore-user-config" in args
assert "--ignore-rules" in args
assert args[-1] == "-"
assert kwargs["input"] == prompt
assert kwargs["shell"] is False
assert kwargs["timeout"] == 4.5
```

Add tests for optional `--model`, executable selection, nonzero exit status,
`subprocess.TimeoutExpired`, missing output, and bounded stderr diagnostics.

- [ ] **Step 2: Run backend tests and verify RED**

Run: `UV_CACHE_DIR=/tmp/garboid-uv-cache uv run pytest tests/bots/llm/test_codex_cli.py -q`

Expected: collection fails because `codex_cli.py` does not exist.

- [ ] **Step 3: Implement isolated subprocess execution**

Use `tempfile.TemporaryDirectory`, a list argument vector, stdin, text mode,
captured output, `shell=False`, and the caller's timeout. The argument vector
must include:

```python
[
    executable,
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
    temporary_directory,
    "--output-last-message",
    output_path,
    *(["--model", model] if model else []),
    "-",
]
```

Raise `CodexCLIError` for timeout, nonzero status, missing output, and empty
output. Bound included stderr to 2,000 characters.

- [ ] **Step 4: Run backend tests and verify GREEN**

Run: `UV_CACHE_DIR=/tmp/garboid-uv-cache uv run pytest tests/bots/llm/test_codex_cli.py -q`

Expected: all backend tests pass.

### Task 4: Codex Bot, Live CLI, and Simulator Registry

**Files:**
- Create: `src/garboid_pocketrocks/bots/llm/codex_bot.py`
- Create: `src/garboid_pocketrocks/bots/llm/__init__.py`
- Create: `tests/bots/llm/test_codex_bot.py`
- Modify: `src/garboid_pocketrocks/bots/__init__.py`
- Modify: `src/garboid_pocketrocks/simulator/cli.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `CodexBot`, with `BOT_NAME = "codex"` and a development-only ID
- Produces: `CODEX_BOT_SPEC`
- Produces: `garboid-codex-bot` console script
- Consumes: all Task 1-3 interfaces

- [ ] **Step 1: Write failing integration tests**

Assert:

```python
assert CodexBot.BOT_NAME == "codex"
assert isinstance(CodexBot.build_brain(None), StatelessLLMBrain)
assert _bot_names("codex,random") == ("codex", "random")
```

Use an injected scripted backend to prove the wrapper returns legal decisions.
Patch `asyncio.to_thread` to prove `CodexBot.choose_decision` offloads the
synchronous bridge. Patch `CodexBot.run` and parse CLI arguments to prove
model, timeout, and executable settings reach `CodexCLIBackend` without making
a live connection.

- [ ] **Step 2: Run integration tests and verify RED**

Run: `UV_CACHE_DIR=/tmp/garboid-uv-cache uv run pytest tests/bots/llm/test_codex_bot.py -q`

Expected: collection fails because `codex_bot.py` does not exist.

- [ ] **Step 3: Implement exports and runtime wiring**

Compose `CodexCLIBackend`, `PocketRocksPromptSkill`, and `StatelessLLMBrain`.
Use `asyncio.to_thread` in the live async bridge. Add `codex` to the simulator
registry and help copy. Add:

```toml
garboid-codex-bot = "garboid_pocketrocks.bots.llm.codex_bot:main"
```

Export the public LLM interfaces through both `bots.llm` and `bots`.

- [ ] **Step 4: Run integration and existing bot/CLI tests**

Run: `UV_CACHE_DIR=/tmp/garboid-uv-cache uv run pytest tests/bots tests/simulator/test_cli.py -q`

Expected: all selected tests pass.

### Task 5: Documentation, Packaging, and Full Verification

**Files:**
- Modify: `README.md`
- Modify: `tests/test_neural_packaging.py` or create a focused package-data test

**Interfaces:**
- Consumes: completed feature
- Produces: documented live and simulator commands and verified wheel contents

- [ ] **Step 1: Write a failing package-data assertion**

Build/install through the project's packaging test boundary and assert
`garboid_pocketrocks/bots/llm/skills/pocketrocks/SKILL.md` is included and
loadable through `PocketRocksPromptSkill`.

- [ ] **Step 2: Run the packaging test and verify RED if package data is absent**

Run: `UV_CACHE_DIR=/tmp/garboid-uv-cache uv run pytest tests/test_package.py tests/test_neural_packaging.py -q`

Expected: the new assertion either fails for missing package data or passes
because Hatch already includes package resources; if it passes, retain it as
coverage and proceed without production packaging changes.

- [ ] **Step 3: Document usage and limitations**

Add examples:

```bash
uv run garboid-codex-bot --model MODEL --timeout-seconds 30
uv run garboid-simulate --bots codex,random,random --games 1 --players 3 --seed 42
```

Document authentication through the local Codex CLI, development-only bot ID,
usage/latency/nondeterminism, stateless behavior, SDK snapshot omissions, and
how to inject another `LLMBackend`.

- [ ] **Step 4: Run full verification**

Run:

```bash
UV_CACHE_DIR=/tmp/garboid-uv-cache uv run ruff check .
UV_CACHE_DIR=/tmp/garboid-uv-cache uv run ruff format --check .
UV_CACHE_DIR=/tmp/garboid-uv-cache uv run mypy
UV_CACHE_DIR=/tmp/garboid-uv-cache uv run pytest -q
git diff --check
```

Expected: all commands exit zero with no lint, format, type, test, or whitespace
failures.

- [ ] **Step 5: Run an opt-in Codex smoke if credentials are available**

Invoke `CodexCLIBackend().complete` with a prompt that requires the answer `0`
and a bounded timeout. If the environment blocks network or lacks auth, report
that limitation separately; do not weaken the automated verification.
