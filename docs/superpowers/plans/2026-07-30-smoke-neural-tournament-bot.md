# Smoke Neural Tournament Bot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use braze-superpowers:subagent-driven-development (recommended) or braze-superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze `vector_ppo_small_v1_g1500` as a reproducible inference bot, include it in the standard tournament field, run the 15,000-game default tournament, and merge the verified result into local `main`.

**Architecture:** A lazy, process-local neural runtime loads the committed inference-only checkpoint and exposes a history-aware decision method. `MatchRunner` supplies exact SDK public history to brains that implement the optional protocol while preserving the existing interface for all other bots.

**Tech Stack:** Python 3.14, PocketRocks SDK simulation, PyTorch 2.13, pytest, Ruff, mypy, multiprocessing tournament runner.

## Global Constraints

- Preserve the immutable local identity `vector_ppo_small_v1_g1500`.
- The checkpoint must declare 1,500 completed games and contain only `manifest.json` and `model.pt`.
- The policy must consume exact public history and use deterministic masked argmax inference.
- Keep Torch optional at registry import time; tournament execution with this default bot uses `uv run --extra neural`.
- Preserve all released heuristic names and behaviors.
- Preserve the existing unstaged `tests/neural/test_rollout.py` edit in the primary `main` worktree.

---

### Task 1: Freeze and validate the inference checkpoint

**Files:**
- Modify: `.gitignore`
- Create: `src/garboid_pocketrocks/neural/checkpoints/vector_ppo_small_v1_g1500/manifest.json`
- Create: `src/garboid_pocketrocks/neural/checkpoints/vector_ppo_small_v1_g1500/model.pt`
- Create: `tests/neural/test_smoke_tournament_bot.py`

**Interfaces:**
- Consumes: `export_inference_checkpoint(training_checkpoint: Path, output_path: Path, *, device: torch.device) -> Path`
- Produces: `SMOKE_BOT_NAME`, `SMOKE_CHECKPOINT_PATH`, and a validated inference bundle available to later tasks.

- [ ] **Step 1: Write the failing checkpoint identity test**

```python
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from garboid_pocketrocks.neural.checkpoint import load_inference_checkpoint
from garboid_pocketrocks.neural.tournament_bot import (
    SMOKE_BOT_NAME,
    SMOKE_CHECKPOINT_PATH,
)


def test_smoke_checkpoint_is_frozen_at_named_training_age() -> None:
    loaded = load_inference_checkpoint(
        SMOKE_CHECKPOINT_PATH,
        device=torch.device("cpu"),
    )

    assert SMOKE_BOT_NAME == "vector_ppo_small_v1_g1500"
    assert {item.name for item in SMOKE_CHECKPOINT_PATH.iterdir()} == {
        "manifest.json",
        "model.pt",
    }
    assert loaded.manifest.completed_episodes == 1_500
    assert loaded.manifest.completed_updates == 1
    assert loaded.manifest.supported_ruleset_names == tuple(
        f"live-{chart}" for chart in "ABCDE"
    )
    assert loaded.manifest.supported_player_counts == (3, 4, 5)
    assert len(loaded.manifest.parameter_digest) == 64
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
uv run --extra neural pytest tests/neural/test_smoke_tournament_bot.py -q
```

Expected: collection fails because `garboid_pocketrocks.neural.tournament_bot`
and its checkpoint do not exist.

- [ ] **Step 3: Add the immutable identity and path shell**

Create `src/garboid_pocketrocks/neural/tournament_bot.py` with only import-safe
constants at this stage:

```python
from __future__ import annotations

from pathlib import Path

SMOKE_BOT_NAME = "vector_ppo_small_v1_g1500"
SMOKE_CHECKPOINT_PATH = (
    Path(__file__).with_name("checkpoints") / SMOKE_BOT_NAME
)
```

- [ ] **Step 4: Permit only packaged neural checkpoints through `.gitignore`**

Append:

```gitignore
!src/garboid_pocketrocks/neural/checkpoints/
!src/garboid_pocketrocks/neural/checkpoints/**/
!src/garboid_pocketrocks/neural/checkpoints/**/model.pt
```

- [ ] **Step 5: Export the validated training checkpoint**

Run:

```bash
uv run --extra neural python -c 'from pathlib import Path; import torch; from garboid_pocketrocks.neural.training_checkpoint import export_inference_checkpoint; export_inference_checkpoint(Path("artifacts/neural-vector-smoke-final/checkpoints/latest"), Path("src/garboid_pocketrocks/neural/checkpoints/vector_ppo_small_v1_g1500"), device=torch.device("cpu"))'
```

Expected: the destination contains exactly `manifest.json` and `model.pt`.

- [ ] **Step 6: Run the test and verify GREEN**

Run:

```bash
uv run --extra neural pytest tests/neural/test_smoke_tournament_bot.py -q
```

Expected: `1 passed`.

- [ ] **Step 7: Commit the frozen artifact**

```bash
git add .gitignore tests/neural/test_smoke_tournament_bot.py \
  src/garboid_pocketrocks/neural/tournament_bot.py \
  src/garboid_pocketrocks/neural/checkpoints/vector_ppo_small_v1_g1500
git commit -m "feat: freeze smoke neural policy"
```

### Task 2: Deliver exact public history to capable brains

**Files:**
- Modify: `src/garboid_pocketrocks/bots/base.py`
- Modify: `src/garboid_pocketrocks/bots/__init__.py`
- Modify: `src/garboid_pocketrocks/simulator/runner.py`
- Modify: `tests/simulator/test_runner.py`

**Interfaces:**
- Produces: `HistoryAwareBotBrain.choose_decision_with_history(context, ruleset, history) -> BotDecision`
- Consumes: `public_history_from_sdk_events(session.events) -> PublicHistory`

- [ ] **Step 1: Write the failing history-delivery test**

Add a top-level test brain whose `choose_decision` raises and whose
`choose_decision_with_history` records each history before returning a legal
pass. Run a three-player match and assert:

```python
assert _RECORDED_HISTORIES
assert all(isinstance(history[0], PublicGameSetup) for history in _RECORDED_HISTORIES)
for history in _RECORDED_HISTORIES:
    assert history == public_history_from_sdk_events(match.events[: len(history)])
```

This compares adapter output for each observed prefix because `PublicHistory`
and SDK event objects are different types.

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
uv run pytest tests/simulator/test_runner.py::test_match_runner_supplies_exact_public_history -q
```

Expected: the ordinary `choose_decision` raises because the runner does not yet
detect the history-aware interface.

- [ ] **Step 3: Add the optional runtime-checkable protocol**

In `bots/base.py`:

```python
@runtime_checkable
class HistoryAwareBotBrain(Protocol):
    def choose_decision_with_history(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
        history: PublicHistory,
    ) -> BotDecision:
        """Return one decision using exact immutable public history."""
```

Re-export it from `bots/__init__.py`.

- [ ] **Step 4: Route exact history in `MatchRunner`**

Before iterating pending seats, parse the current session events once:

```python
history = public_history_from_sdk_events(session.events)
```

For each brain:

```python
if isinstance(brain, HistoryAwareBotBrain):
    decision = brain.choose_decision_with_history(context, knowledge, history)
else:
    decision = brain.choose_decision(context, knowledge)
```

Keep existing validation, fault recording, and fallback behavior unchanged.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```bash
uv run pytest tests/simulator/test_runner.py tests/adapters/test_public_history.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit history-aware simulation**

```bash
git add src/garboid_pocketrocks/bots/base.py \
  src/garboid_pocketrocks/bots/__init__.py \
  src/garboid_pocketrocks/simulator/runner.py \
  tests/simulator/test_runner.py
git commit -m "feat: deliver public history to simulation bots"
```

### Task 3: Implement deterministic smoke-policy inference

**Files:**
- Modify: `src/garboid_pocketrocks/neural/tournament_bot.py`
- Modify: `tests/neural/test_smoke_tournament_bot.py`

**Interfaces:**
- Produces: `VectorPpoSmallV1G1500Brain` and `VECTOR_PPO_SMALL_V1_G1500_BOT_SPEC`
- Consumes: the history-aware protocol, frozen checkpoint, encoder, masked policy evaluator, and action codec.

- [ ] **Step 1: Write failing inference and pickling tests**

Use `SdkGameSession.start(player_count=3, seed=...)` to obtain an exact pending
context and event history. Assert two calls return the same legal decision:

```python
brain = VectorPpoSmallV1G1500Brain(seed=7)
context = session.pending.contexts[0][1]
history = public_history_from_sdk_events(session.events)
first = brain.choose_decision_with_history(context, knowledge, history)
second = brain.choose_decision_with_history(context, knowledge, history)

assert first == second
context.validate(first)
assert pickle.loads(pickle.dumps(VECTOR_PPO_SMALL_V1_G1500_BOT_SPEC)) == (
    VECTOR_PPO_SMALL_V1_G1500_BOT_SPEC
)
```

Also run one fixed-seed match with the smoke bot and two random bots under
`FaultMode.RAISE` and assert it completes.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
uv run --extra neural pytest tests/neural/test_smoke_tournament_bot.py -q
```

Expected: imports fail because the brain and spec are not defined.

- [ ] **Step 3: Implement a lazy process-local runtime**

Use `functools.cache` and local imports so registry import does not eagerly
import Torch:

```python
@dataclass(frozen=True, slots=True)
class _Runtime:
    model: NeuralPolicy
    encoder: NeuralObservationEncoder
    codec: ActionCodec
    device: torch.device


@cache
def _runtime() -> _Runtime:
    device = torch.device("cpu")
    loaded = load_inference_checkpoint(SMOKE_CHECKPOINT_PATH, device=device)
    bounds = EnvironmentBounds(
        loaded.manifest.encoder_config.max_bid,
        loaded.manifest.encoder_config.max_hand_size,
    )
    codec = ActionCodec(bounds)
    return _Runtime(
        model=loaded.model,
        encoder=NeuralObservationEncoder(
            loaded.manifest.encoder_config,
            bounds,
            action_codec=codec,
        ),
        codec=codec,
        device=device,
    )
```

Keep Torch and neural implementation imports inside `_runtime` or guarded by
`TYPE_CHECKING`.

- [ ] **Step 4: Implement deterministic legal decisions**

```python
def choose_decision_with_history(
    self,
    context: DecisionContext,
    ruleset: RulesetKnowledge,
    history: PublicHistory,
) -> BotDecision:
    runtime = self._runtime
    observation = runtime.encoder.encode(context, ruleset, history)
    batch = batch_observations((observation,), runtime.device)
    with torch.inference_mode():
        output = runtime.model(batch)
        selection = evaluate_masked_policy(
            output,
            batch,
            generator=None,
            deterministic=True,
        )
    return runtime.codec.decode(int(selection.actions[0].item()))
```

The ordinary `choose_decision` raises `RuntimeError("smoke neural policy requires public history")`.

Define:

```python
VECTOR_PPO_SMALL_V1_G1500_BOT_SPEC = BotSpec.for_simulation(
    SMOKE_BOT_NAME,
    VectorPpoSmallV1G1500Brain,
)
```

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```bash
uv run --extra neural pytest \
  tests/neural/test_smoke_tournament_bot.py \
  tests/simulator/test_runner.py \
  tests/simulator/test_monte_carlo.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit inference**

```bash
git add src/garboid_pocketrocks/neural/tournament_bot.py \
  tests/neural/test_smoke_tournament_bot.py
git commit -m "feat: run frozen smoke policy in simulations"
```

### Task 4: Register the smoke bot in the standard tournament

**Files:**
- Modify: `src/garboid_pocketrocks/bots/registry.py`
- Modify: `src/garboid_pocketrocks/bots/__init__.py`
- Modify: `tests/bots/test_registry.py`
- Modify: `tests/tournament/test_cli.py`
- Modify: `README.md`
- Modify: `src/garboid_pocketrocks/neural/README.md`

**Interfaces:**
- Consumes: `VECTOR_PPO_SMALL_V1_G1500_BOT_SPEC`
- Produces: registry and default-field selection containing the immutable smoke identity.

- [ ] **Step 1: Update registry expectations first**

Append `vector_ppo_small_v1_g1500` to both expected tuples in
`tests/bots/test_registry.py`. Update the curated-default CLI expectation.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
uv run --extra neural pytest tests/bots/test_registry.py tests/tournament/test_cli.py -q
```

Expected: expected bot tuples contain the smoke identity but the registry does
not.

- [ ] **Step 3: Add the spec to registry and exports**

Import the spec from `garboid_pocketrocks.neural.tournament_bot`, append it to
`BOT_SPECS`, and append it to `DEFAULT_TOURNAMENT_BOT_SPECS`. Re-export the
brain and spec from `bots/__init__.py`.

- [ ] **Step 4: Preserve the base-dependency CLI smoke test**

Pass:

```text
--exclude-bots vector_ppo_small_v1_g1500
```

to the subprocess test that intentionally runs `uv run garboid-tournament`
without neural extras. Add a neural-enabled direct default-resolution assertion
through the registry tests.

- [ ] **Step 5: Document standard invocation**

Document:

```bash
uv run --extra neural garboid-tournament
```

and explain that the default field includes the frozen 1,500-game smoke policy.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run:

```bash
uv run --extra neural pytest \
  tests/bots/test_registry.py \
  tests/tournament/test_cli.py \
  tests/neural/test_smoke_tournament_bot.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit standard registration**

```bash
git add src/garboid_pocketrocks/bots/registry.py \
  src/garboid_pocketrocks/bots/__init__.py \
  tests/bots/test_registry.py tests/tournament/test_cli.py \
  README.md src/garboid_pocketrocks/neural/README.md
git commit -m "feat: add smoke policy to default tournament"
```

### Task 5: Verify and smoke-test the tournament harness

**Files:**
- No production changes expected.

**Interfaces:**
- Consumes: registered smoke bot through the same spawned worker path as the full tournament.

- [ ] **Step 1: Run formatting, lint, and type checks**

```bash
uv run ruff format .
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run --extra neural mypy --config-file mypy.neural.ini src tests
```

Expected: every command exits zero.

- [ ] **Step 2: Run the complete test suite**

```bash
uv run --extra neural pytest -q
```

Expected: at least 645 tests pass with no failures.

- [ ] **Step 3: Run a 120-game spawned tournament smoke**

```bash
uv run --extra neural garboid-tournament \
  --games 120 \
  --bootstrap-samples 0 \
  --output-dir /private/tmp/vector-ppo-small-v1-g1500-tournament-smoke
```

Expected: all 15 chart/player-count cells complete, the smoke bot records zero
faults, and three report files are written.

### Task 6: Run and record the full default tournament

**Files:**
- Create: `docs/benchmarks/tournaments/2026-07-30-vector-ppo-small-v1-g1500-default/ratings.csv`
- Create: `docs/benchmarks/tournaments/2026-07-30-vector-ppo-small-v1-g1500-default/summary.json`
- Create: `docs/benchmarks/tournaments/2026-07-30-vector-ppo-small-v1-g1500-default/report.html`
- Create: `docs/benchmarks/2026-07-30-vector-ppo-small-v1-g1500-tournament.md`

**Interfaces:**
- Produces: fixed-seed strength and throughput evidence for the frozen policy.

- [ ] **Step 1: Run the unchanged 15,000-game default**

```bash
/usr/bin/time -p uv run --extra neural garboid-tournament \
  --output-dir docs/benchmarks/tournaments/2026-07-30-vector-ppo-small-v1-g1500-default
```

Expected configuration in `summary.json`:

```json
{
  "games": 15000,
  "player_counts": [3, 4, 5],
  "charts": ["A", "B", "C", "D", "E"],
  "root_seed": 0,
  "batch_size": 64,
  "bootstrap_samples": 200
}
```

- [ ] **Step 2: Validate the generated evidence**

Read `summary.json` and assert:

- all eight default bot names are present;
- the smoke row has `faults == 0`;
- every bot has nonzero game exposure;
- pair exposure minimum is nonzero;
- model diagnostic numbers are finite and the optimizer message is nonempty;
- at least one bootstrap sample converged and every interval is finite.

- [ ] **Step 3: Write the benchmark summary**

Record the exact command, commit SHA, machine worker count, wall seconds,
games/second, complete leaderboard, and the smoke bot's rank, rating interval,
games, win rate, mean money, and faults.

- [ ] **Step 4: Commit benchmark evidence**

```bash
git add docs/benchmarks/2026-07-30-vector-ppo-small-v1-g1500-tournament.md \
  docs/benchmarks/tournaments/2026-07-30-vector-ppo-small-v1-g1500-default
git commit -m "bench: rank smoke neural policy"
```

### Task 7: Final verification and local-main merge

**Files:**
- Preserve: primary-worktree `tests/neural/test_rollout.py` unstaged edit.

**Interfaces:**
- Produces: a verified merge commit on local `main`.

- [ ] **Step 1: Run final verification on the feature branch**

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run --extra neural mypy --config-file mypy.neural.ini src tests
uv run --extra neural pytest -q
git diff --check
```

Expected: all checks exit zero.

- [ ] **Step 2: Preserve the primary-worktree edit**

Save a backup patch and stash only `tests/neural/test_rollout.py` with a unique
message before merging.

- [ ] **Step 3: Merge the focused branch into local `main`**

```bash
git pull --ff-only
git merge --no-ff codex/smoke-bot-tournament
```

- [ ] **Step 4: Verify the merged result**

Run the same formatting, lint, type, test, and diff checks on local `main`.

- [ ] **Step 5: Restore the preserved unstaged edit**

Pop the named preservation stash, verify it is the only unstaged change, and
confirm the smoke-bot merge is an ancestor of `main`.
