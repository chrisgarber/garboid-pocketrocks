# Clean SDK Migration PR Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use braze-superpowers:subagent-driven-development (recommended) or braze-superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild Garboid PR #1 from current `main` as a conflict-free scalar SDK-engine migration that preserves merged PR #2 behavior and excludes vector, fast-context, neural-throughput, launcher, and unrelated work.

**Architecture:** Start a clean reconstruction branch from `origin/main`; never merge the contaminated PR branch. Replace Garboid's local rules engine with the SDK `SimEngine`, retain `garboid_pocketrocks.simulator` as an orchestration package, and port consumers in dependency order while treating current `main` as authoritative for heuristics and bot versioning.

**Tech Stack:** Python 3.14, pocketrocks Python SDK commit `48373524c61665c3b73ca91a2ae6420127f7da81`, NumPy, Gymnasium, PettingZoo, PyTorch, pytest, Ruff, strict mypy, GitHub Actions

## Global Constraints

- Current `origin/main`—including merged PR #2—is authoritative for future-cash heuristics, v1/v2/latest profiles, bot identities, CLI registration, and their tests.
- The SDK owns all setup, legal-action, transition, objective, reveal, scoring, event, and turn-record rules.
- `garboid_pocketrocks.simulator` remains only as session, runner, replay, Monte Carlo, CLI, error, and seeding orchestration.
- Use upstream SDK scalar simulation APIs; do not copy SDK PR #8 or PR #9 implementations into Garboid.
- Do not include `BatchSimEngine`, `batch_runner.py`, `vector_env.py`, fast-context code, neural-throughput optimizations, live-launcher work, or unrelated visible-resource work.
- Preserve competition ranking for equal terminal totals independently from the SDK's deterministic ordered-seat ranking.
- Keep the contaminated branch tip recoverable until the rebuilt PR passes hosted CI.
- Use test-first changes at every observable adapter boundary.

---

### Task 1: Create a Recoverable Clean Reconstruction Worktree

**Files:**
- Consume: `docs/superpowers/specs/2026-07-29-sdk-migration-pr-scope-design.md`
- Create in reconstruction branch: `docs/superpowers/specs/2026-07-29-sdk-migration-pr-scope-design.md`
- Create in reconstruction branch: `docs/superpowers/plans/2026-07-29-clean-sdk-migration-pr-rebuild.md`

**Interfaces:**
- Consumes: current branch tip and current `origin/main`
- Produces: isolated branch `codex/sdk-engine-migration-rebuild` rooted at `origin/main`

- [ ] **Step 1: Record and back up the contaminated tip**

```bash
git fetch origin main
git status --short --branch
git rev-parse HEAD
git branch codex/sdk-engine-migration-pre-rebuild-20260729 HEAD
```

Expected: the working tree is clean and the backup branch points at the old PR tip.

- [ ] **Step 2: Create an isolated reconstruction worktree**

Use the git-worktree skill if available. Otherwise use native Git:

```bash
git worktree add -b codex/sdk-engine-migration-rebuild \
  /private/tmp/garboid-sdk-migration-rebuild origin/main
```

Expected: `/private/tmp/garboid-sdk-migration-rebuild` is on a named branch whose parent is current `origin/main`.

- [ ] **Step 3: Port only the approved design and this plan**

```bash
git -C /private/tmp/garboid-sdk-migration-rebuild checkout \
  codex/sdk-engine-migration-pre-rebuild-20260729 -- \
  docs/superpowers/specs/2026-07-29-sdk-migration-pr-scope-design.md \
  docs/superpowers/plans/2026-07-29-clean-sdk-migration-pr-rebuild.md
git -C /private/tmp/garboid-sdk-migration-rebuild diff --check
```

Expected: only the two approved migration documents are new.

- [ ] **Step 4: Commit the reconstruction scaffold**

```bash
git -C /private/tmp/garboid-sdk-migration-rebuild add \
  docs/superpowers/specs/2026-07-29-sdk-migration-pr-scope-design.md \
  docs/superpowers/plans/2026-07-29-clean-sdk-migration-pr-rebuild.md
git -C /private/tmp/garboid-sdk-migration-rebuild commit \
  -m "docs: define clean SDK migration rebuild"
```

### Task 2: Pin the Upstream Scalar SDK and Introduce Public Rules Knowledge

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `src/garboid_pocketrocks/knowledge.py`
- Create: `tests/test_knowledge.py`

**Interfaces:**
- Consumes: SDK `DecisionContext` and `pocketrocks.sim.constants`
- Produces: `RulesetKnowledge`, `canonical_knowledge(...)`, and `knowledge_for_context(...)`

- [ ] **Step 1: Write knowledge boundary tests**

Create `tests/test_knowledge.py` with these contracts:

```python
from dataclasses import replace

import pytest
from pocketrocks.sim import SimEngine
from pocketrocks.sim.context import build_sim_context

from garboid_pocketrocks.knowledge import canonical_knowledge, knowledge_for_context


@pytest.mark.parametrize("player_count", (3, 4, 5))
@pytest.mark.parametrize("chart", ("A", "B", "C", "D", "E"))
def test_canonical_knowledge_matches_sdk_variants(
    player_count: int,
    chart: str,
) -> None:
    knowledge = canonical_knowledge(player_count, value_chart=chart)
    assert knowledge.player_count == player_count
    assert knowledge.name == f"live-{chart}"
    assert len(knowledge.value_chart) == 6
    assert sum(knowledge.resource_counts) > 0
    assert sum(knowledge.action_counts) > 0


def test_context_knowledge_uses_public_sdk_state() -> None:
    engine = SimEngine(3, "knowledge", value_chart="E")
    engine.flip_action()
    context = build_sim_context(engine, 0, "submitBid", budget_ms=60_000)
    context = replace(context, received_at=0, deadline_at=2**63 - 1)
    knowledge = knowledge_for_context(context)
    assert knowledge.value_chart == context.value_chart
    assert knowledge.player_count == context.player_count
    assert knowledge.starting_cash == context.starting_cash
```

- [ ] **Step 2: Run the tests to verify the boundary is missing**

Run:

```bash
uv run pytest tests/test_knowledge.py -q
```

Expected: collection fails because `garboid_pocketrocks.knowledge` does not exist.

- [ ] **Step 3: Pin the upstream SDK simulation toolkit**

Set the SDK dependency in `pyproject.toml` to the accepted upstream scalar
simulation commit:

```toml
"pocketrocks-python-sdk @ git+https://github.com/jaiparera/pocketrocks-python-sdk.git@48373524c61665c3b73ca91a2ae6420127f7da81",
```

This commit contains `pocketrocks.sim.SimEngine`, `LocalGame`, `ScoreRow`,
`TurnRecord`, and `pocketrocks.sim.context.build_sim_context`. Use the upstream
repository, not the fork or SDK PR #8/#9 branches. Then run:

```bash
uv lock
uv sync --locked --extra neural
uv run python -c \
  "from pocketrocks.sim import LocalGame, SimEngine; from pocketrocks.sim.context import build_sim_context; print(SimEngine, LocalGame, build_sim_context)"
```

Expected: all three public SDK symbols import from the locked environment.

- [ ] **Step 4: Implement immutable public knowledge**

Create `RulesetKnowledge` with:

```python
@dataclass(frozen=True, slots=True)
class RulesetKnowledge:
    name: str
    player_count: int
    starting_cash: int
    private_cards_per_player: int
    resource_counts: tuple[int, ...]
    action_counts: tuple[int, ...]
    value_chart: tuple[int, ...]
    objective_pool: tuple[int, ...]
    active_objective_count: int
    objectives_enabled: bool
```

Implement `canonical_knowledge(player_count, *, value_chart="A",
objectives_enabled=True) -> RulesetKnowledge` by indexing
`STARTING_CASH`, `INFO_CARDS_PER_PLAYER`, and `VALUE_CHARTS`; count
`ITEM_DECK_SUITS` for suit IDs 1–5; count `ACTION_DECK` in ascending
`ACTION_WIRE_IDS` order; and use sorted public objective IDs.

Implement `knowledge_for_context(context) -> RulesetKnowledge` by matching
`context.value_chart` to an SDK chart, calling `canonical_knowledge`, and
replacing starting cash, player count, active objective count, and the current
bot's total dealt-information count with public context fields. Reject player
counts outside 3–5, charts outside A–E, and unknown context charts.

- [ ] **Step 5: Run the knowledge tests**

```bash
uv run pytest tests/test_knowledge.py -q
uv run mypy src/garboid_pocketrocks/knowledge.py tests/test_knowledge.py
```

Expected: all knowledge tests pass and strict mypy reports no issues.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/garboid_pocketrocks/knowledge.py tests/test_knowledge.py
git commit -m "feat: derive public rules knowledge from the SDK"
```

### Task 3: Adapt Public History, PR #2 Bots, and Heuristics Without Changing Behavior

**Files:**
- Modify: `src/garboid_pocketrocks/adapters/__init__.py`
- Modify: `src/garboid_pocketrocks/adapters/public_history.py`
- Modify: `src/garboid_pocketrocks/bots/base.py`
- Modify: `src/garboid_pocketrocks/bots/heuristic.py`
- Modify: `src/garboid_pocketrocks/bots/random_bot.py`
- Modify: `src/garboid_pocketrocks/heuristics/belief.py`
- Modify: `src/garboid_pocketrocks/heuristics/reveals.py`
- Modify: `src/garboid_pocketrocks/heuristics/valuation.py`
- Modify only for imports/types if required: `tests/bots/test_heuristic_bots.py`
- Modify only for imports/types if required: `tests/heuristics/helpers.py`
- Modify only for imports/types if required: `tests/heuristics/test_belief.py`
- Modify only for imports/types if required: `tests/heuristics/test_reveals.py`
- Modify only for imports/types if required: `tests/heuristics/test_valuation.py`
- Modify: `tests/adapters/test_public_history.py`
- Create: `tests/adapters/test_sdk_history.py`

**Interfaces:**
- Consumes: SDK `DecisionContext`, public `CommonEvent` values,
  `RulesetKnowledge`,
  `knowledge_for_context`, PR #2 profiles and bot generations
- Produces: SDK-derived public history and unchanged bot decisions for the same
  public context and profile

- [ ] **Step 1: Add SDK public-history coverage and retain PR #2 regressions**

In `tests/adapters/test_sdk_history.py`, drive a seeded `SimEngine` through a
turn and assert `public_history_from_sdk_events(engine.events)` equals the
history parsed from an equivalent SDK decision request. Verify the public
auction action, bids, offered resources, and reveal without reading debug deck
order or private opponent hands.

Keep current `main`'s existing
`test_versioned_profiles_have_exact_constants` and
`test_unversioned_profiles_alias_latest_generation` unchanged. They assert the
authoritative `HEURISTIC_V1`, `HEURISTIC_V2`, `LATEST_HEURISTICS`,
`AGGRESSIVE_PROFILE`, `BALANCED_PROFILE`, and `PASSIVE_PROFILE` contracts. Do
not rename or redefine those symbols to fit the old PR branch.

- [ ] **Step 2: Run the focused suite before adaptation**

```bash
uv run pytest \
  tests/bots/test_heuristic_bots.py \
  tests/heuristics/test_belief.py \
  tests/heuristics/test_reveals.py \
  tests/heuristics/test_valuation.py -q
```

Expected: imports or type contracts fail once tests use SDK history records and
`RulesetKnowledge`.

- [ ] **Step 3: Adapt public history and rules-knowledge dependencies**

Start `adapters/public_history.py` from current `main`. Add
`public_history_from_sdk_events(events)` by placing the SDK event sequence
behind the existing strict `common_events` parser. Keep the same immutable
public-history records used by heuristics and neural encoding. Do not port the
old `simulator_history.py` adapter.

Replace `Ruleset` parameters with `RulesetKnowledge`. In
`PocketRocksFastBot`, obtain knowledge with:

```python
def _knowledge_for_context(self, context: DecisionContext) -> RulesetKnowledge:
    return knowledge_for_context(context)
```

Keep PR #2's profile objects, versioned brain classes, bot IDs, CLI names,
future-cash calculations, and decision ordering unchanged. Do not copy
`bots/launcher.py` or old branch profile constants.

- [ ] **Step 4: Run authoritative heuristic suites**

```bash
uv run pytest \
  tests/adapters/test_public_history.py \
  tests/adapters/test_sdk_history.py \
  tests/bots/test_heuristic_bots.py \
  tests/heuristics/test_cash.py \
  tests/heuristics/test_profiles.py \
  tests/heuristics/test_belief.py \
  tests/heuristics/test_reveals.py \
  tests/heuristics/test_sanity.py \
  tests/heuristics/test_valuation.py \
  tests/benchmarks/test_heuristic_tournament.py -q
```

Expected: all PR #2 behavior remains green.

- [ ] **Step 5: Commit**

```bash
git add src/garboid_pocketrocks/adapters/__init__.py \
  src/garboid_pocketrocks/adapters/public_history.py \
  src/garboid_pocketrocks/bots/base.py \
  src/garboid_pocketrocks/bots/heuristic.py \
  src/garboid_pocketrocks/bots/random_bot.py \
  src/garboid_pocketrocks/heuristics/belief.py \
  src/garboid_pocketrocks/heuristics/reveals.py \
  src/garboid_pocketrocks/heuristics/valuation.py \
  tests/adapters/test_public_history.py tests/adapters/test_sdk_history.py \
  tests/bots/test_heuristic_bots.py tests/heuristics/helpers.py \
  tests/heuristics/test_belief.py tests/heuristics/test_reveals.py \
  tests/heuristics/test_valuation.py
git commit -m "refactor: drive heuristics with SDK rules knowledge"
```

### Task 4: Add the Scalar SDK Session Adapter

**Files:**
- Create: `src/garboid_pocketrocks/simulator/session.py`
- Create: `src/garboid_pocketrocks/simulator/seeding.py`
- Modify: `src/garboid_pocketrocks/simulator/errors.py`
- Create: `tests/simulator/test_session.py`

**Interfaces:**
- Consumes: SDK `SimEngine`, `build_sim_context`, `ScoreRow`, `TurnRecord`
- Produces: `SdkGameSession`, `PlayerSnapshot`, `SessionSnapshot`,
  `SessionTransition`, `SessionResult`, `SessionScore`, `PendingDecisions`, and
  `derive_seed`

- [ ] **Step 1: Write session contract tests**

Cover these named public behaviors in `tests/simulator/test_session.py`:

- `test_start_exposes_one_sdk_bid_context_per_seat`: assert acting seats and
  context bot seats equal `(0, 1, 2)`.
- `test_step_rejects_missing_seats_without_mutating_the_sdk_engine`: save the
  snapshot, submit only seat zero, assert `ActingSeatsError`, and compare the
  unchanged snapshot.
- `test_step_rejects_illegal_decision_without_mutating_the_sdk_engine`: submit
  an over-limit bid, assert `IllegalDecisionError`, and compare the unchanged
  snapshot.
- `test_session_matches_sdk_local_game`: finish identical all-pass games through
  `SdkGameSession` and SDK `LocalGame`, then compare score rows, ordered ranking,
  and history.
- `test_same_seed_produces_the_same_complete_game`: parameterize player counts
  3, 4, and 5; finish two sessions and compare snapshots, results, and history.

Add this complete tie regression:

```python
def test_result_uses_competition_ranks_for_equal_totals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = SimEngine(3, "equal-total-result")
    rows = [
        ScoreRow(0, "A", 30, 0, 0, 0, 0, 30),
        ScoreRow(1, "B", 30, 0, 0, 0, 0, 30),
        ScoreRow(2, "C", 20, 0, 0, 0, 0, 20),
    ]
    monkeypatch.setattr(engine, "score", lambda: rows)
    monkeypatch.setattr(engine, "ranking", lambda: [0, 1, 2])
    result = _result(engine)
    assert result.ranking == (0, 1, 2)
    assert tuple(score.rank for score in result.scores) == (1, 1, 3)
```

- [ ] **Step 2: Run tests to verify the adapter is missing**

```bash
uv run pytest tests/simulator/test_session.py -q
```

Expected: collection fails because `simulator.session` does not exist.

- [ ] **Step 3: Implement deterministic seeding**

In `seeding.py` implement:

```python
def derive_seed(root_seed: int, namespace: str, index: int) -> int:
    payload = f"{root_seed}:{namespace}:{index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
```

Preserve the exact established algorithm from the final old branch if it
differs; seed compatibility is observable and must be covered by fixed-value
tests.

- [ ] **Step 4: Implement the thin session**

`SdkGameSession.start(...)` constructs `SimEngine`. Each phase delegates to
`flip_action`, `legal_max_bid`, `resolve`, and `apply_reveal`. Build contexts
with the public SDK:

```python
context = build_sim_context(
    engine,
    seat,
    decision_kind,
    budget_ms=60_000,
    turn_index=turn_index,
)
context = replace(context, deadline_at=2**63 - 1, received_at=0)
```

Do not create `sdk_fast_context.py`. `_result` must use:

```python
rank = 1 + sum(other.total > row.total for other in rows)
```

while storing `tuple(engine.ranking())` separately.

Copy SDK state into immutable orchestration views only:

- `PlayerSnapshot`: seat, cash, hand suits, won suits, revealed suits, loans,
  investments, and objective IDs;
- `SessionSnapshot`: turn index, tiebreak seat, current action, terminal flag,
  and player snapshots;
- `SessionTransition`: before/after snapshots, next pending decisions, optional
  terminal result, submitted decisions, newly emitted SDK events, and changed
  SDK turn records;
- `SessionResult`: competition-ranked `SessionScore` values, SDK `ScoreRow`
  values, and SDK ordered-seat ranking.

Expose cumulative SDK `events` and `history` directly as immutable tuples.
Validate the complete acting-seat set and every `DecisionContext` before
calling a mutating SDK method. Apply SDK automatic reveals immediately; expose
only choice reveals as pending policy decisions.

- [ ] **Step 5: Run the session suite**

```bash
uv run pytest tests/simulator/test_session.py -q
uv run mypy src/garboid_pocketrocks/simulator/session.py \
  src/garboid_pocketrocks/simulator/seeding.py \
  tests/simulator/test_session.py
```

Expected: all session contracts pass.

- [ ] **Step 6: Commit**

```bash
git add src/garboid_pocketrocks/simulator/session.py \
  src/garboid_pocketrocks/simulator/seeding.py \
  src/garboid_pocketrocks/simulator/errors.py \
  tests/simulator/test_session.py
git commit -m "feat: add scalar SDK game session"
```

### Task 5: Port Runner, Replay, and Monte Carlo Orchestration

**Files:**
- Modify: `src/garboid_pocketrocks/simulator/runner.py`
- Modify: `src/garboid_pocketrocks/simulator/replay.py`
- Modify: `src/garboid_pocketrocks/simulator/monte_carlo.py`
- Modify: `src/garboid_pocketrocks/simulator/cli.py`
- Modify: `tests/simulator/test_runner.py`
- Modify: `tests/simulator/test_replay.py`
- Modify: `tests/simulator/test_monte_carlo.py`
- Modify only for SDK API adaptation: `tests/simulator/test_cli.py`

**Interfaces:**
- Consumes: `SdkGameSession`, `SessionResult`, `RulesetKnowledge`, `derive_seed`
- Produces: unchanged match, replay, statistics, worker determinism, and CLI result contracts

- [ ] **Step 1: Convert tests to the session boundary**

Update factories to construct canonical knowledge with:

```python
canonical_knowledge(player_count, value_chart=chart, objectives_enabled=enabled)
```

Keep assertions for deterministic worker counts, replay round trips, fault
handling, behavior statistics, and CLI tables. Add:

```python
def test_worker_count_does_not_change_sdk_monte_carlo_result() -> None:
    config = _small_random_config()
    assert MonteCarloRunner.run(config, workers=1) == MonteCarloRunner.run(
        config,
        workers=2,
    )
```

- [ ] **Step 2: Run focused tests to expose old-engine dependencies**

```bash
uv run pytest \
  tests/simulator/test_runner.py \
  tests/simulator/test_replay.py \
  tests/simulator/test_monte_carlo.py \
  tests/simulator/test_cli.py -q
```

Expected: failures reference `GameEngine`, `GameState`, ruleset samplers, or old
event/model types.

- [ ] **Step 3: Port orchestration without rules**

`MatchRunner` must drive `SdkGameSession.pending` until termination.
`MatchReplay` stores seed, supported SDK variant, decisions, SDK turn records,
and terminal result; replay re-executes through the session. Monte Carlo jobs
carry chart/objective configuration and use deterministic seed derivation.

Preserve PR #2's bot CLI registry exactly. Resolve `simulator/cli.py` conflicts
by starting from current `main` and changing only simulator configuration and
result field access.

- [ ] **Step 4: Run focused orchestration tests with multiprocessing access**

```bash
uv run pytest \
  tests/simulator/test_runner.py \
  tests/simulator/test_replay.py \
  tests/simulator/test_monte_carlo.py \
  tests/simulator/test_cli.py -q
```

Expected: all focused tests pass. Run outside a restrictive sandbox when
process semaphores are required.

- [ ] **Step 5: Commit**

```bash
git add src/garboid_pocketrocks/simulator/runner.py \
  src/garboid_pocketrocks/simulator/replay.py \
  src/garboid_pocketrocks/simulator/monte_carlo.py \
  src/garboid_pocketrocks/simulator/cli.py \
  tests/simulator/test_runner.py tests/simulator/test_replay.py \
  tests/simulator/test_monte_carlo.py tests/simulator/test_cli.py
git commit -m "refactor: run evaluation through the SDK session"
```

### Task 6: Port Scalar Training Environments and Rewards

**Files:**
- Modify: `src/garboid_pocketrocks/training/observations.py`
- Modify: `src/garboid_pocketrocks/training/rewards.py`
- Modify: `src/garboid_pocketrocks/training/single_agent_env.py`
- Modify: `src/garboid_pocketrocks/training/multi_agent_env.py`
- Modify: `tests/training/test_observations.py`
- Modify: `tests/training/test_rewards.py`
- Modify: `tests/training/test_single_agent_env.py`
- Modify: `tests/training/test_multi_agent_env.py`

**Interfaces:**
- Consumes: `SdkGameSession`, session snapshot/transition/result records, `RulesetKnowledge`
- Produces: unchanged Gymnasium and PettingZoo public contracts

- [ ] **Step 1: Add SDK-backed environment assertions**

Update fixtures to use `SessionSnapshot` and `SessionTransition`. Preserve these
exact behaviors:

- two resets with the same seed produce equal observations and session
  snapshots;
- a legal environment step exposes the same transition and terminal result as
  the underlying SDK session;
- `pettingzoo.test.api_test` completes for the AEC environment;
- totals `(30, 30, 25)` with ranks `(1, 1, 3)` split one configured win bonus
  across the two first-place seats;
- accumulated accounting plus terminal-resource rewards equal
  `(final_money - starting_cash) / starting_cash` for every seat in a complete
  SDK-backed game.

- [ ] **Step 2: Run environment tests to expose type mismatches**

```bash
uv run pytest \
  tests/training/test_observations.py \
  tests/training/test_rewards.py \
  tests/training/test_single_agent_env.py \
  tests/training/test_multi_agent_env.py -q
```

Expected: failures reference old `GameState`, `EngineTransition`, or `Ruleset`.

- [ ] **Step 3: Port scalar environments**

Keep standard Gymnasium `reset`/`step` and PettingZoo AEC APIs unchanged.
Replace old engine state with session snapshot/context/result fields. Derive
knowledge with `canonical_knowledge`; never access SDK debug deck order or
private hidden state from observations or rewards.

- [ ] **Step 4: Run environment and reward suites**

```bash
uv run pytest \
  tests/training/test_observations.py \
  tests/training/test_rewards.py \
  tests/training/test_single_agent_env.py \
  tests/training/test_multi_agent_env.py -q
```

Expected: all tests pass; the two existing PettingZoo observation warnings may remain.

- [ ] **Step 5: Commit**

```bash
git add src/garboid_pocketrocks/training/observations.py \
  src/garboid_pocketrocks/training/rewards.py \
  src/garboid_pocketrocks/training/single_agent_env.py \
  src/garboid_pocketrocks/training/multi_agent_env.py \
  tests/training/test_observations.py tests/training/test_rewards.py \
  tests/training/test_single_agent_env.py tests/training/test_multi_agent_env.py
git commit -m "refactor: back scalar training environments with the SDK"
```

### Task 7: Make Only Necessary Neural API Adaptations

**Files:**
- Modify only when required: `src/garboid_pocketrocks/neural/config.py`
- Modify only when required: `src/garboid_pocketrocks/neural/encoding.py`
- Modify: `src/garboid_pocketrocks/neural/rollout.py`
- Modify only when required: `src/garboid_pocketrocks/neural/smoke.py`
- Modify corresponding tests only for session API changes

**Interfaces:**
- Consumes: scalar `PocketRocksEnv`, `SessionResult`, `RulesetKnowledge`
- Produces: unchanged checkpoint schema, model architecture, rollout order, and deterministic smoke behavior

- [ ] **Step 1: Establish the no-throughput-change guard**

Before modifying neural files, record:

```bash
git diff --name-only origin/main -- \
  src/garboid_pocketrocks/neural/model.py \
  tests/neural/test_model.py
```

Expected: no output. `model.py` and `test_model.py` must remain byte-for-byte
from `main`; GRU cropping is out of scope.

- [ ] **Step 2: Run neural tests against the new scalar environment**

```bash
uv run --extra neural pytest tests/neural -m "not neural_smoke" -q
```

Expected: failures are limited to old simulator result/state imports or changed
environment metadata.

- [ ] **Step 3: Port neural types without batching rollout inference**

Start from `main`'s sequential `collect_rollout`. Replace only old
`GameResult`/`Score`/ruleset access with `SessionResult`/`SessionScore` and
`RulesetKnowledge`. Do not introduce active-environment lockstep batching,
inference-mode changes, GRU cropping, or benchmark-specific configuration.

- [ ] **Step 4: Run neural unit and smoke suites**

```bash
uv run --extra neural pytest tests/neural -m "not neural_smoke" -q
uv run --extra neural pytest \
  tests/neural/test_smoke.py::test_two_by_sixteen_smoke_is_deterministic -q
uv run --extra neural mypy --config-file mypy.neural.ini src tests
```

Expected: neural unit tests, deterministic smoke, and complete neural mypy pass.

- [ ] **Step 5: Audit neural diff and commit**

```bash
git diff --name-only origin/main -- src/garboid_pocketrocks/neural tests/neural
```

Expected: only files required for SDK session/result adaptation appear; model
throughput files do not.

```bash
git add src/garboid_pocketrocks/neural/config.py \
  src/garboid_pocketrocks/neural/encoding.py \
  src/garboid_pocketrocks/neural/rollout.py \
  src/garboid_pocketrocks/neural/smoke.py \
  tests/neural/test_encoding.py tests/neural/test_rollout.py \
  tests/neural/test_smoke.py
git commit -m "refactor: consume SDK session results in neural rollout"
```

### Task 8: Remove the Local Rules Engine and Narrow Public Exports

**Files:**
- Delete: `src/garboid_pocketrocks/adapters/simulator_history.py`
- Delete: `src/garboid_pocketrocks/rules.py`
- Delete: `src/garboid_pocketrocks/simulator/context.py`
- Delete: `src/garboid_pocketrocks/simulator/engine.py`
- Delete: `src/garboid_pocketrocks/simulator/events.py`
- Delete: `src/garboid_pocketrocks/simulator/model.py`
- Delete: `src/garboid_pocketrocks/simulator/sampling.py`
- Delete: `src/garboid_pocketrocks/simulator/setup.py`
- Modify: `src/garboid_pocketrocks/simulator/__init__.py`
- Delete: `tests/simulator/helpers.py`
- Delete: `tests/simulator/test_context.py`
- Delete: `tests/simulator/test_engine.py`
- Delete: `tests/simulator/test_invariants.py`
- Delete: `tests/simulator/test_sampling.py`
- Delete: `tests/simulator/test_scoring.py`
- Delete: `tests/simulator/test_sdk_conformance.py`
- Delete: `tests/simulator/test_setup.py`
- Delete: `tests/test_rules.py`
- Modify: `tests/test_package.py`
- Modify: `tests/test_integration.py`

**Interfaces:**
- Consumes: completed SDK-backed consumers from Tasks 2–7
- Produces: orchestration-only `garboid_pocketrocks.simulator` public namespace

- [ ] **Step 1: Add package-removal regression tests**

In `tests/test_package.py` add:

```python
@pytest.mark.parametrize(
    "module_name",
    (
        "garboid_pocketrocks.rules",
        "garboid_pocketrocks.adapters.simulator_history",
        "garboid_pocketrocks.simulator.context",
        "garboid_pocketrocks.simulator.engine",
        "garboid_pocketrocks.simulator.events",
        "garboid_pocketrocks.simulator.model",
        "garboid_pocketrocks.simulator.sampling",
        "garboid_pocketrocks.simulator.setup",
    ),
)
def test_private_rules_modules_are_removed(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        import_module(module_name)


def test_simulator_does_not_export_a_rules_engine() -> None:
    assert not hasattr(simulator, "GameEngine")
    assert not hasattr(simulator, "RulesetVariationSampler")
```

- [ ] **Step 2: Run removal tests before deletion**

```bash
uv run pytest tests/test_package.py -q
```

Expected: FAIL because the old modules still import.

- [ ] **Step 3: Delete private rules code and tests**

Delete the adapter, seven rules modules, and eight obsolete test files listed
above. Keep public-history, runner, replay, Monte Carlo, CLI, and new session
tests. Rewrite `tests/test_integration.py` to configure canonical SDK variants
directly and retain its complete-game heuristic, Gymnasium, and PettingZoo
coverage.

Rewrite `simulator/__init__.py` to export only orchestration records and
functions from `errors`, `monte_carlo`, `replay`, `runner`, `seeding`, and
`session`.

- [ ] **Step 4: Prove no runtime imports remain**

```bash
rg -n \
  "garboid_pocketrocks\\.rules|adapters\\.simulator_history|simulator\\.(context|engine|events|model|sampling|setup)" \
  src tests
```

Expected: only the intentional missing-module strings in `tests/test_package.py`.

- [ ] **Step 5: Run package and scalar integration tests**

```bash
uv run pytest \
  tests/test_package.py \
  tests/test_integration.py \
  tests/simulator \
  tests/training -q
```

Expected: all retained tests pass.

- [ ] **Step 6: Commit**

```bash
git add -A src/garboid_pocketrocks/rules.py \
  src/garboid_pocketrocks/adapters/simulator_history.py \
  src/garboid_pocketrocks/simulator tests/simulator \
  tests/test_rules.py tests/test_package.py tests/test_integration.py
git commit -m "refactor: remove the duplicated Garboid rules engine"
```

### Task 9: Update Migration Documentation and Enforce the PR Allowlist

**Files:**
- Modify: `README.md`
- Modify only if SDK variant examples require it: `.env.example`
- Keep: `docs/superpowers/specs/2026-07-29-sdk-migration-pr-scope-design.md`
- Keep: `docs/superpowers/plans/2026-07-29-clean-sdk-migration-pr-rebuild.md`

**Interfaces:**
- Consumes: final scalar architecture
- Produces: reviewer-facing documentation and a scope-audited diff

- [ ] **Step 1: Update README from current main**

Document that the SDK owns game rules and Garboid's simulator package owns
orchestration. Preserve PR #2's bot-generation, future-cash, analysis, and CLI
sections. Do not copy the old branch's launcher, fast-context, vector-engine,
or neural-throughput sections.

- [ ] **Step 2: Run the forbidden-file audit**

```bash
git diff --name-only origin/main...HEAD | rg \
  "batch_runner|vector_env|sdk_fast_context|bots/launcher|neural-training-throughput|vectorized-batch-engine|rl-fast-simulation|live-visible-resources"
```

Expected: no output.

Also verify PR #2-owned source files contain only necessary SDK type/API
adaptations:

```bash
git diff origin/main...HEAD -- \
  src/garboid_pocketrocks/heuristics/cash.py \
  src/garboid_pocketrocks/heuristics/profiles.py \
  src/garboid_pocketrocks/heuristics/valuation.py \
  src/garboid_pocketrocks/bots/heuristic.py \
  src/garboid_pocketrocks/simulator/cli.py
```

Expected: no profile constants, bot identities, future-cash formulas, or CLI
registry entries regress to the old PR #1 versions.

Verify PR #2's design and plan documents are byte-for-byte unchanged:

```bash
git diff --quiet origin/main -- \
  docs/superpowers/specs/2026-07-29-future-cash-heuristic-design.md \
  docs/superpowers/plans/2026-07-29-future-cash-heuristic.md \
  docs/superpowers/specs/2026-07-29-versioned-heuristic-bots-design.md \
  docs/superpowers/plans/2026-07-29-versioned-heuristic-bots.md
```

Expected: exit 0 with no output.

- [ ] **Step 3: Format and commit documentation**

```bash
uv run ruff format README.md \
  docs/superpowers/specs/2026-07-29-sdk-migration-pr-scope-design.md \
  docs/superpowers/plans/2026-07-29-clean-sdk-migration-pr-rebuild.md
uv run ruff format --check .
git diff --check
```

If `.env.example` has no required SDK migration change, omit it. Otherwise add
its exact path to this command:

```bash
git add README.md \
  docs/superpowers/specs/2026-07-29-sdk-migration-pr-scope-design.md \
  docs/superpowers/plans/2026-07-29-clean-sdk-migration-pr-rebuild.md
git commit -m "docs: explain the SDK orchestration boundary"
```

### Task 10: Verify, Replace PR #1 Safely, and Monitor Hosted CI

**Files:**
- No new runtime files
- Modify PR #1 title/body through GitHub after verification

**Interfaces:**
- Consumes: fully reconstructed branch
- Produces: conflict-free existing PR #1 at the clean branch tip

- [ ] **Step 1: Synchronize the exact locked environment**

```bash
uv sync --locked --extra neural
```

Expected: exit 0 with the upstream scalar SDK commit installed.

- [ ] **Step 2: Run all local quality gates**

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run --extra neural mypy --config-file mypy.neural.ini src tests
uv run pytest
uv run --extra neural pytest \
  tests/neural/test_smoke.py::test_two_by_sixteen_smoke_is_deterministic -q
git diff --check
```

Expected: every command exits 0. Run pytest outside a restrictive sandbox if
multiprocessing semaphore access is required.

- [ ] **Step 3: Audit the final change set**

```bash
git diff --name-status origin/main...HEAD
git log --oneline origin/main..HEAD
git merge-tree "$(git merge-base HEAD origin/main)" HEAD origin/main | \
  rg "changed in both|CONFLICT|<<<<<<<|>>>>>>>"
```

Expected: only core migration files and the two approved documents appear; the
conflict search returns no output.

- [ ] **Step 4: Update the existing remote branch with a lease**

Record the current remote tip and verify it still matches the backed-up
contaminated tip:

```bash
git fetch origin codex/sdk-engine-migration
git rev-parse origin/codex/sdk-engine-migration
git rev-parse codex/sdk-engine-migration-pre-rebuild-20260729
```

Only when they match:

```bash
git push --force-with-lease \
  origin codex/sdk-engine-migration-rebuild:codex/sdk-engine-migration
```

- [ ] **Step 5: Rewrite PR #1 description**

The PR body must state:

- SDK is the game-rule source of truth;
- Garboid retains orchestration only;
- merged PR #2 behavior is preserved;
- vector and fast-context work are explicit follow-ups;
- exact local test, Ruff, mypy, and neural-smoke results.

Use `gh pr edit`; do not create another PR.

- [ ] **Step 6: Verify PR metadata and hosted checks**

```bash
gh pr view 1 --repo chrisgarber/garboid-pocketrocks \
  --json url,state,isDraft,mergeable,headRefOid,baseRefName,title
gh pr checks 1 --repo chrisgarber/garboid-pocketrocks --watch
```

Expected: PR #1 targets `main`, is mergeable, and both `quality` and `neural`
checks pass.

- [ ] **Step 7: Preserve the reconstruction worktree**

Do not remove the backup branch or reconstruction worktree until PR #1 merges.
Report both paths/branches in the handoff.
