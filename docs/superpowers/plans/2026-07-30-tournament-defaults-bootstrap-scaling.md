# Tournament Defaults and Bootstrap Scaling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use braze-superpowers:subagent-driven-development (recommended) or braze-superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Default the comparator to 15,000 games over `random` plus versioned bots, quantify bootstrap stability and scaling, and deliver the verified changes on `main`.

**Architecture:** Keep the complete bot registry for explicit simulation choices and add a curated default-tournament tuple beside it. The CLI chooses that tuple only when `--bots` is omitted. Benchmark one deterministic 15,000-game batch result, reuse it for bootstrap measurements, and use stratified prefixes to estimate game-count scaling without changing the production analysis path.

**Tech Stack:** Python 3.14, NumPy, SciPy, PocketRocks batch SDK, pytest, Ruff, mypy, uv.

## Global Constraints

- Default games: 15,000.
- Default batch size: 64.
- Default bots: `random` and every explicitly versioned simulation bot.
- Unsuffixed public aliases remain selectable through `--bots`, but are excluded by default.
- Retain 200 bootstrap samples unless its interval endpoints differ from the 500-sample reference by more than 10 PL rating points or changes a substantive overlap conclusion.
- Preserve the uncommitted main-worktree edit in `tests/neural/test_rollout.py`.
- Push directly to `main` only after fresh full verification.

---

### Task 1: Curate the default tournament roster

**Files:**
- Modify: `src/garboid_pocketrocks/bots/registry.py`
- Modify: `src/garboid_pocketrocks/bots/__init__.py`
- Modify: `src/garboid_pocketrocks/tournament/cli.py`
- Test: `tests/bots/test_registry.py`
- Test: `tests/tournament/test_cli.py`

**Interfaces:**
- Produces: `DEFAULT_TOURNAMENT_BOT_SPECS: tuple[BotSpec, ...]`.
- Consumes: the existing complete `BOT_SPECS` and `BOT_SPECS_BY_NAME` registry.

- [ ] **Step 1: Write failing registry and CLI tests**

Add assertions that the curated tuple is exactly:

```python
(
    "random",
    "aggressive-v1",
    "balanced-v1",
    "passive-v1",
    "aggressive-v2",
    "balanced-v2",
    "passive-v2",
)
```

Add a resolver test that calls `_resolve_bot_specs` with `include=None`,
the full registry, and the curated tuple, then asserts the same names. Retain
the existing explicit-include test to prove unsuffixed aliases remain valid.

- [ ] **Step 2: Verify the tests fail for the missing curated interface**

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache \
  uv run pytest tests/bots/test_registry.py tests/tournament/test_cli.py -q
```

Expected: collection or assertion failure because
`DEFAULT_TOURNAMENT_BOT_SPECS` and the default resolver argument do not exist.

- [ ] **Step 3: Add the curated tuple and use it for omitted `--bots`**

In `bots/registry.py`, define the curated tuple from the already imported
spec constants:

```python
DEFAULT_TOURNAMENT_BOT_SPECS = (
    BOT_SPECS[0],
    AGGRESSIVE_HEURISTIC_V1_BOT_SPEC,
    BALANCED_HEURISTIC_V1_BOT_SPEC,
    PASSIVE_HEURISTIC_V1_BOT_SPEC,
    AGGRESSIVE_HEURISTIC_V2_BOT_SPEC,
    BALANCED_HEURISTIC_V2_BOT_SPEC,
    PASSIVE_HEURISTIC_V2_BOT_SPEC,
)
```

Export it from `bots/__init__.py`. Extend `_resolve_bot_specs` with:

```python
defaults: tuple[BotSpec, ...] | None = None
```

When `include is None`, use `defaults` when supplied, otherwise preserve the
existing full-registry behavior. Pass `DEFAULT_TOURNAMENT_BOT_SPECS` from
`main()`.

- [ ] **Step 4: Verify focused tests pass**

Run the command from Step 2.

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/garboid_pocketrocks/bots/registry.py \
  src/garboid_pocketrocks/bots/__init__.py \
  src/garboid_pocketrocks/tournament/cli.py \
  tests/bots/test_registry.py tests/tournament/test_cli.py
git commit -m "feat: curate default tournament bots"
```

### Task 2: Raise the default tournament to 15,000 games

**Files:**
- Modify: `src/garboid_pocketrocks/tournament/cli.py`
- Modify: `src/garboid_pocketrocks/tournament/schedule.py`
- Test: `tests/tournament/test_cli.py`
- Test: `tests/tournament/test_schedule.py`

**Interfaces:**
- Produces: matching CLI and `TournamentConfig` defaults of `15_000`.
- Consumes: the existing `--games` positive-integer override.

- [ ] **Step 1: Change tests to require 15,000 games**

Update both default assertions from `10_000` to `15_000`. Rename
`test_default_plan_allocates_exactly_ten_thousand_games` to
`test_default_plan_allocates_exactly_fifteen_thousand_games` and assert the
quota sum and job count are `15_000`.

- [ ] **Step 2: Verify the tests fail with the current 10,000 default**

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache \
  uv run pytest \
    tests/tournament/test_cli.py::test_parser_defaults_to_full_tournament \
    tests/tournament/test_schedule.py::test_default_config_describes_full_tournament \
    tests/tournament/test_schedule.py::test_default_plan_allocates_exactly_fifteen_thousand_games \
    -q
```

Expected: failures showing `10000 != 15000`.

- [ ] **Step 3: Change both production defaults**

Set `TournamentConfig.games` and the parser's `--games` default to `15_000`.
Do not change the batch size or bootstrap default in this task.

- [ ] **Step 4: Verify the focused tests pass**

Run the command from Step 2.

Expected: three tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/garboid_pocketrocks/tournament/cli.py \
  src/garboid_pocketrocks/tournament/schedule.py \
  tests/tournament/test_cli.py tests/tournament/test_schedule.py
git commit -m "perf: scale default tournament to fifteen thousand games"
```

### Task 3: Measure bootstrap stability and computation scaling

**Files:**
- Create temporarily: `/private/tmp/benchmark_tournament_scaling.py`
- Modify: `docs/benchmarks/2026-07-30-batch-tournament-comparator.md`

**Interfaces:**
- Consumes: `DEFAULT_TOURNAMENT_BOT_SPECS`,
  `TournamentPlanner.plan`, `MonteCarloRunner.run_jobs`,
  `fit_plackett_luce`, and `bootstrap_rating_intervals`.
- Produces: measured phase timings, bootstrap convergence, and interval
  endpoint drift for 50/100/200/500 samples.

- [ ] **Step 1: Write the benchmark driver**

The temporary driver must:

1. construct the default 15,000-game seed-42 plan;
2. time planning, 16-worker batch simulation at batch size 64, and the primary
   fit separately;
3. time bootstrap counts 50, 100, 200, and 500 on the same game summaries;
4. compare every smaller interval endpoint with the 500-sample endpoint;
5. create condition-stratified 5,000- and 10,000-game subsets and time 200
   bootstrap fits on them;
6. emit one JSON object containing all timings, convergence counts, interval
   widths, maximum endpoint drift, and leaderboard ratings.

Use `time.perf_counter()` around each phase and serialize only primitive JSON
values. Assert that all requested bootstrap samples converge.

- [ ] **Step 2: Run the benchmark once**

Run outside the filesystem sandbox because multiprocessing needs OS
semaphores:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache \
  uv run python /private/tmp/benchmark_tournament_scaling.py
```

Expected: exit 0 and JSON results for all requested game/sample counts.

- [ ] **Step 3: Apply the bootstrap selection rule**

Keep 200 as the default if its maximum lower/upper endpoint drift from the
500-sample reference is at most 10 PL rating points and its intervals imply the
same overlap/non-overlap conclusions. Otherwise, test 300 and 400 and select
the first count satisfying both requirements, with 500 as the upper bound.

If the default changes, follow a fresh RED/GREEN cycle in
`tests/tournament/test_cli.py` and `tests/tournament/test_schedule.py` before
editing production defaults.

- [ ] **Step 4: Record results**

Extend the benchmark report with:

- phase timing and percent of total time;
- bootstrap time by sample count;
- 5,000/10,000/15,000 game scaling;
- endpoint stability and the selected default;
- the inverse-square-root interpretation of additional games;
- comparison with the earlier 10,000-game all-bot result, explicitly noting
  that the roster changed.

- [ ] **Step 5: Commit**

```bash
git add docs/benchmarks/2026-07-30-batch-tournament-comparator.md \
  src/garboid_pocketrocks/tournament/cli.py \
  src/garboid_pocketrocks/tournament/schedule.py \
  tests/tournament/test_cli.py tests/tournament/test_schedule.py
git commit -m "docs: benchmark tournament bootstrap scaling"
```

Omit unchanged files from `git add` if 200 remains the default.

### Task 4: Run the final default tournament

**Files:**
- Produce outside git: `/private/tmp/garboid-versioned-bots-results-15000/ratings.csv`
- Produce outside git: `/private/tmp/garboid-versioned-bots-results-15000/summary.json`
- Produce outside git: `/private/tmp/garboid-versioned-bots-results-15000/report.html`

**Interfaces:**
- Consumes: the finalized CLI defaults.
- Produces: the user-facing 15,000-game leaderboard and confidence intervals.

- [ ] **Step 1: Run the CLI using defaults except seed, workers, and output**

```bash
.venv/bin/garboid-tournament \
  --seed 42 \
  --workers 16 \
  --output-dir /private/tmp/garboid-versioned-bots-results-15000
```

Expected: seven leaderboard rows, including `random`, with no unsuffixed
aggressive/balanced/passive rows.

- [ ] **Step 2: Validate machine-readable output**

Use `jq` to assert:

- `configuration.games == 15000`;
- `configuration.batch_size == 64`;
- bot names equal the curated seven-name roster;
- bootstrap requested equals the selected default and converged equals
  requested;
- every leaderboard row has zero faults.

### Task 5: Verify, merge, and push

**Files:**
- Verify: the complete repository.
- Preserve: `/Users/Christopher.Garber/Documents/garboid-pocketrocks/tests/neural/test_rollout.py`

**Interfaces:**
- Consumes: all feature-branch commits and final artifacts.
- Produces: updated `origin/main`.

- [ ] **Step 1: Run fresh verification on the feature branch**

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run ruff format --check .
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run ruff check .
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run mypy
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run pytest -q
git diff --check
git status --short
```

Expected: format, lint, types, tests, and whitespace checks pass; feature
worktree is clean.

- [ ] **Step 2: Preserve the main-worktree edit and merge**

Confirm the only main-worktree change is
`tests/neural/test_rollout.py`. Save it in a named stash:

```bash
git -C /Users/Christopher.Garber/Documents/garboid-pocketrocks \
  stash push -m "preserve rollout test before tournament merge" -- \
  tests/neural/test_rollout.py
git -C /Users/Christopher.Garber/Documents/garboid-pocketrocks \
  merge --no-ff codex/optimize-bot-comparator
git -C /Users/Christopher.Garber/Documents/garboid-pocketrocks \
  stash pop
```

Resolve merge conflicts without discarding either branch. If stash restoration
conflicts, preserve the user's rollout-test hunk and the merged committed
content.

- [ ] **Step 3: Verify the merged main branch**

Run the full verification commands from Step 1 in the main worktree. Confirm
that `git status --short` shows only the preserved rollout-test edit.

- [ ] **Step 4: Push main**

```bash
git -C /Users/Christopher.Garber/Documents/garboid-pocketrocks push origin main
```

Expected: `origin/main` advances to the verified local main merge commit.
