# Batch Tournament Comparator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use braze-superpowers:subagent-driven-development (recommended) or braze-superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the all-bot tournament use the SDK vector engine for game execution and sufficient-statistic Plackett–Luce fits for fast deterministic bootstrap intervals.

**Architecture:** Preserve the tournament scheduler and public result types. Add a batch-context/match adapter around `BatchSimEngine`, feed its rich-enough compact results into the existing Monte Carlo aggregator, and compile ranking stages by remaining-bot set so every optimizer evaluation reuses denominators.

**Tech Stack:** Python 3.14, NumPy, SciPy L-BFGS-B, PocketRocks `BatchSimEngine`, pytest, Ruff, strict mypy, uv

## Global Constraints

- Preserve the existing bot registry, schedule, generalized Plackett–Luce model, Davidson–Luce ties, and whole-game bootstrap semantics.
- Preserve deterministic results across worker counts and batch sizes.
- Keep exact replay capture on the scalar SDK path.
- Use the existing fault fallback contract.
- Do not add timing assertions to CI.
- Open any pull request as a draft.

---

### Task 1: Compile Plackett–Luce Choice Sets

**Files:**
- Modify: `src/garboid_pocketrocks/tournament/rating.py`
- Modify: `tests/tournament/test_rating.py`

**Interfaces:**
- Consumes: `RankingObservation`, indexed bot IDs, ghost pseudo-rankings, tie prevalence parameters
- Produces: `_compile_choice_sets(indexed, parameter_count, maximum_tie_order) -> tuple[_ChoiceSet, ...]`
- Produces: `_negative_log_likelihood(parameters, problem)` with the unchanged SciPy callback signature

- [ ] **Step 1: Add a reference-equivalence test**

Add a test helper implementing the existing per-observation likelihood loop and
a test containing weighted untied, second-place-tie, and first-place-tie
observations. Build the new compiled problem and assert:

```python
compiled_value, compiled_gradient = rating_module._negative_log_likelihood(
    parameters,
    problem,
)
reference_value, reference_gradient = _reference_negative_log_likelihood(
    parameters,
    indexed_observations,
    bot_count=len(bot_ids),
    ghost_index=len(bot_ids),
    maximum_tie_order=maximum_tie_order,
)
assert compiled_value == pytest.approx(reference_value, abs=1e-10)
np.testing.assert_allclose(compiled_gradient, reference_gradient, atol=1e-10)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
uv run pytest tests/tournament/test_rating.py -q
```

Expected: FAIL because the compiled choice-set interface is absent.

- [ ] **Step 3: Add compiled sufficient-statistic types**

Replace `_FitProblem.observations` with:

```python
@dataclass(frozen=True, slots=True)
class _ChoiceSet:
    feature_matrix: NDArray[np.float64]
    chosen_weights: NDArray[np.float64]
    total_weight: float


@dataclass(frozen=True, slots=True)
class _FitProblem:
    choice_sets: tuple[_ChoiceSet, ...]
    parameter_count: int
```

Compile every ranking stage by its sorted remaining-index tuple. Candidate
subsets contain orders `1..maximum_tie_order`; each feature row contains
`1 / len(subset)` for real bot log-worths and `1` for the applicable tie
parameter. Add each observation's weight to the selected candidate.

- [ ] **Step 4: Vectorize the likelihood callback**

For each choice set calculate:

```python
log_weights = choice_set.feature_matrix @ parameters
probabilities = np.exp(log_weights - logsumexp(log_weights))
value += choice_set.total_weight * float(logsumexp(log_weights)) - float(
    choice_set.chosen_weights @ log_weights
)
gradient += (
    choice_set.total_weight * (probabilities @ choice_set.feature_matrix)
    - choice_set.chosen_weights @ choice_set.feature_matrix
)
```

Keep bounds, pseudo-rankings, convergence validation, normalized worths, and
diagnostics unchanged.

- [ ] **Step 5: Verify GREEN and regression coverage**

Run:

```bash
uv run pytest tests/tournament/test_rating.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/garboid_pocketrocks/tournament/rating.py tests/tournament/test_rating.py
git commit -m "perf: compile Plackett-Luce choice sets"
```

---

### Task 2: Reuse Weighted Observations in Bootstrap Fits

**Files:**
- Modify: `src/garboid_pocketrocks/tournament/analysis.py`
- Modify: `tests/tournament/test_analysis.py`

**Interfaces:**
- Consumes: `observations_from_games(games)`
- Produces: `_resample_observations(observations, root_seed, replicate) -> tuple[RankingObservation, ...]`
- Produces: unchanged `bootstrap_rating_intervals(...) -> BootstrapSummary`

- [ ] **Step 1: Add exact-resampling tests**

For each replicate, construct the old reference resample:

```python
rng = random.Random(derive_seed(42, "bootstrap", replicate))
reference_games = tuple(games[rng.randrange(len(games))] for _ in games)
reference = fit_plackett_luce(observations_from_games(reference_games), bot_ids)
optimized = analysis_module._fit_bootstrap_replicate(
    observations_from_games(games),
    bot_ids,
    42,
    replicate,
)
```

Assert the optimized rating mapping equals the reference mapping with absolute
tolerance `1e-8`. Retain the existing serial/parallel equality assertion.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
uv run pytest tests/tournament/test_analysis.py -q
```

Expected: FAIL because bootstrap replicates still consume `GameSummary` tuples.

- [ ] **Step 3: Precompute and weight observations**

Convert games once in `bootstrap_rating_intervals`. In each replicate, preserve
the existing `random.Random` index draws and their first-seen order:

```python
counts: dict[int, int] = {}
for _ in observations:
    index = rng.randrange(len(observations))
    counts[index] = counts.get(index, 0) + 1
resampled = tuple(
    RankingObservation(
        observations[index].rank_groups,
        weight=observations[index].weight * count,
    )
    for index, count in counts.items()
)
```

Initialize workers with observations rather than full game summaries.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
uv run pytest tests/tournament/test_analysis.py tests/tournament/test_rating.py -q
```

Expected: PASS with deterministic serial and parallel intervals.

- [ ] **Step 5: Commit**

```bash
git add src/garboid_pocketrocks/tournament/analysis.py tests/tournament/test_analysis.py
git commit -m "perf: reuse weighted bootstrap observations"
```

---

### Task 3: Add the SDK Batch Dependency and Context Adapter

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `src/garboid_pocketrocks/simulator/batch_context.py`
- Create: `tests/simulator/test_batch_context.py`

**Interfaces:**
- Consumes: `pocketrocks.sim.BatchSimEngine`
- Produces:

```python
def build_batch_context(
    engine: BatchSimEngine,
    *,
    row: int,
    seat: int,
    decision_kind: DecisionKind,
    action_id: int,
    resource_ids: tuple[int, int],
    turn_index: int,
    legal_max_amount: int | None,
) -> DecisionContext
```

- [ ] **Step 1: Add full-game scalar/batch context parity tests**

Parameterize player counts `3, 4, 5`, charts `A, E`, and objectives
`True, False`. Start scalar and batch engines with the same seed, advance them
with identical deterministic legal bids and first-card reveals, and compare
every bid and choice-reveal context after normalizing only deadline timestamps.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
uv run pytest tests/simulator/test_batch_context.py -q
```

Expected: FAIL because the pinned SDK has no `BatchSimEngine` and the adapter is
absent.

- [ ] **Step 3: Pin the SDK batch commit**

Change the dependency to:

```toml
"pocketrocks-python-sdk @ git+https://github.com/chrisgarber/pocketrocks-python-sdk.git@51cad378ee1e70a78e39ebbb25957ea003444873",
```

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv lock
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv sync
```

- [ ] **Step 4: Implement the direct context adapter**

Materialize public state from one batch row. Generate the request ID with:

```python
namespace = uuid.uuid5(uuid.NAMESPACE_URL, "pocketrocks-sim")
request_id = str(
    uuid.uuid5(
        namespace,
        f"{engine.seeds[row]}:{turn_index}:{seat}:{decision_kind}",
    )
)
```

Use `deadline_at=2**63 - 1`, `received_at=0`, SDK constants for starting cash,
array counts for public state, and nonzero hand cards for private state.

- [ ] **Step 5: Verify GREEN**

Run:

```bash
uv run pytest tests/simulator/test_batch_context.py -q
```

Expected: PASS for every full-game parity case.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock \
  src/garboid_pocketrocks/simulator/batch_context.py \
  tests/simulator/test_batch_context.py
git commit -m "feat: build bot contexts from batch state"
```

---

### Task 4: Execute Tournament Matches in Batches

**Files:**
- Create: `src/garboid_pocketrocks/simulator/batch_match.py`
- Modify: `src/garboid_pocketrocks/simulator/monte_carlo.py`
- Create: `tests/simulator/test_batch_match.py`
- Modify: `tests/simulator/test_monte_carlo.py`

**Interfaces:**
- Consumes: `build_batch_context`, a homogeneous-player-count sequence of job-like records
- Produces: `run_batch_matches(jobs) -> tuple[MatchResult, ...]`
- Extends:

```python
MonteCarloRunner.run_jobs(
    config,
    jobs,
    *,
    workers=1,
    batch_size: int | None = None,
) -> MonteCarloResult
```

- [ ] **Step 1: Add match-level parity tests**

Create mixed chart/objective jobs for each player count using registered random
and heuristic specs. Compare each batched match with `MatchRunner.run` for:

- terminal `SessionResult`;
- recorded decisions and turn records;
- fault records; and
- brain-seed determinism.

- [ ] **Step 2: Add Monte Carlo parity tests**

Plan at least 60 tournament jobs and assert:

```python
scalar = MonteCarloRunner.run_jobs(config, jobs, workers=1, batch_size=None)
for batch_size in (1, 7, 32):
    assert (
        MonteCarloRunner.run_jobs(
            config,
            jobs,
            workers=1,
            batch_size=batch_size,
        )
        == scalar
    )
```

Also verify `capture_replays=True` returns exact scalar replays even when a
batch size is requested.

- [ ] **Step 3: Run focused tests and verify RED**

Run:

```bash
uv run pytest tests/simulator/test_batch_match.py \
  tests/simulator/test_monte_carlo.py -q
```

Expected: FAIL because batch match execution and `batch_size` are absent.

- [ ] **Step 4: Implement batched brain execution**

Create brains with the same per-game `random.Random(seed).randrange(2**63)`
sequence as `MatchRunner`. At each vector phase, call existing brains with
parity-tested contexts, validate decisions, apply `FaultMode`, and record
decisions. Materialize `TurnRecord`, `RevealRecord`, score rows, ranks, and
faults from batch outcomes and before/after array deltas.

- [ ] **Step 5: Chunk jobs in Monte Carlo**

When `batch_size` is set and replay capture is false:

- bucket jobs by player count;
- split buckets into chunks no larger than `batch_size`;
- process chunks serially or through the existing process pool;
- flatten `_CompletedGame` tuples; and
- rely on `_aggregate`'s `game_index` sort.

When replay capture is true or `batch_size is None`, keep the scalar path.
Validate `batch_size > 0`.

- [ ] **Step 6: Verify GREEN, including process workers**

Run:

```bash
uv run pytest tests/simulator/test_batch_match.py \
  tests/simulator/test_monte_carlo.py \
  tests/tournament/test_runner.py -q
```

Expected: PASS, including scalar/batch and worker-count equality.

- [ ] **Step 7: Commit**

```bash
git add src/garboid_pocketrocks/simulator/batch_match.py \
  src/garboid_pocketrocks/simulator/monte_carlo.py \
  tests/simulator/test_batch_match.py \
  tests/simulator/test_monte_carlo.py
git commit -m "perf: execute tournament matches in batches"
```

---

### Task 5: Wire Batch Execution into the Comparator

**Files:**
- Modify: `src/garboid_pocketrocks/tournament/cli.py`
- Modify: `src/garboid_pocketrocks/tournament/runner.py`
- Modify: `src/garboid_pocketrocks/tournament/schedule.py`
- Modify: `src/garboid_pocketrocks/tournament/reporting.py`
- Modify: `tests/tournament/test_cli.py`
- Modify: `tests/tournament/test_runner.py`
- Modify: `tests/tournament/test_reporting.py`

**Interfaces:**
- Produces: `TournamentConfig.batch_size: int = 256`
- Consumes: `MonteCarloRunner.run_jobs(..., batch_size=config.batch_size)`
- Produces: CLI `--batch-size`

- [ ] **Step 1: Add CLI/config/report tests**

Assert:

- the default batch size is 256;
- zero and negative values are rejected;
- `--batch-size 64` reaches `TournamentConfig`;
- the runner passes it to Monte Carlo; and
- summary configuration includes `batch_size`.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
uv run pytest tests/tournament/test_cli.py \
  tests/tournament/test_runner.py \
  tests/tournament/test_reporting.py -q
```

Expected: FAIL because batch size is not configurable.

- [ ] **Step 3: Implement configuration and wiring**

Add the defaulted validated field to `TournamentConfig`, the parser option, the
runner argument, and additive JSON/report configuration output. Do not change
the scheduler's lineup or seed generation.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
uv run pytest tests/tournament -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/garboid_pocketrocks/tournament \
  tests/tournament
git commit -m "feat: enable batch tournament execution"
```

---

### Task 6: Benchmark, Tune, and Run the Full Comparator

**Files:**
- Create: `docs/benchmarks/2026-07-29-batch-tournament-comparator.md`
- Generate: `tournament-results/ratings.csv`
- Generate: `tournament-results/summary.json`
- Generate: `tournament-results/report.html`

**Interfaces:**
- Consumes: optimized `garboid-tournament`
- Produces: reproducible timings, chosen batch size, and final all-bot report

- [ ] **Step 1: Run focused and full verification**

```bash
uv run pytest tests/tournament tests/simulator tests/bots tests/heuristics -q
uv run pytest -q
uv run ruff check .
uv run mypy
git diff --check
```

Expected: all commands exit zero.

- [ ] **Step 2: Benchmark bootstrap before and after**

Use the deterministic 300-game/20-bootstrap configuration and record elapsed
time and identical leaderboard/interval evidence. Then run 10,000 games with
bootstrap disabled for scalar and batch sizes `64, 256, 1024`, using the same
seed and worker count.

- [ ] **Step 3: Select the measured batch size**

Keep 256 unless another tested size is faster without excessive memory. If the
winner differs, update the default and its tests under a red-green cycle.

- [ ] **Step 4: Run the requested all-bot tournament**

```bash
uv run garboid-tournament \
  --games 10000 \
  --players 3,4,5 \
  --charts A,B,C,D,E \
  --seed 42 \
  --workers 16 \
  --bootstrap-samples 200 \
  --output-dir tournament-results
```

Expected: the leaderboard includes `random`, every row reports zero faults,
and all three artifacts are created.

- [ ] **Step 5: Record benchmark evidence**

Document environment, exact commands, scalar/batch timings, bootstrap timing,
speedups, selected batch size, result parity, final leaderboard, and artifact
paths in the benchmark report.

- [ ] **Step 6: Run final verification and commit**

```bash
uv run pytest -q
uv run ruff check .
uv run mypy
git diff --check
git add docs/benchmarks/2026-07-29-batch-tournament-comparator.md
git commit -m "docs: benchmark optimized bot comparator"
```

Do not commit generated tournament artifacts unless explicitly requested.
