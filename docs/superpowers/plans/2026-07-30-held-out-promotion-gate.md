# Held-Out Promotion Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> braze-superpowers:subagent-driven-development (recommended) or
> braze-superpowers:executing-plans to implement this plan task-by-task. Steps
> use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a strategy-neutral, paired held-out final exam that writes a
deterministic, machine-readable promotion decision for any two `BotSpec`
identities.

**Architecture:** A new `promotion` package loads and hashes committed corpus
recipes, expands held-out cases into candidate/incumbent twin `GameJob` values,
validates real simulator results, bootstraps complete pairs, and writes atomic
JSON artifacts. A dedicated `garboid-promote` CLI keeps promotion separate
from exploratory tournaments.

**Tech Stack:** Python 3.14, frozen dataclasses, PocketRocks SDK simulation,
NumPy, SciPy Plackett-Luce fitting, multiprocessing, JSON, pytest, mypy, Ruff.

## Global Constraints

- Public names must explain their role: use `PromotionCorpus`,
  `PromotionCase`, `PairedGamePlan`, `PromotionAnalysis`, and
  `PromotionReport`; do not introduce unexplained statistical abbreviations.
- User-facing docs must define “held-out” as “not used for tuning” and a
  bootstrap interval as an uncertainty range from resampled complete pairs.
- The gate is strategy-neutral and accepts arbitrary `BotSpec` values.
- Do not change any bot strategy, coefficient, alias, checkpoint, identity, or
  ordinary tournament default.
- Development and held-out expanded engine-seed sets must be disjoint.
- Candidate and incumbent twins must preserve engine seed, chart, player
  count, focal seat, opponent identities, opponent seats, and fault mode.
- Bootstrap sampling keeps both games in a pair together.
- Promotion requires a finite 95% interval whose lower endpoint is greater
  than zero, at least 90% bootstrap convergence, complete expected games,
  exact identities, and zero faults.
- JSON must use sorted keys, `allow_nan=False`, a terminal newline, and atomic
  replacement.
- Implement observable behavior with failing tests first.

---

### Task 1: Load, expand, hash, and compare immutable corpora

**Files:**

- Create: `src/garboid_pocketrocks/promotion/__init__.py`
- Create: `src/garboid_pocketrocks/promotion/corpus.py`
- Create: `configs/promotion/development-v1.json`
- Create: `configs/promotion/held-out-v1.json`
- Create: `tests/promotion/__init__.py`
- Create: `tests/promotion/test_corpus.py`

**Interfaces:**

- Produces:

```python
CorpusPurpose = Literal["development", "held_out"]


@dataclass(frozen=True, slots=True)
class PromotionCorpusRecipe:
    schema_version: int
    name: str
    purpose: CorpusPurpose
    root_seed: int
    repetitions_per_seat_cell: int
    charts: tuple[str, ...]
    player_counts: tuple[int, ...]
    opponent_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PromotionCase:
    case_id: str
    chart: str
    player_count: int
    focal_seat: int
    engine_seed: int
    opponent_names_by_seat: tuple[str | None, ...]


@dataclass(frozen=True, slots=True)
class PromotionCorpus:
    recipe: PromotionCorpusRecipe
    cases: tuple[PromotionCase, ...]
    digest: str

    @property
    def engine_seeds(self) -> tuple[int, ...]: ...


class PromotionCorpusError(ValueError):
    code: str


def load_promotion_corpus(
    path: Path,
    *,
    registry: Mapping[str, BotSpec],
) -> PromotionCorpus: ...


def validate_corpus_separation(
    development: PromotionCorpus,
    held_out: PromotionCorpus,
) -> None: ...


def corpus_snapshot_payload(corpus: PromotionCorpus) -> dict[str, object]: ...
```

- Consumes `derive_seed(root_seed, namespace, index)` from
  `simulator.seeding`.
- Later tasks consume the expanded immutable cases and canonical digest.

- [ ] **Step 1: Write failing parsing and exact-expansion tests**

Create fixture recipes in `tmp_path`. Assert:

```python
corpus = load_promotion_corpus(path, registry=BOT_SPECS_BY_NAME)

assert corpus.recipe.name == "fixture-development-v1"
assert corpus.recipe.purpose == "development"
assert len(corpus.cases) == 5 * (3 + 4 + 5) * 2
assert len(set(corpus.engine_seeds)) == len(corpus.cases)
assert corpus.cases[0].case_id == "fixture-development-v1:A:3:seat-0:repeat-0"
assert corpus.cases[0].opponent_names_by_seat[0] is None
assert corpus.digest == load_promotion_corpus(path, registry=BOT_SPECS_BY_NAME).digest
```

Use four opponent names and assert every case has exactly one `None` at
`focal_seat`, distinct known opponents in all other seats, uppercase charts,
and deterministic ring rotation.

- [ ] **Step 2: Write failing validation tests**

Parametrize malformed payloads and assert stable `PromotionCorpusError.code`
values:

```text
unsupported_schema
invalid_corpus_name
invalid_purpose
invalid_root_seed
invalid_repetitions
unsupported_chart
unsupported_player_count
unknown_opponent
duplicate_opponent
insufficient_opponents
duplicate_engine_seed
```

Assert unknown JSON keys and missing required keys fail instead of being
ignored. Assert a development recipe cannot declare `held_out` and vice versa
when loaded from the committed expected path in the packaging test.

- [ ] **Step 3: Write the failing separation test**

Construct two otherwise valid frozen corpus fixtures whose cases share one
expanded seed and assert:

```python
with pytest.raises(PromotionCorpusError) as captured:
    validate_corpus_separation(development, held_out)
assert captured.value.code == "corpus_seed_overlap"
```

Also reject reversed purposes and identical corpus names. A valid pair returns
`None`.

- [ ] **Step 4: Run RED**

Run:

```bash
uv run pytest -n 0 tests/promotion/test_corpus.py -q
```

Expected: collection fails because `promotion.corpus` does not exist.

- [ ] **Step 5: Implement strict recipe decoding**

Decode JSON without permissive coercion. Reject booleans where integer fields
are expected. Normalize charts to uppercase only after checking each raw chart
is a one-character string. Require unique charts/player counts/opponents and
at least four opponents because five-player cases need four non-focal seats.

Use a helper with an explicit expected-key set:

```python
def _require_exact_keys(
    payload: Mapping[str, object],
    expected: set[str],
    *,
    subject: str,
) -> None:
    missing = expected - set(payload)
    unknown = set(payload) - expected
    if missing or unknown:
        raise PromotionCorpusError(
            "invalid_recipe_keys",
            f"{subject} has missing keys {sorted(missing)} and unknown keys {sorted(unknown)}",
        )
```

`PromotionCorpusError.__init__(code, message)` stores `code` and passes the
message to `ValueError`.

- [ ] **Step 6: Expand cases deterministically**

Loop in this exact order:

```python
for repetition in range(recipe.repetitions_per_seat_cell):
    for chart_index, chart in enumerate(recipe.charts):
        for player_count in recipe.player_counts:
            for focal_seat in range(player_count):
                case_index = len(cases)
```

Derive:

```python
engine_seed = derive_seed(
    recipe.root_seed,
    f"promotion-corpus:{recipe.name}",
    case_index,
)
rotation = (repetition + chart_index + player_count + focal_seat) % len(recipe.opponent_names)
```

Rotate the opponent ring, take `player_count - 1` names, and fill seats in
ascending order while leaving `None` at `focal_seat`.

- [ ] **Step 7: Hash the canonical expanded form**

Build one JSON-safe dictionary containing the recipe and every expanded case.
Serialize with:

```python
encoded = (
    json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
).encode("utf-8")
digest = hashlib.sha256(encoded).hexdigest()
```

The public snapshot payload uses the same normalized fields and includes the
digest. Validate unique seeds within a corpus and disjoint seed sets across
the pair.

- [ ] **Step 8: Add committed corpus recipes**

`development-v1.json` contains purpose `development`, root seed `9001`, four
repetitions, charts A-E, player counts 3-5, and opponents:

```json
["random", "aggressive-v1", "balanced-v1", "passive-v1"]
```

`held-out-v1.json` contains purpose `held_out`, root seed `90001`, eight
repetitions, and the same coverage/opponent ring. Add tests asserting 240
development cases, 480 held-out cases, stable nonempty digests, and disjoint
expanded seeds.

- [ ] **Step 9: Run GREEN and quality checks**

Run:

```bash
uv run pytest -n 0 tests/promotion/test_corpus.py -q
uv run ruff check src/garboid_pocketrocks/promotion/corpus.py tests/promotion/test_corpus.py
uv run mypy src/garboid_pocketrocks/promotion/corpus.py tests/promotion/test_corpus.py
git diff --check
```

Expected: all pass.

- [ ] **Step 10: Commit**

```bash
git add \
  configs/promotion \
  src/garboid_pocketrocks/promotion \
  tests/promotion
git commit -m "feat: define immutable promotion corpora"
```

---

### Task 2: Plan exact candidate/incumbent twin games

**Files:**

- Create: `src/garboid_pocketrocks/promotion/planning.py`
- Create: `tests/promotion/test_planning.py`

**Interfaces:**

- Consumes `PromotionCorpus`, its cases, a candidate/incumbent `BotSpec`, and
  the registered opponent mapping.
- Produces:

```python
@dataclass(frozen=True, slots=True)
class PairedGamePlan:
    pair_index: int
    case: PromotionCase
    candidate_game: GameJob
    incumbent_game: GameJob


@dataclass(frozen=True, slots=True)
class PromotionPlan:
    candidate: BotSpec
    incumbent: BotSpec
    opponents: tuple[BotSpec, ...]
    pairs: tuple[PairedGamePlan, ...]
    monte_carlo_config: MonteCarloConfig

    @property
    def jobs(self) -> tuple[GameJob, ...]: ...


class PromotionPlanningError(ValueError):
    code: str


def plan_paired_games(
    held_out: PromotionCorpus,
    *,
    candidate: BotSpec,
    incumbent: BotSpec,
    registry: Mapping[str, BotSpec],
) -> PromotionPlan: ...
```

- `jobs` flattens each pair as candidate then incumbent so game indices are
  `2 * pair_index` and `2 * pair_index + 1`.

- [ ] **Step 1: Write failing twin-equivalence tests**

Use a one-case corpus and local `BotSpec.for_simulation` candidate/incumbent.
Assert:

```python
pair = plan.pairs[0]
assert pair.candidate_game.seed == pair.incumbent_game.seed == pair.case.engine_seed
assert pair.candidate_game.player_count == pair.incumbent_game.player_count
assert pair.candidate_game.value_chart == pair.incumbent_game.value_chart
assert pair.candidate_game.fault_mode is FaultMode.RECORD_AND_PASS
assert pair.incumbent_game.fault_mode is FaultMode.RECORD_AND_PASS
assert pair.candidate_game.lineup[pair.case.focal_seat] == candidate
assert pair.incumbent_game.lineup[pair.case.focal_seat] == incumbent
```

For every non-focal seat, assert candidate/incumbent jobs have the same
opponent name, bot ID, and seat. Assert jobs are contiguous and the
`MonteCarloConfig` contains each distinct identity exactly once.

- [ ] **Step 2: Write failing identity tests**

Assert stable codes for:

```text
candidate_incumbent_identity_collision
candidate_opponent_identity_collision
incumbent_opponent_identity_collision
opponent_identity_mismatch
held_out_corpus_required
```

Check both names and bot IDs: two specs with different names but the same
`bot_id` are not distinct identities.

- [ ] **Step 3: Run RED**

Run:

```bash
uv run pytest -n 0 tests/promotion/test_planning.py -q
```

Expected: import fails because `promotion.planning` does not exist.

- [ ] **Step 4: Implement lineup construction**

Resolve each non-`None` case opponent from `registry`. Build twin lineups with
one focal substitution and otherwise identical tuple positions. Create
`GameJob` values with:

```python
GameJob(
    game_index=2 * pair_index + variant_offset,
    root_seed=held_out.recipe.root_seed,
    seed=case.engine_seed,
    player_count=case.player_count,
    value_chart=case.chart,
    objectives_enabled=True,
    lineup=lineup,
    fault_mode=FaultMode.RECORD_AND_PASS,
)
```

- [ ] **Step 5: Build the exact simulator configuration**

Preserve candidate, incumbent, then first-seen opponent order while
deduplicating by bot ID. Use:

```python
MonteCarloConfig(
    bot_specs=all_specs,
    games=2 * len(held_out.cases),
    player_counts=held_out.recipe.player_counts,
    value_charts=held_out.recipe.charts,
    root_seed=held_out.recipe.root_seed,
    objectives_enabled=(True,),
    fault_mode=FaultMode.RECORD_AND_PASS,
)
```

- [ ] **Step 6: Run GREEN and commit**

Run:

```bash
uv run pytest -n 0 tests/promotion/test_corpus.py tests/promotion/test_planning.py -q
uv run ruff check src/garboid_pocketrocks/promotion tests/promotion
uv run mypy src/garboid_pocketrocks/promotion tests/promotion
git diff --check
```

Commit:

```bash
git add \
  src/garboid_pocketrocks/promotion/planning.py \
  tests/promotion/test_planning.py
git commit -m "feat: plan paired promotion games"
```

---

### Task 3: Validate results and bootstrap whole paired cases

**Files:**

- Create: `src/garboid_pocketrocks/promotion/analysis.py`
- Create: `tests/promotion/helpers.py`
- Create: `tests/promotion/test_analysis.py`

**Interfaces:**

- Consumes `PromotionPlan`, `MonteCarloResult`, and bootstrap configuration.
- Produces:

```python
@dataclass(frozen=True, slots=True)
class PromotionFailure:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class RatingDifferenceInterval:
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class PromotionAnalysis:
    requested_pairs: int
    completed_pairs: int
    requested_games: int
    completed_games: int
    rating_difference: float | None
    interval: RatingDifferenceInterval | None
    bootstrap_requested: int
    bootstrap_converged: int
    faults_by_identity: tuple[tuple[str, int], ...]
    warnings: tuple[str, ...]
    failures: tuple[PromotionFailure, ...]
    promoted: bool


def analyze_promotion(
    plan: PromotionPlan,
    result: MonteCarloResult,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    workers: int = 1,
) -> PromotionAnalysis: ...


def bootstrap_paired_rating_differences(
    pairs: Sequence[tuple[GameSummary, GameSummary]],
    bot_ids: tuple[str, ...],
    *,
    candidate_id: str,
    incumbent_id: str,
    samples: int,
    root_seed: int,
    workers: int = 1,
) -> tuple[tuple[float, ...], int, tuple[str, ...]]: ...
```

- Later tasks serialize `PromotionAnalysis` without recomputing the decision.

- [ ] **Step 1: Add helpers for exact synthetic summaries**

`tests/promotion/helpers.py` creates a `PromotionCase`, `PromotionPlan`, and
`GameSummary` twins from explicit final-money/rank tuples. Helpers must retain
the production game indices, seeds, charts, lineups, and focal seat instead of
mocking validation away.

- [ ] **Step 2: Write failing result-validation tests**

Start from one valid pair and mutate one field at a time. Assert ordered,
deduplicated failures for:

```text
missing_paired_game
unexpected_game
identity_mismatch
seed_mismatch
ruleset_mismatch
player_count_mismatch
bot_fault
```

Assert duplicated `game_index` is `unexpected_game`, a pair counts complete
only when both exact twins exist, and any nonzero value in any
`fault_counts` entry yields `bot_fault`.

- [ ] **Step 3: Write failing point-estimate and paired-bootstrap tests**

Build connected synthetic games containing candidate/incumbent and common
opponents. Assert:

```python
analysis = analyze_promotion(
    plan,
    result,
    bootstrap_samples=100,
    bootstrap_seed=42,
)
assert analysis.rating_difference is not None
assert analysis.interval is not None
assert analysis.bootstrap_converged >= 90
```

Monkeypatch or wrap `fit_plackett_luce` to record each bootstrap replicate's
game indices. Assert every candidate game appears the same number of times as
its incumbent twin. Compare worker counts 1 and 2 for exact equality.

- [ ] **Step 4: Write failing decision tests**

Pin synthetic fixtures for:

- a passing interval with `lower > 0`;
- `interval_includes_zero`;
- `bootstrap_incomplete` when fewer than 90% converge;
- `nonfinite_analysis` for nonfinite point/interval values;
- `rating_fit_failed` when the fitter raises `TournamentRatingError`;
- `bootstrap_samples <= 0` and `workers <= 0` input rejection.

Assert `promoted` is exactly:

```python
not failures and interval is not None and interval.lower > 0.0
```

- [ ] **Step 5: Run RED**

Run:

```bash
uv run pytest -n 0 tests/promotion/test_analysis.py -q
```

Expected: import fails because `promotion.analysis` does not exist.

- [ ] **Step 6: Implement exact result validation**

Index expected jobs by `game_index`; index summaries while detecting
duplicates. For each expected summary compare:

- root seed and engine seed;
- `ruleset_name(job.value_chart, job.objectives_enabled)`;
- player count;
- exact bot names and IDs by seat;
- score seat set `range(player_count)`;
- decision/fault tuple lengths.

Return validated pairs only when both twins are exact. Sort failures by
`(code, message)` and deduplicate exact duplicates for deterministic reports.

- [ ] **Step 7: Fit the point rating difference**

Use `observations_from_games` and `fit_plackett_luce` with
`tuple(spec.bot_id for spec in plan.monte_carlo_config.bot_specs)`. Compute:

```python
candidate_rating = fit.ratings_by_id[plan.candidate.bot_id].rating
incumbent_rating = fit.ratings_by_id[plan.incumbent.bot_id].rating
rating_difference = candidate_rating - incumbent_rating
```

Catch `TournamentRatingError`, add `rating_fit_failed`, and do not publish a
point estimate or interval.

- [ ] **Step 8: Implement whole-pair bootstrap**

For replicate `i`, seed `random.Random(derive_seed(root_seed,
"promotion-bootstrap", i))`. Draw `len(pairs)` pair indices with replacement.
Flatten both summaries from each selected pair into observations, fit, and
return one rating difference. Catch `TournamentRatingError` per replicate and
count it as non-converged.

Use a process pool only when `workers > 1`; initialize immutable pair
observations and IDs once per worker. If process startup fails, retry the
entire bootstrap serially and return a warning string. Preserve replicate
order so serial and parallel results match exactly.

- [ ] **Step 9: Build the interval and fail-closed analysis**

Require `ceil(samples * 0.9)` converged values. When sufficient, compute:

```python
RatingDifferenceInterval(
    lower=float(np.quantile(values, 0.025, method="linear")),
    upper=float(np.quantile(values, 0.975, method="linear")),
)
```

Reject any nonfinite point or endpoint. Add
`interval_includes_zero` when `lower <= 0.0`. A bootstrap fallback warning is
informational and does not fail the gate when convergence and results remain
valid.

- [ ] **Step 10: Run GREEN and commit**

Run:

```bash
uv run pytest -n 0 tests/promotion/test_analysis.py -q
uv run ruff check src/garboid_pocketrocks/promotion tests/promotion
uv run mypy src/garboid_pocketrocks/promotion tests/promotion
git diff --check
```

Commit:

```bash
git add \
  src/garboid_pocketrocks/promotion/analysis.py \
  tests/promotion/helpers.py \
  tests/promotion/test_analysis.py
git commit -m "feat: analyze paired promotion results"
```

---

### Task 4: Write authoritative deterministic artifacts

**Files:**

- Create: `src/garboid_pocketrocks/promotion/reporting.py`
- Create: `tests/promotion/test_reporting.py`

**Interfaces:**

- Consumes corpora, plan, completed summaries, analysis, execution
  configuration, and repository commit.
- Produces:

```python
@dataclass(frozen=True, slots=True)
class PromotionReport:
    schema_version: int
    repository_commit: str
    candidate: BotSpec
    incumbent: BotSpec
    opponents: tuple[BotSpec, ...]
    development: PromotionCorpus
    held_out: PromotionCorpus
    bootstrap_samples: int
    bootstrap_seed: int
    workers: int
    batch_size: int
    analysis: PromotionAnalysis
    artifact_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PromotionArtifacts:
    report_json: Path
    paired_games_jsonl: Path
    corpus_snapshot_json: Path


def build_promotion_report(
    *,
    repository_commit: str,
    candidate: BotSpec,
    incumbent: BotSpec,
    opponents: tuple[BotSpec, ...],
    development: PromotionCorpus,
    held_out: PromotionCorpus,
    bootstrap_samples: int,
    bootstrap_seed: int,
    workers: int,
    batch_size: int,
    analysis: PromotionAnalysis,
) -> PromotionReport: ...


def write_promotion_artifacts(
    output_dir: Path,
    *,
    report: PromotionReport,
    game_summaries: Sequence[GameSummary],
    development: PromotionCorpus,
    held_out: PromotionCorpus,
    overwrite: bool = False,
) -> PromotionArtifacts: ...


def promotion_report_payload(report: PromotionReport) -> dict[str, object]: ...
```

- [ ] **Step 1: Write failing complete-schema tests**

Build a report and assert the JSON payload includes:

```text
schema_version
repository_commit
candidate
incumbent
opponents
execution
corpora
coverage
rating_difference
confidence_interval_95
bootstrap
faults
failures
promoted
artifacts
```

Assert corpus entries include names, digests, purposes, root seeds, and every
expanded engine seed. Assert failures include stable code and plain-English
message.

- [ ] **Step 2: Write failing deterministic artifact tests**

Write the same inputs to two directories and assert byte equality for each
corresponding artifact. Parse every JSON line and assert game summaries are
ordered by `game_index`. Assert `promotion-report.json` ends in `\n`.

Pass a report with `math.nan` in a numeric field and assert writing fails
without leaving a partial final artifact.

- [ ] **Step 3: Write failing output safety tests**

Assert:

- a nonempty directory fails without `overwrite`;
- `overwrite=True` replaces only the three known artifact files;
- an unrelated existing file is never deleted;
- a simulated `os.replace` failure leaves the previous valid artifact intact
  and removes the temporary file.

- [ ] **Step 4: Run RED**

Run:

```bash
uv run pytest -n 0 tests/promotion/test_reporting.py -q
```

Expected: import fails because `promotion.reporting` does not exist.

- [ ] **Step 5: Implement JSON-safe explicit payloads**

Do not serialize `BotSpec.brain_factory`. Convert identities, summaries,
scores, analysis, failures, and corpora through explicit functions. Use
`asdict` only for leaf dataclasses whose fields are all part of the public
schema.

Render `paired-games.jsonl` with one sorted-key JSON object and terminal
newline per game. Sort by `game_index`.

- [ ] **Step 6: Implement atomic artifact writing**

Validate the directory before simulation and again before writing. Render all
three payloads to strings before replacing any file so `allow_nan=False`
cannot create a partial artifact set. Use a temporary file in `output_dir`,
flush, `os.fsync`, then `os.replace`. Preserve unrelated files.

- [ ] **Step 7: Run GREEN and commit**

Run:

```bash
uv run pytest -n 0 tests/promotion/test_reporting.py -q
uv run ruff check src/garboid_pocketrocks/promotion tests/promotion
uv run mypy src/garboid_pocketrocks/promotion tests/promotion
git diff --check
```

Commit:

```bash
git add \
  src/garboid_pocketrocks/promotion/reporting.py \
  tests/promotion/test_reporting.py
git commit -m "feat: write promotion decision artifacts"
```

---

### Task 5: Orchestrate simulation and expose `garboid-promote`

**Files:**

- Create: `src/garboid_pocketrocks/promotion/runner.py`
- Create: `src/garboid_pocketrocks/promotion/cli.py`
- Create: `tests/promotion/test_runner.py`
- Create: `tests/promotion/test_cli.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**

- Produces:

```python
@dataclass(frozen=True, slots=True)
class PromotionRunConfig:
    candidate: BotSpec
    incumbent: BotSpec
    development: PromotionCorpus
    held_out: PromotionCorpus
    bootstrap_samples: int = 1_000
    bootstrap_seed: int = 0
    batch_size: int = 64


@dataclass(frozen=True, slots=True)
class PromotionRun:
    config: PromotionRunConfig
    plan: PromotionPlan | None
    monte_carlo_result: MonteCarloResult | None
    report: PromotionReport
    artifacts: PromotionArtifacts


class PromotionRunner:
    @staticmethod
    def run(
        config: PromotionRunConfig,
        *,
        registry: Mapping[str, BotSpec],
        workers: int,
        output_dir: Path,
        overwrite: bool = False,
        repository_commit: str | None = None,
    ) -> PromotionRun: ...


def main(argv: Sequence[str] | None = None) -> int: ...
```

- [ ] **Step 1: Write failing real-simulator integration tests**

Use tiny development/held-out fixture corpora and importable random-brain
`BotSpec` fixtures. Run with workers 1 and 2 into separate directories.
Assert:

```python
assert serial.plan == parallel.plan
assert serial.monte_carlo_result == parallel.monte_carlo_result
assert serial.report == parallel.report
assert serial.artifacts.report_json.read_bytes() == parallel.artifacts.report_json.read_bytes()
assert (
    serial.artifacts.paired_games_jsonl.read_bytes()
    == parallel.artifacts.paired_games_jsonl.read_bytes()
)
```

Use `repository_commit="test-commit"` to avoid environment differences.

- [ ] **Step 2: Write failing fail-closed runner tests**

Cover:

- overlapping corpus seeds;
- candidate/incumbent identity collision;
- a brain returning an illegal bid;
- a brain raising at construction or decision time;
- `MonteCarloRunner.run_jobs` raising `SimulationError`;
- an analyzer returning a nonfinite value;
- missing game summaries.

For each expected domain failure, assert the runner returns a report with
`promoted is False`, the stable reason code, and a written
`promotion-report.json`. Filesystem errors still raise.

- [ ] **Step 3: Write failing CLI tests**

Assert parser defaults:

```python
assert args.development_corpus == Path("configs/promotion/development-v1.json")
assert args.held_out_corpus == Path("configs/promotion/held-out-v1.json")
assert args.bootstrap_samples == 1_000
assert args.bootstrap_seed == 0
assert args.batch_size == 64
```

Patch the runner for fast command tests and assert:

- promoted prints “passed the held-out final exam” and exits 0;
- not promoted prints “did not pass” plus each plain-English reason and exits
  1;
- unknown bot, same identity, bad corpus, or operational failure prints a
  concise error and exits 2;
- output always identifies `promotion-report.json`.

Assert `garboid-promote --help` defines held-out, bootstrap interval, candidate,
incumbent, development corpus, and output directory in plain English.

- [ ] **Step 4: Run RED**

Run:

```bash
uv run pytest -n 0 \
  tests/promotion/test_runner.py \
  tests/promotion/test_cli.py -q
```

Expected: imports fail because runner and CLI do not exist.

- [ ] **Step 5: Implement fail-closed orchestration**

Call `validate_artifact_output_dir` before simulation. Resolve
`repository_commit` once. Then:

1. validate corpus separation;
2. plan paired games;
3. run `MonteCarloRunner.run_jobs` using configured batch size/workers;
4. analyze complete results;
5. build and write all artifacts.

Catch `PromotionCorpusError`, `PromotionPlanningError`, `SimulationError`, and
`TournamentRatingError` at this boundary. Convert them to a
`PromotionAnalysis` with `promoted=False` and the relevant stable failure.
When no plan/result exists, write an empty games artifact and both corpus
snapshots. Do not catch `OSError`, `FileExistsError`, `KeyboardInterrupt`, or
`SystemExit`.

- [ ] **Step 6: Implement the CLI**

Resolve candidate/incumbent through `BOT_SPECS_BY_NAME`. Load corpora with the
same registry. Use `max(1, (os.cpu_count() or 2) - 1)` as the worker default.
Require a positive bootstrap sample count, worker count, and batch size.

After the run:

```python
if run.report.analysis.promoted:
    print(f"{candidate.name} passed the held-out final exam.")
    interval = run.report.analysis.interval
    assert interval is not None
    print(f"95% uncertainty interval: {interval.lower:.2f} to {interval.upper:.2f} rating points.")
    print(f"Report: {run.artifacts.report_json}")
    return 0
print(f"{candidate.name} did not pass the held-out final exam.")
for failure in run.report.analysis.failures:
    print(f"- {failure.message}")
print(f"Report: {run.artifacts.report_json}")
return 1
```

Catch invocation/domain errors, print parser usage plus a direct message, and
return 2.

- [ ] **Step 7: Register and lock the command**

Add:

```toml
garboid-promote = "garboid_pocketrocks.promotion.cli:main"
```

Run `uv lock` only if the entry-point metadata changes the lock. Add a
packaging test that reads `pyproject.toml` and asserts the exact script target.

- [ ] **Step 8: Run GREEN and commit**

Run:

```bash
uv run pytest -n 0 tests/promotion -q
uv run garboid-promote --help
uv run ruff check src/garboid_pocketrocks/promotion tests/promotion
uv run mypy src/garboid_pocketrocks/promotion tests/promotion
git diff --check
```

Commit:

```bash
git add \
  pyproject.toml uv.lock \
  src/garboid_pocketrocks/promotion/runner.py \
  src/garboid_pocketrocks/promotion/cli.py \
  tests/promotion/test_runner.py \
  tests/promotion/test_cli.py \
  tests/test_neural_packaging.py
git commit -m "feat: run held-out promotion gate"
```

---

### Task 6: Document, exercise, and verify the complete final exam

**Files:**

- Create: `src/garboid_pocketrocks/promotion/README.md`
- Modify: `README.md`
- Modify: `docs/README.md`
- Modify: `src/garboid_pocketrocks/tournament/README.md`
- Modify: `tests/test_documentation.py` only if the new link exposes a test
  defect
- Create: `tests/promotion/test_integration.py`

**Interfaces:**

- Consumes the complete package and committed corpora.
- Produces current user documentation and byte-stability/fail-closed evidence.

- [ ] **Step 1: Write the package README in plain English**

Lead with:

```text
Promotion is a fair final exam for a new bot. Development games may be used
while tuning. Held-out games are fixed games that were not used for tuning.
The candidate and incumbent play matched copies of each held-out case.
```

Then document:

- the exact `garboid-promote` command;
- the three artifacts and authoritative report;
- every failure reason and exit code;
- “bootstrap interval” as an uncertainty range from resampling complete matched
  cases;
- why failure to promote does not prove the candidate is worse;
- how to version a new corpus without editing an old corpus;
- the rule that strategy/identity changes occur only in later issues after a
  passing report.

- [ ] **Step 2: Link the hierarchy**

Add one short promotion command to the root README. Link the promotion README
from `docs/README.md` and the tournament README. Do not duplicate the detailed
statistical explanation.

- [ ] **Step 3: Add deterministic integration coverage**

Use a tiny real corpus and importable bot specs. Assert:

- serial and parallel executions produce identical reports and artifacts;
- candidate/incumbent twins cover every requested chart/player count/focal
  seat;
- the same inputs and `repository_commit` reproduce exact bytes;
- changing the held-out seed changes the corpus digest and executed game
  seeds;
- an illegal bot produces `promoted=False`, a `bot_fault` reason, and zero
  false promotion claims;
- no test asserts that a released candidate is stronger.

- [ ] **Step 4: Run focused integration and command checks**

Run:

```bash
uv run pytest -n 0 tests/promotion tests/test_documentation.py -q
uv run garboid-promote --help
uv run garboid-tournament --help
```

Expected: all pass and both commands exit zero for help.

- [ ] **Step 5: Run the complete quality gate**

Run outside the restricted sandbox where multiprocessing requires it:

```bash
uv run --extra neural pytest -n 0 -q
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run --extra neural mypy --config-file mypy.neural.ini src tests
git diff --check
```

Expected: all tests and static checks pass.

- [ ] **Step 6: Audit the issue acceptance criteria**

Inspect the committed configs, production interfaces, tests, and one tiny
generated report. Record evidence that:

- both corpus recipes expand immutably and reject seed overlap;
- twins preserve chart/player count/focal seat/opponents/seed;
- the report includes identities, configuration, seeds, commit, artifacts,
  faults, rating difference, interval, and decision;
- identity mismatch, missing games, illegal actions/faults, and nonfinite
  analysis fail closed;
- deterministic reruns reproduce the same report.

- [ ] **Step 7: Commit**

```bash
git add \
  README.md docs/README.md \
  src/garboid_pocketrocks/promotion/README.md \
  src/garboid_pocketrocks/tournament/README.md \
  tests/promotion/test_integration.py
git commit -m "docs: explain the held-out promotion gate"
```

- [ ] **Step 8: Request review and prepare the stacked draft PR**

Review the complete range from the issue #8 branch tip. Address every Critical
or Important finding, rerun the complete gate after fixes, push
`codex/issue-9-promotion-gate`, and create a draft PR with base
`codex/issue-8-cleanup` and `Closes #9`.
