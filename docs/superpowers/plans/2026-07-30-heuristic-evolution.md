# Heuristic Evolution Implementation Plan

> **For agentic workers:** Implement observable behavior with failing tests
> first. Do not change released identities, aliases, or heuristic formulas
> during search implementation.

**Goal:** Deterministically search the existing four heuristic coefficients
on development games, publish complete evidence, freeze explicit v3
candidates, and route them through the existing held-out promotion gate.

**Architecture:** Strict decimal-grid manifests propose one personality at a
time. Picklable candidate specs play matched development cases against a
cached v2 baseline. An immutable report owns every proposal, game, score, and
selection; a separate content-addressed catalog exposes only frozen winners
to promotion.

**Tech stack:** Python 3.14, frozen dataclasses, `Decimal`, SHA-256, existing
batch simulator and Plackett-Luce fitter, JSONL, pytest, mypy, Ruff.

## Global constraints

- Search exactly the four existing `HeuristicProfile` coefficients.
- Never read a held-out corpus from evolution code.
- Preserve every v1/v2 constant, class, identity, registry entry, and
  representative decision.
- Use explicit candidate identities; no live `BOT_ID`.
- Evaluate each policy once per decision through the existing simulator.
- Record every proposal, evaluation, ranking key, and selection.
- Development improvement may freeze a candidate but cannot promote it.
- Use only committed catalog candidates in the held-out CLI.
- Do not move `LATEST_HEURISTICS` or aliases unless promotion passes.
- Canonical finite JSON, deterministic ordering, terminal newlines, and
  transactional complete-generation output are required.

---

### Task 1: Strict manifests and fixed search recipes

**Files:**

- Create `src/garboid_pocketrocks/evolution/__init__.py`
- Create `src/garboid_pocketrocks/evolution/manifest.py`
- Create `configs/evolution/*-v3-search-v1.json`
- Create `tests/evolution/__init__.py`
- Create `tests/evolution/test_manifest.py`

**Work:**

- Decode exact schema keys and reject duplicates, booleans, nonfinite values,
  invalid decimals, off-grid values, unknown coefficients, and any held-out
  key.
- Bind personality, matching v2 predecessor, initial v2 coefficients, and
  development corpus name/digest.
- Normalize and hash with stable sorted JSON.
- Commit aggressive, balanced, and passive manifests with the design's fixed
  ranges and algorithm settings.

**Gate:**

```bash
uv run pytest -n 0 tests/evolution/test_manifest.py -q
```

---

### Task 2: Deterministic candidates and picklable specs

**Files:**

- Create `src/garboid_pocketrocks/evolution/candidates.py`
- Create `tests/evolution/test_candidates.py`
- Extend v1/v2 regression tests where needed

**Work:**

- Define immutable coefficient genomes and content digests.
- Implement the golden generation-zero grid samples and later one-field
  mutations.
- Record parent identity and stable generation/slot candidate identity.
- Materialize the existing `HeuristicProfile` and a `BotSpec` through a
  top-level helper plus `functools.partial`.
- Prove worker pickling, exact direct-brain decision agreement, unchanged
  reveals, and unchanged v1/v2 snapshots.

**Gate:**

```bash
uv run pytest -n 0 tests/evolution/test_candidates.py \
  tests/heuristics/test_profiles.py tests/bots/test_heuristic_bots.py -q
```

---

### Task 3: Development-only matched planning and evaluation

**Files:**

- Create `src/garboid_pocketrocks/evolution/planning.py`
- Create `src/garboid_pocketrocks/evolution/evaluation.py`
- Create `tests/evolution/test_planning.py`
- Create `tests/evolution/test_evaluation.py`

**Work:**

- Require a development corpus and reject held-out corpora.
- Build one reusable v2 baseline job and one candidate job for each exact
  development case.
- Validate seeds, chart, player count, focal seat, lineup, identity, coverage,
  completion, and faults before scoring.
- Fit the existing tie-aware rating model and calculate rating delta,
  normalized-finish delta, and final-money delta.
- Mark candidate faults ineligible; invalidate the run for
  incumbent/opponent/infrastructure evidence faults.
- Prove serial/batch/worker equivalence on a small real corpus.

**Gate:**

```bash
uv run pytest -n 0 tests/evolution/test_planning.py \
  tests/evolution/test_evaluation.py -q
```

---

### Task 4: Generation orchestration and complete selection log

**Files:**

- Create `src/garboid_pocketrocks/evolution/search.py`
- Create `src/garboid_pocketrocks/evolution/runner.py`
- Create `tests/evolution/test_search.py`
- Create `tests/evolution/test_runner.py`

**Work:**

- Evaluate generation zero, rank the complete pool, and select four elites.
- Generate every later population from prior elites in stable order.
- Rank previous elites plus new children by the exact documented key.
- Record every proposed candidate and every pool/elite decision.
- Freeze only a complete, fault-free candidate with positive development
  rating delta.
- Fail closed on missing or mismatched evidence.
- Prove repeated fixed runs and worker counts produce the same sequence,
  evaluations, selections, and winner.

**Gate:**

```bash
uv run pytest -n 0 tests/evolution/test_search.py \
  tests/evolution/test_runner.py -q
```

---

### Task 5: Transactional evidence artifacts and CLI

**Files:**

- Create `src/garboid_pocketrocks/evolution/reporting.py`
- Create `src/garboid_pocketrocks/evolution/cli.py`
- Create `src/garboid_pocketrocks/evolution/README.md`
- Create `tests/evolution/test_reporting.py`
- Create `tests/evolution/test_cli.py`
- Modify `pyproject.toml`
- Update runbook links

**Work:**

- Define one `SearchReport` that owns all normalized sources and results.
- Render the six or seven documented artifacts from that report alone.
- Use the existing complete-generation transaction and rollback pattern.
- Add `garboid-evolve-heuristic` with only execution-neutral CLI options.
- Return zero for a frozen improvement, one for a complete no-improvement
  search, and two for invalid/operational failure.
- Prove canonical bytes, failure rollback, stale freeze cleanup, plain-English
  output, and help text.

**Gate:**

```bash
uv run pytest -n 0 tests/evolution/test_reporting.py \
  tests/evolution/test_cli.py -q
```

---

### Task 6: Run the fixed searches and install frozen winners

**Files:**

- Create committed evidence under `docs/benchmarks/evolution/`
- Create `src/garboid_pocketrocks/heuristics/frozen_candidates/index.json`
- Create selected candidate JSON files
- Create `src/garboid_pocketrocks/heuristics/frozen.py`
- Create `tests/heuristics/test_frozen_candidates.py`

**Work:**

- Run all three fixed manifests on `development-v1`.
- Preserve complete search artifacts and their SHA-256 digests.
- Copy only eligible selected winners into the frozen catalog.
- Validate identity/content/catalog/search/corpus provenance and construct
  picklable local-only specs.
- Prove frozen candidates are absent from released/default registries and
  complete A-E × 3/4/5 development coverage with zero faults.

---

### Task 7: Bind frozen provenance into promotion

**Files:**

- Modify `src/garboid_pocketrocks/promotion/cli.py`
- Modify promotion report/configuration files
- Modify focused promotion tests and docs

**Work:**

- Resolve candidate names from the released registry or frozen catalog;
  resolve incumbents/opponents only from released registry.
- Require frozen predecessor and development digest to match the invocation.
- Record freeze, profile, manifest, evaluation-record, and corpus digests in
  promotion reports.
- Preserve ordinary registered-candidate promotion behavior.
- Reject arbitrary files, aliases, wrong predecessors, and tampered freezes.

**Gate:**

```bash
uv run pytest -n 0 tests/promotion tests/heuristics/test_frozen_candidates.py -q
```

---

### Task 8: Run held-out gates and release only passing v3 behavior

- Commit frozen candidates and search evidence before reading held-out
  results.
- Run each frozen winner exactly once against its matching v2 predecessor on
  unchanged `held-out-v1`, using 1,000 paired bootstrap samples.
- Preserve every promotion artifact.
- If the intended profile set passes, add exact `HEURISTIC_V3` constants,
  explicit v3 brains/specs, registry entries, and latest aliases while keeping
  v1/v2 and remote IDs immutable.
- If any intended behavior fails, do not create or advance that v3 behavior.
- Run all charts A-E and player counts 3/4/5 with zero faults.

---

### Task 9: Full verification and draft PR

- Run the full default and neural test suites.
- Run Ruff, format check, core mypy, strict neural mypy, lockfile validation,
  all relevant CLI help, documentation links, and `git diff --check`.
- Review the complete branch for reproducibility, development/held-out
  separation, provenance, version immutability, and issue acceptance.
- Push and open a draft PR stacked on
  `codex/issue-10-decision-reports` with `Closes #11`.
