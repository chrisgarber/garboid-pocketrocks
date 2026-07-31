# Phase-Aware Heuristic v4 Implementation Plan

> **For agentic workers:** Use test-driven development and implement this plan
> task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, tune, diagnose, freeze, promote, and release three
phase-aware heuristic v4 policies without changing any v1-v3 behavior or
letting held-out results influence development choices.

**Architecture:** One shared selector derives early, middle, or late solely
from the public resource horizon. Each personality is one composite policy
with three ordinary four-coefficient experts, while schema-v2 evolution
jointly searches the resulting twelve loci. Development search, winner-only
diagnostics, frozen-candidate provenance, held-out promotion, and release are
separate reviewable stages.

**Tech Stack:** Python 3.14, frozen dataclasses, `Decimal`, SHA-256, JSON/JSONL
and CSV artifacts, the existing simulator/evolution/promotion pipelines,
pytest, mypy, and Ruff.

## Global Constraints

- Let `R` be future biddable resources and `T` be total biddable resources.
  Select early when `3 * R >= 2 * T`, middle when `3 * R >= T`, and late
  otherwise. Comparisons are inclusive integer arithmetic.
- For the canonical 3/4/5-player games, this is early at `R >= 10`, middle at
  `R == 5..9`, and late at `R == 0..4`.
- Derive `R` and `T` only from public SDK context plus immutable public
  ruleset knowledge. Never read a private hand, hidden deck, simulator state,
  seed, or opponent model.
- Each aggressive, balanced, and passive candidate contains exactly three
  experts and twelve loci, ordered early/middle/late and then
  liquidity/future-cash/objective-progress/bid-shading.
- Schema-v2 evolution is development-only. Its manifest precommits the phase
  rule, predecessor, corpus digest, grids, seed, and algorithm settings.
- Preserve schema-v1 normalized bytes, digests, v3 candidate identities,
  search artifacts, and exact rerun behavior.
- Evaluate each policy once per decision. Winner-only diagnostics reuse that
  selection and record `selected_expert_phase`; they do not rerun valuation.
- Freeze explicit v4 candidate identities. Do not add candidate identities to
  the released registry or use a live bot ID.
- Run the common unchanged held-out gate against the exact matching v3
  predecessor. Held-out evidence may accept or reject a frozen candidate but
  may not alter boundaries, coefficients, manifests, or development winners.
- Add `HEURISTIC_V4` and advance latest/unversioned aliases only after all
  three intended personalities pass. Keep all v1-v3 constants, brains, specs,
  registry entries, representative decisions, and remote bot IDs unchanged.
- Canonical finite JSON, deterministic ordering, terminal newlines,
  transactional artifact replacement, complete A-E × 3/4/5 coverage, and
  zero illegal decisions or bot faults are required.
- Every observable behavior change starts RED, becomes GREEN with the minimum
  implementation, and is committed before the next task.

---

### Task 1: Shared public resource-horizon phase selector

**Files:**

- Create: `src/garboid_pocketrocks/heuristics/phases.py`
- Modify: `src/garboid_pocketrocks/diagnostics/analysis.py`
- Create: `tests/heuristics/test_phases.py`
- Modify: `tests/diagnostics/test_analysis.py`

**Interfaces:**

- Consumes: `DecisionContext`, `RulesetKnowledge`, and the existing public
  `won_resource_counts_by_seat` and `current_resource_ids`.
- Produces: `HeuristicPhase = Literal["early", "middle", "late"]`,
  `PublicResourceHorizon(total_biddable_resources: int,
  future_biddable_resources: int)`,
  `public_resource_horizon(context, ruleset) -> PublicResourceHorizon`, and
  `select_expert_phase(horizon) -> HeuristicPhase`.

- [ ] **Step 1: Write failing selector and diagnostic-boundary tests**

  Add table tests for `(R, T) = (10, 15) -> early`, `(9, 15) -> middle`,
  `(5, 15) -> middle`, `(4, 15) -> late`, `(10, 14) -> early`, and
  `(4, 14) -> late`. Build equivalent public contexts for 3, 4, and 5 players
  and assert private-hand changes cannot alter the returned horizon or phase.
  At the same won-resource count, assert Auction 1 excludes only its first
  offered card while Auction 2 excludes every nonzero offered card, including
  the valid one-card deck-tail case. Assert the following reveal context
  retains the same horizon after the award moves into public won counts.
  Keep the existing tests that pin `DecisionSlice.game_phase` to turns 1–5,
  6–12, and 13 onward. Add tests proving the diagnostic cash-horizon fields
  come from the shared public accounting without changing that legacy label.

- [ ] **Step 2: Run the focused tests and verify RED**

  Run:

  ```bash
  uv run pytest -n 0 tests/heuristics/test_phases.py \
    tests/diagnostics/test_analysis.py -q
  ```

  Expected: FAIL because `heuristics.phases` and the shared selector do not
  exist.

- [ ] **Step 3: Implement the selector once**

  In `phases.py`, calculate:

  ```python
  total = sum(ruleset.resource_counts) - (
      context.player_count * ruleset.private_cards_per_player
  )
  already_won = sum(sum(row) for row in context.won_resource_counts_by_seat)
  currently_offered = public_resource_count_awarded_by_current_action(context)
  future = total - already_won - currently_offered
  ```

  Decode the public current action and count the first offered resource for
  Auction 1, every nonzero offered resource for Auction 2, and zero for
  non-resource or reveal decisions. Validate matching player counts, public
  row widths, `T > 0`, and `0 <= R <= T`. Implement the two inclusive integer
  comparisons exactly.
  Replace diagnostics' private `_cash_horizon` calculation with the shared
  public accounting. Preserve its turn-index `_game_phase` function exactly;
  the v4 expert phase is a separate dimension added in Task 6.

- [ ] **Step 4: Run focused tests and verify GREEN**

  Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add src/garboid_pocketrocks/heuristics/phases.py \
    src/garboid_pocketrocks/diagnostics/analysis.py \
    tests/heuristics/test_phases.py tests/diagnostics/test_analysis.py
  git commit -m "feat: select heuristic experts from public game phase"
  ```

---

### Task 2: Composite profiles and a single-evaluation phase-aware brain

**Files:**

- Modify: `src/garboid_pocketrocks/heuristics/profiles.py`
- Modify: `src/garboid_pocketrocks/bots/heuristic.py`
- Modify: `src/garboid_pocketrocks/diagnostics/trace.py`
- Modify: `tests/heuristics/test_profiles.py`
- Modify: `tests/bots/test_heuristic_bots.py`
- Modify: `tests/diagnostics/test_trace.py`

**Interfaces:**

- Consumes: `select_expert_phase` and the unchanged `HeuristicValuator`.
- Produces: `PhaseAwareHeuristicProfile(name, early, middle, late)`,
  `profile_for_phase(phase) -> HeuristicProfile`, and
  `PhaseAwareHeuristicBotBrain`. Adds a separate
  `PhaseAwareHeuristicBidExplanation` containing the ordinary bid breakdown,
  `selected_expert_phase`, `future_biddable_resources`, and
  `total_biddable_resources`.

- [ ] **Step 1: Write failing composite-policy tests**

  Assert invalid personality/expert names and missing phases are rejected.
  For fixed public contexts at `R=10`, `R=9`, `R=5`, and `R=4`, assert the
  brain's decision equals a direct `HeuristicBotBrain` using the selected
  expert, and its explanation records the same phase. Add a counting
  valuator/brain regression proving one valuation call per decision. Snapshot
  all existing v1-v3 representative decisions and explanation payloads. Pin
  existing trace-schema-v1 bytes, then require phase-aware bid traces to use
  schema v2 and explanation kind `phase_aware_heuristic_bid`. Strictly reject
  v1/v2 field mixing, alternate phases, and inconsistent horizon counts.

- [ ] **Step 2: Run tests and verify RED**

  ```bash
  uv run pytest -n 0 tests/heuristics/test_profiles.py \
    tests/bots/test_heuristic_bots.py tests/diagnostics/test_trace.py -q
  ```

  Expected: FAIL because composite profiles, the phase-aware brain, and the
  explanation field are absent.

- [ ] **Step 3: Implement composite selection without duplicating policy**

  Construct one valuator per expert in `PhaseAwareHeuristicBotBrain.__init__`.
  Select the expert before `_choose_raw`, pass the already-selected phase into
  explanation construction, and keep reveal behavior and input-error fallback
  identical to `HeuristicBotBrain`. Version-dispatch trace encoding and
  decoding so legacy explanations remain exact schema v1 while phase-aware
  bid explanations use strict schema v2. Shared reveals, invalid-input passes,
  and fault fallbacks carry no expert phase. Do not modify `HEURISTIC_V1`,
  `HEURISTIC_V2`, `HEURISTIC_V3`, or latest aliases in this task.

- [ ] **Step 4: Run tests and verify GREEN**

  Run the Step 2 command. Expected: PASS with unchanged v1-v3 snapshots.

- [ ] **Step 5: Commit**

  ```bash
  git add src/garboid_pocketrocks/heuristics/profiles.py \
    src/garboid_pocketrocks/bots/heuristic.py \
    src/garboid_pocketrocks/diagnostics/trace.py \
    tests/heuristics/test_profiles.py tests/bots/test_heuristic_bots.py \
    tests/diagnostics/test_trace.py
  git commit -m "feat: add phase-aware heuristic policies"
  ```

---

### Task 3: Strict schema-v2 manifests with schema-v1 byte compatibility

**Files:**

- Create: `docs/benchmarks/tournaments/2026-07-30-heuristic-v3-phase-boundaries-development/`
- Create: `docs/benchmarks/2026-07-30-heuristic-v4-phase-boundaries.md`
- Modify: `src/garboid_pocketrocks/evolution/manifest.py`
- Modify: `tests/evolution/test_manifest.py`
- Create: `configs/evolution/aggressive-v4-search-v2.json`
- Create: `configs/evolution/balanced-v4-search-v2.json`
- Create: `configs/evolution/passive-v4-search-v2.json`

**Interfaces:**

- Consumes: existing `CoefficientValues`, `CoefficientGrids`,
  `SearchAlgorithm`, and development corpus binding.
- Produces: `PhaseCoefficientValues(early, middle, late)`,
  `PhaseCoefficientGrids(early, middle, late)`,
  `PhaseSearchManifest`, and
  `SearchRecipe = SearchManifest | PhaseSearchManifest`. Its canonical
  `as_loci()` order is phase first, then `COEFFICIENT_NAMES`.

- [ ] **Step 1: Commit development-only v3 boundary evidence**

  Run a v3-only tournament with `--decision-reports` across charts A-E and
  player counts 3/4/5 using a new recorded development seed. Record exact
  `(future_biddable_resources, total_biddable_resources)` selection counts and
  phase-sliced outcomes, explain why equal resource thirds give nonempty,
  useful ranges, and state that no alternate thresholds were searched. Commit
  only the privacy-safe derived `phase-boundary-slices.csv`, a distilled
  report, and their SHA-256 digests before constructing a v4 manifest. Keep
  the high-dimensional source slices, raw traces, and game summaries in
  temporary storage: publishing them beside the reproducible development seed
  would reconstruct private game state. Do not use the held-out corpus or
  promotion output.

- [ ] **Step 2: Pin schema-v1 golden bytes and write schema-v2 RED tests**

  Record the normalized payload bytes and digest for all three existing v3
  manifests. For schema v2, assert exact twelve-locus order, exact phase rule
  payload:

  ```json
  {
    "kind": "public-resource-horizon-v1",
    "early": "3*future>=2*total",
    "middle": "3*future>=total",
    "late": "otherwise"
  }
  ```

  plus the exact boundary-evidence path and digest. Assert rejection of
  missing/extra phases, duplicate keys, booleans,
  nonfinite/off-grid values, a v2 predecessor other than matching
  `<personality>-v3`, any held-out key, or any alternative boundary.

- [ ] **Step 3: Run manifest tests and verify RED**

  ```bash
  uv run pytest -n 0 tests/evolution/test_manifest.py -q
  ```

  Expected: FAIL on unsupported schema 2.

- [ ] **Step 4: Decode v2 through a version-specific path**

  Leave the schema-v1 payload builder and hash path untouched. Dispatch only
  after reading `schema_version`; decode v2's `initial_experts`,
  `expert_coefficient_grids`, and fixed `phase_selector`. Use explicit
  dataclasses rather than nested untyped mappings. Give v2 its own canonical
  payload builder and digest function.

- [ ] **Step 5: Add the three fixed development manifests**

  Start every expert from the matching v3 profile, use the same committed
  development corpus digest, fixed seeds, fixed grids, and fixed algorithm
  budget across personalities: twelve generations, sixteen proposals per
  generation, four elites, and a four-step mutation radius. The files must
  differ only where personality, predecessor, seed, and initial v3
  coefficients require it.

- [ ] **Step 6: Run tests and verify GREEN**

  Run the Step 3 command. Expected: PASS, including exact schema-v1 golden
  bytes and digests.

- [ ] **Step 7: Commit**

  ```bash
  git add docs/benchmarks/tournaments/2026-07-30-heuristic-v3-phase-boundaries-development \
    docs/benchmarks/2026-07-30-heuristic-v4-phase-boundaries.md \
    src/garboid_pocketrocks/evolution/manifest.py \
    tests/evolution/test_manifest.py configs/evolution/*-v4-search-v2.json
  git commit -m "feat: describe twelve-locus heuristic searches"
  ```

---

### Task 4: Deterministic twelve-locus candidates and joint mutation

**Files:**

- Modify: `src/garboid_pocketrocks/evolution/candidates.py`
- Modify: `src/garboid_pocketrocks/evolution/search.py`
- Modify: `tests/evolution/test_candidates.py`
- Modify: `tests/evolution/test_search.py`

**Interfaces:**

- Consumes: `PhaseSearchManifest`, `PhaseCoefficientValues`, and
  `PhaseAwareHeuristicBotBrain`.
- Produces: `PhaseCoefficientGenome`,
  `PhaseAwareHeuristicCandidate`, `phase_candidate_profile(candidate)`, and
  a picklable local-only candidate `BotSpec`. Candidate identities use
  `<personality>-v4-candidate-gNNN-sNNN-<12-hex-digest>`.

- [ ] **Step 1: Write failing deterministic-candidate tests**

  Pin generation-zero identities and the full twelve-value genomes for all
  three manifests. Require slot zero to be the all-v3 composite, slots 1–12
  to make one seeded nonzero perturbation at each distinct locus, and slots
  13–15 to make one broader seeded proposal for each phase. Assert each later
  child changes exactly one of twelve loci, uses deterministic stratified
  locus selection, records its parent, stays on the configured phase-specific
  grid, and is identical across repeated runs. Prove pickling and direct-brain
  decision agreement. Retain golden schema-v1 populations, v3 identities, and
  four-coefficient digests unchanged.

- [ ] **Step 2: Run tests and verify RED**

  ```bash
  uv run pytest -n 0 tests/evolution/test_candidates.py \
    tests/evolution/test_search.py -q
  ```

  Expected: FAIL because phase-aware candidate construction is absent.

- [ ] **Step 3: Implement version-dispatched population construction**

  Flatten phase values only at sampling/mutation/ranking boundaries, then
  reconstruct named `early`, `middle`, and `late` values. Generate identities
  from the twelve-locus canonical digest. Extend the ranking key with a
  version-neutral coefficient tuple while preserving the exact four-value
  schema-v1 key and ordering.

- [ ] **Step 4: Run tests and verify GREEN**

  Run the Step 2 command. Expected: PASS, including schema-v1 goldens.

- [ ] **Step 5: Commit**

  ```bash
  git add src/garboid_pocketrocks/evolution/candidates.py \
    src/garboid_pocketrocks/evolution/search.py \
    tests/evolution/test_candidates.py tests/evolution/test_search.py
  git commit -m "feat: evolve phase experts as one policy"
  ```

---

### Task 5: Run schema-v2 searches through unchanged development games

**Files:**

- Modify: `src/garboid_pocketrocks/evolution/runner.py`
- Modify: `src/garboid_pocketrocks/evolution/evaluation.py`
- Modify: `src/garboid_pocketrocks/evolution/reporting.py`
- Modify: `src/garboid_pocketrocks/evolution/cli.py`
- Modify: `src/garboid_pocketrocks/evolution/README.md`
- Modify: `tests/evolution/test_runner.py`
- Modify: `tests/evolution/test_evaluation.py`
- Modify: `tests/evolution/test_reporting.py`
- Modify: `tests/evolution/test_cli.py`

**Interfaces:**

- Consumes: `SearchRecipe` and either candidate type from Tasks 3-4.
- Produces: the existing `SearchRun`/`CandidateEvaluation`/artifact set with
  version-specific normalized candidate payloads and no held-out access.

- [ ] **Step 1: Write failing schema-v2 pipeline tests**

  On a tiny development corpus, assert serial, batched, and multi-worker runs
  have identical proposals, game plans, evaluations, selection logs, winner,
  and canonical bytes. Assert all twelve coefficients and the phase rule
  appear in reports. Assert incomplete/faulted evidence cannot freeze. Rerun
  the existing fixed schema-v1 fixtures and compare their complete artifact
  SHA-256 values.

- [ ] **Step 2: Run tests and verify RED**

  ```bash
  uv run pytest -n 0 tests/evolution/test_runner.py \
    tests/evolution/test_evaluation.py tests/evolution/test_reporting.py \
    tests/evolution/test_cli.py -q
  ```

  Expected: FAIL when the runner receives `PhaseSearchManifest`.

- [ ] **Step 3: Generalize orchestration, not game semantics**

  Dispatch manifest/candidate serialization by schema version, but continue
  to use `plan_development_games`, the cached exact v3 baseline, matched cases,
  the existing tie-aware ratings, selection key, complete-generation writer,
  and CLI exit codes. Do not add a held-out path or a trace flag to candidate
  evaluation.

- [ ] **Step 4: Run tests and verify GREEN**

  Run the Step 2 command, then:

  ```bash
  uv run pytest -n 0 tests/evolution -q
  ```

  Expected: all evolution tests PASS and schema-v1 artifact hashes match.

- [ ] **Step 5: Commit**

  ```bash
  git add src/garboid_pocketrocks/evolution tests/evolution
  git commit -m "feat: run phase-aware development searches"
  ```

---

### Task 6: Winner-only decision diagnostics and phase outcomes

**Files:**

- Modify: `src/garboid_pocketrocks/diagnostics/analysis.py`
- Modify: `src/garboid_pocketrocks/diagnostics/reporting.py`
- Modify: `tests/diagnostics/test_analysis.py`
- Modify: `tests/diagnostics/test_reporting.py`
- Create: `src/garboid_pocketrocks/evolution/diagnostics.py`
- Modify: `src/garboid_pocketrocks/evolution/cli.py`
- Modify: `src/garboid_pocketrocks/evolution/reporting.py`
- Create: `tests/evolution/test_diagnostics.py`
- Modify: `tests/evolution/test_cli.py`
- Modify: `tests/evolution/test_reporting.py`

**Interfaces:**

- Consumes: `PhaseAwareHeuristicBidExplanation` and a selected,
  complete development winner.
- Produces: `DecisionSlice.selected_expert_phase:
  HeuristicPhase | None`; additive selection counts and the existing
  final-money, normalized-finish, win, tie, and fault outcome sums grouped by
  phase. After search selection, the evolution command reruns only the winner
  on its exact 240 development cases with tracing enabled, reconciles that
  result in temporary storage, then writes only privacy-safe aggregate
  `winner-decision-slices.csv`, `winner-diagnostics.json`, and
  `winner-diagnostics.md` artifacts. The frozen winner report links every
  retained diagnostic artifact digest.

- [ ] **Step 1: Write failing reconciliation/reporting tests**

  Build phase-aware traces covering all three phases. Assert the selected
  expert equals the shared public phase, per-phase `decision_count` sums to
  total selections, all outcome sums reconcile, reversed inputs render
  byte-identically, and a mismatched/missing phase on a phase-aware
  explanation fails closed. Pin v1-only report and CSV bytes. Require mixed
  or phase-aware traces to produce report schema v2 and a
  `selected_expert_phase` column. Ordinary bots, reveals, invalid-input
  passes, and fault fallbacks leave that column empty and remain included in
  ordinary reconciliation totals without increasing expert selection counts.

- [ ] **Step 2: Run tests and verify RED**

  ```bash
  uv run pytest -n 0 tests/diagnostics/test_analysis.py \
    tests/diagnostics/test_reporting.py \
    tests/evolution/test_diagnostics.py tests/evolution/test_cli.py \
    tests/evolution/test_reporting.py -q
  ```

  Expected: FAIL because slices do not expose selected expert phase.

- [ ] **Step 3: Add the phase dimension and winner evidence link**

  Populate the slice only from the already-returned phase-aware explanation.
  Validate its phase and exact horizon counts against
  `select_expert_phase(public_resource_horizon(...))`. Version-dispatch report
  rendering: preserve the v1-only schema and columns exactly, while schema v2
  adds the expert-phase column and stable sort key. Add a winner-only evolution
  diagnostic runner that rebuilds the selected candidate's exact development
  plan, enables tracing for those jobs, constructs the ordinary statistics
  needed for reconciliation, and transactionally writes only the aggregate
  slice/summary artifacts. Raw per-decision and per-game rows stay temporary
  and are removed from the retained generation. Add retained content digests
  to the frozen-winner report. Do not enable diagnostics for the baseline or
  any losing proposal.

- [ ] **Step 4: Run tests and verify GREEN**

  Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add src/garboid_pocketrocks/diagnostics \
    src/garboid_pocketrocks/evolution/diagnostics.py \
    src/garboid_pocketrocks/evolution/cli.py \
    src/garboid_pocketrocks/evolution/reporting.py \
    tests/diagnostics tests/evolution/test_diagnostics.py \
    tests/evolution/test_cli.py tests/evolution/test_reporting.py
  git commit -m "feat: report phase expert selections and outcomes"
  ```

---

### Task 7: Execute development searches and freeze explicit v4 winners

**Files:**

- Create: `docs/benchmarks/evolution/*-v4-search-v2/`
- Modify: `src/garboid_pocketrocks/heuristics/frozen.py`
- Modify: `src/garboid_pocketrocks/heuristics/frozen_candidates/index.json`
- Create: `src/garboid_pocketrocks/heuristics/frozen_candidates/*-v4-candidate-*.json`
- Modify: `tests/heuristics/test_frozen_candidates.py`

**Interfaces:**

- Consumes: the three committed schema-v2 manifests and `development-v1`.
- Produces: three immutable `FrozenPhaseAwareCandidate` records with composite
  profiles, picklable local-only specs, search/corpus/report/diagnostic
  digests, and exact matching `<personality>-v3` predecessors.

- [ ] **Step 1: Add RED catalog tests before generating artifacts**

  Assert v4 identity/content/profile/manifest/search/diagnostic digests are
  cross-checked; all three experts and twelve values are required; v3
  predecessors are exact; v4 candidates are absent from released/default
  registries; and all existing v3 catalog files still load unchanged. Add
  tampering cases for phase swaps, alternate boundaries, substituted v3
  specs, and candidate/spec identity mismatches.

- [ ] **Step 2: Run catalog tests and verify RED**

  ```bash
  uv run pytest -n 0 tests/heuristics/test_frozen_candidates.py -q
  ```

  Expected: FAIL because schema-v2 frozen candidates are unsupported.

- [ ] **Step 3: Extend the strict catalog loader**

  Decode frozen schema version before version-specific exact-key validation.
  Keep the schema-v1/v3 path and hashes untouched. Build v4 candidate specs
  only from catalog-validated phase profiles and the fixed selector.

- [ ] **Step 4: Run each fixed development search exactly once**

  ```bash
  uv run garboid-evolve-heuristic \
    --manifest configs/evolution/aggressive-v4-search-v2.json \
    --corpus configs/promotion/development-v1.json \
    --output docs/benchmarks/evolution/aggressive-v4-search-v2
  uv run garboid-evolve-heuristic \
    --manifest configs/evolution/balanced-v4-search-v2.json \
    --corpus configs/promotion/development-v1.json \
    --output docs/benchmarks/evolution/balanced-v4-search-v2
  uv run garboid-evolve-heuristic \
    --manifest configs/evolution/passive-v4-search-v2.json \
    --corpus configs/promotion/development-v1.json \
    --output docs/benchmarks/evolution/passive-v4-search-v2
  ```

  Expected: each run completes without faults and freezes one positive-rating
  development winner. Preserve every proposal, game, ranking, selection,
  source snapshot, and digest.

- [ ] **Step 5: Validate the three winner-only diagnostic generations**

  Each search command must have created diagnostics only for its selected
  winner. Commit canonical traces/slices plus a short boundary summary showing
  the precommitted integer rule, expert coefficients, selection counts, and
  phase-sliced outcomes. Verify exact development-case coverage across A-E and
  3/4/5, zero illegal decisions/faults, and content digests bound into each
  frozen record.

- [ ] **Step 6: Install the three frozen records and verify GREEN**

  Update the catalog only from complete search and diagnostic evidence. Run:

  ```bash
  uv run pytest -n 0 tests/heuristics/test_frozen_candidates.py \
    tests/evolution -q
  ```

  Expected: PASS.

- [ ] **Step 7: Commit development evidence before any held-out run**

  ```bash
  git add configs/evolution docs/benchmarks/evolution \
    src/garboid_pocketrocks/heuristics/frozen.py \
    src/garboid_pocketrocks/heuristics/frozen_candidates \
    tests/heuristics/test_frozen_candidates.py
  git commit -m "data: freeze phase-aware development winners"
  ```

---

### Task 8: Bind frozen v4 provenance into the common held-out gate

**Files:**

- Modify: `src/garboid_pocketrocks/promotion/candidates.py`
- Modify: `src/garboid_pocketrocks/promotion/cli.py`
- Modify: `src/garboid_pocketrocks/promotion/reporting.py`
- Modify: `tests/promotion/test_candidates.py`
- Modify: `tests/promotion/test_cli.py`
- Modify: `tests/promotion/test_reporting.py`
- Modify: `tests/heuristics/test_promotion_evidence.py`

**Interfaces:**

- Consumes: catalog-loaded `FrozenPhaseAwareCandidate` and canonical
  `BOT_SPECS_BY_NAME["<personality>-v3"]`.
- Produces: ordinary promotion plans/reports that additionally bind all three
  expert digests, selector rule, development diagnostics digest, and exact
  v3 predecessor identity.

- [ ] **Step 1: Write failing provenance and bypass tests**

  Accept only the exact catalog object and exact canonical matching v3 spec.
  Reject arbitrary files, caller-created lookalikes, alternate catalogs,
  phase/profile substitution, another personality's predecessor, aliases,
  changed selector text, and tampered development/search/diagnostic evidence.
  Keep registered and frozen v3 promotion tests unchanged.

- [ ] **Step 2: Run tests and verify RED**

  ```bash
  uv run pytest -n 0 tests/promotion \
    tests/heuristics/test_promotion_evidence.py -q
  ```

  Expected: FAIL because promotion does not understand v4 frozen provenance.

- [ ] **Step 3: Generalize candidate provenance with a strict common protocol**

  Share only the immutable fields promotion actually consumes
  (`identity`, `bot_spec`, `predecessor_name`, and provenance digests).
  Keep version-specific validation in the frozen loader and serialize the
  complete v4 expert/selector provenance in the promotion report.

- [ ] **Step 4: Run tests and verify GREEN**

  Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add src/garboid_pocketrocks/promotion tests/promotion \
    tests/heuristics/test_promotion_evidence.py
  git commit -m "feat: gate phase-aware candidates against v3"
  ```

---

### Task 9: Run the three held-out promotions without tuning

**Files:**

- Create: `docs/benchmarks/promotions/2026-07-30-*-v4-candidate-*-vs-*-v3/`
- Create: `docs/benchmarks/2026-07-30-heuristic-v4-candidate-promotions.md`
- Modify: `tests/heuristics/test_promotion_evidence.py`

**Interfaces:**

- Consumes: unchanged `held-out-v1`, each exact frozen v4 candidate, and its
  exact v3 predecessor.
- Produces: complete paired-game evidence and one pass/fail decision per
  personality. It does not produce new coefficients or boundaries.

- [ ] **Step 1: Verify the freeze commit and record its SHA**

  Confirm manifests, search artifacts, diagnostics, frozen JSON, and catalog
  are committed and the worktree is clean. Record that commit in all three
  invocations.

- [ ] **Step 2: Run each frozen candidate exactly once**

  Use `garboid-promote` with the unchanged held-out config, the frozen
  candidate identity, its matching `<personality>-v3` incumbent, and 1,000
  paired bootstrap samples. Do not inspect partial results to modify any
  development artifact.

- [ ] **Step 3: Validate and summarize the evidence**

  Require complete A-E × 3/4/5 matched coverage, all requested game pairs and
  bootstrap samples, a strictly positive rating improvement with its
  confidence interval excluding zero, and zero candidate/incumbent/opponent/
  infrastructure faults. The summary must state the frozen identity, v3
  predecessor, boundaries, twelve coefficients, selection counts, and phase
  outcomes.

- [ ] **Step 4: Add evidence-integrity tests**

  Pin all promotion/report/corpus digests and assert each report names the
  exact frozen candidate and v3 predecessor. Run:

  ```bash
  uv run pytest -n 0 tests/heuristics/test_promotion_evidence.py \
    tests/promotion -q
  ```

  Expected: PASS. If any personality fails, commit the truthful failure
  evidence and stop before Task 10; do not advance any alias.

- [ ] **Step 5: Commit**

  ```bash
  git add docs/benchmarks/promotions \
    docs/benchmarks/2026-07-30-heuristic-v4-candidate-promotions.md \
    tests/heuristics/test_promotion_evidence.py
  git commit -m "data: record heuristic v4 promotion results"
  ```

---

### Task 10: Release v4 atomically only after all three pass

**Files:**

- Modify: `src/garboid_pocketrocks/heuristics/profiles.py`
- Modify: `src/garboid_pocketrocks/heuristics/__init__.py`
- Modify: `src/garboid_pocketrocks/bots/heuristic.py`
- Modify: `src/garboid_pocketrocks/bots/registry.py`
- Modify: `src/garboid_pocketrocks/bots/__init__.py`
- Modify: `tests/heuristics/test_profiles.py`
- Modify: `tests/bots/test_heuristic_bots.py`
- Modify: `tests/bots/test_registry.py`
- Modify: `tests/benchmarks/test_heuristic_tournament.py`

**Interfaces:**

- Consumes: the three passing frozen profiles and their exact coefficients.
- Produces: `HEURISTIC_V4`, explicit
  `Aggressive/Balanced/PassiveHeuristicV4Brain`, explicit `*-v4` simulation
  specs, `LATEST_HEURISTIC_POLICY_SET` plus the three latest `*_POLICY`
  aliases, and latest/unversioned brain and bot aliases pointing to v4. The
  existing scalar-profile aliases remain pinned to v3 for compatibility.

- [ ] **Step 1: Write failing atomic-release tests**

  Assert v4 constants exactly match the three promoted frozen records; each
  explicit v4 brain selects the documented expert at all inclusive
  boundaries; `*-v4` specs are registered and in default tournaments; and
  unversioned brains/specs use v4. Assert `LATEST_HEURISTICS` and the existing
  `*_PROFILE` aliases retain their scalar v3 values while the accurately named
  v4 policy aliases advance. Snapshot every v1-v3 constant, decision, bot spec
  name/ID, registry position, and frozen schema-v1 hash.

- [ ] **Step 2: Run tests and verify RED**

  ```bash
  uv run pytest -n 0 tests/heuristics/test_profiles.py \
    tests/bots/test_heuristic_bots.py tests/bots/test_registry.py \
    tests/benchmarks/test_heuristic_tournament.py -q
  ```

  Expected: FAIL because v4 release symbols do not exist.

- [ ] **Step 3: Add explicit v4 symbols, then move latest aliases together**

  Copy the promoted twelve coefficients into a
  `PhaseAwareHeuristicProfileSet(version="v4", ...)`. Add explicit v4 brains
  and simulation specs first. Only after all three exist, set
  `LATEST_HEURISTIC_POLICY_SET`, the three `*_POLICY` aliases, unversioned
  brains, live wrappers, and default tournament entries to v4. Preserve the
  scalar `LATEST_HEURISTICS`/`*_PROFILE` compatibility aliases, remote
  `BOT_ID` values, and every explicit v1-v3 object.

- [ ] **Step 4: Run tests and verify GREEN**

  Run the Step 2 command. Expected: PASS with every v1-v3 snapshot unchanged.

- [ ] **Step 5: Run A-E × 3/4/5 release smoke coverage**

  Run the focused benchmark/sanity command that schedules each v4 personality
  across every chart and player count. Expected: complete coverage, no illegal
  decisions, and zero faults.

- [ ] **Step 6: Commit**

  ```bash
  git add src/garboid_pocketrocks/heuristics \
    src/garboid_pocketrocks/bots tests/heuristics tests/bots \
    tests/benchmarks/test_heuristic_tournament.py
  git commit -m "feat: release promoted heuristic v4 bots"
  ```

---

### Task 11: Full verification, review, and stacked draft PR

**Files:**

- Modify documentation only if verification exposes a broken link or an
  omitted v4 command/result.

**Interfaces:**

- Consumes: the complete issue branch.
- Produces: verified commits and a draft PR stacked on
  `codex/issue-11-heuristic-evolution` with `Closes #12`.

- [ ] **Step 1: Run focused suites**

  ```bash
  uv run pytest -n 0 tests/heuristics tests/bots tests/diagnostics \
    tests/evolution tests/promotion tests/benchmarks/test_heuristic_tournament.py -q
  ```

  Expected: PASS.

- [ ] **Step 2: Run both complete test environments**

  ```bash
  uv run pytest -q
  uv run --extra neural pytest -q
  ```

  Expected: PASS with only already-documented third-party warnings.

- [ ] **Step 3: Run static and packaging gates**

  ```bash
  uv run ruff check .
  uv run ruff format --check .
  uv run mypy src
  uv run --extra neural mypy src
  uv lock --check
  uv run garboid-evolve-heuristic --help
  uv run garboid-promote --help
  uv run garboid-tournament --help
  git diff --check
  ```

  Expected: every command exits zero.

- [ ] **Step 4: Review invariants**

  Review the complete branch for public-state-only phase selection,
  single-evaluation decisions, twelve-locus ordering, schema-v1 byte/hash
  compatibility, development/held-out separation, exact v3 predecessors,
  winner-only diagnostics, catalog/provenance hardening, atomic release, and
  unchanged v1-v3/remote identities. Address every actionable finding with a
  failing regression test first.

- [ ] **Step 5: Push and create the draft stacked PR**

  Push `codex/issue-12-phase-aware-v4` and create a **draft** PR based on
  `codex/issue-11-heuristic-evolution`. Include development winners,
  phase-selection counts/outcomes, held-out deltas/confidence intervals, exact
  verification commands, and `Closes #12` in the body.
