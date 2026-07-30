# Decision Diagnostics Implementation Plan

> **For agentic workers:** Implement each task with failing tests first. Keep
> commits small enough to review and do not change bot strategy, coefficients,
> identities, aliases, checkpoints, or promotion behavior.

**Goal:** Add opt-in, privacy-safe, deterministic decision traces and
filterable tournament reports that reconcile with existing results.

**Architecture:** Capture the already-selected action and optional typed
explanation at the shared bot-execution boundary. Scalar and batch runners
finalize public outcomes into immutable traces. A strategy-neutral diagnostics
package validates and aggregates traces, and tournament reporting writes the
new artifacts as one safe generation.

**Tech Stack:** Python 3.14, frozen dataclasses and protocols, PocketRocks SDK,
JSONL, CSV, pytest, mypy, Ruff.

## Global constraints

- Invoke each policy exactly once per decision.
- Preserve all existing action validation and fault fallback behavior.
- Use explicit allowlists; never serialize an SDK context or metadata with
  `asdict`.
- Exclude raw hands, snapshots, engines, neural observations, belief states,
  request IDs, deadlines, and arbitrary metadata.
- Explanations are typed closed schemas and contain only finite values.
- Tracing is opt-in for tournaments and unavailable to held-out promotion.
- Stable sort keys, sorted JSON keys, fixed separators, finite JSON, terminal
  newlines, and transactional output are required.
- Implement observable behavior with failing tests first.

---

### Task 1: Define strict decision-trace types

**Files:**

- Create `src/garboid_pocketrocks/diagnostics/__init__.py`
- Create `src/garboid_pocketrocks/diagnostics/trace.py`
- Create `tests/diagnostics/__init__.py`
- Create `tests/diagnostics/test_trace.py`

**Work:**

- Define immutable `PublicDecisionContext`, `RecordedAction`,
  `HeuristicBidExplanation`, `NeuralPolicyExplanation`,
  `ExplainedBotDecision`, `PendingDecisionTrace`, `PublicDecisionOutcome`,
  and `DecisionTrace`.
- Build the public context with an explicit field-by-field allowlist.
- Enumerate legal bid and reveal actions in canonical order and normalize a
  zero bid to pass.
- Encode public-history events through their existing typed public schema.
- Reject unknown fields, nonfinite explanation values, and selected actions
  outside the legal set.
- Prove request IDs, deadlines, metadata, and raw private hands cannot change
  the recorded public context or enter decoded trace schemas.

**RED/GREEN command:**

```bash
uv run pytest -n 0 tests/diagnostics/test_trace.py -q
```

---

### Task 2: Capture the chosen action and optional explanation once

**Files:**

- Modify `src/garboid_pocketrocks/simulator/bot_execution.py`
- Modify `src/garboid_pocketrocks/bots/heuristic.py`
- Modify `src/garboid_pocketrocks/neural/tournament_bot.py`
- Modify focused bot and neural tests

**Work:**

- Add an optional explanation-aware protocol without changing the ordinary
  `BotBrain` contract.
- Add `execute_brain_decision`, returning a `DecisionExecution` with decision,
  explanation, and `policy` or `fault_fallback` source.
- Keep `choose_brain_decision` as a compatibility wrapper.
- Reuse the heuristic bid evaluation and neural masked inference output that
  already select the action; never evaluate either policy twice.
- Ensure explanations agree with the returned action and contain only their
  closed, finite fields.
- Prove ordinary brains and fault fallbacks retain identical decisions.

**RED/GREEN commands:**

```bash
uv run pytest -n 0 tests/simulator/test_bot_execution.py \
  tests/bots/test_heuristic_bots.py -q
uv run --extra neural pytest -n 0 tests/neural/test_smoke_tournament_bot.py -q
```

---

### Task 3: Collect equivalent scalar and batch traces

**Files:**

- Modify `src/garboid_pocketrocks/simulator/runner.py`
- Modify `src/garboid_pocketrocks/simulator/batch_match.py`
- Modify `src/garboid_pocketrocks/simulator/monte_carlo.py`
- Modify simulator tests

**Work:**

- Add an opt-in `capture_decision_traces` flag through match, game-job, and
  Monte Carlo configuration.
- Build pending traces immediately around `execute_brain_decision`.
- Finalize each trace only after the public session result exists.
- Add ordered traces to `MatchResult` and `MonteCarloResult`.
- Preserve batch execution when traces are enabled.
- Prove tracing on/off yields identical replay decisions, turns, faults,
  summaries, and bot statistics.
- Prove scalar and batch paths, different batch sizes, and one/multiple
  workers produce identical traces for fixed jobs.

**RED/GREEN commands:**

```bash
uv run pytest -n 0 tests/simulator/test_runner.py \
  tests/simulator/test_batch_match.py \
  tests/simulator/test_monte_carlo.py -q
```

---

### Task 4: Build slices and reconcile every total

**Files:**

- Create `src/garboid_pocketrocks/diagnostics/analysis.py`
- Create `tests/diagnostics/test_analysis.py`

**Work:**

- Define immutable `DecisionSlice`, `DecisionReconciliation`, and
  `DecisionReport`.
- Implement documented phase, cash-horizon, objective-state, seat, action,
  chart, player-count, and opponent-composition dimensions.
- Store additive counts and sums rather than averages.
- Validate unique trace keys, game identity, decision counts, outcomes, bot
  statistics, tournament rows, and condition statistics.
- Fail with clear messages before rendering when any source disagrees.

**RED/GREEN command:**

```bash
uv run pytest -n 0 tests/diagnostics/test_analysis.py -q
```

---

### Task 5: Write deterministic diagnostic artifacts safely

**Files:**

- Create `src/garboid_pocketrocks/diagnostics/reporting.py`
- Create `tests/diagnostics/test_reporting.py`
- Modify `src/garboid_pocketrocks/tournament/reporting.py`
- Modify tournament reporting tests

**Work:**

- Render canonical `game-summaries.jsonl`, `decision-traces.jsonl`, and
  `decision-slices.csv`.
- Extend `summary.json` with artifact names and reconciliation totals.
- Extend `report.html` with a short diagnostics status and artifact links.
- Render all known files before replacement and use the promotion writer's
  staged-generation rollback pattern.
- Preserve unrelated output files and restore the previous generation after
  a replacement failure.
- Prove reversed input ordering is byte-identical and all JSON rejects
  nonfinite values.

**RED/GREEN commands:**

```bash
uv run pytest -n 0 tests/diagnostics/test_reporting.py \
  tests/tournament/test_reporting.py -q
```

---

### Task 6: Integrate the opt-in tournament workflow

**Files:**

- Modify `src/garboid_pocketrocks/tournament/schedule.py`
- Modify `src/garboid_pocketrocks/tournament/runner.py`
- Modify `src/garboid_pocketrocks/tournament/cli.py`
- Modify tournament runner and CLI tests
- Update tournament documentation

**Work:**

- Add `decision_reports: bool = False` to `TournamentConfig` and
  `--decision-reports` to `garboid-tournament`.
- Thread the flag through planning and simulation without disabling batches.
- Build and validate the diagnostic report before writing artifacts.
- Print the diagnostic artifact paths when enabled.
- Leave default output and promotion behavior unchanged.
- Prove fixed tournaments have identical ordinary jobs, decisions, summaries,
  ratings, analysis, and bootstrap results with the flag on or off.

**RED/GREEN commands:**

```bash
uv run pytest -n 0 tests/tournament/test_runner.py \
  tests/tournament/test_cli.py -q
```

---

### Task 7: Full verification and draft PR

- Run focused diagnostics, simulator, bot, tournament, promotion, and neural
  tests.
- Run the full default and neural test suites.
- Run Ruff format check, Ruff lint, core mypy, strict neural mypy,
  lock-file validation, CLI help, and `git diff --check`.
- Review the complete branch diff for privacy, behavior preservation,
  deterministic output, naming, and issue acceptance criteria.
- Commit, push, and create a draft PR stacked on
  `codex/issue-7-neural-baseline` with `Closes #10`.

