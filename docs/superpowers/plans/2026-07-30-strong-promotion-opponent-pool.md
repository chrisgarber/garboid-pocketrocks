# Strong Promotion Opponent Pool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use braze-superpowers:subagent-driven-development (recommended) or braze-superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace random promotion opposition with every frozen v1/v2 heuristic and the released 350k PPO bot, while deterministically excluding exact candidate/incumbent identities and recording the resulting effective plan.

**Architecture:** Keep committed corpora as candidate-independent recipes and seed schedules. Resolve and filter their eligible opponent pools during planning, build exact effective paired cases, hash the canonical effective plan, and record the pool and plan in the authoritative promotion report.

**Tech Stack:** Python 3.14, frozen dataclasses, canonical JSON/SHA-256, pytest, Ruff, mypy.

## Global Constraints

- Update `development-v1` and `held-out-v1` in place because this PR introduces them.
- The ordered eligible pool is all aggressive/balanced/passive v1 and v2 heuristics plus `vector_ppo_large_v1_g350k`; remove `random`.
- Exclude only complete `(name, bot_id)` matches for candidate/incumbent; partial matches fail closed.
- Preserve case IDs, ordering, charts, player counts, focal seats, repetitions, and engine seeds.
- Both twins use exactly the same effective non-focal lineup.
- Keep source corpus snapshots candidate-independent; put effective-pool and plan evidence in `promotion-report.json`.
- Implement every behavior change with a failing test first.

---

### Task 1: Update the committed eligible pools

**Files:**
- Modify: `configs/promotion/development-v1.json`
- Modify: `configs/promotion/held-out-v1.json`
- Modify: `tests/promotion/test_corpus.py`

**Interfaces:**
- Produces: both committed recipes with the same seven-name ordered pool.
- Consumes: existing `load_promotion_corpus()` validation and digest generation.

- [ ] Add a packaging assertion that both recipes expose exactly:

```python
expected = (
    "aggressive-v1",
    "balanced-v1",
    "passive-v1",
    "aggressive-v2",
    "balanced-v2",
    "passive-v2",
    "vector_ppo_large_v1_g350k",
)
assert development.recipe.opponent_names == expected
assert held_out.recipe.opponent_names == expected
```

- [ ] Run `uv run pytest -n 0 tests/promotion/test_corpus.py -q` and confirm the new assertion fails because `random` is still configured.
- [ ] Replace each JSON recipe's `opponent_names` with the approved ordered pool.
- [ ] Re-run the corpus test and confirm it passes.

### Task 2: Build a deterministic effective opponent pool and plan

**Files:**
- Modify: `src/garboid_pocketrocks/promotion/planning.py`
- Modify: `tests/promotion/test_planning.py`
- Modify: `tests/promotion/helpers.py`

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True, slots=True)
class OpponentExclusion:
    opponent: BotSpec
    reason: Literal["candidate", "incumbent"]


@dataclass(frozen=True, slots=True)
class EffectiveOpponentPool:
    configured: tuple[BotSpec, ...]
    exclusions: tuple[OpponentExclusion, ...]
    remaining: tuple[BotSpec, ...]


class PromotionPlanningError(ValueError):
    code: str
    opponent_pool: EffectiveOpponentPool | None


@dataclass(frozen=True, slots=True)
class PromotionPlan:
    ...
    opponent_pool: EffectiveOpponentPool
    digest: str


def promotion_plan_payload(plan: PromotionPlan) -> dict[str, object]: ...
```

- [ ] Add failing planning tests for exact candidate and incumbent exclusion, partial name/ID collision rejection, pinned filtered rotations, insufficient remaining capacity, twin equality, and a stable 64-character plan digest.
- [ ] Run `uv run pytest -n 0 tests/promotion/test_planning.py -q` and confirm failures identify the missing effective-pool behavior.
- [ ] Resolve all configured identities first. Exclude only exact compared identities; retain existing collision failures for partial matches.
- [ ] Rebuild each effective case with:

```python
rotation = (repetition + chart_index + case.player_count + case.focal_seat) % len(pool.remaining)
selected = rotated(pool.remaining, rotation)[: case.player_count - 1]
```

Preserve the source case metadata and replace only `opponent_names_by_seat`.
- [ ] Canonically serialize the source corpus identity, compared identities, configured/excluded/remaining pools, and complete twin lineups; compute `PromotionPlan.digest` from compact sorted JSON with a terminal newline.
- [ ] Re-run the focused planning tests and confirm they pass.

### Task 3: Record effective evidence and failed-pool context

**Files:**
- Modify: `src/garboid_pocketrocks/promotion/reporting.py`
- Modify: `src/garboid_pocketrocks/promotion/runner.py`
- Modify: `tests/promotion/test_reporting.py`
- Modify: `tests/promotion/test_runner.py`

**Interfaces:**
- Consumes: `EffectiveOpponentPool`, `PromotionPlan.digest`, and `promotion_plan_payload()`.
- Produces: `PromotionReport.opponent_pool`, `PromotionReport.plan`, and JSON fields `opponent_pool` and `effective_plan`.

- [ ] Add failing report tests asserting configured identities, exclusion reasons, remaining identities, exact effective plan payload/digest, and `null` plan evidence on pre-plan failures.
- [ ] Add a failing runner test where filtering leaves too few opponents and assert the failed report retains configured/excluded/remaining pool evidence with `insufficient_eligible_opponents`.
- [ ] Run `uv run pytest -n 0 tests/promotion/test_reporting.py tests/promotion/test_runner.py -q` and confirm the assertions fail for missing fields.
- [ ] Extend `PromotionReport` and `build_promotion_report()` with structured pool and optional plan values.
- [ ] Have `PromotionPlanningError` carry a successfully resolved pool on capacity failure; preserve it in `PromotionRunner.run()` and pass it to reporting even when `plan` remains `None`.
- [ ] Serialize the effective pool and plan in `promotion-report.json`. Keep `corpus-snapshot.json` unchanged and candidate-independent.
- [ ] Re-run the reporting and runner tests and confirm they pass.

### Task 4: Update operator documentation and verify the PR

**Files:**
- Modify: `src/garboid_pocketrocks/promotion/README.md`
- Modify: `README.md`
- Modify: `tests/promotion/test_integration.py`

**Interfaces:**
- Consumes: the completed eligible-pool planning and report schema.
- Produces: documented behavior and end-to-end coverage for an incumbent present in the configured pool.

- [ ] Add an integration test whose incumbent is a configured opponent; assert it is absent from every non-focal seat, both twins remain matched, and repeated plans have the same digest.
- [ ] Run the integration test and confirm it fails before the completed planner/report path is wired through.
- [ ] Update the runbooks: held-out means unseen cases/seeds, configured names are an eligible pool, exact compared identities are filtered, and reports identify matchup-specific effective plans.
- [ ] Run the focused promotion suite:

```bash
uv run pytest -n 0 tests/promotion -q
```

- [ ] Run repository quality checks:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run --extra neural mypy --config-file pyproject-neural.toml
git diff --check
```

- [ ] Run the full test suite with the neural extra:

```bash
uv run --extra neural pytest -q
```

- [ ] Commit the implementation, ensure PR #23 is a draft, and push `codex/issue-9-promotion-gate`.
