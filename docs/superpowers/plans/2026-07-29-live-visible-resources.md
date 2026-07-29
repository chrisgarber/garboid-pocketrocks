# Live Visible Resources Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use braze-superpowers:subagent-driven-development (recommended) or braze-superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make live heuristic bots interpret visible resource cards according to the active action instead of passing on financial actions.

**Architecture:** Add one action-aware offered-resource helper in the belief module and reuse it from validation, belief construction, and valuation. Preserve the SDK context and the existing loan and investment economics.

**Tech Stack:** Python 3.14, PocketRocks SDK, pytest

## Global Constraints

- Auction 1 offers only the first visible resource.
- Auction 2 offers up to both visible resources.
- Financial actions offer no resources even when the board contains visible cards.
- Do not change loan or investment economics.
- Keep the opportunity-aware loan/endgame heuristic deferred.

---

### Task 1: Reproduce live offered-lot behavior

**Files:**
- Modify: `tests/heuristics/test_belief.py`
- Modify: `tests/heuristics/test_valuation.py`

**Interfaces:**
- Consumes: `build_belief(context, knowledge)` and `HeuristicValuator.evaluate_bid(context, knowledge)`
- Produces: Regression coverage for live financial and Auction 1 contexts

- [ ] **Step 1: Add failing regression tests**

Add a belief test asserting an Invest 10 context with `(2, 4)` is accepted
without removing either visible card from the future pool. Add Auction 1 tests
asserting `(3, 2)` removes and values only suit 3.

- [ ] **Step 2: Verify the tests fail for the diagnosed reason**

Run:

```bash
uv run pytest tests/heuristics/test_belief.py tests/heuristics/test_valuation.py -q
```

Expected: the investment test fails with `offered resource is invalid for a
financial action`, and the Auction 1 test fails with `offered resource count
exceeds auction capacity`.

### Task 2: Share action-aware offered-resource counts

**Files:**
- Modify: `src/garboid_pocketrocks/heuristics/belief.py`
- Modify: `src/garboid_pocketrocks/heuristics/valuation.py`
- Test: `tests/heuristics/test_belief.py`
- Test: `tests/heuristics/test_valuation.py`

**Interfaces:**
- Produces: `offered_resource_counts(context: DecisionContext, action: ActionId) -> tuple[int, ...]`
- Consumes: `DecisionContext.current_resource_ids` as the visible two-card board

- [ ] **Step 1: Implement the minimal shared helper**

Return zero counts for financial actions, count only the first visible card for
Auction 1, and count both nonzero visible cards for Auction 2.

- [ ] **Step 2: Update validation and consumers**

Use the helper to validate auction availability and to construct beliefs.
Import and use the same helper from `valuation.py`; remove its duplicated
offered-count implementation.

- [ ] **Step 3: Verify focused tests**

Run:

```bash
uv run pytest tests/heuristics/test_belief.py tests/heuristics/test_valuation.py tests/bots/test_heuristic_bots.py -q
```

Expected: all selected tests pass.

- [ ] **Step 4: Verify the full project**

Run the repository's test, formatting, lint, and type-check commands. Expected:
all commands exit successfully without modifying unrelated files.
