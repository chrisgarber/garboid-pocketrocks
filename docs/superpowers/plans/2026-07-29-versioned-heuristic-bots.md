# Versioned Heuristic Bots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use braze-superpowers:subagent-driven-development (recommended) or braze-superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve v1 and v2 heuristic generations as reproducible simulation opponents while keeping unversioned names and live bots on the latest generation, and add a repository skill that asks before substantial bot behavior changes.

**Architecture:** Store each generation in a frozen `HeuristicProfileSet` and use the shared evaluator for both; v1 is exactly reproduced by a zero future-cash weight. Provide explicit versioned brains, wrappers, and specs for simulation, while existing unversioned APIs subclass the latest generation and retain their live identities. A repository-local skill governs future behavior-changing bot work.

**Tech Stack:** Python 3.14, frozen dataclasses, pytest, deterministic multiprocessing simulator, repository-local Codex skills

## Global Constraints

- Released v1 and v2 coefficients, names, and stable IDs are immutable.
- `aggressive`, `balanced`, and `passive` remain latest aliases.
- Existing unversioned live bot IDs do not change.
- The live launcher starts only random and the three latest unversioned bots.
- The simulation CLI accepts all six explicit version names.
- Neural training continues to consume the unversioned latest specs.
- The evaluator, simulator, game engine, and SDK adapters are not duplicated.
- All observable changes use red-green TDD.

---

### Task 1: Freeze heuristic profile generations

**Files:**
- Modify: `tests/heuristics/test_profiles.py`
- Modify: `src/garboid_pocketrocks/heuristics/profiles.py`

**Interfaces:**
- Produces: `HeuristicProfileSet(version, aggressive, balanced, passive)`
- Produces: `HEURISTIC_V1`, `HEURISTIC_V2`, `LATEST_HEURISTICS`
- Preserves: `AGGRESSIVE_PROFILE`, `BALANCED_PROFILE`, `PASSIVE_PROFILE`

- [ ] **Step 1: Write failing generation tests**

Add tests that import the new profile-set API and assert:

```python
assert HEURISTIC_V1.version == "v1"
assert HEURISTIC_V1.aggressive == HeuristicProfile("aggressive", 0.75, 0.0, 0.25, 0.05)
assert HEURISTIC_V1.balanced == HeuristicProfile("balanced", 0.40, 0.0, 0.20, 0.25)
assert HEURISTIC_V1.passive == HeuristicProfile("passive", 0.15, 0.0, 0.15, 0.50)

assert HEURISTIC_V2.version == "v2"
assert HEURISTIC_V2.aggressive == HeuristicProfile("aggressive", 0.75, 1.50, 0.25, 0.05)
assert HEURISTIC_V2.balanced == HeuristicProfile("balanced", 0.40, 0.75, 0.20, 0.25)
assert HEURISTIC_V2.passive == HeuristicProfile("passive", 0.15, 0.60, 0.15, 0.30)

assert LATEST_HEURISTICS is HEURISTIC_V2
assert AGGRESSIVE_PROFILE is HEURISTIC_V2.aggressive
assert BALANCED_PROFILE is HEURISTIC_V2.balanced
assert PASSIVE_PROFILE is HEURISTIC_V2.passive
```

Also require invalid versions and mismatched personality names to raise
`ValueError`.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
PYTHONPATH=src .venv/bin/pytest tests/heuristics/test_profiles.py -q
```

Expected: import failure because `HeuristicProfileSet` and the generation
constants do not exist.

- [ ] **Step 3: Implement the frozen profile-set contract**

Add:

```python
@dataclass(frozen=True, slots=True)
class HeuristicProfileSet:
    version: str
    aggressive: HeuristicProfile
    balanced: HeuristicProfile
    passive: HeuristicProfile

    def __post_init__(self) -> None:
        if (
            len(self.version) < 2
            or self.version[0] != "v"
            or not self.version[1:].isdigit()
            or int(self.version[1:]) < 1
        ):
            raise ValueError("heuristic version must use canonical vN form")
        names = (
            self.aggressive.name,
            self.balanced.name,
            self.passive.name,
        )
        if names != ("aggressive", "balanced", "passive"):
            raise ValueError("profile set must contain canonical personalities")
```

Construct `HEURISTIC_V1` and `HEURISTIC_V2` with the pinned coefficients, set
`LATEST_HEURISTICS = HEURISTIC_V2`, and point the three legacy profile
constants at the corresponding latest members.

- [ ] **Step 4: Run profile and valuation tests and verify GREEN**

Run:

```bash
PYTHONPATH=src .venv/bin/pytest \
  tests/heuristics/test_profiles.py \
  tests/heuristics/test_valuation.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the profile generations**

```bash
git add src/garboid_pocketrocks/heuristics/profiles.py \
  tests/heuristics/test_profiles.py
git commit -m "feat: freeze heuristic profile generations"
```

### Task 2: Add versioned brains, wrappers, and bot specs

**Files:**
- Modify: `tests/bots/test_heuristic_bots.py`
- Modify: `src/garboid_pocketrocks/bots/heuristic.py`
- Modify: `src/garboid_pocketrocks/bots/__init__.py`

**Interfaces:**
- Consumes: `HEURISTIC_V1`, `HEURISTIC_V2`
- Produces: `AggressiveHeuristicV1Brain` through `PassiveHeuristicV2Brain`
- Produces: `AggressiveHeuristicV1Bot` through `PassiveHeuristicV2Bot`
- Produces: six `*_HEURISTIC_VN_BOT_SPEC` constants
- Preserves: all existing unversioned brain, bot, and spec exports

- [ ] **Step 1: Write failing versioned-bot tests**

Add parametrized tests requiring:

```python
(
    AggressiveHeuristicV1Bot.BOT_NAME,
    BalancedHeuristicV1Bot.BOT_NAME,
    PassiveHeuristicV1Bot.BOT_NAME,
) == ("aggressive-v1", "balanced-v1", "passive-v1")

(
    AggressiveHeuristicV2Bot.BOT_NAME,
    BalancedHeuristicV2Bot.BOT_NAME,
    PassiveHeuristicV2Bot.BOT_NAME,
) == ("aggressive-v2", "balanced-v2", "passive-v2")
```

Pin the six UUID-like simulation IDs. Verify each built brain's
`valuator.profile` is the matching member of `HEURISTIC_V1` or
`HEURISTIC_V2`. For identical contexts, require each unversioned bot decision
to equal its v2 bot decision. Pickle every explicit `BotSpec`.

- [ ] **Step 2: Run the bot tests and verify RED**

Run:

```bash
PYTHONPATH=src .venv/bin/pytest tests/bots/test_heuristic_bots.py -q
```

Expected: import failures for the versioned classes and specs.

- [ ] **Step 3: Implement explicit versioned bot APIs**

Add six profile-specific brain classes. Add an internal wrapper base:

```python
class _VersionedHeuristicBot(PocketRocksFastBot):
    BRAIN_CLASS: ClassVar[type[HeuristicBotBrain]]

    @classmethod
    def build_brain(cls, seed: int | None) -> HeuristicBotBrain:
        del seed
        return cls.BRAIN_CLASS()
```

Define six explicit versioned wrappers with these stable IDs:

```text
aggressive-v1 bot_10000000-0000-4000-8000-000000000001
balanced-v1   bot_10000000-0000-4000-8000-000000000002
passive-v1    bot_10000000-0000-4000-8000-000000000003
aggressive-v2 bot_20000000-0000-4000-8000-000000000001
balanced-v2   bot_20000000-0000-4000-8000-000000000002
passive-v2    bot_20000000-0000-4000-8000-000000000003
```

Keep the existing unversioned wrappers and IDs, but make their brain classes
subclass the corresponding v2 brain classes. Export all new public symbols
from `garboid_pocketrocks.bots`.

- [ ] **Step 4: Run bot and integration tests and verify GREEN**

Run:

```bash
PYTHONPATH=src .venv/bin/pytest \
  tests/bots/test_heuristic_bots.py \
  tests/test_integration.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the versioned bot APIs**

```bash
git add src/garboid_pocketrocks/bots/heuristic.py \
  src/garboid_pocketrocks/bots/__init__.py \
  tests/bots/test_heuristic_bots.py
git commit -m "feat: expose versioned heuristic bots"
```

### Task 3: Expose versions in simulation, not the live launcher

**Files:**
- Modify: `tests/simulator/test_cli.py`
- Modify: `tests/bots/test_launcher.py`
- Modify: `src/garboid_pocketrocks/simulator/cli.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: six explicit versioned `BotSpec` constants
- Preserves: `bots.launcher.BOT_REGISTRY` latest-only order
- Extends: simulator `_BOT_REGISTRY` with `*-v1` and `*-v2`

- [ ] **Step 1: Write failing registry and simulation tests**

Add a parser test that calls `_bot_names` and requires:

```python
assert _bot_names(
    "aggressive-v1,balanced-v1,passive-v1,"
    "aggressive-v2,balanced-v2,passive-v2"
) == (
    "aggressive-v1",
    "balanced-v1",
    "passive-v1",
    "aggressive-v2",
    "balanced-v2",
    "passive-v2",
)
```

Keep the live launcher assertion exactly:

```python
assert tuple(BOT_REGISTRY) == ("random", "aggressive", "balanced", "passive")
```

Add a six-game, three-player mixed-generation CLI simulation with
`balanced-v1,balanced-v2,passive-v2`, two workers, and assert all fault counts
are zero and result names remain versioned.

- [ ] **Step 2: Run CLI and launcher tests and verify RED**

Run:

```bash
PYTHONPATH=src .venv/bin/pytest \
  tests/simulator/test_cli.py \
  tests/bots/test_launcher.py -q
```

Expected: `_bot_names` rejects versioned names.

- [ ] **Step 3: Register explicit versions in the simulator**

Import the six versioned specs into `simulator/cli.py`, add each by its
`spec.name`, and generate help text from the registry keys. Do not modify
`bots/launcher.py`.

Document:

- unversioned names track latest;
- explicit version names are reproducible;
- the live launcher remains latest-only;
- an example mixed-generation simulation command.

- [ ] **Step 4: Run CLI and launcher tests and verify GREEN**

Run the command from Step 2.

Expected: PASS, including the two-worker mixed-generation simulation.

- [ ] **Step 5: Commit simulation version selection**

```bash
git add src/garboid_pocketrocks/simulator/cli.py \
  tests/simulator/test_cli.py tests/bots/test_launcher.py README.md
git commit -m "feat: select heuristic generations in simulations"
```

### Task 4: Add the repository bot-versioning skill

**Files:**
- Create: `.agents/skills/versioning-bots/SKILL.md`
- Create: `.agents/skills/versioning-bots/agents/openai.yaml`

**Interfaces:**
- Trigger: substantial bot decision or strength changes
- Decision: new version versus deliberate in-place update
- Exemption: behavior-preserving changes

- [ ] **Step 1: Run RED pressure scenarios without the skill**

Use three fresh, read-only agents with no access to a bot-versioning skill:

1. "Raise passive objective-progress weight from 0.15 to 0.20; this is small,
   please implement immediately."
2. "Fix a bidding bug that materially changes legal bids; we are late, patch
   the current bot now."
3. "Rename a private helper without changing any decisions."

Ask each to state its next action without editing files. Record whether it
asks about preserving the current bot version. Expected RED: at least one of
the first two proceeds without asking.

- [ ] **Step 2: Initialize the repository skill**

Run:

```bash
/Users/Christopher.Garber/.codex/skills/.system/skill-creator/scripts/init_skill.py \
  versioning-bots \
  --path .agents/skills \
  --interface 'display_name=Versioning Bots' \
  --interface 'short_description=Preserve bot generations as behavior evolves' \
  --interface 'default_prompt=Use $versioning-bots when changing a bot strategy or strength.'
```

No scripts, references, examples, or assets are needed.

- [ ] **Step 3: Write the minimal skill**

Use frontmatter:

```yaml
---
name: versioning-bots
description: Use when changing bot strategy, decisions, coefficients, training, checkpoints, information inputs, action selection, or fixing a bug that materially changes bot behavior or strength
---
```

The body must require this sequence before implementation:

1. Identify the affected bot and behavior change.
2. Ask the user whether to create a new version or update the current version
   in place.
3. If new: freeze the current version, add explicit names, advance latest
   aliases, and add reproducibility tests and a benchmark record.
4. If in place: record the choice and deliberately update pinned tests.
5. Skip the question only when decisions and strength are demonstrably
   unchanged.

Include a compact quick-reference table and explicit red flags for pressures
such as "small change," "bug fix," or "implement immediately."

- [ ] **Step 4: Validate and GREEN-test the skill**

Run:

```bash
/Users/Christopher.Garber/.codex/skills/.system/skill-creator/scripts/quick_validate.py \
  .agents/skills/versioning-bots
```

Then repeat the three read-only scenarios with fresh agents explicitly using
the repository skill. Expected:

- scenarios 1 and 2 ask the versioning question before implementation;
- scenario 3 proceeds without asking.

If a pressure scenario exposes a loophole, patch only the necessary wording
and rerun that scenario.

- [ ] **Step 5: Commit the repository skill**

```bash
git add .agents/skills/versioning-bots
git commit -m "docs: require bot versioning decisions"
```

### Task 5: Verify historical reproducibility and the complete repository

**Files:**
- Modify only if verification exposes a defect in the files above

**Interfaces:**
- Consumes: completed v1/v2 implementation and repository skill
- Produces: verified branch ready for integration

- [ ] **Step 1: Run focused deterministic simulations**

Run the same fixed seed twice for v1 and v2 lineups and require exact JSON
equality per generation:

```bash
PYTHONPATH=src .venv/bin/python -m garboid_pocketrocks.simulator.cli \
  --bots aggressive-v1,balanced-v1,passive-v1 \
  --games 100 --players 3 --seed 20260729 --workers 2 --format json

PYTHONPATH=src .venv/bin/python -m garboid_pocketrocks.simulator.cli \
  --bots aggressive-v2,balanced-v2,passive-v2 \
  --games 100 --players 3 --seed 20260729 --workers 2 --format json
```

Confirm v1 and v2 produce distinct results and zero faults.

- [ ] **Step 2: Run static verification**

```bash
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
PYTHONPATH=src .venv/bin/mypy src tests
git diff --check
```

Expected: all commands exit zero.

- [ ] **Step 3: Run the complete test suite**

```bash
PYTHONPATH=src .venv/bin/pytest -q
```

Expected: all tests pass; only documented third-party warnings may remain.

- [ ] **Step 4: Review branch scope**

Run:

```bash
git status --short
git diff --stat main...HEAD
git log --oneline main..HEAD
```

Confirm only heuristic versioning, the repository skill, tests, README,
design, plan, and prior future-cash implementation are present.

- [ ] **Step 5: Commit any verification-only corrections**

If verification required changes, stage only the affected versioning files
and commit:

```bash
git commit -m "fix: complete heuristic versioning verification"
```
