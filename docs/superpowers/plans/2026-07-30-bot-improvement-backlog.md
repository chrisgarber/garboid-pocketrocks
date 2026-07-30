# Bot Improvement Backlog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use braze-superpowers:subagent-driven-development (recommended) or braze-superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the approved bot-improvement portfolio into eight new GitHub
issues, a structured extension to existing issue #7, and a verified closing
summary on issue #6.

**Architecture:** Treat the approved design document as the content authority,
but stage every issue body in local temporary Markdown so it can be inspected
before it changes GitHub. Create the eight new issues first, read back their
assigned numbers, then replace their dependency descriptions with real links,
extend #7 without removing its original request, and close #6 only after a
complete external-state audit.

**Tech Stack:** Markdown, GitHub CLI (`gh`), `jq`, Git

## Global Constraints

- Every issue begins with `## What this solves` and `## Why it matters` in
  simple, human-readable language.
- Explain technical terms the first time they appear; call project 1 a "fair
  final exam" before calling it a held-out promotion gate.
- The primary objective is held-out tournament strength.
- Existing heuristic generations and neural checkpoints remain immutable.
- Behavior-changing projects use new explicit local identities and compare
  against their immediate predecessor.
- Candidate tuning uses development games only; final evidence uses untouched
  held-out games.
- Promotion requires the held-out 95% bootstrap interval for rating delta to
  be above zero, with no illegal actions or bot faults.
- Create exactly eight new issues for ranked projects 1, 2, 3, 5, 6, 7, 8,
  and 9.
- Reuse issue #7 for ranked project 4; do not create a duplicate neural
  baseline issue.
- Do not modify unrelated issue #8.
- Preserve issue #7's original problem statement when extending its body.
- Close issue #6 only after every new/updated issue passes read-back
  verification.

---

### Task 1: Establish the immutable preflight snapshot

**Files:**
- Read: `docs/superpowers/specs/2026-07-30-bot-improvement-portfolio-design.md`
- Read: `docs/superpowers/plans/2026-07-30-bot-improvement-backlog.md`

**Interfaces:**
- Consumes: approved portfolio specification at commit `c2664b2`
- Produces: authoritative pre-mutation issue snapshot for #6, #7, and #8

- [ ] **Step 1: Confirm the local source documents and clean state**

Run:

```bash
git status --short --branch
git log -2 --oneline
rg -n "^### [1-9]\\.|^## GitHub issue writing standard|^## Backlog delivery" \
  docs/superpowers/specs/2026-07-30-bot-improvement-portfolio-design.md
```

Expected: detached `HEAD`, no uncommitted files except this plan if it has not
yet been committed, and exactly nine numbered project headings.

- [ ] **Step 2: Verify GitHub authentication**

Run:

```bash
gh auth status
```

Expected: authenticated to `github.com` with permission to write issues in
`chrisgarber/garboid-pocketrocks`.

- [ ] **Step 3: Capture the current issue snapshot**

Run:

```bash
gh issue list \
  --repo chrisgarber/garboid-pocketrocks \
  --state all \
  --limit 100 \
  --json number,title,state,url
gh issue view 6 \
  --repo chrisgarber/garboid-pocketrocks \
  --json number,title,body,state,comments,url
gh issue view 7 \
  --repo chrisgarber/garboid-pocketrocks \
  --json number,title,body,state,comments,url
gh issue view 8 \
  --repo chrisgarber/garboid-pocketrocks \
  --json number,title,body,state,comments,url
```

Expected: #6, #7, and #8 are open; there are no existing issues with any of
the eight titles in Task 2; #7 contains its original RL-harness request; #8 is
recorded for later byte-for-byte comparison.

### Task 2: Stage and inspect all issue bodies

**Files:**
- Create temporarily: `/private/tmp/garboid-pocketrocks-issue-6-20260730/project-1.md`
- Create temporarily: `/private/tmp/garboid-pocketrocks-issue-6-20260730/project-2.md`
- Create temporarily: `/private/tmp/garboid-pocketrocks-issue-6-20260730/project-3.md`
- Create temporarily: `/private/tmp/garboid-pocketrocks-issue-6-20260730/project-5.md`
- Create temporarily: `/private/tmp/garboid-pocketrocks-issue-6-20260730/project-6.md`
- Create temporarily: `/private/tmp/garboid-pocketrocks-issue-6-20260730/project-7.md`
- Create temporarily: `/private/tmp/garboid-pocketrocks-issue-6-20260730/project-8.md`
- Create temporarily: `/private/tmp/garboid-pocketrocks-issue-6-20260730/project-9.md`
- Create temporarily: `/private/tmp/garboid-pocketrocks-issue-6-20260730/issue-7.md`

**Interfaces:**
- Consumes: project scope, RICE values, dependencies, and common rules from
  the approved specification
- Produces: nine complete Markdown bodies ready for GitHub

- [ ] **Step 1: Create one new empty temporary directory**

First verify that the fixed task-local path does not exist:

```bash
test ! -e /private/tmp/garboid-pocketrocks-issue-6-20260730
mkdir /private/tmp/garboid-pocketrocks-issue-6-20260730
```

Expected: both commands succeed. If the path already exists, stop and inspect
it rather than deleting or overwriting it.

- [ ] **Step 2: Write the eight new issue bodies**

Use `apply_patch` to create one file per project. Directly beneath the first
three headings, copy the complete matching paragraphs from the approved
project scope. Under the remaining headings, use the exact RICE table and
project-specific boundaries later in this task. Each body must use this exact
heading order:

```markdown
## What this solves

## Why it matters

## Proposed outcome

## RICE and rank

RICE is a planning score: reach × impact × confidence ÷ effort. Here, reach is
the number of bot experiments likely to benefit in six months, impact is the
expected tournament-strength gain, confidence reflects current evidence, and
effort is estimated engineer-weeks.

## Dependencies

Parent portfolio: #6.

## In scope

## Out of scope

## Acceptance criteria

- [ ] Charts A-E and three-, four-, and five-player games complete with no
      illegal actions or bot faults when this project changes behavior.

## Versioning and promotion

Released identities remain immutable. Any changed policy receives a new
explicit simulation identity, is tuned only on development games, and is
frozen before the held-out comparison. It advances a latest alias only when
the held-out 95% bootstrap interval for rating delta is above zero.
```

For projects 1 and 2, replace the final paragraph with: "This is
behavior-preserving infrastructure and does not create or advance a bot
identity. Its outputs support the common promotion rules used by later
projects."

For each issue, add three or more concrete checklist items under `Acceptance
criteria` by converting its project-specific boundaries below into observable
assertions. Do not use generic phrases such as "works correctly" or "add
tests."

Use these exact titles and RICE tuples:

| File | GitHub title | Rank | Reach | Impact | Confidence | Effort | Score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `project-1.md` | Create a fair final exam before promoting new bots | 1 | 10 | 1.0 | 95% | 1.5 | 6.33 |
| `project-2.md` | Add decision-by-decision reports for bot performance | 2 | 9 | 1.5 | 85% | 2.5 | 4.59 |
| `project-3.md` | Use evolutionary search to produce heuristic v3 candidates | 3 | 8 | 2.0 | 80% | 3.0 | 4.27 |
| `project-5.md` | Build phase-aware heuristic v4 candidates | 5 | 6 | 2.0 | 70% | 4.0 | 2.10 |
| `project-6.md` | Adapt bids to opponents using public game history | 6 | 6 | 2.0 | 65% | 4.5 | 1.73 |
| `project-7.md` | Bootstrap neural training from heuristic knowledge | 7 | 5 | 2.5 | 65% | 5.0 | 1.63 |
| `project-8.md` | Search late-game possibilities using public beliefs | 8 | 4 | 3.0 | 55% | 6.0 | 1.10 |
| `project-9.md` | Choose among promoted bot experts with a hybrid policy | 9 | 5 | 3.0 | 50% | 7.0 | 1.07 |

Project-specific boundaries:

- Project 1: create immutable development/final game corpora, paired
  candidate-incumbent comparisons, rating-delta intervals, and a
  machine-readable promotion report; do not change a bot.
- Project 2: record public context, legal actions, chosen action, policy
  explanation, and eventual public outcome; never expose hidden simulator
  state; do not change strategy.
- Project 3: search only existing coefficient families; preserve v1/v2; defer
  phase-specific coefficients to project 5.
- Project 5: use public remaining-game state to select early/middle/late
  experts; depend on projects 2 and 3; preserve v3.
- Project 6: model opponent bids from public history and cash; use a
  deterministic sparse-data prior; defer general expert selection to project
  9.
- Project 7: compare behavior cloning, heuristic value targets, and heuristic
  opponents through separate ablations; depend on projects 2 and 4/#7; do not
  promote based on training loss or shaped reward.
- Project 8: search only states compatible with public information and the
  Bayesian belief model; bound deterministic work and latency; depend on
  project 2.
- Project 9: select only among promoted experts using live-compatible
  information; depend on projects 3 and 4/#7; treat projects 5, 6, and 8 as
  optional expert providers.

- [ ] **Step 3: Extend a copy of issue #7 without replacing its original text**

Start `issue-7.md` with the exact current body captured in Task 1. Append:

```markdown

---

## Bot improvement portfolio extension

### What this solves

The repository can train and checkpoint neural policies, but it does not yet
expose a registered neural brain or demonstrate that a checkpoint is
competitive with the heuristic field.

### Why it matters

Without the training-to-tournament loop, neural progress is measured mainly by
training metrics rather than actual playing strength. Closing that loop tells
us whether faster training is producing a better bot.

### Proposed outcome

Complete a competitive neural baseline by adding a history-aware inference
adapter, immutable checkpoint identities, committed training runs, and
tournament evaluation. Use the fair final exam from ranked project 1 before
advancing any neural alias.

### RICE and rank

- **Rank:** 4
- **Reach:** 7
- **Impact:** 2.0
- **Confidence:** 80%
- **Effort:** 3.5 engineer-weeks
- **RICE score:** 3.20

RICE is a planning score: reach × impact × confidence ÷ effort.

### Dependencies

Ranked project 1, "Create a fair final exam before promoting new bots."
Parent portfolio: #6.

### Acceptance criteria

- [ ] A checkpoint can be loaded as a deterministic, history-aware bot brain.
- [ ] Every candidate has an immutable identity containing its model profile
      and completed-game count.
- [ ] The committed training profiles produce auditable checkpoints and raw
      training artifacts.
- [ ] Candidate and incumbent complete identical charts, player counts, seats,
      and opponent mixtures without illegal actions or bot faults.
- [ ] A machine-readable held-out promotion report records the rating delta,
      confidence interval, configuration, seeds, commit, and decision.

### Versioning and promotion

Existing checkpoints remain immutable. A candidate advances a latest neural
alias only when the held-out 95% bootstrap interval for rating delta is above
zero. Inconclusive checkpoints remain available under their explicit research
identity.
```

- [ ] **Step 4: Validate readability and completeness before external writes**

Run:

```bash
for body in /private/tmp/garboid-pocketrocks-issue-6-20260730/project-*.md; do
  rg -q '^## What this solves$' "$body"
  rg -q '^## Why it matters$' "$body"
  rg -q '^## Proposed outcome$' "$body"
  rg -q '^## RICE and rank$' "$body"
  rg -q '^## Dependencies$' "$body"
  rg -q '^## In scope$' "$body"
  rg -q '^## Out of scope$' "$body"
  rg -q '^## Acceptance criteria$' "$body"
  rg -q '^## Versioning and promotion$' "$body"
done
rg -n 'TBD|TODO|FIXME|PLACEHOLDER|\\[[^]]+\\]' \
  /private/tmp/garboid-pocketrocks-issue-6-20260730/*.md
```

Expected: every heading check succeeds. The final search may show Markdown
checklist markers and links, but must show no prose placeholders or unfinished
instructions. Read every file top-to-bottom before continuing.

### Task 3: Create the eight new issues

**Files:**
- Read: the eight staged `project-*.md` files from Task 2

**Interfaces:**
- Consumes: inspected issue bodies
- Produces: eight new GitHub issue URLs and title-to-number mapping

- [ ] **Step 1: Re-query titles immediately before creation**

Run:

```bash
gh issue list \
  --repo chrisgarber/garboid-pocketrocks \
  --state all \
  --limit 100 \
  --json number,title,state,url
```

Expected: none of the eight exact titles from Task 2 exists. If any title now
exists, stop and inspect it rather than creating a duplicate.

- [ ] **Step 2: Create each issue exactly once**

Run one command per row in Task 2:

```bash
gh issue create \
  --repo chrisgarber/garboid-pocketrocks \
  --title "Create a fair final exam before promoting new bots" \
  --body-file "/private/tmp/garboid-pocketrocks-issue-6-20260730/project-1.md"
```

Repeat with the exact remaining title/body pairs from the Task 2 table.
Expected: each command returns one distinct issue URL ending in a numeric issue
ID.
Record all eight outputs before continuing. Never rerun a create command after
an ambiguous timeout; query by exact title first.

- [ ] **Step 3: Prove exact cardinality and uniqueness**

Run:

```bash
gh issue list \
  --repo chrisgarber/garboid-pocketrocks \
  --state all \
  --limit 100 \
  --json number,title,state,url
```

Expected: each exact title from Task 2 appears once, producing eight distinct
numbers. #6, #7, and #8 remain open.

### Task 4: Link dependencies and extend issue #7

**Files:**
- Modify temporarily: the eight staged `project-*.md` files
- Modify temporarily: staged `issue-7.md`

**Interfaces:**
- Consumes: title-to-number mapping from Task 3
- Produces: nine GitHub issues with direct, correct dependency links

- [ ] **Step 1: Replace named dependencies with actual issue links**

Use `apply_patch` on the temporary Markdown files. Render every dependency as
a Markdown link whose label is the exact project title and whose target is the
actual issue URL from Task 3. Use the title-to-number mapping returned by
GitHub, not predicted issue numbers. Link every body to
`https://github.com/chrisgarber/garboid-pocketrocks/issues/6` as the parent.
Update `issue-7.md` so ranked project 1 is also a direct link.

- [ ] **Step 2: Push the final linked bodies**

Resolve each number from its exact title:

```bash
issue_number_for_title() {
  gh issue list \
    --repo chrisgarber/garboid-pocketrocks \
    --state all \
    --limit 100 \
    --json number,title \
    --jq ".[] | select(.title == \"$1\") | .number"
}
```

Then run all eight exact edits:

```bash
gh issue edit "$(issue_number_for_title 'Create a fair final exam before promoting new bots')" --repo chrisgarber/garboid-pocketrocks --body-file /private/tmp/garboid-pocketrocks-issue-6-20260730/project-1.md
gh issue edit "$(issue_number_for_title 'Add decision-by-decision reports for bot performance')" --repo chrisgarber/garboid-pocketrocks --body-file /private/tmp/garboid-pocketrocks-issue-6-20260730/project-2.md
gh issue edit "$(issue_number_for_title 'Use evolutionary search to produce heuristic v3 candidates')" --repo chrisgarber/garboid-pocketrocks --body-file /private/tmp/garboid-pocketrocks-issue-6-20260730/project-3.md
gh issue edit "$(issue_number_for_title 'Build phase-aware heuristic v4 candidates')" --repo chrisgarber/garboid-pocketrocks --body-file /private/tmp/garboid-pocketrocks-issue-6-20260730/project-5.md
gh issue edit "$(issue_number_for_title 'Adapt bids to opponents using public game history')" --repo chrisgarber/garboid-pocketrocks --body-file /private/tmp/garboid-pocketrocks-issue-6-20260730/project-6.md
gh issue edit "$(issue_number_for_title 'Bootstrap neural training from heuristic knowledge')" --repo chrisgarber/garboid-pocketrocks --body-file /private/tmp/garboid-pocketrocks-issue-6-20260730/project-7.md
gh issue edit "$(issue_number_for_title 'Search late-game possibilities using public beliefs')" --repo chrisgarber/garboid-pocketrocks --body-file /private/tmp/garboid-pocketrocks-issue-6-20260730/project-8.md
gh issue edit "$(issue_number_for_title 'Choose among promoted bot experts with a hybrid policy')" --repo chrisgarber/garboid-pocketrocks --body-file /private/tmp/garboid-pocketrocks-issue-6-20260730/project-9.md
```

Then run:

```bash
gh issue edit 7 \
  --repo chrisgarber/garboid-pocketrocks \
  --body-file "/private/tmp/garboid-pocketrocks-issue-6-20260730/issue-7.md"
```

Expected: each command returns its issue URL. #7 retains its complete original
opening body and gains exactly one `## Bot improvement portfolio extension`
section.

- [ ] **Step 3: Read back and validate all nine bodies**

Run `gh issue view` with `--json number,title,body,state,url` for each of the
eight numbers returned by `issue_number_for_title`, and for #7. Verify:

- the title is exact;
- the first headings are `What this solves` and `Why it matters`;
- no prose placeholders remain;
- every declared dependency URL targets the intended issue;
- every RICE tuple matches the approved table;
- every body links to #6;
- #7 still contains its original request;
- all nine issues are open.

### Task 5: Publish the portfolio summary and close issue #6

**Files:**
- Create temporarily: `/private/tmp/garboid-pocketrocks-issue-6-20260730/issue-6-summary.md`

**Interfaces:**
- Consumes: verified URLs and bodies from Task 4
- Produces: final linked summary on closed issue #6

- [ ] **Step 1: Write the closing summary with actual links**

Use `apply_patch` to create `issue-6-summary.md` with:

- a plain-language opening that explains the portfolio prioritizes the
  fastest path to stronger tournament results;
- the RICE definition from the approved specification;
- the full nine-row ranked table with each project title linked to its new
  issue or existing #7;
- the dependency tree in plain language;
- the common immutable-version and held-out final-exam rules;
- a note that #7 was reused to avoid duplicate RL work;
- a link to the committed design document in the repository.

Do not claim that any proposed bot is already stronger; the portfolio only
prioritizes future work.

- [ ] **Step 2: Comment on issue #6**

Run:

```bash
gh issue comment 6 \
  --repo chrisgarber/garboid-pocketrocks \
  --body-file "/private/tmp/garboid-pocketrocks-issue-6-20260730/issue-6-summary.md"
```

Expected: one new comment URL.

- [ ] **Step 3: Close issue #6**

Run:

```bash
gh issue close 6 \
  --repo chrisgarber/garboid-pocketrocks \
  --reason completed
```

Expected: issue #6 is reported closed as completed.

### Task 6: Audit the final external and local state

**Files:**
- Read: `docs/superpowers/specs/2026-07-30-bot-improvement-portfolio-design.md`
- Read: `docs/superpowers/plans/2026-07-30-bot-improvement-backlog.md`

**Interfaces:**
- Consumes: every artifact created by Tasks 1-5
- Produces: requirement-by-requirement completion evidence

- [ ] **Step 1: Verify issue #6 state and closing comment**

Run:

```bash
gh issue view 6 \
  --repo chrisgarber/garboid-pocketrocks \
  --json number,title,body,state,stateReason,comments,url
```

Expected: state `CLOSED`, reason `COMPLETED`, and the latest comment contains
all nine correct issue links and the RICE table.

- [ ] **Step 2: Verify the new issue set and #7**

Run:

```bash
gh issue list \
  --repo chrisgarber/garboid-pocketrocks \
  --state all \
  --limit 100 \
  --json number,title,state,url
gh issue view 7 \
  --repo chrisgarber/garboid-pocketrocks \
  --json number,title,body,state,url
gh issue view 8 \
  --repo chrisgarber/garboid-pocketrocks \
  --json number,title,body,state,url
```

Expected: exactly eight new open issues with the approved titles, #7 open with
the portfolio extension, and #8 byte-for-byte unchanged from Task 1.

- [ ] **Step 3: Verify local documentation**

Run:

```bash
git diff --check
git status --short --branch
git log -3 --oneline
```

Expected: no whitespace errors, both design and plan are committed, and the
worktree is clean.
