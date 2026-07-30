# RL Experiment Backlog Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use braze-superpowers:subagent-driven-development (recommended) or braze-superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four independently ranked RL experiment issues, update the two
existing ranks they displace, and publish a verified 13-project ordering on
closed parent issue #6.

**Architecture:** Stage every final Markdown body in a new fixed temporary
directory and inspect it before any GitHub mutation. Create the four issues
only after an exact-title duplicate check, resolve the issue numbers assigned
by GitHub, then add real dependency links, update #15/#16, publish the
superseding parent comment, push the committed design and plan, and read back
every external artifact.

**Tech Stack:** Markdown, GitHub CLI (`gh`), `jq`, Git

## Global Constraints

- Every new issue begins with `## What this solves` and `## Why it matters` in
  simple language before technical implementation detail.
- Create exactly four issues with the titles fixed in the approved expansion
  specification.
- Preserve all existing checkpoints, configurations, bot IDs, and issue
  content except the explicitly approved #15/#16 rank/dependency changes.
- Every frozen experimental candidate receives an immutable explicit identity.
- Training loss, shaped reward, throughput, parameter count, and development
  performance cannot trigger promotion.
- Promotion uses issue #9's untouched held-out tournament and requires its 95%
  bootstrap interval rule.
- Update #15 from rank 8 to 10 without changing its RICE inputs or score.
- Update #16 from rank 9 to 11 without changing its RICE inputs or score.
- Add the new population and architecture projects as optional future neural
  experts in #16 using their actual GitHub URLs.
- Add one new comment to #6 that explicitly supersedes the earlier
  nine-project ordering and contains all 13 linked projects.
- Leave issue #6 closed with reason `COMPLETED`.
- Do not modify issue #8.
- Do not create, update, or close a pull request.
- Push only the existing `codex/bot-improvement-portfolio` branch, without
  force.

---

### Task 1: Capture the pre-mutation state and prove no duplicates exist

**Files:**
- Read: `docs/superpowers/specs/2026-07-30-rl-experiment-portfolio-expansion-design.md`
- Read: `docs/superpowers/plans/2026-07-30-rl-experiment-backlog-expansion.md`

**Interfaces:**
- Consumes: approved expansion specification at commit `2e738ba`
- Produces: exact original bodies/states for #6, #8, #15, and #16 plus a
  duplicate-title audit

- [ ] **Step 1: Verify local branch and documents**

Run:

```bash
git status --short --branch
git branch --show-current
git log -4 --oneline
rg -n "^## Project|^## GitHub delivery|^\\| (8|9|10|11|12|13) \\|" \
  docs/superpowers/specs/2026-07-30-rl-experiment-portfolio-expansion-design.md
```

Expected: branch `codex/bot-improvement-portfolio`, no uncommitted changes
except this plan if not yet committed, and four project sections.

- [ ] **Step 2: Verify GitHub authentication**

Run:

```bash
gh auth status
```

Expected: authenticated with issue and branch write access to
`chrisgarber/garboid-pocketrocks`.

- [ ] **Step 3: Snapshot the issue backlog**

Run:

```bash
gh issue list \
  --repo chrisgarber/garboid-pocketrocks \
  --state all \
  --limit 100 \
  --json number,title,body,state,url
gh issue view 6 \
  --repo chrisgarber/garboid-pocketrocks \
  --json number,title,body,state,stateReason,comments,url
gh issue view 8 \
  --repo chrisgarber/garboid-pocketrocks \
  --json number,title,body,state,url
gh issue view 15 \
  --repo chrisgarber/garboid-pocketrocks \
  --json number,title,body,state,url
gh issue view 16 \
  --repo chrisgarber/garboid-pocketrocks \
  --json number,title,body,state,url
```

Expected:

- #6 is `CLOSED/COMPLETED`;
- #8, #15, and #16 are open;
- #15 says rank 8 and RICE 1.10;
- #16 says rank 9 and RICE 1.07;
- none of these four exact titles exists:
  - `Use population-based training and PSRO to build a stronger neural league`
  - `Compare GRU, transformer, attention, and state-space neural policies`
  - `Evaluate IMPALA and V-trace for scalable neural training`
  - `Explore MuZero-style learned-model search`

Store the exact #8, #15, and #16 bodies for byte-level or structured
post-mutation comparison. If any target title exists, stop and inspect it
instead of creating a duplicate.

### Task 2: Stage and inspect all final Markdown

**Files:**
- Create temporarily: `/private/tmp/garboid-pocketrocks-rl-expansion-20260730/population.md`
- Create temporarily: `/private/tmp/garboid-pocketrocks-rl-expansion-20260730/architecture.md`
- Create temporarily: `/private/tmp/garboid-pocketrocks-rl-expansion-20260730/impala.md`
- Create temporarily: `/private/tmp/garboid-pocketrocks-rl-expansion-20260730/muzero.md`
- Create temporarily: `/private/tmp/garboid-pocketrocks-rl-expansion-20260730/issue-15.md`
- Create temporarily: `/private/tmp/garboid-pocketrocks-rl-expansion-20260730/issue-16.md`
- Create temporarily: `/private/tmp/garboid-pocketrocks-rl-expansion-20260730/issue-6-expansion.md`

**Interfaces:**
- Consumes: approved project sections and exact original #15/#16 bodies
- Produces: complete human-readable bodies ready for external mutation

- [ ] **Step 1: Create a new task-local temporary directory**

Run:

```bash
test ! -e /private/tmp/garboid-pocketrocks-rl-expansion-20260730
mkdir /private/tmp/garboid-pocketrocks-rl-expansion-20260730
```

Expected: both commands succeed. If the path exists, inspect it and choose a
new date-suffixed path rather than deleting or overwriting unknown files;
update every later command consistently before proceeding.

- [ ] **Step 2: Write the four issue bodies with `apply_patch`**

Each body uses this exact heading order:

```markdown
## What this solves

## Why it matters

## Proposed outcome

## RICE and rank

## Dependencies

## In scope

## Out of scope

## Acceptance criteria

## Versioning and promotion
```

Copy the complete plain-language and technical requirements from the matching
approved spec section, expanding prose into concrete scope/checklist bullets
without changing its meaning. Explain these terms on first use:

- population-based training and policy-space response oracles (PSRO);
- gated recurrent unit (GRU), transformer, cross-attention, and state-space
  encoder;
- IMPALA and V-trace;
- MuZero representation, dynamics, reward, policy, and value models.

Use these exact RICE tuples:

| File | Rank | Reach | Impact | Confidence | Effort | Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `population.md` | 8 | 6 | 2.0 | 60% | 5.0 | 1.44 |
| `architecture.md` | 9 | 5 | 2.0 | 55% | 4.0 | 1.38 |
| `impala.md` | 12 | 4 | 1.5 | 55% | 5.0 | 0.66 |
| `muzero.md` | 13 | 4 | 3.0 | 35% | 10.0 | 0.42 |

The first drafts may name dependencies by exact title. Task 4 replaces them
with actual GitHub URLs after creation. Every body links parent portfolio #6.

- [ ] **Step 3: Stage exact updates for #15 and #16**

Create `issue-15.md` from the exact original #15 body with only:

- `- **Rank:** 8` changed to `- **Rank:** 10`.

Create `issue-16.md` from the exact original #16 body with:

- `- **Rank:** 9` changed to `- **Rank:** 11`;
- one sentence in `Dependencies` saying the population and architecture
  experiments may provide optional promoted neural experts, initially named by
  exact title and later linked in Task 4.

Do not change any RICE input, RICE score, existing dependency, acceptance
criterion, versioning rule, or other prose.

- [ ] **Step 4: Stage the superseding #6 comment**

Create `issue-6-expansion.md` containing:

- a plain-language opening that says this 13-project ordering supersedes the
  earlier nine-project table;
- the unchanged RICE definition;
- the complete 13-row table from the approved spec;
- actual links for existing #7 and #9-#16;
- exact-title dependency names for the four not-yet-created issues, replaced
  with actual links in Task 4;
- short explanations of why population training ranks 8, architecture ranks
  9, IMPALA ranks 12, and MuZero ranks 13;
- a link to the published expansion design commit;
- an explicit statement that #6 stays closed and the original portfolio issues
  remain valid.

- [ ] **Step 5: Inspect the staged Markdown**

Run:

```bash
for body in /private/tmp/garboid-pocketrocks-rl-expansion-20260730/{population,architecture,impala,muzero}.md; do
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
rg -n 'TBD|TODO|FIXME|PLACEHOLDER' \
  /private/tmp/garboid-pocketrocks-rl-expansion-20260730
```

Expected: all heading checks succeed and the final search returns no matches.
Read every staged file completely before the first GitHub write.

### Task 3: Create exactly four new issues

**Files:**
- Read: the four staged new issue bodies

**Interfaces:**
- Consumes: inspected issue bodies
- Produces: four unique GitHub issue URLs and an exact title-to-number mapping

- [ ] **Step 1: Repeat the duplicate-title check immediately before creation**

Run:

```bash
gh issue list \
  --repo chrisgarber/garboid-pocketrocks \
  --state all \
  --limit 100 \
  --json number,title,state,url
```

Expected: none of the four exact titles exists. Stop on any match.

- [ ] **Step 2: Create one issue per exact title/body pair**

Run each command once:

```bash
gh issue create --repo chrisgarber/garboid-pocketrocks \
  --title "Use population-based training and PSRO to build a stronger neural league" \
  --body-file /private/tmp/garboid-pocketrocks-rl-expansion-20260730/population.md
gh issue create --repo chrisgarber/garboid-pocketrocks \
  --title "Compare GRU, transformer, attention, and state-space neural policies" \
  --body-file /private/tmp/garboid-pocketrocks-rl-expansion-20260730/architecture.md
gh issue create --repo chrisgarber/garboid-pocketrocks \
  --title "Evaluate IMPALA and V-trace for scalable neural training" \
  --body-file /private/tmp/garboid-pocketrocks-rl-expansion-20260730/impala.md
gh issue create --repo chrisgarber/garboid-pocketrocks \
  --title "Explore MuZero-style learned-model search" \
  --body-file /private/tmp/garboid-pocketrocks-rl-expansion-20260730/muzero.md
```

Expected: four distinct issue URLs ending in numeric IDs. After any ambiguous
timeout, query by exact title and never retry blindly.

- [ ] **Step 3: Prove exact cardinality**

Query all issues again. Expected: each target title appears exactly once, all
four new issues are open, and they have four distinct numbers.

### Task 4: Add real links and update #15/#16

**Files:**
- Modify temporarily: all seven staged Markdown files

**Interfaces:**
- Consumes: actual issue URLs from Task 3
- Produces: directly linked new issue bodies and exact approved #15/#16 updates

- [ ] **Step 1: Resolve and apply actual dependency links**

Use the issue URLs returned by GitHub, not predicted numbers. Patch:

- population: dependencies #7 and #9;
- architecture: dependencies #7 and #9;
- IMPALA: dependencies #7 and #9;
- MuZero: dependencies #7, #9, #10, and #15;
- #16: optional expert links to the population and architecture issues;
- #6 comment: all four new project links and the complete 13-project table.

- [ ] **Step 2: Push final bodies**

Define an exact-title resolver:

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

Run all six explicit edits:

```bash
gh issue edit "$(issue_number_for_title 'Use population-based training and PSRO to build a stronger neural league')" --repo chrisgarber/garboid-pocketrocks --body-file /private/tmp/garboid-pocketrocks-rl-expansion-20260730/population.md
gh issue edit "$(issue_number_for_title 'Compare GRU, transformer, attention, and state-space neural policies')" --repo chrisgarber/garboid-pocketrocks --body-file /private/tmp/garboid-pocketrocks-rl-expansion-20260730/architecture.md
gh issue edit "$(issue_number_for_title 'Evaluate IMPALA and V-trace for scalable neural training')" --repo chrisgarber/garboid-pocketrocks --body-file /private/tmp/garboid-pocketrocks-rl-expansion-20260730/impala.md
gh issue edit "$(issue_number_for_title 'Explore MuZero-style learned-model search')" --repo chrisgarber/garboid-pocketrocks --body-file /private/tmp/garboid-pocketrocks-rl-expansion-20260730/muzero.md
gh issue edit 15 --repo chrisgarber/garboid-pocketrocks --body-file /private/tmp/garboid-pocketrocks-rl-expansion-20260730/issue-15.md
gh issue edit 16 --repo chrisgarber/garboid-pocketrocks --body-file /private/tmp/garboid-pocketrocks-rl-expansion-20260730/issue-16.md
```

Expected: six successful issue URLs.

- [ ] **Step 3: Read back and compare**

Run `gh issue view` for all four new issues, #15, and #16 with
`--json number,title,body,state,url`.

Verify:

- every new title/body/state exactly matches its final staged Markdown;
- each declared dependency URL targets the intended issue;
- each RICE tuple and rank matches the 13-project table;
- #15 differs from its snapshot only at rank 10;
- #16 differs only at rank 11 and the approved optional-expert sentence;
- all six issues remain open.

### Task 5: Publish the expanded ordering on #6

**Files:**
- Read: final staged `issue-6-expansion.md`

**Interfaces:**
- Consumes: verified real URLs and expanded ranking
- Produces: one superseding parent comment while #6 remains closed

- [ ] **Step 1: Comment on #6**

Run:

```bash
gh issue comment 6 \
  --repo chrisgarber/garboid-pocketrocks \
  --body-file /private/tmp/garboid-pocketrocks-rl-expansion-20260730/issue-6-expansion.md
```

Expected: one new comment URL.

- [ ] **Step 2: Verify parent state and comment**

Run:

```bash
gh issue view 6 \
  --repo chrisgarber/garboid-pocketrocks \
  --json number,title,state,stateReason,comments,url
```

Expected: `CLOSED/COMPLETED`; latest comment exactly matches the staged file,
says it supersedes the earlier table, and links all 13 projects once.

### Task 6: Publish documentation and audit final state

**Files:**
- Read: expansion design and this plan

**Interfaces:**
- Consumes: verified local commits and GitHub state
- Produces: published documentation plus requirement-by-requirement evidence

- [ ] **Step 1: Verify branch safety**

Run:

```bash
git status --short --branch
git branch --show-current
git rev-parse HEAD
git ls-remote --heads origin codex/bot-improvement-portfolio
```

Expected: clean branch `codex/bot-improvement-portfolio`. Remote is an ancestor
of local HEAD. Stop if the remote has diverged.

- [ ] **Step 2: Push without force**

Run:

```bash
git push origin codex/bot-improvement-portfolio
```

Expected: fast-forward update. Do not create a pull request.

- [ ] **Step 3: Verify published design and local/remote equality**

Run:

```bash
gh api repos/chrisgarber/garboid-pocketrocks/commits/2e738ba \
  --jq '{sha: .sha, html_url: .html_url, message: .commit.message}'
git rev-parse HEAD
git rev-parse origin/codex/bot-improvement-portfolio
git diff --check
git status --short --branch
```

Expected: expansion design commit resolves on GitHub, local and remote SHAs
match, no whitespace errors, and the worktree is clean.

- [ ] **Step 4: Final external audit**

Re-query the complete issue list and #6 comments. Assert:

- exactly four new titles, each once and open;
- #15 rank 10/RICE 1.10;
- #16 rank 11/RICE 1.07 with both optional expert links;
- #8 byte-for-byte unchanged from Task 1;
- #6 `CLOSED/COMPLETED` with the verified superseding 13-project comment;
- no pull request was created or modified.
