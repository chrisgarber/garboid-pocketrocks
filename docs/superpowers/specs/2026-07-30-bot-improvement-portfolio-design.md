# Bot Improvement Portfolio Design

## Plain-language summary

Garboid already has several useful bot-building tools: deterministic games,
fast tournaments, statistical ratings, heuristic bots, and a neural-network
training harness. The next step is not to build every interesting idea at
once. It is to create a reliable way to tell whether a new bot is actually
better, use that feedback to pursue the cheapest promising improvements first,
and leave the expensive experimental ideas until the evidence supports them.

The first project, the "held-out promotion gate," is simply a fair final exam
for bots. A candidate is developed using one known set of games, then tested
once on a different set of games it has never seen. It becomes the new default
only if that final test gives strong evidence that it is better than the
current bot. This prevents luck, cherry-picking, or tuning to the test from
being mistaken for progress.

This design ranks nine projects by their likely contribution to tournament
strength relative to their cost. Eight will become new GitHub issues. Existing
issue #7 will own the competitive neural-baseline project so that the new
portfolio does not duplicate work already in the backlog.

## Goal and success criterion

The primary objective is to maximize held-out tournament strength as quickly
as practical. Research breadth and live deployment remain valuable, but they
do not drive the initial ordering.

Issue #6 is complete when:

1. the opportunities are described and stack-ranked with RICE scores;
2. their dependencies and common validation rules are recorded;
3. eight nonduplicative follow-on issues are created;
4. existing issue #7 is updated to represent the neural-baseline project;
5. issue #6 contains the final ranking and links, then is closed.

## Current baseline

The repository already provides more than the original issue assumed:

- a deterministic SDK-backed simulator and fast batch engine;
- balanced multi-bot tournaments across charts A-E and three to five players;
- tie-aware Plackett-Luce ratings and bootstrap confidence intervals;
- immutable heuristic v1 and v2 generations;
- a Bayesian reveal policy that minimizes useful information exposed to
  opponents;
- vectorized recurrent PPO with checkpointing, evaluation-plan, metric, and
  league primitives.

The largest current gaps are:

- no standard development-versus-final-test protocol for bot promotion;
- limited decision-level evidence explaining where strategies gain or lose;
- no automated search over heuristic candidates;
- no registered, strength-tested neural bot;
- no phase-aware, opponent-aware, endgame-search, or hybrid policy.

## Chosen approach

Use a measurement-and-search funnel:

1. make candidate comparisons trustworthy;
2. expose strategy weaknesses at the decision level;
3. search cheap heuristic improvements;
4. close the neural training-to-tournament loop;
5. use the evidence to justify phase, opponent, learning, search, and hybrid
   projects.

This is preferred over a neural-first portfolio because current neural strength
is unproven, and over building many strategy families at once because that
would dilute effort before the repository can reliably select winners.

## RICE method

- **Reach:** expected number of bot generations or experiments that benefit
  during the next six months, capped at 10.
- **Impact:** expected effect on held-out tournament strength: 0.5 low, 1
  medium, 2 high, and 3 massive.
- **Confidence:** evidence-adjusted probability that the expected impact is
  real.
- **Effort:** estimated engineer-weeks.
- **RICE:** `reach * impact * confidence / effort`.

These are comparative planning estimates, not measured results. Re-score later
projects when earlier diagnostics produce better evidence.

| Rank | Project | Reach | Impact | Confidence | Effort | RICE |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Fair final exam and benchmark games for new bots | 10 | 1.0 | 95% | 1.5 | 6.33 |
| 2 | Decision telemetry and conditional performance diagnostics | 9 | 1.5 | 85% | 2.5 | 4.59 |
| 3 | Evolutionary heuristic search producing v3 candidates | 8 | 2.0 | 80% | 3.0 | 4.27 |
| 4 | Competitive neural baseline tracked by issue #7 | 7 | 2.0 | 80% | 3.5 | 3.20 |
| 5 | Phase-aware heuristic mixture producing v4 candidates | 6 | 2.0 | 70% | 4.0 | 2.10 |
| 6 | Public-history opponent model and first-price bid shading | 6 | 2.0 | 65% | 4.5 | 1.73 |
| 7 | Heuristic-informed PPO bootstrap and auxiliary value targets | 5 | 2.5 | 65% | 5.0 | 1.63 |
| 8 | Belief-state endgame search bot | 4 | 3.0 | 55% | 6.0 | 1.10 |
| 9 | Hybrid meta-policy combining promoted experts | 5 | 3.0 | 50% | 7.0 | 1.07 |

## Dependencies

```text
1 promotion gate
├── 2 decision diagnostics
│   ├── 5 phase-aware heuristic v4 (also depends on 3)
│   ├── 6 opponent-aware bidding
│   ├── 7 heuristic-informed PPO (also depends on 4)
│   └── 8 endgame belief search
├── 3 evolutionary heuristic v3
│   ├── 5 phase-aware heuristic v4
│   └── 9 hybrid meta-policy
└── 4 competitive neural baseline (#7)
    ├── 7 heuristic-informed PPO
    └── 9 hybrid meta-policy

5, 6, and 8 also provide optional experts to 9.
```

## Common validation and versioning rules

Every behavior-changing project must:

- keep released heuristic generations and neural checkpoints immutable;
- create a new explicit local simulation identity;
- keep fixed-input decisions deterministic;
- complete charts A-E with three, four, and five players without illegal
  actions or bot faults;
- compare the candidate and immediate predecessor using identical games,
  seats, charts, player counts, and opponent mixtures;
- select and tune only with development games;
- freeze the candidate before running the held-out final comparison;
- promote only when the held-out 95% bootstrap interval for the candidate's
  rating delta is above zero;
- publish configuration, seeds, commit, artifacts, and the promotion decision.

An inconclusive or failed candidate remains available under its versioned
research identity but does not advance an unversioned latest alias.

Infrastructure projects require unit, property, and deterministic integration
tests. Strategy projects additionally require development tournaments and one
untouched held-out tournament.

## Project scopes

### 1. Create a fair final exam before promoting new bots

**What this solves:** Today a developer can run a tournament, change a bot,
and run another tournament, but there is no standard separation between games
used to tune the bot and games used to prove the final result. Reusing the same
examples can make a lucky or overfit candidate look stronger than it is.

**Why it matters:** Every later strategy needs a fair final exam. This project
makes "better" a reproducible claim instead of a judgment based on a convenient
tournament run.

Define immutable development and held-out game corpora, paired candidate versus
incumbent evaluation, rating-delta intervals, and a machine-readable promotion
report. The report must fail closed on identity mismatches, overlapping
development/held-out seeds, illegal actions, faults, missing games, or
nonfinite analysis results. This project does not change bot behavior.

### 2. Add privacy-safe bot decision diagnostics

**What this solves:** Tournament ratings show which bot wins but provide
limited evidence about which decisions caused the difference. That makes
strategy work slower and encourages coefficient guessing.

**Why it matters:** Decision-level evidence can identify whether a bot loses
money early, mishandles loans, overpays for objectives, reveals useful
information, or fails only in specific charts and player counts.

Record public decision context, legal options, selected action, optional
policy explanation/value components, and eventual public outcome. Produce
reports sliced by game phase, chart, player count, action type, cash horizon,
objective state, seat, and opponent composition. Hidden simulator state must
never enter policy inputs or diagnostic explanations presented as deployable
knowledge.

### 3. Use evolutionary search to produce heuristic v3 candidates

**What this solves:** Current heuristic coefficients were manually calibrated.
The fast simulator can evaluate far more combinations than a person can
reasonably try.

**Why it matters:** Automated search is a relatively cheap way to find stronger
behavior while keeping the existing policy understandable and fast.

Use an evolutionary algorithm to search the existing heuristic coefficient
families on development games. Freeze the best candidates before held-out
evaluation. Preserve v1 and v2, publish the full search manifest, and create v3
only for candidates that pass the promotion gate. Do not add phase-specific
parameters here; project 5 tests that extra complexity separately.

### 4. Complete a competitive neural baseline in issue #7

**What this solves:** The repository can train and checkpoint neural policies,
but it does not yet expose a registered neural brain or demonstrate that a
checkpoint is competitive with the heuristic field.

**Why it matters:** Without the training-to-tournament loop, neural progress is
measured mainly by training metrics rather than actual playing strength.

Extend existing issue #7 with a history-aware inference adapter, immutable
checkpoint identities, committed training runs, tournament evaluation, and
the common promotion rules. This is an update to #7, not a duplicate new issue.

### 5. Build phase-aware heuristic v4 candidates

**What this solves:** One coefficient set currently governs the entire game
even though cash, information, remaining auctions, and objective urgency change
substantially from the opening to the endgame.

**Why it matters:** A conservative early policy and decisive late policy may
outperform any single compromise while retaining transparent, inexpensive
decisions.

Add early, middle, and late heuristic experts selected from public
remaining-game state. Use diagnostics to define the phase boundary and
evolutionary search to tune experts. Preserve v3 and promote a v4 alias only
after held-out improvement.

### 6. Add public-history opponent modeling and bid shading

**What this solves:** Current first-price bid shading does not adapt to what
opponents have publicly bid or how much cash they retain. The same private
value can justify different bids against passive and aggressive fields.

**Why it matters:** Predicting the chance that each legal bid wins can reduce
both unnecessary overpayment and avoidable losses.

Add a history-aware brain protocol and a deterministic opponent-bid model
using public history, cash, action, chart, player count, and game phase. Select
the bid with the best expected surplus under the estimated winning
probability. Use an explicit deterministic prior when data is sparse.

### 7. Bootstrap PPO from heuristic policies and value targets

**What this solves:** PPO currently has to discover useful bidding and reveal
behavior largely through self-play. Strong heuristic knowledge may shorten
that expensive discovery period.

**Why it matters:** Faster learning produces stronger checkpoints for the same
compute budget and makes neural experiments cheaper to repeat.

Compare behavior-cloning initialization, heuristic value auxiliary targets,
and heuristic-opponent curricula through controlled ablations. PPO is the
reinforcement-learning algorithm currently used to train the neural policy.
Use issue #7's competitive neural baseline as the incumbent. Training loss or
shaped reward alone cannot trigger promotion; held-out raw tournament utility
must improve.

### 8. Build a belief-state endgame search bot

**What this solves:** Near the end of a game there are fewer unknown cards and
future decisions, but current heuristics still reduce the position to immediate
hand-written values.

**Why it matters:** Bounded imperfect-information search can reason through the
small remaining decision tree and find tactical gains that static coefficients
miss.

Search sampled states consistent with public information and the existing
Bayesian belief model. Enforce deterministic sampling, a configured work
budget, legal live-compatible inputs, and a documented fallback. Measure both
strength and decision latency.

### 9. Build a hybrid meta-policy across promoted bot experts

**What this solves:** Heuristic, neural, opponent-aware, and search policies
have different strengths, but choosing one globally discards the others.

**Why it matters:** A small selector may obtain more value from already-built
experts than another large standalone strategy project.

Select among promoted experts using only live-compatible information. Train or
tune the selector only on development data, preserve deterministic fallbacks,
and report both aggregate strength and which expert was selected in each
condition. Begin only after at least the heuristic v3 and competitive neural
baseline are available.

## GitHub issue writing standard

Every issue must be understandable before implementation details begin. Use
this order:

1. `## What this solves` — two or three plain-language sentences describing
   the current limitation.
2. `## Why it matters` — two or three plain-language sentences connecting the
   work to stronger bots or more trustworthy decisions.
3. `## Proposed outcome` — the concrete end state without unnecessary jargon.
4. `## RICE and rank` — values, score, and the assumptions behind them.
5. `## Dependencies` — linked predecessor issues and what is required from
   each.
6. `## In scope` and `## Out of scope`.
7. `## Acceptance criteria` — observable behavior, tests, and artifacts.
8. `## Versioning and promotion` — identity and benchmark requirements.

Technical terms must be explained the first time they appear. In particular,
the first issue must call its promotion gate a "fair final exam" before using
the formal term. Issue titles should prefer familiar verbs and concrete
outcomes; put necessary technical terminology in the body after the
plain-language opening.

## Backlog delivery

Create eight new issues for projects 1, 2, 3, 5, 6, 7, 8, and 9. Update issue
#7 with project 4's plain-language problem statement, rank, RICE score,
dependency on project 1, and acceptance criteria. Do not modify unrelated issue
#8.

After creation, comment on issue #6 with:

- the scoring definition and final table;
- an ordered list of all nine linked projects;
- the dependency summary;
- the immutable-version and held-out promotion rules;
- a note that #7 was reused to avoid duplicate work.

Then close issue #6 as completed.
