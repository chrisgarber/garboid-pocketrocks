# RL Experiment Portfolio Expansion Design

## Plain-language summary

The original bot-improvement portfolio deliberately prioritized trustworthy
evaluation and cheap strategy improvements before expensive neural research.
That ordering was sound, but it left several promising reinforcement-learning
experiments hidden inside broad issues instead of making them visible,
comparable projects.

This expansion adds four focused experiments:

1. train against a changing population of old and newly discovered opponents;
2. compare the current GRU network with transformer, attention, and
   state-space alternatives;
3. test a more asynchronous training algorithm when synchronous PPO becomes a
   measured scaling bottleneck;
4. explore a learned game model that can plan ahead.

Each experiment changes one major part of the system while holding the others
fixed. None can claim success from faster training, lower loss, or a more
complex model alone. The final decision still comes from the fair held-out
tournament defined by issue #9.

## Current baseline and terminology

The current neural system uses:

- a gated recurrent unit (GRU) to encode at most 77 public history events;
- proximal policy optimization (PPO) to update the policy;
- synchronous rollout collection from one frozen policy snapshot;
- immutable checkpoint and league data structures, although the main trainer
  currently ignores `league_fraction` and always plans mirror self-play;
- a policy and value model that predicts actions and returns directly, without
  a learned dynamics model or search.

The four additions target different layers:

- **architecture:** how observations and history become policy features;
- **population:** which opponents generate training experience;
- **learning algorithm:** how distributed experience updates the policy;
- **planning:** whether the policy learns dynamics and searches future
  possibilities.

## RICE method

Use the portfolio's existing definitions:

- **Reach:** expected number of bot generations or experiments benefiting in
  six months, capped at 10.
- **Impact:** expected effect on held-out tournament strength: 0.5 low, 1
  medium, 2 high, and 3 massive.
- **Confidence:** evidence-adjusted probability that the impact is real.
- **Effort:** estimated engineer-weeks.
- **RICE:** `reach * impact * confidence / effort`.

## Expanded stack rank

| Rank | Project | Reach | Impact | Confidence | Effort | RICE |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Fair final exam for new bots | 10 | 1.0 | 95% | 1.5 | 6.33 |
| 2 | Decision-by-decision reports | 9 | 1.5 | 85% | 2.5 | 4.59 |
| 3 | Evolutionary heuristic v3 search | 8 | 2.0 | 80% | 3.0 | 4.27 |
| 4 | Competitive neural baseline | 7 | 2.0 | 80% | 3.5 | 3.20 |
| 5 | Phase-aware heuristic v4 | 6 | 2.0 | 70% | 4.0 | 2.10 |
| 6 | Opponent-aware bidding | 6 | 2.0 | 65% | 4.5 | 1.73 |
| 7 | Heuristic-informed PPO | 5 | 2.5 | 65% | 5.0 | 1.63 |
| 8 | Population-based training and PSRO neural league | 6 | 2.0 | 60% | 5.0 | 1.44 |
| 9 | GRU versus transformer, attention, and state-space bakeoff | 5 | 2.0 | 55% | 4.0 | 1.38 |
| 10 | Belief-state endgame search | 4 | 3.0 | 55% | 6.0 | 1.10 |
| 11 | Hybrid expert policy | 5 | 3.0 | 50% | 7.0 | 1.07 |
| 12 | IMPALA/V-trace off-policy training | 4 | 1.5 | 55% | 5.0 | 0.66 |
| 13 | MuZero-style learned-model search | 4 | 3.0 | 35% | 10.0 | 0.42 |

The lower RICE scores do not mean that the later projects lack upside. They
mean their cost and uncertainty are high relative to the evidence currently
available.

## Shared experiment contract

Every experiment must:

- preserve existing checkpoints, model configurations, and bot identities;
- assign an explicit immutable identity to each frozen candidate;
- use only information available to a live SDK bot during inference and
  search;
- predeclare training seeds, development evaluation, compute budget, hardware,
  checkpoint selection, and stopping rules;
- compare candidates under matched charts, player counts, seats, games,
  opponents, environment steps, and either matched compute or matched wall
  time;
- separate development selection from the untouched held-out final exam;
- report throughput, sample efficiency, memory, inference latency, illegal
  actions, faults, raw utility, and tournament rating;
- advance a latest alias only when issue #9's held-out promotion rule passes.

Training loss, shaped reward, development performance, throughput, or
parameter count alone cannot trigger promotion.

## Project 8: Population-based training and PSRO neural league

### What this solves

Training only against the current policy can produce a bot that is strong
against itself but exploitable by older or different strategies. It can also
cycle: one checkpoint beats the previous one while losing to an even older
checkpoint.

### Why it matters

A diverse league makes the learner practice against persistent weaknesses
instead of chasing one moving opponent. Population-based training can also
search training settings while policy-space response oracles (PSRO) add new
best responses to the opponent population.

### Proposed outcome

Wire the existing immutable league planner into the main trainer and compare
four matched-compute arms:

1. mirror-self-play PPO control;
2. checkpoint-league PPO without adaptive population search;
3. population-based training that mutates and selects predeclared training
   hyperparameters;
4. PSRO that iteratively trains approximate best responses and updates the
   opponent mixture.

If a combined population-based/PSRO arm is tested, it is a fifth arm and
cannot replace the separate ablations.

### Dependencies

- issue #7: competitive neural baseline and working checkpoint inference;
- issue #9: fair held-out promotion gate.

Decision diagnostics from issue #10 are helpful but not a hard dependency.

### Acceptance criteria

- `league_fraction` and immutable historical checkpoint identities affect
  actual training episodes rather than only configuration files;
- fixed population manifests and seeds reproduce the same opponent mixture,
  mutations, selections, and planned games;
- PBT and PSRO each have a separate matched-compute ablation;
- reports include exploitability proxies, pairwise population results,
  diversity, checkpoint lineage, raw utility, and held-out tournament
  strength;
- no current checkpoint or alias is mutated;
- a population-trained candidate advances only through issue #9.

## Project 9: Neural architecture bakeoff

### What this solves

The GRU is a sensible baseline, but it has not been shown to be the best way to
combine current state, player information, and public history for
PocketRocks.

### Why it matters

Transformers or structured attention may capture relationships between
auctions, players, and reveals that a recurrent summary loses. A state-space
encoder may retain long-context benefits with lower inference cost. A fair
bakeoff shows whether any added complexity produces stronger play.

### Proposed outcome

Compare these model families:

1. current GRU control;
2. transformer history encoder;
3. cross-attention between current state, player slots, and history;
4. state-space history encoder;
5. no-history control to measure the value of historical information.

Hold PPO, observation schema, legal-action masking, reward, rollout schedule,
seeds, model-parameter envelope, training environment steps, and accelerator
budget fixed. Target trainable parameter counts within 5% of the GRU control.
If an architecture cannot fit that tolerance, use its closest viable
configuration, explain the mismatch before training, and report both raw and
parameter-normalized results.

### Dependencies

- issue #7: competitive neural baseline;
- issue #9: fair held-out promotion gate.

### Acceptance criteria

- every architecture uses the same live-compatible observations and action
  masks;
- matched-budget runs record parameter count, FLOPs or an explicit work proxy,
  training time, peak memory, decisions per second, sample efficiency,
  inference latency, value calibration, and tournament strength;
- the no-history control quantifies whether history encoding materially helps;
- fixed inputs and checkpoint identities reproduce the same greedy actions;
- every frozen architecture has a distinct immutable checkpoint identity;
- no architecture advances from training metrics alone.

## Project 12: IMPALA/V-trace off-policy training

### What this solves

Synchronous PPO requires rollout collection to use one frozen policy snapshot.
At larger scales, actors, the learner, or an accelerator may wait for one
another instead of using the available hardware continuously.

### Why it matters

IMPALA separates actors from the learner so both can run continuously.
V-trace corrects for the fact that actors may generate experience using a
slightly older policy. This can improve wall-clock learning if policy lag stays
controlled.

### Proposed outcome

Build a bounded-policy-lag actor/learner experiment and compare it with the
synchronous PPO baseline under both:

- equal environment steps, measuring sample efficiency;
- equal wall-clock budget on the same hardware, measuring time to strength.

This project begins only when issue #7 profiling either attributes at least
20% of end-to-end update time to actor/learner waiting or coordination, or
measures accelerator utilization below 70% because of pipeline stalls. Record
the qualifying evidence before implementation.

### Dependencies

- issue #7: profiled competitive PPO baseline;
- issue #9: fair held-out promotion gate.

### Acceptance criteria

- the issue records the measured bottleneck that justifies starting;
- actor policy versions, learner versions, queue delay, and lag distribution
  are observable and bounded;
- V-trace targets have unit/property tests against hand-computed examples;
- equal-step and equal-wall-time comparisons use identical hardware,
  environments, seeds, and evaluation gates;
- reports include hardware utilization, decisions per second, policy lag,
  sample efficiency, raw utility, and held-out strength;
- synchronous PPO remains available and unchanged as the control.

## Project 13: MuZero-style learned-model search

### What this solves

The current neural policy chooses an action directly from observations. It
does not learn a dynamics model that can imagine future decisions and search
for an action with a better long-term outcome.

### Why it matters

A learned model and bounded search could discover tactics beyond the reach of
static heuristics or a reactive policy. The upside is high, but hidden
information and multiplayer opponents make a faithful model difficult.

### Proposed outcome

Prototype a MuZero-style representation, dynamics, reward, policy, and value
model operating on live-compatible public/private observations. Run bounded
search in the learned latent state without reading hidden opponent hands, deck
order, unresolved bids, or simulator RNG.

The first mechanical milestone may use live-A with three players to prove
legality, determinism, model-target integrity, and search-budget behavior.
Results from that milestone cannot support promotion. Full promotion still
requires charts A-E and three to five players.

### Dependencies

- issue #7: competitive neural policy/value baseline;
- issue #9: fair held-out promotion gate;
- issue #10: decision-level data and diagnostics;
- issue #15: public-belief and deterministic search boundaries.

### Acceptance criteria

- model inputs and training targets have an explicit information-boundary
  audit proving no hidden-state leakage;
- deterministic seeds and work budgets reproduce the same latent search and
  action;
- representation, dynamics, reward, policy, and value losses are reported
  separately;
- multi-step latent predictions are compared with held-out public trajectories
  without claiming that low model loss proves playing strength;
- search reports node count, depth, latency, fallback, and legal-action
  violations;
- the direct neural policy remains available as the matched control;
- promotion requires issue #9's full held-out tournament.

## GitHub delivery

Create four new issues with these human-readable titles:

1. `Use population-based training and PSRO to build a stronger neural league`
2. `Compare GRU, transformer, attention, and state-space neural policies`
3. `Evaluate IMPALA and V-trace for scalable neural training`
4. `Explore MuZero-style learned-model search`

Each issue begins with `What this solves` and `Why it matters`, then includes
the proposed outcome, RICE tuple, dependencies with real links, in/out of
scope, acceptance criteria, and versioning/promotion rules.

Update existing issue #15 from rank 8 to rank 10 and issue #16 from rank 9 to
rank 11 without changing their RICE inputs or scores. Add the new architecture
and population projects as optional future neural experts in #16.

Add a new comment to closed parent issue #6 that:

- clearly says it supersedes the earlier nine-project ordering;
- contains the full 13-project table with links;
- explains why the four additions rank where they do;
- leaves issue #6 closed as completed.

Do not modify unrelated issue #8. Do not reopen issue #6. Do not create or
modify a pull request.
