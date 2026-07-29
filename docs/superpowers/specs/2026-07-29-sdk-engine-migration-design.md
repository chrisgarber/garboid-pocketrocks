# SDK Engine Migration Design

## Goal

Make the PocketRocks SDK the sole source of truth for local game rules. Migrate
Monte Carlo evaluation, replay, Gymnasium, and PettingZoo to the SDK's published
`pocketrocks.sim.SimEngine`, then remove this project's game-engine
reimplementation and configurable noncanonical rulesets.

The target SDK revision is
`48373524c61665c3b73ca91a2ae6420127f7da81`, which publishes
`pocketrocks.sim`.

## Scope

This migration includes:

- deterministic single-game execution;
- Monte Carlo planning, process workers, statistics, and behavior reporting;
- replay recording and deterministic replay verification;
- Gymnasium single-agent training;
- PettingZoo multi-agent training;
- reward tracking and public-history production;
- heuristic strategy knowledge derived from canonical SDK rules;
- CLI and README documentation;
- removal of project-owned setup, transition, objective, auction, reveal,
  financial, and scoring rules.

Noncanonical resource decks, action decks, player setups, objective pools, and
ruleset variation sampling are intentionally removed. The SDK's canonical live
rules, charts A-E, and objectives on/off options are authoritative.

## Architecture

All local execution uses one project-owned orchestration adapter over the SDK
engine:

```text
Monte Carlo / Replay / Gymnasium / PettingZoo
                         |
                         v
                 SDK game session
                 - pending contexts
                 - decision batches
                 - fault policy
                 - transition snapshots
                         |
                         v
             pocketrocks.sim.SimEngine
                 - setup and shuffling
                 - auction resolution
                 - reveals and objectives
                 - scoring and ranking
                 - canonical wire events
```

The adapter may sequence the public SDK operations `flip_action()`, `resolve()`,
`apply_reveal()`, context construction, and scoring. It must not contain deck
construction, winner selection, objective matching, financial rules, or score
formulas.

## SDK Session

The session owns one mutable `SimEngine` and exposes a stepwise interface needed
by RL environments and deterministic replay:

- construction accepts player count, seed, chart, objectives flag, and player
  names;
- `pending` exposes SDK `DecisionContext` values keyed by acting seat;
- a bidding step requires one legal SDK decision per seat;
- a choice-reveal step requires one legal SDK decision from the winner;
- SDK automatic reveals occur immediately and do not create a policy decision;
- each step returns an immutable transition view containing before/after
  snapshots, new SDK events/turn records, decisions, and an optional result;
- termination and scores come directly from SDK scoring and ranking.

Snapshots are observation and reward inputs copied from SDK state. They are not
accepted as engine input and do not implement rules.

## Strategy Knowledge

Brains continue to receive immutable public knowledge because heuristic and
neural policies need canonical deck counts and setup metadata omitted from
`DecisionContext`.

The configurable `Ruleset` type is removed. A canonical knowledge helper derives
metadata from SDK simulation constants and reconciles context-visible chart,
starting cash, player count, hand size, and objective state. Knowledge can never
configure the engine.

## Match and Monte Carlo Execution

`MatchRunner` uses the SDK session and retains the project's two fault modes:

- `raise` propagates bot construction, decision, and legality failures;
- `record_and_pass` records the failure and applies the SDK-equivalent fallback.

Fallback is bid zero for bidding and reveal index zero for a choice reveal.
Automatic SDK reveals are not counted as bot decisions.

Monte Carlo retains stable job planning, seat rotation, bot IDs, seeded brain
construction, process workers, detailed statistics, and optional replay
capture. Job configuration contains only canonical chart and objectives
settings rather than a ruleset sampler. Per-ruleset reporting becomes
per-variant reporting keyed by chart/objective configuration.

## Replay and Events

Replay schema version 2 records:

- SDK revision-compatible canonical configuration;
- player count and seed provenance;
- bot identities;
- submitted SDK decisions in step order;
- canonical SDK turn records and terminal scores;
- project fault records where applicable.

Replay drives a new SDK session with the recorded decisions and compares the
canonical SDK result. Schema version 1 is rejected clearly because it encodes
the removed engine's configuration and shuffle semantics.

SDK wire events and `TurnRecord` values replace project-authored rule events.
Project reporting may derive immutable statistics from these records, but may
not create a second normative event stream.

Public history is reconstructed from SDK wire requests/events using the same
SDK protocol path used for live contexts. The simulator-specific adapter over
project rule events is removed.

## RL Environments and Rewards

Both environments own an SDK session.

PettingZoo collects the sealed bidding batch across agents before submitting it
to the session; choice reveals expose only the winning seat. Gymnasium supplies
the learner decision, asks opponent brains for other pending decisions, and
advances until the learner acts again or the game terminates.

Reward tracking compares immutable before/after snapshots copied from SDK state.
It preserves the existing reward configuration and terminal-rank behavior while
removing dependencies on project engine types.

Observation and action encoding continue to consume SDK contexts. Environment
constructors select player count, chart, and objectives directly; arbitrary
ruleset samplers are removed.

## Error Handling

- Invalid session decision batches fail before mutating the SDK engine.
- SDK context validation is authoritative for individual decisions.
- Configuration validation delegates to `SimEngine`.
- Replay divergence names the first mismatching step or terminal result.
- Worker pickling and process failures retain the project's contextual errors.

## Testing

Implementation follows test-driven development.

Tests cover:

- the SDK dependency revision and canonical knowledge;
- deterministic SDK session setup, bidding, choice/automatic reveal, and
  termination;
- direct parity between session results and SDK `LocalGame` for fixed bots and
  seeds;
- strict and fallback fault modes;
- replay versioning, round trips, and divergence;
- serial/parallel Monte Carlo equivalence and behavior statistics;
- Gymnasium and PettingZoo API behavior and determinism;
- reward calculations from SDK snapshots;
- absence of project-owned rule execution modules and configurable samplers.

The final gate runs pytest, Ruff, strict mypy, and the existing neural mypy
configuration.

## Compatibility

Preserved where it does not conflict with SDK authority:

- bot brains and live wrappers;
- CLI bot names and chart choices;
- detailed Monte Carlo statistics;
- fixed action and observation encodings;
- Gymnasium/PettingZoo environment behavior;
- public neural-training interfaces.

Intentional breaking changes:

- configurable rulesets and ruleset samplers are removed;
- replay schema version 1 is no longer executable;
- automatic reveals no longer request a bot decision, matching the SDK;
- project engine/model/setup/context/event types are removed.
