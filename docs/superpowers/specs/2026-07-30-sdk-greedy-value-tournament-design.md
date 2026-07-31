# SDK Greedy Value Tournament Bot Design

## Goal

Add the PocketRocks SDK's published `GreedyValueBot` to Garboid's shared bot
registry and curated default tournament without copying or changing the SDK
policy.

The new local-only simulation identity is `sdk-greedy-value-v1`. It is the
first frozen Garboid identity for this external policy, so it has no preceding
generation to preserve or benchmark against. Existing bot identities and
behavior remain unchanged.

## Policy source and versioning

`GreedyValueBot` comes from
`pocketrocks.sim.sample_bots` in the already-pinned SDK dependency at commit
`51cad378ee1e70a78e39ebbb25957ea003444873`. The class was originally published
in upstream SDK commit `48373524c61665c3b73ca91a2ae6420127f7da81`.

Garboid imports the SDK class rather than copying its bid and reveal formulas.
The versioned simulation name remains immutable. Reproducibility tests pin:

- the exact SDK Git revision in `pyproject.toml`;
- the local name and name-based simulation identity;
- representative bid, pass, capped-bid, and reveal decisions;
- parity between the synchronous Garboid adapter and the SDK bot.

A future SDK upgrade that changes any pinned decision must either retain this
SDK revision for `sdk-greedy-value-v1` or introduce a new Garboid generation.
It must not silently change v1 behavior.

The bot is local-only. It uses `BotSpec.for_simulation`, does not define a
remote `BOT_ID`, and is not added to the live launcher.

## Synchronous adapter

Add a focused `bots.sdk_samples` module containing
`SdkGreedyValueV1Brain` and `SDK_GREEDY_VALUE_V1_BOT_SPEC`.

Garboid's simulator expects synchronous `BotBrain.choose_decision` calls,
while the SDK sample exposes an asynchronous method. The published
`GreedyValueBot.choose_decision` coroutine performs no asynchronous work. The
adapter owns one SDK bot instance and advances its decision coroutine once to
extract the returned `BotDecision` without creating an event loop for every
tournament action.

The bridge must fail clearly if a future SDK implementation suspends instead
of returning immediately. It closes the unexpected coroutine and raises a
runtime error rather than guessing how to resume it or silently changing
tournament timing.

The adapter ignores `RulesetKnowledge` because the SDK bot derives its entire
decision from the supplied `DecisionContext`. It also ignores the optional
brain seed because the published policy is deterministic.

## Registration and tournament behavior

Add `sdk-greedy-value-v1` to `BOT_SPECS` and
`DEFAULT_TOURNAMENT_BOT_SPECS`. The curated default field becomes:

- `random`;
- `aggressive-v1`, `balanced-v1`, and `passive-v1`;
- `aggressive-v2`, `balanced-v2`, and `passive-v2`;
- `sdk-greedy-value-v1`;
- `vector_ppo_small_v1_g1500`;
- `vector_ppo_large_v1_g350k`.

The complete name registry makes the bot selectable through both
`garboid-simulate --bots` and `garboid-tournament --bots`. Existing
unversioned heuristic aliases remain registered but excluded from the curated
field because they duplicate v2 behavior.

No schedule, rating, reporting, or fault-handling logic changes. The standard
tournament automatically balances the added identity across charts, player
counts, lineups, and seats using its existing deterministic scheduler.

## Tests

Use test-driven development.

Create focused SDK-sample bot tests that:

- demonstrate the desired versioned name and local identity;
- compare adapter and SDK decisions for bid and reveal contexts;
- pin representative expected decisions independently of the imported SDK
  class;
- prove the adapter is deterministic and its `BotSpec` is pickle-safe;
- prove an unexpectedly suspending SDK coroutine raises the bridge error.

Update registry tests to pin the complete registry order and curated default
field. Update tournament CLI tests to prove default resolution includes the
new bot. Run a small tournament smoke test through the existing CLI path and
confirm it finishes with zero bot faults and writes all report artifacts.

Run the focused tests first, then formatting, lint, strict type checking, and
the full test suite.

## Documentation

Update the README's simulation and tournament sections to:

- name `sdk-greedy-value-v1` as the frozen SDK policy;
- describe its value-based bidding behavior;
- include it in the shared registry and curated default field;
- state that its implementation is sourced from the pinned SDK rather than a
  Garboid remote bot.

## Compatibility and non-goals

- Existing public bot names, remote IDs, checkpoints, and decisions do not
  change.
- The SDK dependency does not need to advance because the pinned revision
  already contains `GreedyValueBot`.
- Garboid does not copy or modify the SDK policy.
- Garboid does not add general asynchronous bot support to the simulator.
- Garboid does not create or launch a live wrapper for this sample opponent.
- This integration does not run or commit a new full 15,000-game benchmark;
  the next standard tournament run establishes the v1 baseline.
