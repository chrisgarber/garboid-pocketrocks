# SDK Migration PR Scope Design

## Goal

Rebuild Garboid PR #1 on the current `main` so it contains only the core
migration from Garboid's duplicated game engine to the SDK's published scalar
simulation engine.

Merged PR #2 is authoritative for future-cash heuristics, named heuristic
generations, bot versioning, CLI registration, and their tests. The migration
must preserve that behavior while changing the source of game state and rules.

## Architecture Boundary

The SDK owns:

- deck generation and seeded setup;
- legal actions and bid limits;
- auction, loan, investment, reveal, and objective transitions;
- mutable game state;
- scoring and deterministic seat ordering;
- canonical simulation events and turn records.

Garboid continues to own:

- bot brains and public rules knowledge;
- the synchronous per-decision session adapter required by local environments;
- match running, replay files, Monte Carlo aggregation, and CLI presentation;
- Gymnasium, PettingZoo, and neural training integrations;
- Garboid-specific reward accounting and competition-ranked placements.

The `garboid_pocketrocks.simulator` package therefore remains, but it becomes an
orchestration namespace rather than a rules-engine namespace.

## In Scope

### SDK dependency

- Pin an upstream SDK commit that publishes the scalar simulation toolkit.
- Use public SDK state, results, events, and contexts.
- Do not depend on Garboid-local copies of pending SDK PR implementations.

### Remove duplicated rules

Delete the local engine and its private representations:

- `src/garboid_pocketrocks/simulator/engine.py`
- `src/garboid_pocketrocks/simulator/model.py`
- `src/garboid_pocketrocks/simulator/setup.py`
- `src/garboid_pocketrocks/simulator/context.py`
- `src/garboid_pocketrocks/simulator/events.py`
- `src/garboid_pocketrocks/simulator/sampling.py`
- `src/garboid_pocketrocks/rules.py`

Delete tests that exist only to verify those private implementations. Replace
them with SDK adapter, session, replay, runner, and end-to-end conformance tests.

### Keep scalar orchestration

Keep or introduce:

- a thin `SdkGameSession` that exposes deterministic pending decisions,
  snapshots, transitions, terminal results, events, and history;
- deterministic seed derivation;
- match runner, replay, Monte Carlo, CLI, and error types;
- public `RulesetKnowledge` derived from SDK-supported variants and contexts;
- scalar Gymnasium and PettingZoo environments backed by `SdkGameSession`;
- necessary neural encoder and rollout adaptations to the new session/result
  types.

The session may translate SDK state into stable Garboid records, but it must not
reimplement any game rule.

### Preserve merged behavior

Port PR #2's current implementations rather than resolving in favor of the old
PR #1 branch. In particular:

- all v1/v2/latest heuristic profile constants remain unchanged;
- future-cash opportunity cost remains unchanged;
- public bot names, development-only IDs, and CLI registry remain unchanged;
- PR #2 tests remain authoritative and must continue to pass;
- `SessionScore.rank` uses competition ranking derived from final totals, while
  the SDK ranking remains the deterministic ordered seat list.

## Out of Scope

The following move to follow-up work:

- SDK `BatchSimEngine` integration;
- Garboid batch runner and vector environment;
- vector throughput benchmark and vector-engine design documents;
- local fast-context monkey patch and direct-context benchmark;
- neural GRU cropping, batched rollout collection, and neural throughput
  benchmark;
- local live-bot launcher;
- unrelated visible-resource work;
- duplicated or superseded future-cash/versioning documents;
- unrelated formatting-only changes.

SDK PR #8 owns direct scalar context optimization. SDK PR #9 owns the vector
engine. Garboid follow-ups should consume those APIs only after they merge or
their upstream commits are otherwise accepted.

## Branch and PR Strategy

Do not merge or rebase the contaminated PR #1 branch. Instead:

1. Create a clean reconstruction branch at current `origin/main`.
2. Port the approved final scalar migration behavior by subsystem.
3. Use `main` for every PR #2-owned conflict, then make only the minimal SDK
   adapter changes required to compile and preserve behavior.
4. Verify the reconstructed diff contains no out-of-scope files.
5. Force-update `codex/sdk-engine-migration` with `--force-with-lease` so the
   existing PR becomes the clean review surface.

Keep the old branch tip recoverable under a backup branch until the rebuilt PR
passes local and hosted verification.

## Testing Strategy

Use test-first reconstruction for behavior that changes at an adapter boundary.
The required coverage is:

- complete scalar SDK session parity with SDK `LocalGame`;
- deterministic seeded games for three, four, and five players;
- replay round trips and divergence detection;
- Monte Carlo equivalence across worker counts;
- scalar Gymnasium and PettingZoo API contracts;
- PR #2 heuristic/profile/versioning regression suites;
- equal-total terminal competition ranks;
- absence of imports and public exports for the deleted local rules engine.

Before updating the PR branch, run:

- `uv sync --locked --extra neural`;
- `uv run ruff format --check .`;
- `uv run ruff check .`;
- `uv run mypy src tests`;
- `uv run --extra neural mypy --config-file mypy.neural.ini src tests`;
- `uv run pytest`;
- the deterministic neural smoke test used by CI.

After the force-update, wait for both hosted `quality` and `neural` jobs to pass.

## Success Criteria

- PR #1 is based on the merged PR #2 result and is conflict-free.
- The diff contains the SDK migration and scalar consumers only.
- No local rules transition or scoring implementation remains.
- `simulator/` contains orchestration, not a second game engine.
- PR #2 behavior and tests remain intact.
- Vector, fast-context, neural-throughput, launcher, and unrelated work do not
  appear in the PR.
- Local and hosted quality/neural verification pass.
