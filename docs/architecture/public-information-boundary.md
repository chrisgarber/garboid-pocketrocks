# Public-information boundary

## Decision

Bot strategy, evaluation, and learning code may consume only information
available to a deployable SDK bot:

- the current SDK decision context;
- public rules knowledge derived from the pinned SDK;
- cumulative public setup, turn, auction-resolution, and reveal events;
- the acting bot's current private hand;
- the legal-action boundary for the current request.

They must not receive opponent hands, shuffled deck order, unresolved sealed
bids, simulator RNG state, future actions, or an omniscient critic state.

Public-history adapters use an allowlist and fail closed when an SDK frame no
longer matches the supported schema. Batch execution must construct the same
context and history as scalar SDK execution.

## Consequences

- Heuristic beliefs remove known public cards and the acting hand from a
  finite population; they do not inspect engine-private arrays.
- Neural policy and value heads share the deployable information boundary.
- Replay and analysis may record resolved public outcomes, but may not feed
  future or hidden information back into a decision.
- New inputs require an explicit architecture review and, when behavior can
  change, the [identity policy](immutable-bot-identities.md).
