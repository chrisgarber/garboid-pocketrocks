# SDK authority

## Decision

The current [PocketRocks rules](https://pocketrocks.xyz/rules) and the SDK
revision pinned in [`pyproject.toml`](../../pyproject.toml) are the authority
for game behavior.

SDK scalar and batch engines own setup, legal decisions, state transitions,
auctions, priority, reveals, objectives, scoring, and ranking. Garboid does
not implement an alternative rules engine. It owns the boundaries around the
SDK:

- synchronous brain invocation and fault policy;
- public rules knowledge and history adaptation;
- deterministic match, replay, Monte Carlo, and tournament orchestration;
- Gymnasium, PettingZoo, and neural-learning interfaces.

The competition repository and historical designs are useful context, but
they are not runtime dependencies or rule authorities.

## Consequences

- Rules knowledge is derived from pinned SDK constants and public contexts.
- SDK upgrades require conformance and deterministic-parity tests before the
  pin changes.
- Serialized compatibility is handled at replay, checkpoint, and report
  boundaries; it is not a reason to duplicate SDK behavior.
- Unknown SDK history or decision shapes fail explicitly instead of being
  guessed.
