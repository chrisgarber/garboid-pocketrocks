# Public-belief search foundation

This package contains the information-safe first layer of issue #15. It does
not yet choose actions or search a game tree.

`reconstruct_public_search_position` starts from the same inputs a deployed bot
receives: its own hand, the canonical SDK decision context, public history, and
canonical rules knowledge. It replays the public history to check cash,
resources won, revealed cards, objectives, tiebreak order, loans, investments,
the current turn, and legal bid limits. Contradictions fail closed. Request
IDs, deadlines, metadata, simulator seeds, private decks, and real opponent
hands are not accepted as position inputs.

`sample_compatible_worlds` fills the genuinely unknown parts of a validated
position:

- each opponent's remaining hand slots;
- the still-hidden resource order; and
- the still-hidden action order.

It uniformly permutes the exact finite multisets that remain after public card
accounting. This joint sample has the same hypergeometric marginals as the
existing heuristic belief. SHA-256 ranking makes samples deterministic from
the explicit development identity, canonical public-input digest, fixed
search seed, and sample index. Asking for more work extends the same sample
prefix instead of changing earlier samples.

The only identity allowed by this foundation is
`late-game-public-belief-v1-dev`. It is deliberately absent from bot
registries and latest aliases. Released `balanced-v3` behavior and every other
released identity remain unchanged.

## Why tree search stops here

The pinned SDK can start a simulation only from a seed and advance the state it
created. Its public test-only `scenario` helper can derive a hypothetical
`DecisionContext`, but it cannot advance or score that situation. There is no
supported API to restore or fork an arbitrary sampled world. Mutating
`SimEngine` fields or its private batch arrays would couple strategy to hidden
implementation state and violate the SDK-authority boundary.

Canonical search therefore needs a small upstream SDK prerequisite:

1. a frozen `SimPosition` containing the phase, public event stream, setup,
   turn, cash, complete sampled hands and future orders, public holdings,
   objective claimants, loan and investment ledgers, and pending reveal state;
2. `SimEngine.snapshot()`, `SimEngine.from_position(position)`, and
   `SimEngine.fork()`;
3. a public `context_for(seat, decision_kind)` method; and
4. atomic validation that public-history replay, cards, actions, objectives,
   phase, and all ledgers conserve exactly.

SDK conformance tests must snapshot and restore real prefixes for three-,
four-, and five-player games on charts A-E, replay identical suffixes to equal
contexts/events/scores/rankings, reject each tampered field, and prove forks
are isolated. Once that exists, the next stacked change can add the bounded
search policy, deterministic fallback, work and latency reporting, development
tuning, freeze, and the common held-out promotion gate.
