# Fixed-Bid Bot Design

## Goal

Add a deterministic local simulation baseline that bids fixed amounts based
only on the current action type. Existing random, heuristic, and neural bot
behavior remains unchanged.

## Policy

The bot uses these target bids:

| Action | Target bid |
|---|---:|
| One-resource auction | 6 |
| Two-resource auction | 12 |
| Loan | 1 |
| Investment | 7 |

For a bid decision, the bot submits the smaller of the target bid and the
context's legal maximum. It passes when the legal maximum is absent or not
positive.

For an information-reveal decision, the bot selects index `0` when at least
one option is revealable and passes otherwise. Seeds and ruleset knowledge do
not affect its decisions.

## Architecture and API

Implement the policy in a dedicated `FixedBidBotBrain`. Keep it separate from
the valuation-based heuristic engine because it has no beliefs, coefficients,
or state-dependent valuation.

Expose one local-only `BotSpec` named `fixed-bid`. Its simulation identity is
also `fixed-bid`; it does not define a remote-style `BOT_ID` or live bot
wrapper because no server-issued identity was provided.

Export the brain from the bots package, register the spec in the shared bot
registry, and include it in the default tournament field as a deterministic
baseline. Existing live launcher behavior remains unchanged.

## Error Handling

Known PocketRocks actions map directly to the four target bids. If a future or
malformed action identifier reaches the policy, it passes rather than risking
an illegal or unintended bid.

Every returned decision must satisfy the SDK context's legality constraints.

## Testing

Use test-driven development. Focused brain tests cover:

- all four action-to-bid mappings;
- capping each target at the legal maximum;
- passing for missing or nonpositive bid limits;
- deterministic reveal selection and empty reveal handling;
- seed-independent fresh brain construction;
- local-only name-based simulation identity.

Registry tests pin the new name, unique identity, and inclusion in the default
tournament field. Run the focused bot and registry tests first, then the full
test, lint, and type-check suites.

## Compatibility and Non-Goals

- Existing bot names, identities, policies, and latest aliases remain frozen.
- Existing simulator and tournament CLI selection works through the shared
  registry.
- The bot is not remotely launchable.
- Bid amounts are not user-configurable.
- No benchmark or claim about comparative strength is part of this change.

## Design Review Record

The optional automated design-review panel is unavailable in this workspace.
The spec was self-reviewed for placeholders, contradictions, ambiguous action
mapping, and scope.
