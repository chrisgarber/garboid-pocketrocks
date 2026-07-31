# Public opponent-bid model design

## Goal

Use bids that opponents have already revealed to estimate what they may bid
next. The model must make the same decision in live play, scalar simulation,
batch simulation, and replay. It must never receive an opponent's hand,
objective, valuation, random seed, or simulator state.

This work starts from released heuristic v3. The phase-aware v4 experiment in
Issue #12 did not pass promotion, so its policies are not predecessors for
this experiment.

## Release scope

The model and reporting interfaces are personality-independent. The first
promotion candidate is `balanced-v5`: a proof of concept based on
`balanced-v3`. The skipped v4 release number records that the phase-aware v4
experiment produced frozen candidates but no release.

Aggressive and passive can adopt the same model in later experiments. They do
not consume a held-out final exam in this project unless development evidence
first shows a reason to expand the scope.

Released v1, v2, and v3 identities and live bot IDs remain unchanged. The new
candidate is local-only until it passes the common held-out promotion gate.

## Public input boundary

The strategy receives two deliberately narrow immutable values:

1. `PublicHistory`, parsed from the SDK's `common_events` allowlist.
2. A `PublicOpponentBidContext` copied field by field from the public decision
   context.

The model context contains only:

- player count and starting cash;
- the public value chart;
- current action ID;
- current cash by seat;
- current tiebreak seat and the bot's seat;
- the current legal maximum bid; and
- the public game phase.

It contains no current hand, objective ownership, private valuation, request
metadata, deadline, bot identity, seed, deck order, or simulator object.

Live bots obtain history through the SDK's raw-decision callback. The wrapper
parses the raw frame immediately and passes only `PublicHistory` to a
history-aware brain. A malformed frame fails closed; it does not silently run
a different history-free policy.

## Public game phase

Issue #10 already reports game phase by public turn number:

- turns 1 through 5 are early;
- turns 6 through 12 are middle; and
- turn 13 onward is late.

That helper becomes shared strategy code so diagnostics and opponent modeling
cannot drift. This is intentionally different from Issue #12's remaining-
resource expert selector. Reusing a reporting phase does not inherit any of
the failed v4 expert policies.

## Resolved bid history

A pure parser pairs each `turn_opened` event with the following
`auction_resolved` event and returns `PublicResolvedBidRound` records:

- zero-based turn index and public phase;
- action ID and public resource IDs; and
- effective bids by seat.

Only completed rounds become observations. The unresolved current turn is
allowed at the end of history but contributes no opponent bid. Invalid event
ordering, player counts, negative bids, or setup/context contradictions fail
closed.

## Deterministic sparse-history prior

With fewer than two completed rounds, the model uses only a documented public
prior. It assumes opponents tend to bid near a public reference price but may
choose any affordable integer amount.

For auctions, the reference begins with the median adjacent increase in the
public value chart, multiplied by the one- or two-resource action size. For
loan and investment actions, it begins with the public face value. Two small,
fixed adjustments represent public conditions:

- four- and five-player games raise the reference by 10% and 20%; and
- early, middle, and late phases multiply it by 75%, 100%, and 125%.

The chart component uses absolute adjacent gaps, because supported charts are
not all ascending. The reference is rounded deterministically and capped at
the opponent's public legal maximum. Every affordable bid receives positive
prior probability, with additional triangular weight near the reference. No
private estimate affects this prior.

An opponent's support is not always `0..current cash`: a loan action lets the
winner bid against the public principal as well. The model derives the public
credit amount from the acting bot's legal maximum minus its current cash, then
adds the same amount to each opponent's public cash. This produces each seat's
legal support without reading a private field.

The frozen model configuration records the total prior strength, the minimum
history threshold, and history weights. Those values are selected on the
development corpus only.

## Learning from revealed bids

Once at least two rounds are available, each opponent gets a separate
smoothed discrete distribution over `0..current_cash`:

- a past bid from the same action and phase receives the strongest weight;
- a bid matching either action or phase receives a medium weight;
- other public bids receive a small fallback weight; and
- the deterministic prior remains as smoothing.

If an old bid exceeds the opponent's current legal maximum, it maps to that
maximum. This treats it as evidence of willingness to spend everything
currently available without predicting an illegal action.

The model is recomputed as a pure function of the immutable public prefix for
every decision. It has no mutable online state and no randomness.

## Winning probability

For each legal bid, the model calculates the probability that every opponent
submits an amount the bot defeats. Ties follow the public tiebreak order:

- opponents ahead of the bot in that order must bid strictly less; and
- opponents behind the bot may bid the same amount.

Opponent distributions are combined under a documented conditional-
independence approximation. The report exposes every component distribution,
so this assumption remains auditable.

## Expected-surplus choice

The existing v3 valuator already computes `win_delta` for every legal integer
bid: how much better winning at that price is than losing. The history-aware
brain does not create a second valuation system.

For every legal effective bid, including zero:

`expected_surplus = predicted_win_probability * win_delta`

Passing is effective bid zero and can win an all-zero auction through the
public tiebreak rule, so it receives the same calculation as every other bid.
The brain selects the greatest expected surplus and chooses the lower bid on
an exact tie. Reveal decisions keep v3 behavior.

## Decision explanation and privacy

The typed explanation contains:

- the ordinary v3 value breakdown and reservation bid;
- one discrete bid distribution per opponent;
- predicted winning probability, win delta, and expected surplus for every
  legal bid; and
- the selected bid and public phase.

Local decision reports may show this public decision-level reasoning. No
high-dimensional per-decision table is committed as benchmark evidence.
Committed development summaries aggregate across cohorts of at least 30
distinct games and follow the privacy fail-closed rules introduced by Issue
#12. Held-out promotions run with diagnostics disabled.

## Development, freeze, and promotion

The implementation first lands as behavior-neutral infrastructure. The
candidate then uses fixed development games to select its small public model
configuration. The chosen configuration and exact v3 profile are frozen under
an explicit local-only identity with content digests.

Before the final exam, the frozen candidate must complete charts A-E with
three, four, and five players without illegal actions or bot faults. It is
then compared once with canonical `balanced-v3` on the existing held-out
paired corpus and bootstrap rule.

- If the 95% interval for rating delta is entirely above zero, release
  `balanced-v5` and move only the balanced latest alias.
- Otherwise retain the frozen negative result and leave every alias on v3.

No failed result is rerun to hunt for a favorable seed.
