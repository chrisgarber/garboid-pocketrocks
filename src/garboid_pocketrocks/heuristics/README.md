# Heuristic bots

The aggressive, balanced, and passive strategies use deterministic,
auditable valuation over the SDK-visible game state. They follow the
[public-information boundary](../../../docs/architecture/public-information-boundary.md)
and never inspect opponent hands or shuffled engine state.

## Public belief

For each resource suit, the belief removes publicly won cards, public reveals,
the acting bot's hand, and the current offer from the finite deck. Remaining
opponent slots are an exact hypergeometric sample from the unseen cards.
Conditioning on one suit affects the others because all cards share one finite
population.

Offered-resource counts are action-aware: only resource-granting auctions add
the visible lot. Known cards, hidden slots, and future biddable resources must
conserve the configured deck or belief construction fails.

## Valuation

The valuator builds an additive win value for every legal integer bid:

- expected terminal value of offered resources;
- full value for objectives completed by the lot and shaped option value for
  public progress;
- concave liquidity value for cash retained for later auctions;
- future-auction cash value over the public remaining-resource horizon;
- fixed loan and investment cash flows.

The reservation bid is the greatest bid with nonnegative win value. Each
profile shades that reservation bid. `BidEvaluation` exposes the posterior,
complete bid curve, reservation and chosen bids, and the value breakdown for
analysis.

Reveal selection conditions the public belief on each candidate reveal and
chooses the card with the smallest weighted benefit to opponents; ties use the
lowest hand index.

## Released generations

| Generation | Profile | Liquidity | Future cash | Objective progress | Bid shading |
| --- | --- | ---: | ---: | ---: | ---: |
| v1 | aggressive | 0.75 | 0.00 | 0.25 | 0.05 |
| v1 | balanced | 0.40 | 0.00 | 0.20 | 0.25 |
| v1 | passive | 0.15 | 0.00 | 0.15 | 0.50 |
| v2 | aggressive | 0.75 | 1.50 | 0.25 | 0.05 |
| v2 | balanced | 0.40 | 0.75 | 0.20 | 0.25 |
| v2 | passive | 0.15 | 0.60 | 0.15 | 0.30 |
| v3 | aggressive | 1.00 | 1.95 | 0.15 | 0.40 |
| v3 | balanced | 0.25 | 1.55 | 0.30 | 0.35 |
| v3 | passive | 1.50 | 1.80 | 0.95 | 0.45 |

`aggressive`, `balanced`, and `passive` are live remote-capable aliases to v3.
Their public IDs are committed constants:

| Alias | Bot ID |
| --- | --- |
| aggressive | `bot_386b81bb-14df-477a-8d4c-0231cf1b3b1a` |
| balanced | `bot_265c84aa-f28e-4a35-b4de-a4f4ee406415` |
| passive | `bot_9d33c9de-4d90-4608-9a58-d2c77d93e0bd` |

Explicit `*-v1`, `*-v2`, and `*-v3` names are local simulation identities. Released
profiles and identities follow the
[immutability decision](../../../docs/architecture/immutable-bot-identities.md).

## Run and inspect

```bash
uv run garboid-simulate \
  --bots aggressive-v1,balanced-v2,passive-v3 \
  --games 1000 \
  --players 3 \
  --ruleset live-E \
  --seed 42
```

The [v1 benchmark](../../../docs/benchmarks/2026-07-28-heuristic-v1.md) and
[v2 future-cash benchmark](../../../docs/benchmarks/2026-07-29-future-cash-heuristic.md)
record frozen results. The v3 coefficients are the frozen winners that passed
their held-out promotion gates, as recorded in the concise
[v3 promotion benchmark](../../../docs/benchmarks/2026-07-30-heuristic-v3-candidate-promotions.md).
The
[visualization runbook](../../../docs/analysis/heuristic-bot-visualizations.md)
defines the 100,000-game analysis and its metrics.

## Extension points

- Add a new validated profile generation instead of mutating a release.
- Keep belief accounting pure and independently property-tested.
- Add valuation components to the explicit breakdown rather than hiding them
  in bid selection.
- Benchmark a new generation against its predecessor with fixed seeds before
  advancing unversioned aliases.
