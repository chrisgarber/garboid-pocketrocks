# Heuristic v4 candidate promotions

## Decision

The aggressive, balanced, and passive phase-aware v4 candidates each took
the held-out final exam once against their matching v3 predecessor. All three
runs completed cleanly, but none produced a 95% uncertainty interval entirely
above zero. The point estimates were also slightly negative.

The release rule was all-or-nothing: all three candidates had to complete the
full held-out matrix without faults and show a reliably positive rating
advantage. Because every interval included zero, the combined decision is
**fail**. No v4 candidate, latest alias, or default tournament entry advances;
the three v3 identities remain unchanged and released.

These were one-shot statistical decisions. There was no rerun, overwrite, or
retuning after seeing the held-out results.

## Evidence lineage

The development searches ran from clean source commit
`a66c49e559849b35a290827b51b2e5098524e2d1`. The held-out promotion
commands ran from source commit
`109d0602ab035df82b382b92f4a63a133617b5c1`.

All candidates were selected on `development-v1`, digest
`17c016350dbe717641b8cd499b0908e3bc0faa811a3b4f5e574f8713a5bf2b3d`,
before the held-out evidence was opened. The final exams used
`held-out-v1`, digest
`de686b97e9318d840554514d71158e7d30e4b1603c6692d68b73bc77947b10da`.

| Personality | Frozen candidate | Predecessor |
| --- | --- | --- |
| Aggressive | `aggressive-v4-candidate-g011-s004-000d194163fa` | `aggressive-v3` |
| Balanced | `balanced-v4-candidate-g009-s000-4d391ce068d7` | `balanced-v3` |
| Passive | `passive-v4-candidate-g005-s005-cf4f7b924ee3` | `passive-v3` |

## Unchanged public phase selector

The selector is the same for all three candidates and uses only the public
resource horizon. Let `R` be the number of future biddable resources after
the current award and `T` the total number of biddable resources:

- select **early** when `3R >= 2T`;
- otherwise select **middle** when `3R >= T`;
- otherwise select **late**.

The selector kind is `public-resource-horizon-v1`. Its evidence is the
[phase-boundary report](2026-07-30-heuristic-v4-phase-boundaries.md), SHA-256
`9961f26f32270dcebc98df443588e96cbde2f953858cd131c66a37aeecaa9b01`,
and the
[supporting slices](tournaments/2026-07-30-heuristic-v3-phase-boundaries-development/phase-boundary-slices.csv),
SHA-256
`4f8aa60edf31b28c746cb8004a4dd5468ee8ab1b26462550c914b2e3fa50d7ae`.

## Frozen coefficient profiles

These are the 12 coefficients frozen for each personality. The linked frozen
candidate files are the authoritative machine-readable profiles.

### Aggressive

[Frozen profile](evolution/aggressive-v4-search-v2/frozen-candidate.json)

| Phase | Liquidity strength | Future-cash weight | Objective-progress weight | Bid shading |
| --- | ---: | ---: | ---: | ---: |
| Early | 0.9 | 1.95 | 0.15 | 0.6 |
| Middle | 1 | 1.9 | 0.1 | 0.4 |
| Late | 1.4 | 2 | 0.15 | 0.5 |

### Balanced

[Frozen profile](evolution/balanced-v4-search-v2/frozen-candidate.json)

| Phase | Liquidity strength | Future-cash weight | Objective-progress weight | Bid shading |
| --- | ---: | ---: | ---: | ---: |
| Early | 0.25 | 1.35 | 0.3 | 0.35 |
| Middle | 0.3 | 1.55 | 0.35 | 0.35 |
| Late | 0.45 | 1.45 | 0.25 | 0.35 |

### Passive

[Frozen profile](evolution/passive-v4-search-v2/frozen-candidate.json)

| Phase | Liquidity strength | Future-cash weight | Objective-progress weight | Bid shading |
| --- | ---: | ---: | ---: | ---: |
| Early | 1.5 | 1.8 | 0.95 | 0.45 |
| Middle | 1.5 | 1.75 | 0.95 | 0.45 |
| Late | 1.5 | 2 | 0.95 | 0.4 |

## Development selection and phase outcomes

The
[development-winner summary](2026-07-30-heuristic-v4-development-winners.md)
records the full search contract and development deltas. In each winner's
240-game diagnostic run, the fixed selector chose an expert 3,730 times:
1,272 early, 1,238 middle, and 1,220 late.

| Personality | Phase | Selected-expert decisions | Eventual final-money sum | Eventual normalized-finish sum | Outright-win decisions | Tied-first decisions | Faulted-seat decisions |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Aggressive | Early | 1,272 | 100,979 | 1189.8333333333333 | 1,050 | 11 | 0 |
| Aggressive | Middle | 1,238 | 95,568 | 1149.0833333333333 | 1,003 | 18 | 0 |
| Aggressive | Late | 1,220 | 94,118 | 1139.8333333333333 | 988 | 16 | 0 |
| Balanced | Early | 1,272 | 97,481 | 1177.6666666666667 | 1,005 | 29 | 0 |
| Balanced | Middle | 1,238 | 91,322 | 1126.5 | 946 | 19 | 0 |
| Balanced | Late | 1,220 | 90,824 | 1123.4166666666667 | 949 | 18 | 0 |
| Passive | Early | 1,272 | 103,481 | 1204.75 | 1,092 | 9 | 0 |
| Passive | Middle | 1,238 | 98,147 | 1167.25 | 1,050 | 8 | 0 |
| Passive | Late | 1,220 | 95,994 | 1147.0 | 1,021 | 7 | 0 |

These are development diagnostics, not held-out promotion results. They
explain what the selected policies did before the final exam.

## One-shot held-out results

Each run covered charts A through E at three, four, and five players. Every
run completed all 480 requested matched pairs and 960 requested games. All
1,000 bootstrap fits converged, and all three reports recorded zero candidate,
incumbent, opponent, and unattributed faults. There were no warnings or
missing games.

| Personality | Rating delta | 95% interval | Pairs | Games | Bootstrap | Faults | Result |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| Aggressive | -3.3070227531331966 | -38.19114288857132 to 30.14083215446108 | 480 / 480 | 960 / 960 | 1000 / 1000 | 0 | Fail |
| Balanced | -7.903601047803477 | -17.151275621918607 to 0.11911328931605498 | 480 / 480 | 960 / 960 | 1000 / 1000 | 0 | Fail |
| Passive | -0.09074482590222033 | -27.15037852540195 to 29.972291195257924 | 480 / 480 | 960 / 960 | 1000 / 1000 | 0 | Fail |

For each candidate, the recorded failure is
`interval_includes_zero`: the evidence does not show a reliably positive
advantage over v3. A clean run is necessary for promotion, but it is not
sufficient; the lower confidence bound must also be positive.

## Direct evidence and digests

The common held-out corpus snapshot is byte-identical across the three runs,
with SHA-256
`6122d00ba4995580c3f2c4642be8e1a045f371c5b527f8fe967d3f284549cb0d`.

### Aggressive

- [Promotion report](promotions/2026-07-30-aggressive-v4-candidate-g011-s004-000d194163fa-vs-aggressive-v3/promotion-report.json):
  `c3f6faa6f8d70b387d3962e66fc96bdaa6385edb9a3cf40ed40e72cf04689878`
- [Paired games](promotions/2026-07-30-aggressive-v4-candidate-g011-s004-000d194163fa-vs-aggressive-v3/paired-games.jsonl):
  `c426336cff83e6c7d14e99bf668b4e0560ba17a1ffb42d258ca64e24042be07d`
- [Development search report](evolution/aggressive-v4-search-v2/search-report.json):
  `05a77d4bdc177b0aa8b84c43e9ef364ab51533adebd9f2dc67ac3b1da85473bb`
- Frozen/profile/manifest digests:
  `2b63de8569efa4922cbb751ac052a20e46ca2accfdb3b0091d2f67d9858690dd`,
  `000d194163fac76a1e2928631379d7aab9b308025d6edb7583110cd962736b04`,
  `71c06a1a246e81c935156ff818f66dcac454719168fabdd4af63ba94249ca69b`

### Balanced

- [Promotion report](promotions/2026-07-30-balanced-v4-candidate-g009-s000-4d391ce068d7-vs-balanced-v3/promotion-report.json):
  `e61dad7b0116d5a26286169159b0a0fc0d81bb6d055dcf8f3c4c93e2f71eb30b`
- [Paired games](promotions/2026-07-30-balanced-v4-candidate-g009-s000-4d391ce068d7-vs-balanced-v3/paired-games.jsonl):
  `71332fde536a5ac6cac5d8d77ae634712cc1e71f3becc9a9937b8e6b69c65ea3`
- [Development search report](evolution/balanced-v4-search-v2/search-report.json):
  `3c84573a97def0068bc417714232d8c7870a331029037aede73235c8d7b6efab`
- Frozen/profile/manifest digests:
  `126fbbd3d7d20dc66a239c0e7608365352c5077fee81c6b0d88c4410c5b28df3`,
  `4d391ce068d794767aff27aaa2782a63f57255402d41fe3ee7b0196edaed036e`,
  `e1f1bed8f09aef9193ffeb0ed3e0be822be96df7fd69985c9e4111f5c725933c`

### Passive

- [Promotion report](promotions/2026-07-30-passive-v4-candidate-g005-s005-cf4f7b924ee3-vs-passive-v3/promotion-report.json):
  `212d275550ea659bb9be7a70d3fd01f53c51f85ee40dfa16c947d3623bc2e1b8`
- [Paired games](promotions/2026-07-30-passive-v4-candidate-g005-s005-cf4f7b924ee3-vs-passive-v3/paired-games.jsonl):
  `f243a0bf900d7dbf8186b12d563fbfe62b57c7dadceb2d9b426b3c6270452a3e`
- [Development search report](evolution/passive-v4-search-v2/search-report.json):
  `dc6f291668c934bf3f16028d1fb5a03d4b36f4ae34f5209142724202d5fbd78c`
- Frozen/profile/manifest digests:
  `36285933ff9a36b45004a5cfd14dd828a7ebda10c6bf34eb87d7511ae8d68f84`,
  `cf4f7b924ee3759d05eff38f47340951fb51c55827e469d8ea96a14e3cd4ccc4`,
  `334579f896a0d4281c8926bb4cc5d9bffd9b3c63b8be3d0ae3375699792d4bc6`

The promotion reports also bind the candidate-evaluation, selection-log,
development-game, and three winner-diagnostic digests from the development
searches. Those nested bindings make each held-out decision traceable back to
the exact one-shot frozen candidate.

One of those three historical diagnostic digests names
`winner-decision-slices.csv`. The detailed slice bytes were withheld after
review because their high-dimensional grouping left most rows with a single
contributing decision that could be linked to reproducible development-game
seeds. The digest remains as a tombstone so the frozen and promotion records
stay immutable. Each development search publishes the safe three-row
`winner-phase-outcomes.csv` projection of its existing
`winner-diagnostics.json` and a `privacy-redaction.json` mapping instead. No
simulation was rerun. The redaction changed neither the development winner nor
any held-out result reported above.
