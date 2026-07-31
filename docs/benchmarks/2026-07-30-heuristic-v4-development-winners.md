# Heuristic v4 development winners

These are locally frozen candidates selected on the development corpus. They
are not released bots, and they have not been evaluated on the held-out
promotion corpus. The held-out evidence remains untouched, so these results
support only the next promotion-evaluation step; they are not promotion
claims.

## Shared search contract

All three searches ran from clean repository commit
`a66c49e559849b35a290827b51b2e5098524e2d1` against the 240-case
`development-v1` corpus, whose digest is
`17c016350dbe717641b8cd499b0908e3bc0faa811a3b4f5e574f8713a5bf2b3d`.

Each search used 12 generations, 16 proposals per generation, four elites,
and 240 development games per proposal. That is 192 evaluated candidates,
46,080 candidate games, and 240 cached baseline games per personality. Every
requested generation, candidate, and game completed. The search evidence
records no failures and zero candidate, incumbent, opponent, or unattributed
faults.

The phase selector is fixed and uses only the public resource horizon. Let
`R` be the number of future biddable resources after the current award and
`T` the total number of biddable resources:

- use the **early** expert when `3R >= 2T`;
- otherwise, use the **middle** expert when `3R >= T`;
- otherwise, use the **late** expert.

## Development score deltas

Each row compares the selected candidate with its corresponding v3
predecessor on the development corpus.

| Personality | Selected local candidate | Generation / slot | Rating delta | Normalized-finish delta | Final-money delta |
| --- | --- | ---: | ---: | ---: | ---: |
| Aggressive | `aggressive-v4-candidate-g011-s004-000d194163fa` | 11 / 4 | 24.713329019553612 | 2.75 | 259 |
| Balanced | `balanced-v4-candidate-g009-s000-4d391ce068d7` | 9 / 0 | 9.953994694126777 | 0.9166666666666572 | 92 |
| Passive | `passive-v4-candidate-g005-s005-cf4f7b924ee3` | 5 / 5 | 12.092948249983237 | 1.8333333333333428 | 3 |

## Selected coefficient profiles

The phases are shown in selector order. Coefficient names are written out so
the behavior being tuned is explicit.

### Aggressive

| Phase | Liquidity strength | Future-cash weight | Objective-progress weight | Bid shading |
| --- | ---: | ---: | ---: | ---: |
| Early | 0.9 | 1.95 | 0.15 | 0.6 |
| Middle | 1 | 1.9 | 0.1 | 0.4 |
| Late | 1.4 | 2 | 0.15 | 0.5 |

### Balanced

| Phase | Liquidity strength | Future-cash weight | Objective-progress weight | Bid shading |
| --- | ---: | ---: | ---: | ---: |
| Early | 0.25 | 1.35 | 0.3 | 0.35 |
| Middle | 0.3 | 1.55 | 0.35 | 0.35 |
| Late | 0.45 | 1.45 | 0.25 | 0.35 |

### Passive

| Phase | Liquidity strength | Future-cash weight | Objective-progress weight | Bid shading |
| --- | ---: | ---: | ---: | ---: |
| Early | 1.5 | 1.8 | 0.95 | 0.45 |
| Middle | 1.5 | 1.75 | 0.95 | 0.45 |
| Late | 1.5 | 2 | 0.95 | 0.4 |

## Winner diagnostic aggregates

The selected-expert counts reconcile to 3,730 decisions for each winner:
1,272 early, 1,238 middle, and 1,220 late. The retained outcome evidence is
aggregated by phase and does not contain raw per-decision traces or
reproducible game seeds.

| Personality | Phase | Selected-expert decisions | Eventual final-money sum | Eventual normalized-finish sum | Outright-win decisions | Tied-first decisions | Decisions from faulted game seats |
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

The aggregate diagnostics also reconcile all traced, sliced, and summarized
decisions: 18,025 for aggressive, 18,038 for balanced, and 18,014 for
passive. Each diagnostic run covers the same 240 games and 1,000 game seats,
with zero decisions from faulted game seats.

## Direct evidence and key digests

### Aggressive

- [Search report](evolution/aggressive-v4-search-v2/search-report.json) —
  SHA-256
  `05a77d4bdc177b0aa8b84c43e9ef364ab51533adebd9f2dc67ac3b1da85473bb`;
  manifest digest
  `71c06a1a246e81c935156ff818f66dcac454719168fabdd4af63ba94249ca69b`.
- [Frozen candidate](evolution/aggressive-v4-search-v2/frozen-candidate.json) —
  profile digest
  `000d194163fac76a1e2928631379d7aab9b308025d6edb7583110cd962736b04`.
- [Winner diagnostics](evolution/aggressive-v4-search-v2/winner-diagnostics.json) —
  SHA-256
  `83d2a3043dd16d0fc30f7f47ee775e78cbaf781d396c2b8497c1c54ad46e5e0a`;
  [aggregate decision slices](evolution/aggressive-v4-search-v2/winner-decision-slices.csv)
  SHA-256
  `bdb47a31da14ee95cddc9579b2dc477b6d6a04f99420ede12ebc7ccc3e81b07d`.

### Balanced

- [Search report](evolution/balanced-v4-search-v2/search-report.json) —
  SHA-256
  `3c84573a97def0068bc417714232d8c7870a331029037aede73235c8d7b6efab`;
  manifest digest
  `e1f1bed8f09aef9193ffeb0ed3e0be822be96df7fd69985c9e4111f5c725933c`.
- [Frozen candidate](evolution/balanced-v4-search-v2/frozen-candidate.json) —
  profile digest
  `4d391ce068d794767aff27aaa2782a63f57255402d41fe3ee7b0196edaed036e`.
- [Winner diagnostics](evolution/balanced-v4-search-v2/winner-diagnostics.json) —
  SHA-256
  `4ff4b1694b7807e39b58556a050a03d5ed77f825505dff08f70db857712e1029`;
  [aggregate decision slices](evolution/balanced-v4-search-v2/winner-decision-slices.csv)
  SHA-256
  `c6a6372898b25f26b7f34b14bca83743769492a68510f2f2f1aaf77c3f4a6e99`.

### Passive

- [Search report](evolution/passive-v4-search-v2/search-report.json) —
  SHA-256
  `dc6f291668c934bf3f16028d1fb5a03d4b36f4ae34f5209142724202d5fbd78c`;
  manifest digest
  `334579f896a0d4281c8926bb4cc5d9bffd9b3c63b8be3d0ae3375699792d4bc6`.
- [Frozen candidate](evolution/passive-v4-search-v2/frozen-candidate.json) —
  profile digest
  `cf4f7b924ee3759d05eff38f47340951fb51c55827e469d8ea96a14e3cd4ccc4`.
- [Winner diagnostics](evolution/passive-v4-search-v2/winner-diagnostics.json) —
  SHA-256
  `82822bf049202cbaed6db262c5f307e97221cfc1cf7bd34afe1db76c194344d5`;
  [aggregate decision slices](evolution/passive-v4-search-v2/winner-decision-slices.csv)
  SHA-256
  `04d94fb3d302a39517bc383cec38bcf751670ed91c220e31beb3cc9fdfbf4db1`.
