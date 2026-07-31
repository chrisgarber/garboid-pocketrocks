# Broad-field heuristic v3 retraining

Date: 2026-07-31

The v3 evolution searches were rerun after replacing the overall-rating-first
objective with a robust broad-field objective. Each personality now trains on
240 matched games against eight identities: random, fixed-bid, SDK greedy, all
three v1 heuristics, and both non-focal v2 heuristics. Selection first maximizes
the worst challenger-specific normalized-finish improvement, then overall
Plackett-Luce rating, finish, and money. A candidate freezes only when every
challenger slice and the overall rating improve on its matching v2 incumbent.

## Development search

All three deterministic searches evaluated 96 candidates and completed 23,040
candidate games plus 240 reusable baseline games without faults or missing
evidence. Source commit: `70c4bb87959a8012524e5e7460adb058ce0b78e7`.

| Personality | Selected candidate | Coefficients (liquidity, future cash, objective, shading) | Worst challenger finish delta | Overall rating delta |
| --- | --- | --- | ---: | ---: |
| Aggressive | `aggressive-v3-candidate-g007-s009-9c43f610b2f0` | `1.40, 1.05, 0.70, 0.30` | 0.1067 | 153.82 |
| Balanced | `balanced-v3-candidate-g005-s005-90544d0f26d2` | `1.30, 1.90, 0.00, 0.05` | 0.0494 | 62.18 |
| Passive | `passive-v3-candidate-g006-s000-739e30e8d844` | `1.40, 1.10, 0.40, 0.30` | 0.1051 | 136.71 |

## Held-out promotion

Each frozen candidate received one 480-pair held-out comparison against its
matching v2 predecessor. The held-out fields introduced both PPO policies and
used seeds disjoint from development. All 1,000 bootstrap fits converged and no
bot faulted.

| Personality | Rating difference | 95% interval | Decision |
| --- | ---: | ---: | --- |
| Aggressive | 85.51 | 53.15 to 123.38 | Promoted in place to v3 |
| Balanced | 23.21 | -9.11 to 53.76 | Not promoted; existing v3 retained |
| Passive | 81.41 | 46.75 to 114.62 | Promoted in place to v3 |

Promotion report SHA-256 digests are `7db14301e18c1a3fa9fe9e58aff0bd84fd3f14d5d92b4756eae501f74dbf7dd2`,
`12a118df46f86da2b84ce46a0b5b94f43a267411a4022d606293024ad4179f08`,
and `b91222afc9fd1bdf433261d432306d7149ad2c20351ca0491d6a28a74048f209`
for aggressive, balanced, and passive respectively.

## Full tournament

The released set was evaluated over 15,000 games, all charts, three through
five players, seed 0, and 200 bootstrap samples. There were no faults and all
200 bootstrap fits converged.

| Rank | Bot | Rating | 95% interval |
| ---: | --- | ---: | ---: |
| 1 | aggressive-v3 | 1706.30 | 1697.60 to 1714.10 |
| 2 | passive-v3 | 1697.83 | 1690.07 to 1706.95 |
| 3 | fixed-bid | 1680.00 | 1669.63 to 1690.27 |
| 4 | aggressive-v2 | 1675.33 | 1667.35 to 1682.36 |
| 5 | balanced-v2 | 1658.73 | 1649.30 to 1667.40 |
| 6 | balanced-v3 | 1649.73 | 1642.36 to 1656.37 |
| 7 | vector_ppo_large_v1_g350k | 1621.82 | 1612.35 to 1631.53 |
| 8 | passive-v2 | 1619.42 | 1610.33 to 1626.48 |

The tournament summary SHA-256 is
`12ece75bb1a009cc8bd87f71c76f4a0aa1b1316543ee770598bb49091d4c9c66`.

## Behavior check

Across 1,099,635 captured decisions, aggressive-v3 submitted slightly larger
auction bids than passive-v3: 3.77 versus 3.73 in Auction 1 and 7.02 versus
6.92 in Auction 2. Balanced-v3 was lower at 3.61 and 6.48. Fixed-bid retained
its distinctive target-driven behavior at 4.61 and 8.47 with lower submission
rates when cash or legal maxima constrained its fixed targets.

The aggressive label is directionally correct, but the promoted passive policy
has converged close to aggressive behavior and is no longer the most passive of
the three. Broad opponent coverage fixed the competitive overfitting identified
in the prior tournament, but it did not preserve personality separation. Style
constraints or a style-aware secondary objective should be treated as a
separate follow-up rather than inferred from competitive fitness.

The concise behavior summary SHA-256 is
`afd465e8a7b05d159b838d13e1704f8410aaf91a0f5d80c1ca248a9b3e2f04de`.
Complete search and promotion receipts remain under the gitignored
`artifacts/evolution/broad-field-v3` and `artifacts/promotions/broad-field-v3`
directories. Concise tournament ratings, summary, behavior aggregates, and the
analysis script remain under
`artifacts/tournaments/2026-07-31-broad-field-v3-full`; the reproducible 3.4 GB
raw tournament traces were removed after aggregation.
