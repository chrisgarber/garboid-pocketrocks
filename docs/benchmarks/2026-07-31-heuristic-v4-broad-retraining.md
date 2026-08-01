# Heuristic v4 broad-field retraining

Date: 2026-07-31

The phase-aware v4 search was rerun from repository commit
`b306d77de634efba21542b18589946a3fd8fc703` after binding each personality to
the personality-specific broad development corpus introduced for v3. Each
initial v4 genome repeated the released v3 profile in all three phases, so the
search had to improve from the current predecessor while facing the broader
opponent rotation. The prior narrow-field v4 candidates were replaced in
place, as requested; their raw committed search and promotion receipts were
removed.

## Frozen replacements

| Personality | Replacement identity | Broad development rating delta | Finish delta | Money delta |
| --- | --- | ---: | ---: | ---: |
| Aggressive | `aggressive-v4-candidate-g011-s014-9a2908cce71c` | +129.7861 | +28.1667 | +1879 |
| Balanced | `balanced-v4-candidate-g005-s010-ae48ac912b3a` | +12.5529 | +2.1667 | +100 |
| Passive | `passive-v4-candidate-g011-s012-fcf5cb322e51` | +71.5911 | +15.5833 | +721 |

These are development-selection measurements, not final-exam results. Complete
search receipts remain local under `artifacts/evolution/` and are not committed.

## Held-out comparisons with v3

Each comparison used 960 paired games, 1,000 deterministic bootstrap samples,
the matching personality-specific held-out broad corpus, and completed with no
bot faults.

| Personality | Rating difference | 95% interval | Gate |
| --- | ---: | ---: | --- |
| Aggressive | +130.5532 | +100.3683 to +161.9410 | Pass |
| Balanced | -6.9635 | -18.0634 to +4.2704 | Inconclusive |
| Passive | +6.5604 | -19.1844 to +30.1406 | Inconclusive |

The report SHA-256 digests are, respectively,
`291a6fef74a2dd693df42d7c708e50adcea87e922672897432a9db7d7f11c1fe`,
`ea9538d6096709d154f9e4478df26b6b4bfb8c958829ffdb735ae364700d0c09`,
and `0afa54d69a4a220dc04d45681fa9d23442361501f5b284e0e09917049bdb75b3`.
Raw reports remain local under `artifacts/promotions/`.

## Default field plus v4 tournament

The deterministic tournament used seed 0, 15,000 games, charts A through E,
three through five players, 200 bootstrap samples, the 13 curated default bots,
and all three replacement v4 candidates. All 16 bots completed with zero
faults.

| Bot | Rank | Rating | 95% interval |
| --- | ---: | ---: | ---: |
| Aggressive v3 | 3 | 1571.58 | 1563.27 to 1579.58 |
| Aggressive v4 | 5 | 1564.87 | 1556.69 to 1572.67 |
| Balanced v3 | 10 | 1480.97 | 1474.03 to 1488.73 |
| Balanced v4 | 11 | 1478.89 | 1472.39 to 1486.11 |
| Passive v4 | 14 | 1386.28 | 1378.64 to 1394.38 |
| Passive v3 | 15 | 1349.84 | 1342.73 to 1357.51 |

The held-out aggressive result and the full-field ordering answer different
questions: the paired gate isolates v4 versus v3 in its fixed broad opponent
matrix, while the tournament estimates global strength across a 16-bot field.
In the tournament, aggressive v3 and v4 overlap, balanced v3 and v4 are nearly
indistinguishable, and passive v4 is clearly stronger than passive v3.

The tournament summary SHA-256 is
`ad13c6bbe0f6d6c6bf153c075847354d7be013bcd574b8824e3b8891c36c4f39`.
The local interactive report is
`artifacts/tournaments/2026-07-31-default-plus-v4-broad/report.html`.
