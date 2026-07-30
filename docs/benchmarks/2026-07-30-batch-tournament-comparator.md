# Batch tournament comparator benchmark

Date: 2026-07-30

The comparator was benchmarked with all ten registered bots, charts A-E,
3/4/5-player games, and root seed 42. The optimized and scalar executions
produced identical `MonteCarloResult` values.

## Match execution

A single-process comparison on the same 1,500-game schedule isolates engine
performance from process-pool scheduling:

| Execution | Seconds | Relative speed |
| --- | ---: | ---: |
| Scalar SDK engine | 57.77 | 1.00x |
| Batch SDK engine, batch size 64 | 14.98 | 3.86x |

## Parallel batch-size selection

An eight-worker, 1,500-game end-to-end tournament without bootstrapping was
used to select the CLI default:

| Batch size | Seconds |
| ---: | ---: |
| 64 | 4.12 |
| 256 | 7.38 |

Both runs produced identical leaderboard values. Batch size 64 is the default;
`--batch-size` remains configurable for other machines and workloads.

## Plackett-Luce bootstrap optimization

On the 300-game, 20-bootstrap diagnostic workload, compiling repeated choice
sets and reusing weighted observations reduced runtime from 6.23 seconds to
3.34 seconds before the batch engine was enabled, a 1.86x end-to-end speedup.

## Full comparator run

The requested 10,000-game tournament with 200 bootstrap fits completed in
33.20 seconds using 16 workers and batch size 64. All 200 bootstrap fits
converged and no bot faults were recorded.

## Versioned-bot default at 15,000 games

The finalized default roster contains `random` and the six versioned v1/v2
heuristic bots. Unsuffixed public aliases were excluded because they share
brains with the v2 simulation identities.

Direct 16-worker measurements show near-linear planning and simulation scaling:

| Games | Planning seconds | Simulation seconds | 200-bootstrap seconds |
| ---: | ---: | ---: | ---: |
| 5,000 | 1.22 | 5.45 | 7.52 |
| 10,000 | 2.43 | 10.42 | 7.57 |
| 15,000 | 3.78 | 16.46 | 8.06 |

The 200-bootstrap measurements reused condition-stratified subsets of the same
15,000-game result. Bootstrap time grows much more slowly than simulation time
at this scale because the Plackett-Luce fitter aggregates repeated ranking
choice sets; process-pool startup and optimizer work dominate the additional
resampling loop.

For the 15,000-game run, the core measured phases total 28.41 seconds:

| Phase | Seconds | Share |
| --- | ---: | ---: |
| Planning | 3.78 | 13.3% |
| Batch simulation | 16.46 | 58.0% |
| Primary fit | 0.10 | 0.4% |
| 200 bootstrap fits | 8.06 | 28.4% |

Bootstrapping is no longer the majority of runtime. Even 500 samples took
11.42 seconds and would account for about 36% of the corresponding core run.

## Bootstrap sample stability

All requested fits converged. Smaller runs are deterministic prefixes of the
500-sample run.

| Samples | Seconds | Maximum endpoint drift from 500 | Changed overlap conclusions |
| ---: | ---: | ---: | ---: |
| 50 | 4.24 | 2.56 PL points | 0 |
| 100 | 6.14 | 1.23 PL points | 0 |
| 200 | 8.06 | 0.70 PL points | 0 |
| 500 | 11.42 | 0.00 PL points | 0 |

Two hundred samples is retained as the default. It is well inside the
10-point stability threshold, agrees with all 500-sample interval-overlap
conclusions, and saves 3.36 seconds relative to 500 samples.

## Effect of more games

Mean 95% interval width fell from 21.82 PL points at 5,000 games to 15.37 at
10,000 and 11.67 at 15,000. This is broadly consistent with uncertainty
shrinking near the inverse square root of game count.

The broad performance tiers did not change across the stratified 5,000,
10,000, and 15,000 samples. `balanced-v2` and `aggressive-v2` did exchange the
top position: aggressive-v2 led at 5,000 and 10,000 games, while balanced-v2
led by 3.55 points at 15,000. Their 15,000-game intervals still overlap, so
more games could meaningfully change their ordering but are unlikely to alter
the clearly separated v2, v1, and random tiers.

Absolute ratings should not be compared directly with the earlier ten-bot run
because removing duplicated public aliases changed the fitted comparison set
and its mean-rating normalization.
