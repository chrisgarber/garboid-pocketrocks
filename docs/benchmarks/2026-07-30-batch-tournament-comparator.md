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
