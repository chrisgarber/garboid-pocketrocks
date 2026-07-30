# `vector_ppo_large_v1_g350k` held-out promotion

## Result

`vector_ppo_large_v1_g350k` passed the held-out promotion gate against
`vector_ppo_small_v1_g1500`. The comparison ran 480 matched pairs, or 960
games. The candidate's fitted rating was 380.4651404425133 points higher than
the incumbent's. Its 95% bootstrap interval was 333.59571588851003 to
429.92008317501353 points, with all 1,000 of 1,000 bootstrap fits converging.
There were no bot faults and no failed or missing games, so the report records
`promoted: true`.

In plain English, each policy took the same focal seat in a copy of the same
game, with the same chart, player count, opponents, and seed. The uncertainty
interval stayed entirely above zero after resampling those matched pairs.
This supports the large v1 checkpoint over the small v1 checkpoint on this
specific held-out corpus. It does not claim that the large policy is the best
bot overall.

This run recorded evidence only. It did not create, move, or otherwise change
any bot alias.

## Reproduction and provenance

Exact command used:

```bash
uv run --extra neural garboid-promote \
  --candidate vector_ppo_large_v1_g350k \
  --incumbent vector_ppo_small_v1_g1500 \
  --development-corpus configs/promotion/development-v1.json \
  --held-out-corpus configs/promotion/held-out-v1.json \
  --bootstrap-samples 1000 \
  --bootstrap-seed 0 \
  --workers 8 \
  --batch-size 64 \
  --output-dir docs/benchmarks/promotions/2026-07-30-vector-ppo-large-v1-g350k-vs-small-v1-g1500
```

Run it with source commit
`5852176ff3c28b3f469a85a349be40ce41c05aa8` checked out and an empty output
directory. The committed evidence now makes the path shown above nonempty, so
a reproduction must either use a fresh checkout before those artifacts were
added or replace `--output-dir` with a fresh path.

- Promotion source commit:
  `5852176ff3c28b3f469a85a349be40ce41c05aa8`
- Held-out corpus digest:
  `de686b97e9318d840554514d71158e7d30e4b1603c6692d68b73bc77947b10da`
- Development corpus digest:
  `17c016350dbe717641b8cd499b0908e3bc0faa811a3b4f5e574f8713a5bf2b3d`
- Candidate checkpoint: 349,860 games and 196 updates, trained from commit
  `154e17be349670c14342fcaa8b5dc7c7d413f760`, parameter digest
  `088160ad4006b2bac3691980d7f3e9dc56635fd57e6ad2b94068497e199f0e5c`,
  model SHA-256
  `2ff577e25cf20f4217290a18ecf3d23188d0a5cddb392aa571ba54f2e1cd8974`
- Incumbent checkpoint: 1,500 games and one update, trained from commit
  `2cb541c9922f6369fb0e2dbdc8b2372a52541abb`, parameter digest
  `4c75fa7aa08432a7f503d83d23332b0ee5d4f63f8d1e4abb3d26e02d5c0ee16a`,
  model SHA-256
  `cd584e1647363ef51b3bb51f95276acb6334a2a17fe8a4eb9c8d6cc628923862`

## Artifacts

- [`promotion-report.json`](promotions/2026-07-30-vector-ppo-large-v1-g350k-vs-small-v1-g1500/promotion-report.json):
  authoritative identities, configuration, result, and decision
- [`paired-games.jsonl`](promotions/2026-07-30-vector-ppo-large-v1-g350k-vs-small-v1-g1500/paired-games.jsonl):
  all 960 ordered game summaries
- [`corpus-snapshot.json`](promotions/2026-07-30-vector-ppo-large-v1-g350k-vs-small-v1-g1500/corpus-snapshot.json):
  the normalized development and held-out cases used by the run
