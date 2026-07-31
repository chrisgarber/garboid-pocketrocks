# Heuristic v3 candidate promotions

## Decision

The aggressive, balanced, and passive v3 candidates all passed the same
held-out promotion gate against their v2 predecessors. The decision was
all-or-nothing: the three candidates would move forward together only if every
candidate completed every held-out game, had no faults, and had a 95%
confidence interval entirely above zero. All three met those conditions, so
the combined decision is **pass**.

This benchmark retains the durable summary of that decision. It does not
itself change a released bot name, alias, or registry entry.

## Development search

The coefficient searches ran from source commit
`5fb33de9734234ce0902bf79b85a75c3a5585c23`. Each search evaluated 96
candidates across eight generations. That was 23,040 candidate games plus 240
baseline games on `development-v1`, whose digest was
`17c016350dbe717641b8cd499b0908e3bc0faa811a3b4f5e574f8713a5bf2b3d`.
These development results selected candidates for the final exam; they were
not promotion results.

| Personality | Search | Selected candidate | Development rating delta | Normalized finish delta | Final-money delta |
| --- | --- | --- | ---: | ---: | ---: |
| Aggressive | `aggressive-v3-search-v1` | `aggressive-v3-candidate-g007-s008-c70e11540db9` | 252.83828743380695 | 36.58333333333334 | 3609 |
| Balanced | `balanced-v3-search-v1` | `balanced-v3-candidate-g006-s010-e3971899626c` | 176.30500419887812 | 23.25 | 1991 |
| Passive | `passive-v3-search-v1` | `passive-v3-candidate-g006-s001-812832214cd5` | 262.3829631208198 | 32.0 | 3337 |

## Held-out results

The authoritative promotion runs used source commit
`7a59af3e37d5124536f1f7ba7366a1953b929137`. Each run completed 480
ordered matched pairs, or 960 games, covering charts A through E at three,
four, and five players. All 1,000 requested bootstrap fits converged. No run
recorded a bot fault, warning, failure, or missing game, and every report
records `promoted: true`.

| Personality | Candidate versus incumbent | Rating delta | 95% interval | Pairs / games | Bootstrap | Clean? |
| --- | --- | ---: | --- | --- | --- | --- |
| Aggressive | `aggressive-v3-candidate-g007-s008-c70e11540db9` versus `aggressive-v2` | 231.9549699979907 | 191.07134861693405 to 280.1109053312352 | 480 / 960 | 1000 / 1000 | Yes |
| Balanced | `balanced-v3-candidate-g006-s010-e3971899626c` versus `balanced-v2` | 143.35885513014068 | 102.76507930109351 to 188.82823172726077 | 480 / 960 | 1000 / 1000 | Yes |
| Passive | `passive-v3-candidate-g006-s001-812832214cd5` versus `passive-v2` | 303.80330913850275 | 256.50646874507333 to 360.10282152090446 | 480 / 960 | 1000 / 1000 | Yes |

The shared held-out corpus was `held-out-v1`, with digest
`de686b97e9318d840554514d71158e7d30e4b1603c6692d68b73bc77947b10da`.
Within each pair, the candidate and incumbent occupied the same focal seat
against the same opponents, chart, player count, and engine seed. The positive
lower bounds therefore compare each candidate with its own predecessor on the
same unseen matrix.

## Exact commands

The balanced and passive commands each ran once. The aggressive successful
invocation ran once after a zero-game sandbox failure; no held-out outcome had
been exposed. No invocation used `--overwrite`, and there was no statistical
retry after observing an outcome.

```bash
uv run garboid-promote \
  --candidate aggressive-v3-candidate-g007-s008-c70e11540db9 \
  --incumbent aggressive-v2 \
  --development-corpus configs/promotion/development-v1.json \
  --held-out-corpus configs/promotion/held-out-v1.json \
  --bootstrap-samples 1000 \
  --bootstrap-seed 0 \
  --workers 4 \
  --batch-size 64 \
  --output-dir artifacts/promotions/aggressive-v3-vs-v2
```

```bash
uv run garboid-promote \
  --candidate balanced-v3-candidate-g006-s010-e3971899626c \
  --incumbent balanced-v2 \
  --development-corpus configs/promotion/development-v1.json \
  --held-out-corpus configs/promotion/held-out-v1.json \
  --bootstrap-samples 1000 \
  --bootstrap-seed 0 \
  --workers 4 \
  --batch-size 64 \
  --output-dir artifacts/promotions/balanced-v3-vs-v2
```

```bash
uv run garboid-promote \
  --candidate passive-v3-candidate-g006-s001-812832214cd5 \
  --incumbent passive-v2 \
  --development-corpus configs/promotion/development-v1.json \
  --held-out-corpus configs/promotion/held-out-v1.json \
  --bootstrap-samples 1000 \
  --bootstrap-seed 0 \
  --workers 4 \
  --batch-size 64 \
  --output-dir artifacts/promotions/passive-v3-vs-v2
```

To reproduce a run, use the named source commit in a fresh checkout and choose
a fresh path under `artifacts/`. The promotion command deliberately refuses to
reuse a nonempty directory.

## Frozen provenance

The packaged frozen-candidate catalog binds each runnable candidate to its
development search and frozen profile. All three name `development-v1`, its
digest shown above, and search source commit
`5fb33de9734234ce0902bf79b85a75c3a5585c23`.

| Personality | Freeze digest | Profile digest | Manifest digest |
| --- | --- | --- | --- |
| Aggressive | `218e9682d8d174125d4b9e7550fec9afda01ddb4433084143968b6d525d335da` | `c70e11540db92d0c77ce5085670ff48105c91aede1ed52c4abb7874a64687b58` | `627eb77836f8dceace745a8fb7f60573e2dad05aa47a423d902850c32a98f5e0` |
| Balanced | `05bcd898e7fc79062585cb989b67cb2e5641eed6cc59b1a60255de84c8ee2988` | `e3971899626ca3f651b2992d0cc429dc3ffd57fcdbb7cfac8249e6f0f9d9b03e` | `da9e2162eec9dd934dc80e59d9950b49c74a3a4cd4d72e6273134b502e705152` |
| Passive | `0617870c8641e9d25237354b5fe5a1df4f15637af0e46e8c11b1c13e7054adee` | `812832214cd5a16115104c50d33e94cba9929a3cc355b4779d6002b52b25e734` | `bf533a434a4208e7b018606c53488fcc3a09499b6da2fcb4b1d020346001a9c1` |

## Artifact retention

The complete search and promotion receipts—including per-game JSONL,
candidate-evaluation streams, bootstrap samples, and expanded corpus
snapshots—were intentionally not retained in Git. They are deterministic
working data that can be regenerated from the source commits, versioned
manifests and corpus recipes, and commands above.

Git retains this aggregate decision, the three search manifests, and the
packaged frozen candidates needed to reproduce or run the released generation.
The zero-game sandbox failure was discarded because it contained no measured
evidence.
