# Balanced v5 opponent-model development grid

## Decision

This development-only search did not find a balanced v5 candidate that beat
`balanced-v3`. All 18 candidates had a negative rating delta, ranging from
`-52.34363556835501` to `-23.036246981343766`. No candidate was frozen, the
`balanced-v3` aliases remain unchanged, and the held-out corpus was not run.

The run completed cleanly: all 18 proposed candidates were valid and eligible,
with zero candidate, incumbent, opponent, or unattributed faults. It completed
the shared 240-game `balanced-v3` baseline and 4,320 candidate games (240 for
each candidate), exactly matching the requested coverage.

## Fixed development grid

The search used the committed 240-case `development-v1` corpus: charts A
through E, three, four, and five players, every focal seat, and four repetitions
per seat cell. Its digest is
`17c016350dbe717641b8cd499b0908e3bc0faa811a3b4f5e574f8713a5bf2b3d`.

The fixed 18-point parameter grid combined:

- prior strength: `2`, `4`, or `8`;
- minimum public-history rounds: `2` or `3`;
- same-action-and-phase / partial-match / fallback history weights: `3/2/1`,
  `4/2/1`, or `6/3/1`.

Every candidate kept the released `balanced-v3` heuristic profile fixed and
changed only the `public-opponent-bids-v1` model configuration.

## Best development result

The highest-rated configuration was
`balanced-v5-candidate-g000-s007-7afc1b9e6724`, with opponent-model config
digest
`001d5130a98451443ec6bce9760a76dc8a01576e2551ed17b6ce16dd0aeb8f1d`:

| Setting | Value |
| --- | ---: |
| Prior strength | 4 |
| Minimum public-history rounds | 2 |
| Same-action-and-phase weight | 4 |
| Partial-match weight | 2 |
| Fallback weight | 1 |

Against `balanced-v3` on the development corpus, its rating delta was
`-23.036246981343766`, its normalized-finish delta was `-1.25`, and its
aggregate final-money delta was `+509`. The positive money delta is a separate
descriptive total; it does not establish strength, particularly when both the
primary rating measure and normalized finish were worse. The search therefore
records `complete_no_improvement` and leaves `frozen_candidate_identity` null.

## Evidence and reproduction

The compact [search manifest](evolution/balanced-v5-opponent-grid-v1/search-manifest.json)
records the fixed grid, predecessor, model, profile digest, and development
corpus digest. The original evidence was committed at `605fe9f`; its runner
implementation was at `04ef310`.

At that historical implementation commit, the run can be reproduced with:

```console
uv run garboid-evolve-heuristic \
  --manifest configs/evolution/balanced-v5-opponent-grid-v1.json \
  --development-corpus configs/promotion/development-v1.json \
  --batch-size 64 \
  --output-dir artifacts/evolution/balanced-v5-opponent-grid-v1
```

Per-game summaries, corpus snapshots, candidate-evaluation streams, and
selection logs are intentionally omitted from Git. They are run exhaust rather
than a released bot artifact; the aggregate coverage, faults, digests, result,
and no-freeze decision above are the retained record.
