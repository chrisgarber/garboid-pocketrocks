# `vector_ppo_large_v2_g1750k` release

## Result

`vector_ppo_large_v2_g1750k` is the immutable release name for the final
checkpoint from the cold mixed MPS 10-hour run. The rounded `g1750k` suffix
matches the existing release convention; its manifest retains the exact age
of 1,764,480 games and 919 PPO updates.

In the fixed-seed current-main comparison, the policy ranked fourth of nine at
1522.05 (95% bootstrap interval 1515.89–1527.68), with 2,051 outright wins in
6,666 appearances, mean final money of 48.15, and no faults. Its direct
predecessor `vector_ppo_large_v1_g350k` ranked eighth at 1448.69
(1442.74–1455.25), with 1,282 outright wins in 6,667 appearances, mean final
money of 44.18, and no faults. The 73.36-point rating difference and
non-overlapping marginal intervals support v2 over v1 in this field. This is a
multiplayer field comparison, not a matched-pair promotion test.

`monte-the-bookie-v1` ranked first at 1581.36 (1576.52–1588.18), narrowly ahead
of `surplus-v10` at 1580.26 (1573.76–1586.82); those intervals overlap. The
comparison ran 15,000 games at root seed `2026080311`, covering player counts
three through five and charts A through E with 200 bootstrap samples.

In the 15,000-game strong-field diagnostic tournament, the policy ranked third
of seven with a Plackett-Luce rating of 1538.59 (95% bootstrap interval
1533.12–1543.87), 3,068 outright wins in 8,572 appearances, mean final money
of 49.37, and no bot faults. `surplus-v10` ranked first at 1597.92 and
`fixed-objective-overlay-v3` ranked second at 1571.38. The tournament used root
seed 2026080215 and 200 bootstrap samples.

## Provenance

- Training source commit: `0ca22b62f14b3185920f4ba61591fcc0068470f9`
- Training root seed: `2026080203`
- Parameter digest: `896f83b082ddc0f2c8f0045f5f9066d89f56e796d43a3706b507153b10749ba8`
- Model SHA-256: `92a2855c74e1d0a2f568230e70cefc66bb2f65f56cec8cfffd69de800eeccbfb`
- Supported field: value charts A–E with three, four, or five players
- Current-main comparison source commit: `1d13d673ee21df134ec548dd0a471ff49a197805`
- Comparison `summary.json` SHA-256:
  `e35607e2d028f19b456fbe3188e21c6abace0374f915e6bb54ab03e4ef43eaf7`

The authoritative compact comparison is
[`summary.json`](tournaments/2026-08-03-vector-ppo-large-v2-g1750k-current-main/summary.json).
Reproduce it to a fresh ignored artifact directory with:

```bash
uv run --extra neural garboid-tournament \
  --games 15000 \
  --players 3,4,5 \
  --charts A,B,C,D,E \
  --seed 2026080311 \
  --workers 16 \
  --batch-size 64 \
  --bootstrap-samples 200 \
  --bots vector_ppo_large_v2_g1750k,vector_ppo_large_v1_g350k,monte-the-bookie-v1,surplus-v10,fixed-objective-overlay-v3,aggressive-v2,balanced-v3,passive-v3,fixed-bid-tuned-v1 \
  --output-dir artifacts/tournaments/vector-ppo-large-v2-g1750k-pr-current-main-15000
```

The interactive diagnostic report remains a local generated artifact at
`artifacts/tournaments/cold-mixed-mps-10h-v2/final-15000-diagnostics/insights.html`.
Its raw multi-gigabyte event streams were intentionally discarded after the
self-contained report was generated.
