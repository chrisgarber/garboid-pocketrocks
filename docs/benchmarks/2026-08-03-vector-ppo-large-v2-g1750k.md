# `vector_ppo_large_v2_g1750k` release

## Result

`vector_ppo_large_v2_g1750k` is the immutable release name for the final
checkpoint from the cold mixed MPS 10-hour run. The rounded `g1750k` suffix
matches the existing release convention; its manifest retains the exact age
of 1,764,480 games and 919 PPO updates.

In the 15,000-game strong-field diagnostic tournament, the policy ranked third
of eight with a Plackett-Luce rating of 1538.59 (95% bootstrap interval
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

The interactive diagnostic report remains a local generated artifact at
`artifacts/tournaments/cold-mixed-mps-10h-v2/final-15000-diagnostics/insights.html`.
Its raw multi-gigabyte event streams were intentionally discarded after the
self-contained report was generated.
