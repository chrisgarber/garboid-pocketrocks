# Large 350k Neural Tournament Design

## Goal

Add the existing large self-play checkpoint to the standard fixed-seed
tournament and measure it against the current random, heuristic, and smoke
neural field.

The tournament bot is named `vector_ppo_large_v1_g350k`. Its inference
manifest preserves the exact training age of 349,860 completed games and the
repository commit that produced it.

## Checkpoint packaging

Export the validated training checkpoint at
`artifacts/vector_ppo_large_v1_overnight_20260730/checkpoints/vector_ppo_large_v1_g350k`
as a portable inference-only bundle under
`src/garboid_pocketrocks/neural/checkpoints/vector_ppo_large_v1_g350k`.

The committed bundle contains only `manifest.json` and `model.pt`. Optimizer
state, RNG state, rollout metrics, and the multi-gigabyte training history are
not tournament dependencies. The original smoke checkpoint remains immutable.

## Tournament integration

Generalize the existing neural tournament adapter so each explicit neural
version selects its own frozen checkpoint while sharing observation encoding,
action decoding, deterministic masked-policy inference, and per-process runtime
caching.

Expose explicit simulation specs for:

- `vector_ppo_small_v1_g1500`
- `vector_ppo_large_v1_g350k`

Add the large checkpoint to `BOT_SPECS`, `BOT_SPECS_BY_NAME`, and
`DEFAULT_TOURNAMENT_BOT_SPECS`. Do not change the existing bots or aliases.
The default field becomes:

1. random
2. aggressive-v1
3. balanced-v1
4. passive-v1
5. aggressive-v2
6. balanced-v2
7. passive-v2
8. vector_ppo_small_v1_g1500
9. vector_ppo_large_v1_g350k

The SDK batch tournament engine, scheduling, seeding, Plackett-Luce fitting,
bootstrap intervals, and HTML reporting remain unchanged.

## Tournament run

Run the standard 15,000-game tournament with seed 0, charts A-E, player counts
3-5, batch size 64, the default worker count, and 200 bootstrap samples. Write
results to a new artifact directory so the prior smoke benchmark remains
available.

Report:

- elapsed time and games per second;
- rank and Plackett-Luce rating with confidence interval;
- games, outright win rate, mean final money, and faults;
- direct comparison between the 350k and smoke neural checkpoints;
- the generated HTML report and machine-readable artifacts.

No claim that training improved play is made unless the fixed-seed tournament
result supports it.

## Failure handling

Checkpoint export and load must validate model hashes and manifest metadata.
Missing or corrupt bundles fail loudly before tournament play. Tournament bot
exceptions use the existing tournament fault policy and are reported in the
leaderboard.

## Verification

Add focused tests that verify:

- the frozen large checkpoint loads and records 349,860 completed games;
- the large neural simulation name is explicit and unique;
- both neural versions remain registered;
- the default tournament order includes the large checkpoint;
- the CLI resolves the updated default field.

Run the focused neural/tournament tests, formatting, lint, and type checking.
Then run the full fixed-seed tournament and inspect the report for faults and
throughput.

