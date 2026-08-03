# Cold mixed MPS actor/learner pilot

Date: 2026-08-02

Machine: Apple M5 Pro, 18 CPU cores, 20 GPU cores, 64 GB memory

Repository base: `4f224f17fc054b0b8694aff98191608ff5263727`

Profile: `configs/neural/cold-mixed-mps-5m-v1.json`

Profile SHA-256: `38efa4bb3c6fae5932a2a507279f139620196749d28cc111ee173ee3aa179ced`

## Workload

- cold random initialization of a new large model;
- one trainable focal seat per game;
- 50% of opponent seats sampled from `strong-field-pool-v1`, with the rest
  using a frozen copy of the current policy;
- 1,920 balanced games and about 35,500 trainable transitions per update;
- two PPO epochs, MPS minibatches of 8,192;
- eight persistent one-thread CPU actors;
- at most one update of policy lag, with pre-PPO stale-rollout guardrails of
  approximate KL <= 0.01 and clip fraction <= 0.15.

The fixed pool weights `surplus-v10` and `fixed-objective-overlay-v3` twice and
also includes `aggressive-v2`, `balanced-v3`, `passive-v3`, and
`fixed-bid-tuned-v1`.

## Results

| Run | Wall time | Updates | Games | Trainable transitions | Collection decisions/s |
|---|---:|---:|---:|---:|---:|
| synchronous MPS, minibatch 512 | 94.3 s | 1 | 1,920 | 35,464 | 1,773 |
| synchronous MPS, minibatch 8,192 | 255.0 s | 3 | 5,760 | 106,575 | 1,845 |
| guarded actor/learner, minibatch 8,192 | 254.5 s | 11 | 21,120 | 390,311 | 8,255 |

The guarded actor/learner produced 3.67x as many trainable transitions per wall
second as synchronous 8k training. CPU collection throughput improved 4.47x.
Observed actor utilization was roughly 88-98% of one core for each of eight
actors, while sampled GPU utilization repeatedly reached 80-99% during overlap.

The learner rejected two prefetched rollouts and recollected them from the
fresh policy. Across applied updates, maximum approximate KL was 0.00888 and
maximum clip fraction was 0.126. The final value RMSE was 0.660; these short-run
diagnostics establish runtime health, not competitive strength.

An intentionally unguarded diagnostic reached approximate KL 0.045 and clip
fraction 0.368 by update 9. That result motivated the pre-update guard and is
not a candidate checkpoint.

## Raw evidence

Raw outputs remain ignored under `artifacts/training/`:

- `cold-mixed-mps-baseline-512-v1/metrics.jsonl`, SHA-256
  `8edbd87ad2f05d7fc3984d1a526ef6e5e25d2977d754ba27fcfe66bbe87ac777`;
- `cold-mixed-mps-5m-v1/metrics.jsonl`, SHA-256
  `03a143946e1231938e9f6597ceb5bd06e7c1625e71cd021ed56761b0b8e10748`;
- `cold-mixed-mps-actor-learner-guarded-5m-v1/metrics.jsonl`, SHA-256
  `6b4cd7838d339d945becdfbd6f6c469b9173f4d6d2bc8396a7af16794e0150f0`.

The guarded checkpoint is a cold systems-validation artifact only. It is not a
released bot and should not replace or initialize any existing neural version.
