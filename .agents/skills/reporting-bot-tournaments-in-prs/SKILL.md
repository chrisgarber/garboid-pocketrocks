---
name: reporting-bot-tournaments-in-prs
description: Require a concise, evidence-backed tournament results overview in the pull request body whenever a change introduces a new runnable bot version or generation. Use when preparing, creating, updating, or reviewing a PR that adds a versioned bot identity, policy, configuration, coefficients, or checkpoint, or advances a latest alias to a newly preserved version.
---

# Report Bot Tournaments in PRs

Ensure reviewers can judge a new bot generation from the PR body without opening raw tournament artifacts.

## Determine whether the requirement applies

Compare the complete PR diff with its merge base. Treat the PR as introducing a new bot version when it adds a runnable generation or identity through versioned code, policy/configuration, coefficients, a checkpoint, registry entry, simulation name, or a latest alias advanced to a newly preserved version.

Do not trigger for test-only fixtures, documentation-only examples, pure renames, or behavior-preserving refactors that add no runnable version. When behavior itself is changing, also follow `$versioning-bots`.

## Gather tournament evidence

Require a fixed-seed tournament containing both the new version and its direct predecessor. Prefer the committed benchmark note and its `summary.json`; use the tournament runbook at `src/garboid_pocketrocks/tournament/README.md` when evidence must be generated.

Verify that the result identifies the new version exactly and records:

- overall rank and field size;
- Plackett-Luce rating and 95% interval;
- appearances, outright win rate, mean final money, and faults;
- total tournament games, root seed, player counts, and charts;
- the direct comparison with the preceding version;
- a durable evidence link and reproduction command.

Never infer metrics from partial logs or describe a bot as stronger merely because its point estimate is higher. State interval overlap and other limits plainly. Keep raw run exhaust under `artifacts/`; link the concise committed benchmark evidence from the PR.

## Put the overview in the PR body

Before creating or updating the PR, add this section with measured values:

```markdown
## Tournament results

<One to three sentences stating where the new version ranked and how it compared
with its predecessor, including uncertainty or faults that affect the conclusion.>

| Bot | Rank | PL rating (95% interval) | Appearances | Outright win rate | Mean money | Faults |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `<new-version>` | `<rank>/<field>` | `<rating> (<lower>–<upper>)` | `<games>` | `<rate>` | `<money>` | `<count>` |
| `<predecessor>` | `<rank>/<field>` | `<rating> (<lower>–<upper>)` | `<games>` | `<rate>` | `<money>` | `<count>` |

- Configuration: `<total games>` games; root seed `<seed>`; players `<counts>`; charts `<charts>`.
- Evidence: [`<benchmark note>`](https://github.com/OWNER/REPO/blob/COMMIT/path/to/benchmark.md)
  and [`summary.json`](https://github.com/OWNER/REPO/blob/COMMIT/path/to/summary.json).
- Reproduce: `<exact command>`
```

Add the field winner or another baseline row only when it materially clarifies the result. Keep the full leaderboard in the linked evidence rather than copying a large table into the PR.

## Enforce the requirement

- For PR creation or update, inspect the final body and do not submit it without the completed tournament section.
- If evidence is missing, generate it before opening the PR. Do not insert placeholders or fabricate results.
- For PR review, treat a missing, stale, or unsupported overview as a blocking finding and identify the exact evidence needed.
- If later commits change the bot or tournament evidence, refresh the overview before handoff.
