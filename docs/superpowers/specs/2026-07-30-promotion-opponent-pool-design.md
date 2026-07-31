# Strong Promotion Opponent Pool Design

## Status

Approved in conversation on 2026-07-30.

## Purpose

The initial promotion corpora should test new bots against the strongest
stable opponents that already exist. The random bot is too weak to provide
useful discrimination for current candidates, while the released v1 and v2
heuristics and the frozen 350k PPO policy are deterministic, reproducible,
and materially stronger.

This change updates `development-v1` and `held-out-v1` in place because those
corpora are being introduced by the current pull request and have not been
released. It does not change any bot policy, bot identity, checkpoint, or
latest alias.

## Opponent Pool

Both v1 recipes declare the same ordered eligible opponent pool:

1. `aggressive-v1`
2. `balanced-v1`
3. `passive-v1`
4. `aggressive-v2`
5. `balanced-v2`
6. `passive-v2`
7. `vector_ppo_large_v1_g350k`

`random` is removed.

Development and held-out corpora retain their existing charts, player counts,
repetition counts, root seeds, and disjoint expanded engine seeds. Development
games may be used to tune or select a candidate. Held-out games remain the
final exam because their cases and seeds are not used for tuning. Reusing the
same frozen opponent identities in both corpora does not violate held-out
separation.

## Eligible-Pool Semantics

`opponent_names` describes an ordered pool of bots eligible to occupy
non-focal seats. It does not require every named bot to be usable in every
candidate-versus-incumbent comparison.

At promotion planning time:

1. Resolve every configured opponent to its exact registered name and bot ID.
2. Remove an opponent only when its complete `(name, bot_id)` identity equals
   the candidate or incumbent identity. A name-only or bot-ID-only match is
   inconsistent identity evidence and fails closed using the corresponding
   existing candidate/incumbent opponent-collision code.
3. Require at least `max(player_counts) - 1` remaining distinct opponents.
4. Re-expand the held-out cases in their original nested-loop order. For each
   case, calculate:

   ```text
   rotation =
     (repetition + chart_index + player_count + focal_seat)
     modulo len(filtered_opponents)
   ```

   Rotate the filtered pool left by that amount and take the first
   `player_count - 1` opponents for the non-focal seats.
5. Put the candidate in the focal seat of one game and the incumbent in the
   same seat of its twin. Every non-focal seat uses the same effective
   opponent in both games.

Filtering changes only the opponent lineups. Case IDs, case ordering, charts,
player counts, focal seats, repetition counts, and engine seeds remain exactly
the same as the committed source corpus.

This permits the intended comparisons:

- a v3 heuristic can be tuned against every frozen v1 and v2 heuristic;
- `aggressive-v3` can be promoted against `aggressive-v2`;
- `aggressive-v2` serves as the incumbent in that comparison, but is excluded
  from ordinary non-focal seats so one game never contains the same identity
  twice;
- the other v1 and v2 heuristics and the 350k PPO policy remain available as
  strong ordinary opponents.

Candidate and incumbent must still have different names and bot IDs. Distinct
configured opponents must also keep distinct names and bot IDs.

## Effective Opponent Pool and Plan

The committed corpus remains the immutable source recipe and seed schedule.
Its digest continues to identify the unfiltered recipe and source cases.
Because compared identities can change the effective opponent pool, the
source corpus digest alone does not identify the games that actually run.

Planning therefore creates two first-class immutable values:

- `EffectiveOpponentPool` records the candidate and incumbent identities, the
  ordered resolved configured pool, exact-match exclusions with
  `candidate`/`incumbent` reasons, and the ordered remaining pool.
- `PromotionPlan` records that pool, every effective case and per-seat name
  and bot ID, the candidate/incumbent twin jobs, and a canonical SHA-256
  digest over the complete plan.

`PairedGamePlan.case` is the effective case used by both jobs, not the
unfiltered source case. The canonical plan payload includes the source
held-out corpus name and digest so the relationship remains auditable.

The promotion report is the authoritative location for:

- the complete effective-opponent-pool payload;
- the effective-plan payload and digest when pair construction succeeds;
- `null` plan fields when pair construction does not succeed.

`corpus-snapshot.json` remains a candidate-independent snapshot of the
committed development and held-out source corpora. It does not mix
matchup-dependent lineups into the corpus digest. `paired-games.jsonl`
continues to record the identities that actually ran in every seat, and every
completed game is validated against the effective plan.

If filtering leaves too few opponents, the planning failure carries the
resolved pool, exclusions, and remaining pool into the failed report, with no
effective case lineups. If registry resolution fails before a trustworthy
pool exists, the report preserves the configured recipe and the existing
identity failure without claiming an effective pool.

## Reproducibility and Evidence

Filtering is a deterministic function of:

- the committed corpus recipe and digest;
- the candidate name and bot ID;
- the incumbent name and bot ID;
- the repository source commit and executed source state.

Repeating a run with the same source state, corpora, compared identities, and
complete run configuration must produce byte-identical effective plans and
artifacts. The effective plan remains byte-identical across worker counts;
the complete report is byte-identical only when worker count, batch size,
bootstrap settings, and every other recorded execution input also match.

## Failure Behavior

Planning fails closed with `insufficient_eligible_opponents` when filtering
leaves fewer than `max(player_counts) - 1` distinct opponents.

The existing candidate/incumbent identity-collision failure remains.
An exact candidate/opponent or incumbent/opponent identity is excluded from
the configured eligible pool. A partial match on only name or only bot ID
retains the existing fail-closed collision behavior. Registry mismatches,
duplicate opponent identities, missing games, changed identities, bot faults,
and statistical failures retain their existing fail-closed behavior.

## Testing Strategy

Implementation follows test-driven development.

Corpus tests first assert that both committed recipes contain exactly the
seven ordered opponents and no random bot, while retaining disjoint,
deterministic seeds.

Planning tests then prove:

- exact candidate and incumbent identities are excluded;
- name-only and bot-ID-only matches fail closed;
- unrelated v1, v2, and PPO opponents remain eligible;
- the exact filtered rotation formula produces pinned first and last lineups;
- zero-, one-, and two-exclusion plans retain all case IDs, order, charts,
  focal seats, and seeds and have pinned full-pool exposure counts;
- every case has distinct non-focal identities;
- candidate and incumbent twins have identical non-focal lineups;
- the effective plan never places candidate or incumbent in a non-focal seat;
- too few remaining opponents fail closed;
- unrelated registry identity errors are still rejected.

Integration and CLI tests exercise at least one comparison where the
candidate or incumbent is present in the configured pool. Artifact tests
assert a golden effective-plan digest, the exact effective exclusions and
lineups, and validation of `paired-games.jsonl` against that plan. Runner and
report tests cover insufficient capacity after filtering and prove that the
structured pool evidence survives that failed plan. Repeat-run and
cross-worker tests include a filtered incumbent; plan bytes remain equal
across workers while report bytes follow the existing execution-setting
contract.

The promotion runbook explains that configured names form an eligible pool,
that compared identities are removed automatically, and that held-out means
unseen cases and seeds rather than unseen opponent policies.

## Future Corpus Versions

Once v1 is released, it remains immutable. Future corpus versions may become
harder as stronger bots are released, but each version preserves its own
recipe, seeds, and interpretation. Results from different corpus versions are
reported against their named standards rather than treated as identical
experiments.

Even within one corpus version, two comparisons can use different effective
opponent pools because their candidate or incumbent identities differ.
Results are directly comparable only when the effective-plan pool and
schedule are the same; reports use the effective-plan digest rather than the
source corpus digest alone to establish that fact.

## Design Review Record

### Touchpoint 1: Approach Review

- **Tier:** standalone
- **Panel:** user/driver review in conversation
- **Input:** replace random with all frozen v1/v2 heuristics and the released
  350k PPO policy; exclude compared identities deterministically
- **Verdict:** APPROVE
- **Escalation:** none

### Touchpoint 2: Spec Review

- **Tier:** standalone
- **Panel:** architecture and testing/reproducibility reviewers
- **Input:** this specification and the existing corpus, planning, reporting,
  runner, and test boundaries
- **Verdict:** Initial CONCERN; revised specification APPROVED by both
  reviewers
- **Action taken:** require exact-pair identity filtering; fail closed on
  partial identity matches; define the exact filtered rotation formula and
  capacity rule; make the effective pool and canonical digested plan
  first-class report evidence; preserve structured evidence on capacity
  failures; tighten byte-identity requirements; and document comparison
  limits within one corpus version.
