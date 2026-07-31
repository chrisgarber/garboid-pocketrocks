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
2. Remove any configured opponent whose name or bot ID matches the candidate
   or incumbent.
3. Require enough remaining distinct opponents to fill the largest requested
   game.
4. Expand each held-out case deterministically from the filtered pool using
   the existing rotation inputs: repetition, chart position, player count,
   and focal seat.
5. Put the candidate in the focal seat of one game and the incumbent in the
   same seat of its twin. Every non-focal seat uses the same effective
   opponent in both games.

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

## Reproducibility and Evidence

Filtering is a deterministic function of:

- the committed corpus recipe and digest;
- the candidate name and bot ID;
- the incumbent name and bot ID;
- the repository source commit.

The promotion evidence must expose the effective comparison, not only the
unfiltered eligible pool. The corpus snapshot or promotion report records the
compared identities, the excluded pool identities, and the final effective
opponent lineup for every held-out case. `paired-games.jsonl` continues to
record the identities that actually ran in every seat.

Repeating a run with the same source, corpora, candidate, and incumbent must
produce byte-identical effective plans and artifacts.

## Failure Behavior

Planning fails closed when filtering leaves fewer distinct opponents than a
requested player count requires. The failure uses a stable,
plain-English reason code such as `insufficient_eligible_opponents`.

The existing candidate/incumbent identity-collision failure remains.
Candidate/opponent and incumbent/opponent collisions are no longer terminal
when the collision comes from the configured eligible pool; those entries are
deterministically excluded instead. Registry mismatches, duplicate opponent
identities, missing games, changed identities, bot faults, and statistical
failures retain their existing fail-closed behavior.

## Testing Strategy

Implementation follows test-driven development.

Corpus tests first assert that both committed recipes contain exactly the
seven ordered opponents and no random bot, while retaining disjoint,
deterministic seeds.

Planning tests then prove:

- candidate and incumbent matches are excluded by either name or bot ID;
- unrelated v1, v2, and PPO opponents remain eligible;
- filtering and case rotation are deterministic;
- every case has distinct non-focal identities;
- candidate and incumbent twins have identical non-focal lineups;
- the effective plan never places candidate or incumbent in a non-focal seat;
- too few remaining opponents fail closed;
- unrelated registry identity errors are still rejected.

Integration and CLI tests exercise at least one comparison where the
candidate or incumbent is present in the configured pool. Artifact tests
assert that the exact effective exclusions and lineups are recorded and
reproducible.

The promotion runbook explains that configured names form an eligible pool,
that compared identities are removed automatically, and that held-out means
unseen cases and seeds rather than unseen opponent policies.

## Future Corpus Versions

Once v1 is released, it remains immutable. Future corpus versions may become
harder as stronger bots are released, but each version preserves its own
recipe, seeds, and interpretation. Results from different corpus versions are
reported against their named standards rather than treated as identical
experiments.
