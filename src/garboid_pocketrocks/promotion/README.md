# Promotion

Promotion is a fair final exam for a new bot. Development games may be used
while tuning. Held-out games are fixed games that were not used for tuning.
The candidate and incumbent play matched copies of each held-out case.

In each matched pair, the compared bot changes but the chart, player count,
focal seat, opponents, and game seed stay the same. The gate promotes the
candidate only when every game is valid and the measured advantage remains
positive across its full uncertainty range.

## Run the final exam

For the two frozen neural policies, install the neural dependencies and run:

```bash
uv run --extra neural garboid-promote \
  --candidate vector_ppo_large_v1_g350k \
  --incumbent vector_ppo_small_v1_g1500 \
  --development-corpus configs/promotion/development-v1.json \
  --held-out-corpus configs/promotion/held-out-v1.json \
  --bootstrap-samples 1000 \
  --workers 8 \
  --output-dir artifacts/promotions/neural-comparison
```

The candidate may be either a released registered bot name or the exact
identity of a candidate in the committed frozen-candidate catalog. The
incumbent and every corpus opponent must still be released registered bots.
File paths, “latest” aliases, and other names are not accepted as frozen
candidates.

A frozen candidate also carries its development evidence forward. Before the
final exam starts, the command checks the candidate itself and every
provenance field against the committed catalog record. It also checks that the
declared predecessor is the invoked incumbent, that the development-corpus
name and digest match, and that recomputing the digest from the loaded recipe
and cases gives the same result. A mismatch exits with code `2` instead of
loading or running the held-out games.

The committed corpus files cover charts A through E, three through five
players, every focal seat, and fixed opponent mixtures. Development and
held-out seeds must not overlap. Use `--overwrite` only when intentionally
replacing the three known files in an existing output directory.

The process exits with:

- `0` when the candidate passed and `promoted` is `true`;
- `1` when the exam completed but the candidate was not promoted;
- `2` when the command, corpus input, repository lookup, or artifact write
  could not be completed.

Exit `1` is a valid measured result. Exit `2` means there was no usable
decision from that command.

## Read the evidence

Every attempted run that reaches reporting writes three files:

- `promotion-report.json` is the authoritative decision. It records the
  source commit; all bot names and IDs; execution settings; corpus names,
  digests, and seeds; requested and completed coverage; rating difference;
  uncertainty interval; bootstrap progress; faults; failure reasons; artifact
  names; and the final `promoted` value. For a frozen heuristic candidate it
  also records the exact candidate name and bot ID; search name and source
  commit; frozen file, profile, search manifest, search report, and
  candidate-evaluation digests; and the predecessor and development-corpus
  name-and-digest binding.
- `paired-games.jsonl` contains one canonical summary for each executed game.
  For pair `n`, game index `2n` is the candidate game and `2n+1` is its
  incumbent twin. Failed evidence may have gaps where a game did not finish.
- `corpus-snapshot.json` contains the normalized, expanded development and
  held-out corpora and their digests.

The rating difference is the candidate rating minus the incumbent rating. The
bootstrap interval is an uncertainty range made by repeatedly resampling
complete matched cases and refitting the rating model. Both games in a pair
always stay together. The candidate passes only when the lower end of the 95%
interval is above zero.

A failure to promote does not prove that the candidate is worse. It can mean
that the measured advantage was too small or uncertain, that too few
resampled fits converged, or that the evidence was invalid. Read `failures`,
coverage, and faults in the report before interpreting the decision.

Complete promotion receipts are local working data and default to the
gitignored `artifacts/promotions/` directory. Retain historically important
outcomes as concise dated benchmark notes; do not commit raw paired games,
bootstrap samples, repeated corpus snapshots, or failed-run receipts.

## Failure reasons

The report fails closed: any listed reason forces `promoted` to `false`.
Stable report reason codes are:

- Corpus separation: `invalid_purpose`, `duplicate_corpus_name`,
  `duplicate_engine_seed`, and `corpus_seed_overlap`.
- Paired-plan identity: `held_out_corpus_required`,
  `candidate_incumbent_identity_collision`,
  `candidate_opponent_identity_collision`,
  `incumbent_opponent_identity_collision`, and
  `opponent_identity_mismatch`.
- Missing or changed game evidence: `unexpected_game`,
  `missing_paired_game`, `seed_mismatch`, `ruleset_mismatch`,
  `identity_mismatch`, and `player_count_mismatch`.
- Execution and decisions: `simulation_failed` and `bot_fault`.
- Rating and uncertainty: `rating_fit_failed`, `nonfinite_analysis`,
  `bootstrap_incomplete`, and `interval_includes_zero`.

A corpus that cannot be loaded stops with exit `2` before simulation. Its
input code identifies the problem: `duplicate_json_key`, `malformed_json`,
`invalid_recipe`, `invalid_recipe_keys`, `unsupported_schema`,
`invalid_corpus_name`, `invalid_purpose`, `invalid_root_seed`,
`invalid_repetitions`, `unsupported_chart`, `unsupported_player_count`,
`insufficient_opponents`, `unknown_opponent`, `duplicate_opponent`, or
`duplicate_engine_seed`. Unknown command-line bot names, invalid frozen
provenance, a predecessor or development-corpus mismatch, invalid numeric
options, unavailable Git metadata, and filesystem failures also exit `2` with
a direct message.

## Preserve held-out meaning

Development results may guide tuning. Use the held-out corpus only for the
final promotion decision; inspecting repeated held-out outcomes and tuning
against them turns the final exam into development data.

Never edit a committed corpus version in place. To change coverage, opponent
mixtures, repetitions, or seeds, add a new file such as
`held-out-v2.json`, give its recipe a new name such as `held-out-v2`, verify
that its expanded seeds do not overlap the development corpus, and commit it
beside the older version. Old reports then keep their original meaning.

The gate records evidence; it does not change strategy code, bot IDs,
checkpoints, aliases, or tournament defaults. A later strategy issue may
advance an identity only after its own held-out report passes.
