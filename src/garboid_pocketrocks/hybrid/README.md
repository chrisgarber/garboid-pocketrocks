# Promoted expert foundations

The hybrid project may choose only from bots that already passed the common
held-out promotion gate. `promoted_experts-v1.json` is a compact, immutable
receipt catalog for the initial four experts. It records no per-game held-out
outcomes. Instead, each entry binds the retained promotion source, exact
corpus and gate, positive confidence interval, complete coverage, clean run,
and executable profile or checkpoint.

Loading returns an opaque verified catalog that proves **eligibility**.
Selection and name lookup accept only that catalog, so a free-floating or
locally edited expert record cannot enter the roster. Loading does not import
optional neural dependencies or construct a bot. `check_expert_availability`
is the separate runtime probe. This distinction prevents a missing local
dependency from rewriting promotion history.

The selector module only chooses an expert identity and records a deterministic
fallback reason. Nothing in this package is registered as a bot, changes an
existing policy, reads held-out games, or advances a latest alias. A later
behavior-changing PR must train on development inputs, create a new explicit
hybrid identity, freeze it before held-out evaluation, and pass the common
promotion rule.
