# Immutable bot identities

## Decision

A released bot generation is immutable. Its versioned name, strategy
coefficients, checkpoint, inference contract, and any real remote bot ID stay
fixed.

Remote-capable wrappers own public server-issued `BOT_ID` values. Local-only
simulation generations use their versioned names as `BotSpec.bot_id`; they do
not claim remote identities. Unversioned names may track the latest
generation only while every released generation remains explicitly
selectable.

Before a change that can alter a decision or expected strength, follow the
[versioning-bots workflow](../../.agents/skills/versioning-bots/SKILL.md).
The user must choose either a new generation or an in-place update. Renames,
documentation, adapters, and verified behavior-preserving refactors do not
create new generations.

## Consequences

- Tournament and benchmark fields use explicit versioned identities when
  reproducibility matters.
- Historical checkpoints and aliases are never overwritten by a newly trained
  policy.
- Strength claims compare generations with fixed seeds and recorded
  configurations.
- Manifests retain exact training age and provenance even when a public alias
  uses a rounded age.
