# Heuristic-to-neural bootstrapping experiment

## Goal

Test three different ways of transferring the released v3 heuristic bots'
knowledge into the large PPO policy without changing the live bot or weakening
the normal promotion gate.

In plain English, the three experiments ask different questions:

1. **Behavior cloning:** does the network learn faster if it first practices
   copying legal decisions made by `balanced-v3`?
2. **Auxiliary value:** does PPO improve if its value estimate also receives a
   small training-only hint about how valuable `balanced-v3` thinks the current
   lot is?
3. **Opponent curriculum:** does PPO improve if its learning seat practices
   against the released aggressive, balanced, and passive v3 bots early, then
   shifts gradually toward self-play?

Each strategy is a separate ablation. The configuration loader rejects a run
that combines them.

## Immutable teachers and incumbent

- Behavior cloning and auxiliary labels pin `balanced-v3` and its exact profile
  digest. They never resolve the mutable `latest` alias.
- The curriculum pins `aggressive-v3`, `balanced-v3`, and `passive-v3`.
- The competitive neural reference is
  `vector_ppo_large_v1_g350k`, parameter digest
  `088160ad4006b2bac3691980d7f3e9dc56635fd57e6ad2b94068497e199f0e5c`.
- Existing checkpoint bytes, bot identities, registry entries, and aliases are
  immutable.

## Fixed development budget

All arms use root seed 42, the large model, all A-E charts, all 3/4/5-player
counts, and 119 games per chart/player cell.

| Arm | Demonstration rounds | PPO updates | Complete training games |
|---|---:|---:|---:|
| Control | 0 | 196 | 349,860 |
| Behavior cloning | 4 | 192 | 349,860 |
| Auxiliary value | 0 | 196 | 349,860 |
| Opponent curriculum | 0 | 196 | 349,860 |

For behavior cloning, four 1,785-game demonstration rounds replace four PPO
updates instead of increasing the game budget. Wall time and neural optimizer
steps are also reported because different strategies do different work per
game; the game count is the precommitted primary budget.

## Information boundaries

- Demonstrations contain only the public observation the deployable neural
  policy could see and the SDK-validated legal teacher action.
- Auxiliary targets are computed from the original public `DecisionContext`
  and canonical rules. Reveal decisions are masked. The target never enters
  rewards, GAE returns, tournament utility, or legal-action masks.
- Curriculum heuristic seats are non-trainable. Only the focal neural seat's
  trajectory reaches PPO.
- Raw local training artifacts may contain decision-level rows. Committed
  reports contain aggregate curves and provenance only; they must not expose
  game seeds or seed-linkable decision traces.

## Selection, freeze, and held-out rules

Training loss, teacher agreement, shaped reward, and auxiliary loss are
diagnostics, not strength evidence. Each final development checkpoint is given
a new research identity containing its strategy, root seed, exact game count,
and configuration digest.

The development corpus may compare the frozen final checkpoints with the
released large-v1 incumbent. Any selected checkpoint and its selection report
must be written and checksummed before a held-out corpus is loaded. Held-out is
run at most once for that frozen candidate. Promotion requires:

- successful games across A-E at 3, 4, and 5 players;
- zero illegal actions, bot faults, missing games, or failed games; and
- a strictly positive lower endpoint of the 95% bootstrap interval for the
  primary rating delta.

Failure is preserved as a negative experiment. It does not move an alias or
rewrite a released identity.
