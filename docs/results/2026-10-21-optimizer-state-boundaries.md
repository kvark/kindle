# Scaled optimizer arithmetic passes state-boundary qualification

October21,2026. Extends the [original tiny-gradient fixture](2026-10-21-optimizer-denormals.md).

The repaired shader passes **786,432 independent checks over4,096 stateful
optimizer updates**, covering both Adam and LaProp. Epsilon1e-20, variance storage
and the optimizer equation are unchanged. This is correctness qualification,
not a speedup or learning result.

## A real edge case found before GPU execution

The first repair selected its scaled path using the old variance. With beta2=0,
old variance1 and gradient1e-20, that history is discarded but still prevented
selection of the safe path. The new variance1e-40 could therefore flush again.
The repaired helper selects zero history when beta2=0 before choosing/scaling
the variance path. The earlier production-setting fixture remains valid; it
never covered this boundary. No GPU cohort repeats to rediscover the CPU proof.

All8,388,608 subnormal mantissa decodes and16,777,216 integer/halfway encodings
pass an independent IEEE conversion reference on CPU, including nearest-even
rounding into the normal range. These are CPU arithmetic checks, not that many
GPU executions. The initial CPU counterexample mistakenly used3e-18, whose
square is normal at beta2=0; the corrected1e-20 case passes. The subsequent
NumPy-boolean serialization error and partial output remain recorded.

## What the GPU test covers

- Eight gradient magnitudes crossed with eight initial variances: zero, smallest
  and largest subnormals, smallest normal and both sides of the scaled-path guards.
- Adam and LaProp, each with beta2=0,.9,.999,.9999; beta1=.9,LR.1,weight decay.01.
-512 updates per setting: held gradients, reversed signs, zero-gradient decay,
  then alternating tiny/ordinary gradients.64 values in each dispatch.
- Every parameter/momentum/variance saved as binary F32 before numerical checks;
  exact supplied bias corrections and gradients are retained too.

The independent reference computes in F64 and rounds each stored state to F32.
That explicitly tests finite-precision state, not an unrepresentable continuous
variance below the smallest F32 value. Original bounds remain unchanged:
variance1e-5 times reference plus two minimum subnormals; momentum/parameter
1e-5 times one plus reference magnitude. No failed gate was relaxed.

All786,432 comparisons pass independently in Rust and Python; these are paired
checks of the same values, not twice as many GPU values. There are35,884 saved
subnormal variance states. Maximum error/bound ratios are.7424 for variance,
.0826 for momentum and.5301 for parameters. No broad exact-trajectory claim.

One passing GPU guard, zero warnings,4,096 standalone optimizer updates, zero
learner updates/game actions. All634 source receipts pass. Original ShaderData
and segment handling are retained; the only shader change beyond the prior
candidate is zero-history selection for beta2=0. No worker remains in this study.
Artifacts: `runs/optimizer-boundaries-20261021.K8y2M25r`.

## Current-upstream Session and checkpoint qualification passes

The same helper is integrated into current upstream Meganeuraa75c38e, with
Bladeeae0c6f7/Nagaa489554c. Local prototype commit`c4f2cc2cef5500b49153afae2ce3440a4850d500`
is on`fix/optimizer-subnormal-variance` in`/x/Code/meganeura-optimizer-subnormal`.
Only the optimizer shader and existing LaProp regression module change. The
user's dirty neighboring checkout is untouched; no Kindle dependency is repinned.

Two GPU tests pass: the existing ordinary LaProp case and a new actual-backward/
Session/checkpoint regression covering Adam/LaProp,beta2=0/.999,16 gradients and
eight steps. All512 explicit raw gradients match exactly;1,536 independent F64
state checks pass. Four checkpoints preserve all48 values and step counts on
load; continued restored/uninterrupted Sessions match960 state values exactly
including initial restore. Independent CPU parsing checks192 serialized values,
including12 subnormal variances.49 synthetic Session updates total, zero full
Dreamer updates/game actions. One passing guard, zero warnings,338 source receipts.

Build preparation initially encountered the enclosing Kindle workspace and a
copied lockfile incompatible with this upstream manifest. Both failures remain;
move the separate worktree outside Kindle and resolve offline before the locked
build. Only the pinned Blade/Naga/Naga-types source entries change in resolution.
The subsequent release build passes in1m49s, with a distinct copied dependency
cache and one-CPU/2GiB limit. No original cache/source mutation. No GPU retry.
Artifacts: `runs/optimizer-session-20261021.AswkXd38`; no worker remains.

## Remaining correctness and adoption gates

This qualifies finite inputs, not arbitrary invalid state. Source review finds
that selecting zero history at beta2=0 also discards nonfinite history; preserve
the original poisoning behavior and add negative-state coverage before adoption.
No GPU nonfinite-state result is claimed here. The workstation-only headroom
assertion in the prototype test also belongs in the local qualification runner,
not a portable upstream regression.

Current upstream has no retained command-cache extension/API, so these tests
cannot qualify Kindle's isolated cached runtime. Cached-runtime and full-model
numerical/cost checks remain. Prototype is committed locally, not promoted or
submitted as an upstream-ready fix. No Atari restart; the five-game learning
objective remains unachieved. Across this turn: two passing GPU guards,
4,096 standalone optimizer plus49 synthetic Session updates, zero game actions.
