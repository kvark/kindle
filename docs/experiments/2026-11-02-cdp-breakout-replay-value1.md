# Breakout replay-value scale1: bounded development check

Select one8,192-action check after the Qbert confirmation's original closure.
The [current sampled control](../results/2026-11-02-cdp-breakout-behavior8e5.md)
scores59.000 sampled/117.292 greedy in21m44.517s, withzero two-wall successes.
Its behavior rate is8e-5. Change only replay-value loss scale0.3→1.0.

The [critic calibration](../results/2026-11-02-cdp-breakout-current-critic.md)
and [real-trajectory consistency check](../results/2026-11-02-cdp-breakout-value-consistency.md)
show weak complete-return prediction on the neighboring behavior4e-5 controls.
They do not directly measure the behavior8e-5 critic or identify a causal loss
defect. Qbert's scale1 development gain motivates testing this existing
coefficient, but its fresh confirmation missed the gate; transfer of benefit to
Breakout is unproven. This is a separate game-specific development experiment,
not a Qbert retry or a broad coefficient sweep.

## Fixed recipe and interpretation

Keep centered CDP12M64/Auto/N8/B16/T16/fullBPTT and microbatch16/H15/R16;
base/encoder4e-5, behavior8e-5, dynamics4e-4, sampled collection/posteriors,
actorunimix.01, action-effects coefficient1, replay100k and warmup1000.
Keep expert100, unit action weights, expert world gradients, all-state
corrective labels and20% episode guidance/cap1920. Full18/sticky.25/repeat4/
noops0 and one GPU resize from native pooled RGB remain fixed.
There are16,334,353 trainable parameters.

The replay-value coefficient weights critic fitting and representation
gradients through frozen value logits. Existing real rewards, episode boundaries
and imagined-return bootstraps remain. No complete-return target, new auxiliary
head or architecture is introduced. A future gameplay difference would not
separate critic and representation effects or prove a missing-loss defect.

Use development seed1009 and frozen seed4000001009 with the same eight lane
offsets. This cohort is reused; it cannot establish independent reliability.
Start fresh without weights, replay, beliefs, games, RNG or experience reuse.
Native447fe50/SHA35ad0db and the Breakout teacher remain unchanged. Runtime:
`/x/Temp/kindle-breakout-corrective-20261028/runtime`. No build is required.

The exact control runner6c1b69fb receives only the already-qualified ten-line
fresh-only `--replay-value-scale` option. Candidate runnerSHAcf4ea238; no other
runner behavior or defaults change. It rejects invalid, evaluation and restore
overrides. The23 relevant private-runner loss/rate CPU tests pass.

CPU qualification passes in2.186s (2.218s wrapper): both actual commands,17
configuration cases,357 control tensor shapes, first-update/cadence derivation
and all282 retained-control pins. No native module or environment is constructed.
The prepared runner adds ten lines and deletes none; no production source changes.

## Fixed bounds and acceptance

- Train8,192 aggregate actions within180s, with489 updates, first update376
  and residual credit.5. Save actual-initial and final checkpoints.
- Restore the final checkpoint and run1,024 greedy frozen actions within60s,
  guidance/labels off andzero updates. This budget is not a competence test.
- Require357 finite tensors, exact initial values and pre-update semantic
  equality to the retained behavior8e-5 control, and invariant frozen tensors.
- Require the exact configuration, finite and aligned world/behavior imitation
  reports, changed CNN/reward/actor/critic weights, full training pixel/action/
  teacher-label replay, native/source identities and host guards.
- Replay cap120s (process cap150s), total audit180s, main service480s,
  collector600s. No overlapping Kindle GPU allocation.
- Cost gate: measured check training seconds×200000/8192×1.10 must be at most
  1,800s. Passing establishes execution and early cost only. A full learner
  requires a separate selection after this check closes; no automatic promotion.

No retry, cap increase, coefficient sweep, extension or reuse of check experience.
Preserve original failures and inspect saved evidence before any separate
CPU-only repair. The original Breakout two-wall/90% gate remains unchanged;
no gate pass is inferred from finite losses or this short frozen budget.

## Execution package

Root `/x/Temp/kindle-breakout-replay-value1-20261102/native-check`, trial
`breakout-12m-64-b16-r16-replayvalue1-expert100-check`.
Main `kindle-breakout-replay-value1-check-20261102`, with the original
`kindle-breakout-replay-value1-check-collection-20261102` observer.
Pin the actual commands, protocol snapshot, runner diff, schema, source/native/
teacher bytes, completed control and preceding Qbert closure before launch.
Recheck live state; do not restart active or terminal stages.
