# Breakout: bounded check of greedy training actions

Completed:8,192 actions/489 updates in47.418s;1,024 frozen/.869s with zero
updates and no natural episodes. All387 pins,initial/frozen tensors,full replay,
guards and original services pass. Conservative200k projection21m13.430s passes
30min. [Completed result](../results/2026-11-02-cdp-breakout-greedy-training.md).
No full learner selected; do not repeat completed stages. The original selection
and fixed bounds follow.

Select one fresh 8,192-action check within the unfinished five-game goal.
The retained [replay-value1 learner](../results/2026-11-02-cdp-breakout-replay-value1.md)
scores 70.750 sampled and 143.667 greedy, with zero of 24 two-wall successes.
The [coverage diagnosis](../results/2026-11-02-cdp-breakout-score-coverage.md)
finds 97.03% of its training actions below 216 points, zero at 432+, and only
1,262 native actions at scores 216–431. These are score ranges, not verified
wall boundaries. Greedy collection may improve experience or remove useful
diversity; neither gameplay benefit nor a causal coverage explanation is known.

The earlier [Qbert comparison](../results/2026-11-01-cdp-qbert-greedy-training.md)
provides supporting development evidence, but does not establish transfer to
Breakout. No completed experiment is restarted or extended.

## Fixed change and controls

Change only live training action selection from sampled to greedy. Keep centered
CDP, 12M64/Auto/N8/B16/T16/full BPTT/microbatch16/H15/R16, base/encoder4e-5,
behavior8e-5, dynamics4e-4, replay-value1, actor unimix .01 and action-effects
coefficient1. Sampled categorical posteriors and imagined actor sampling stay
fixed. Retain expert100, unit action weights, world expert gradients, all-state
corrective labels, 20% episode guidance and its 1,920-action cap. Full18,
sticky .25, repeat4, noops0 and one GPU resize from native pooled RGB stay fixed.

Reuse development seed1009 and frozen seed4000001009 with eight lane offsets.
Start fresh without weights, replay, beliefs, games, RNG or previous experience.
There are 16,334,353 trainable parameters. Native447fe50/SHA35ad0db and the
Breakout teacher are unchanged, using
`/x/Temp/kindle-breakout-corrective-20261028/runtime`. No build is required.

The retained runner cf4ea238 receives the already-qualified fresh-only
`--training-action-mode {sample,greedy}` parser, validation, header and three
native dispatch changes. New SHA6e03d739 adds nine lines and replaces three.
Omitting the option preserves sampled training and the original header;
frozen `--greedy` behavior remains unchanged. Teacher overrides, passive labels
and actual-action feedback retain their existing paths.

## Qualification and bounds

The 45 targeted CPU tests pass, covering invalid/restored modes, metadata,
corrective labels/feedback/resets, replay-value and actor controls. An added
wrapper sentinel then fails because importing native metadata was mistaken for
GPU construction. Preserve that failed receipt. Separate source/receipt review
qualifies the already-passed tests in .050s, with no test or GPU repetition.
All exercised constructor paths use CPU stubs.

The actual-command preflight passes in 1.048s (1.083s wrapper): two commands,
17 configuration cases, four mode cases, 357 tensor shapes, cadence derivation
and all 338 control pins. It constructs no native model or environment and
loads no native module. All preparation and original failure sources are pinned.

- Train 8,192 aggregate actions within 180s: 489 updates, first update at376,
  residual credit .5. Save actual-initial and final checkpoints.
- Restore final weights for 1,024 greedy frozen actions within60s, with guidance
  and labels off and zero updates. This short budget is not a competence test.
- Require 357 finite control-matching initial tensors and 357 invariant frozen
  tensors. Pre-update training trajectories must differ under changed collection;
  retain all 47 paired semantic events and their difference count.
- Require exact configuration/modes, finite aligned world/behavior imitation
  reports, changed CNN/reward/actor/critic weights, full training pixel/action/
  teacher-label replay, exact native/source identity and host guards.
- Keep replay cap120s/process150s, audit180s, main480s and collector600s.
  Check the idle host and preceding Qbert closure; no overlapping GPU job.
- Cost gate: `training_seconds * 200000 / 8192 * 1.10 <= 1800`.
  Passing establishes execution and early cost only.

No full learner, retry, cap increase, extension or automatic promotion is
selected. A later separately selected fresh200k comparison must preserve
100k recovery, actual-initial controls and both sampled/greedy24-episode frozen
assessments against70.750/143.667. The original two-wall/90% gate remains fixed.
Collection changes exploration and visited experience together; a gain would
not establish an isolated architecture or coverage cause or seed reliability.

Root `/x/Temp/kindle-breakout-greedy-training-20261102/native-check`, trial
`breakout-12m-64-b16-r16-replayvalue1-greedy-expert100-check`.
Main `kindle-breakout-greedy-training-check-20261102`, with the original
`kindle-breakout-greedy-training-check-collection-20261102` observer.
