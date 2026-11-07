# Breakout: four-step direct image supervision

The retained [coverage control](../results/2026-11-07-breakout-world-coverage.md)
reconstructs108/112 observed balls and predicts94/109 at one step, but only14/107
at eight. The closed [H16 motion rollout](../results/2026-11-07-breakout-world-motion-rollout.md)
improves eight to25/107 while damaging observation, one-step, events and primary
fixed-action energy. Observation-only code and its faster combined recipe also
fail. Select a shorter direct image horizon, without importing those changes.
This tests a different bounded objective, not an extension or repeat.

Root `/mnt/data/kindle-breakout-world-short-rollout-20261107`. Reuse exact
qualified native85dbe150/SHA5b112866 and its packaged runtime. No Rust/backend
change or rebuild, new data, actor/critic, pretrained encoder or object/RAM labels.
Python orchestration supports the new horizon and optional matched dense-initial
exports; learning and inference stay native on Meganeura/Blade.

## Fixed recipe and hypothesis

Use `motion_cross_entropy`, prediction length4 instead of16. Preserve the
retained recurrent-conditioned posterior and h/z decoder, stride-two off,
mixing.01, RGB64/12M/B8/T32/full BPTT/Auto, seed41001, flat4e-4 rates, observed
motion reconstruction100, one-step KL1/.1/free-nat1 and observed reward/
continuation1. All68 logical/8,852,675 scalar weights/204 checkpoint tensors
remain. Fresh8,192 updates on the exact original4/2/1/1 source plan and sixteen
fitting/eight development streams. Reused seed/families are not confirmation.

The prior starts from observed index27, then receives only the four submitted
future actions. Detached targets use real previous/current RGB, generic weights
`1+100*abs(current-previous)`, normalized per image. Average the existing image
loss across full Bx4, with reset and subsequent suffix rows masked out and
terminal arrivals retained. The first motion pair uses observed index27 and
target28. No new loss formula, free-nat setting, event term or target feedback.

Shorter paths may be easier to optimize and less damaging to perception; this
is a hypothesis. Changing the horizon also changes observed-prefix length,
suffix positions, valid exposures and categorical RNG consumption. It is not an
isolated gradient-variance estimate. Straight-through categorical gradients and
conditional-mean pressure remain. Target-dependent weights are not a calibrated
generative likelihood. Evaluate all original long horizons, not only four.

## Qualification, allocation and decision

Before GPU use, require affected CPU CLI/sampling/report tests, exact equality
of all8192 source locations and configuration to the H0/H16 controls, matching
non-suffix exposures, and directly recounted four-step validity/terminal totals.
Package matching runner bytes and pin the reused native/runtime; preserve failures.

Separately declare one composition preflight (900s): all204 initial tensors
match the retained control; the original eight-origin initial predictions match;
one finite11-report update has positive future-image loss, exact mask fraction,
weighted total and matching observed component outputs. Require65 first-update
changed parameters (three reward-hidden tensors remain initially inactive),
204 invariant frozen tensors and18 exact initial/trained restore arrays. Verify
saved/restored horizon4. Reuse prior primitive/derivative and legacy-restore
qualification without repeating their native fixtures or querying old models.

Only after reviewed qualification, separately declare one fresh8192 learner
(3600s) and final restore (600s). Retain initial/2048/4096/final checkpoints,
all periodic original cohorts,256 fitting/128 development matched dense-initial
and final exports. Check204 initial tensors after dense-initial inference, full
plan/exposures, finite11-report losses, all trainable families and18 final restore
arrays. Dense-initial queries precede the existing train-loop timer; report the
guard worker span separately. No collector or new emulator actions/frames.

After learning/restore review, separately declare only the new final world's
84 frozen action-branch groups (1800s), history16 primary/32 secondary,eight
draws, all1/4/8/16/32 endpoints. Reuse original oracle and H0/H16 control arrays;
no old-model query, re-collection, selected checkpoint or best-match draw.

Keep observed>=90%, one-step>=89/109, eight-step>=28/107, primary16/32 ball
energy<=.12712666431732942, NOOP serve false balls<=10%, ending32 Brier<=.10,
ending16 Brier<=.01, RIGHT16 ending probability<=.10, non-ending category-mean
false ending<=5%, and freeflight8 ball presence>=90% on every plan. Report all
individual false alarms, horizons/actions, reward/paddle/brick errors and actual-
initial gains. No easier slice or gate. Full world competence additionally needs
precise physical forecasts, persistent brick memory, late/tunnel/second-wall
coverage, untouched families and fresh confirmation.

Use current AMD host guard/Auto, verify live identity, stop real faults/nonfinite
state, no reset/recovery/NVML. No unchanged extension, sweep or automatic successor.
Completed negative learners and their original failures remain closed. Passing
execution or intermediate gates does not complete the world-model goal.

## Actual qualification and allocation

All36 affected CPU tests pass. Preparation matches8192 source locations and
non-suffix counts; H4 has256,201 valid suffix rows and2,382 terminal targets.
First139-pin composition preflight completes in15.354s:204 exact initial
weights/moments,18 original initial arrays,18 initial/trained restore arrays,
65 changed parameters,finite11-report/mask/total and204 frozen tensors pass.
No primitive/legacy fixture repeats or stage failure. A separately declared147-pin
fresh8192 learner/restore is active; final-only branches await reviewed completion.
