# Bounded Breakout check with CDP prediction scale1000

Select one fresh bounded check of the existing centered-CDP prediction loss
weight,500→1000. This coefficient has remained fixed in the retained recipes.
Earlier two-batch gradient attribution found imitation already a major encoder,
posterior and dynamics objective, with partly aligned CDP gradients; it does
not support claiming CDP overwhelms imitation. Stronger prediction training may
help useful state learning, but those old late-checkpoint measurements do not
predict the effect of a new learning history. This is an untested hypothesis,
not a diagnosed cause or a sweep.

Change only `loss_scales.future_prediction`. Keep centered cosine targets,
target stopping, other loss weights, the retained convolution/max-pool encoder
and all parameter/replay/latent/control shapes. No new model capacity, targets,
rollouts or scheduled updates. The completed stride-two comparison remains
closed and its option stays off. No new native build is needed.

Expose optional `--cdp-scale` in the canonical vector runner and a private copy
of retained runnercf4ea238. Require positive finite F32 and fresh learned-CNN
CDP; omission preserves500, and restore rejects overrides while retaining the
checkpoint value. In the canonical runner, effective optional open-loop loss
validation uses the requested CDP scale. The old private runner has no open-loop
option. Preserve exact pre-edit canonical bytes and private-runner diff.

Reuse native447fe50/SHA256
`35ad0db42fc7ed5dbaf6dacddff613fdc04d83ae6e0f38548532f6ab2c7812a2`
and `/x/Temp/kindle-breakout-corrective-20261028/runtime` unchanged. Keep
CDP12M64/Auto/N8/B16/T16/H15/R16, world microbatch16, replay100k, native pooled
frames/one GPU resize, encoder/base4e-5,dynamics4e-4,behavior8e-5, critic1/1,
representation KL.1,dynamics KL1/free nats1,continuation1,disagreement1,
expert100/unit weights/world gradients/all-state corrective labels,20% episode
guidance/cap1920, sampled collection/posteriors, latent/actor mixing.01,
full18/sticky.25/repeat4/noops0. Native policy and optional objectives stay fixed.
Fresh seed1009, frozen seed4000001009; no transferred weights or experience.

CPU checks cover defaults/explicit500/1000, centered and uncentered CDP, invalid
values and non-CDP encoders, evaluation/restore override rejection, checkpoint
value preservation and canonical effective-open-loop overflow. Block native and
emulator construction in exact queue checks; verify the weighted loss arithmetic
with1000 and rejection of500, cadence, guidance, replay inputs and schema.

Run8,192 actions/489 updates within180s. Require first update376, residual credit.5,
all357 initial tensors and47 pre-update events equal retained, finite reports,
correct loss sums and changed encoder/reward/continuation/actor/value parameters.
Independently replay every action, frame, reward, boundary and teacher label/
override. Restore for1,024 frozen greedy actions within60s; guidance and updates
off,357 exact tensors, checkpoint config preserved. This is an execution/cost
check, not a natural-episode competence evaluation.

Use existing host guard on the AMD host with live boot/driver/process checks.
Stop on real faults or nonfinite values; preserve failure, no automatic retry,
reset/reload/reboot. Pin exact native, runtime, runner, protocol and helpers.
Queue cap300s, audit180s and replay120s. Report construction, actions, emulator
frames, training/stage elapsed times and source limitations separately.

Require padded200k projection `seconds*200000/8192*1.1`≤1800s and at least2GiB
conservative full-replay headroom after subtracting the entire5,939,200,000-byte
replay footprint from minimum observed native headroom. Compare against saved
47.689684s bounded timing without claiming a paired speed result.

Root `/x/Temp/kindle-breakout-cdp1000-20261106`. No full learner is selected here.
Any fresh full comparison needs a separate declaration,100k recovery, own
actual-initial/final sampled and greedy24-natural-episode evaluations, with
greedy raw score and own-initial gain/min primary against143.667/6.590132.
Keep the original864/90% two-wall gate and unfinished five-game goal. No sweep,
extension, check-experience transfer or automatic successor.
