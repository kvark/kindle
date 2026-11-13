# Loss-only attention to observed RGB changes

The [no-prior-image result](../results/2026-11-07-breakout-world-no-prior-image.md)
improves ball reconstruction and training cost, but only5/112 development balls
are within1 pixel. The image/latent probes remain inaccurate. The next selected
change directs more observed reconstruction error toward visual changes while
preserving the H0 architecture and latent dynamics.

## Objective

Select standalone `pixel_loss="motion_cross_entropy"`, prediction length0.
For each observed RGB channel/pixel, form `u=1+100*abs(image_t-image_t-1)` and
`weight=u/mean_image(u)`. Multiply observed soft-target cross-entropy by this
detached weight, sum pixels and average batch/time. Use uniform weights for
clipped starts and natural reset rows. The previous RGB observation is already
part of the fixed training sequence; no extra replay field or resize is added.

Weights use all RGB pixels, with no object masks, RAM or emulator-state targets.
The scale100 is fixed for this one experiment. No labels/weights enter imagined
state or inference, and no prior-image term is allowed with this objective.
Keep RSSM KL/dynamics, reward/continuation, RGB64/12M/B8/T32/Auto,4e-4 rates,
coefficient100, seed41001, data, initialization and source sampler unchanged.
There are no new trainable parameters, encoders or actor/critic dependencies.
Default MSE and plain cross-entropy remain available.

Fitting-only output arithmetic puts mean red weight40.29 at checked ball cores,
raising their absolute output-logit gradient share2.78%→31.18% in H0 exports.
This measures allocation, not upstream parameter gradients or a proven cause.
Motion also emphasizes paddle/brick/header changes. Target-dependent weighting
is an attention objective, not calibrated generative likelihood. Frozen physical
tests, rather than weighted training-loss values, determine useful learning.

New writes use checkpoint formatv2 with an explicit objective so older runtimes
reject unsupported training semantics. New code still reads v1 MSE checkpoints
without an objective and v1 cross-entropy artifacts with one. Version/invalid
objective validation happens before GPU construction. Use exact matching native
and runner bytes; do not mix package overlays.

## Qualification

CPU checks cover independent F64 loss/derivatives, normalized image-wise weights,
uniform reset rows, detached targets, unchanged parameter shapes and graph
compilation. Check v1/v2 objective metadata and default paths. Run affected world
tests, strict workspace/bindings Clippy and the packaged Python checks. No GPU
work may begin before the new source/package/binary identities are pinned.

Only two new guarded qualification workers are selected initially:

1. Native motion-weight/loss/derivative fixture: two1024-element images, different
   motion patterns, a reset row, normalization across workgroups. Compare weights
   with independent F64 at2e-5, derivatives2e-6 and summed loss2e-3 tolerances.
2. Full-size motion-H0 preflight: match204 initial tensors and18 initial arrays
   to the completed plain-H0 control. Restore that v1 control correctly. Require
   one finite nine-loss update, correct weighted sum,204 invariant frozen tensors
   and18 exact trained-restore arrays with age1/prediction length0/new objective.

Use the AMD host guard, current boot/module identity and Auto. Stop on actual
faults/nonfinite values. No NVML loop, reset/reload/reboot, automatic retry or
repeat of a completed phase. Preserve failures and declare separate repairs.
This is objective qualification, not world-model competence.

## Fixed learning after review

After qualification review, separately declare one fresh8,192-update learner and
its final frozen restore. Fit on the retained16,384 actions/65,503 emulator frames;
development8,192/32,755 stays separate. No collector or closed learner repeats.
Exact initial tensors, uniform source plan and evaluation origins must match H0.
All ordinary exposure counts match; prior-image terminal exposure stays0.
Replayed arrivals are not new environment actions or frames. Keep0/2048/4096/8192
checkpoints and all8,192 finite weighted reports.

Keep four-draw0/1024/.../8192 natural/contact/ending curves,1/4/8/16/32 forecasts,
final256 fitting/128 development exports, RGB and physical measurements, reward/
endings, same-support controls and frozen fitting-only readouts. Verify204 frozen
tensors and18 exact restore arrays. Report native and total loop times separately.
These remain retrospective recorded-control forecasts, not prospective planning.

Primary intermediate gate remains90% observed supported development balls within
1 resized pixel. Report misses, false positives, fitting/generalization gap and
other objects; do not lower the gate or select a favorable draw. Passing it alone
does not complete the world-model goal. Untouched tests, prospective branches,
calibrated stochastic outcomes, late-game/tunnel/second-wall coverage and fresh
confirmation remain unfinished. No capacity increase, broad seed campaign,
unchanged extension, actor/critic resume or automatic successor is selected.

Root `/mnt/data/kindle-breakout-world-motion-20261107`. Implementation,37 CPU
world tests,61 packaged Python checks and strict workspace/bindings Clippy pass.
Native/source3fa37b6d, SHA256
`eafe8359ae5019ea0a5a7517b7acfe75aacaf64b4e84a3bcd9ec4548cf939dc5`.
Both new GPU workers pass306 pins: native F64 weights/loss/derivatives and
full-size matched-init/one-update/frozen/v1+v2 restore. Initial204 tensors/18
arrays, trained18 arrays and204 frozen tensors are exact; v2 objective retained.
One matched preflight update changes99 optimizer moments (including encoder)
versus plain H0, verifying integration without a competence claim.

After review, the fresh8,192-update learner/final restore is separately declared
and active under307 pins. Queue session71177; learner PID1609131 confirmed live,
first736 updates finite. Initial weights/source plan/origins match H0. Stop at
the fixed budget; no automatic successor. The completed H0 source36a5d3e0 is
archived before these changes; original artifacts/receipts remain unchanged.
