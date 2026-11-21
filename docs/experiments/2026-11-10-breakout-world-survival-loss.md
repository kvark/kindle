# Optional cumulative survival Brier loss: bounded qualification

The [closed timing review](../results/2026-11-10-breakout-world-rgb-event-heads-timing.md)
finds premature game-over risk even on several accurately drawn short paths.
Select an additional cumulative-survival objective for implementation and bounded
qualification. It is not a claim that pointwise continuation BCE is mathematically
wrong or that this change will fix serving, physics, calibration or the full goal.

Root `/x/Temp/kindle-world-survival-loss-20261110`.
No learner, collector, full assessment or automatic successor is selected here.
Use native Meganeura/Blade, Auto and the existing AMD host guard. Pin source,
native, runner, scripts, fixture/config and controls before any GPU execution;
stop on real faults/nonfinite outputs. Preserve failures under their original
names; repairs need new filenames/declarations. No NVML/reset/reload/reboot.

## Definition and scope

Add default-absent `world_survival_scale: Option<f32>`, requiring a finite positive
value and the currently qualified1M/RGB64/H16 `motion_event_rollout` with RGB
event heads and all-world training. Unsupported compositions/whole-agent uses
reject before GPU allocation. Keep RGB3 replay, encoder/RSSM/local decoder,
two-image history, physical command features, parameter shapes and inference.
No extra parameter, pretrained input, object label or emulator state.

At suffix step t, let c_t be the head's sigmoid continuation probability and
gamma the original float32 continuation discount. Define
q_t=clip(c_t/gamma,0,1) and S_t=product(q_1…q_t). Real continuation targets are
gamma when alive and0 at terminal. Their stopped, undiscounted cumulative
product Y_t is1 until game over and0 thereafter. Add

`world_survival_scale * sum_t,lane(keep_t * (S_t - Y_t)^2) / (H * B)`.

Use exactly the existing stopped suffix validity weights: retain a terminal
arrival, exclude resets and everything after the first reset. Target products
and validity never enter latent states, generated images or inference. Loss
gradients flow through own cumulative continuation forecasts and their existing
RGB/latent readout. Preserve all existing losses/scales/report slots and append
one unweighted survival-Brier report. This is a per-draw training term, not a
proof of calibration for the latent-marginal distribution. H16 training does
not certify H32 extrapolation.

Give the changed objective explicit checkpoint v29, wrapping v28 and recording
the positive coefficient. Reject mismatched flags/coefficient/base/scope before
allocation; restore retains the coefficient. Default-off v28 checkpoints and
graph behavior remain compatible. No changed-shape or old-checkpoint warm start
is used for qualification.

## Meaningful bounded checks

Check values and every continuation-logit derivative against independent F64
analytic products, with mixed lanes, survival/terminal/reset masks, all-masked
cases and probabilities both below and above the clipping boundary. Check zero
target/mask derivatives and no future-target path into states/images. Finite
differences stay away from the clipping kink. Validate configuration/metadata
and reporting/weighted-total integration on CPU. Reuse existing CNN/RSSM backend
qualification; no new broad parity project.

Package an isolated matching runtime/runner and retain its validation executable.
Bounded guarded native work includes the small loss value/derivative fixture,
a default-off comparison with sealed original RGB-event qualification controls,
and at most four fresh enabled updates on the retained eight-lane fixture.
Check actual initial/first/final tensors, all finite14-component reports,
weighted totals and the raw coefficient, freeze/query invariants, v29 metadata
and byte-exact restore arrays. Qualification tests the new scalar term and
plumbing, not useful learned physics. Release disposable build caches only
after package/executable hashes verify.

Any learning decision requires its own fixed protocol after qualification.
The candidate coefficient is10 (no sweep); no inference discount/risk threshold
or evaluator gate changes. A future learner must preserve an actual-initial
control, natural two-cycle plan/counts, dense and original joint/event gates,
all regressions, wall time and frozen chosen-action assessment if separately
selected. No automatic extension or successor; previous failed curricula remain
closed. Precise stochastic physics, natural events, persistent brick memory,
late/above-top and untouched/fresh confirmation still define the full goal.
