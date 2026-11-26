# Removing prior-image supervision helps, but the world remains unqualified

The [fixed ablation](../experiments/2026-11-07-breakout-world-no-prior-image.md)
removes the extra16-step prior-image term while retaining observed cross-entropy
and RSSM KL/dynamics learning. It improves physical readouts and costs less, but
observed development ball accuracy is only **5/112 (4.46%)**, far below the90%
intermediate diagnostic gate. The full world-model goal remains active.

Retain H0 for the next isolated investigation. Do not extend this learner or
resume actor/critic. The next selected experiment is generic motion-weighted
observed cross-entropy, with unchanged architecture and no object/RAM labels in
training. Its implementation/qualification is separate from this completed run.

## Matched learning and cost

H16 control is the [completed cross-entropy learner](2026-11-07-breakout-world-pixel-cross-entropy.md).
Both start from the same204 tensors, source-sampling plan and evaluation origins.
RGB64,8,852,675 parameters,12M/B8/T32/full BPTT/Auto,seed41001,4e-4 rates,
reconstruction100,KL1/.1/free-nat1 and reward/continuation remain fixed. Only the
extra prior-image horizon changes16→0. Posterior and rollout RNG streams are
separate. No actor/critic, imitation, replay-value or intrinsic objectives learn.

Both reuse fitting16,384 actions/65,503 emulator frames and separate development
8,192/32,755. No collector repeats.8,192 updates replay2,097,152 arrivals, covering
all16,414 fitting arrivals; replay is not new environment interaction. Exposures
remain2,649 terminal/52,753 positive-reward/2,656 natural-reset rows. Prior-image
terminal rows become0, as intended. Source masks and true clipped-start labels
remain intact. Open-loop tests still cover1/4/8/16/32 decisions.

| Cost | H16 | H0 |
|---|---:|---:|
| Native training |765.084s|527.850s (8m47.850s)|
| Whole train/eval/checkpoint loop |1,003.581s|768.072s|

Native training time falls31.01%; whole-loop time falls23.47%. These are observed
single-run timings with less training work, not isolated kernel benchmarks or
seed reliability. Neither speed nor finite losses establishes competence.

## Frozen physical results

Final dense exports use256 fitting/128 development origins and one fixed draw;
all source arrays match. Periodic natural/contact/ending curves use four fixed
draws. Forecasts condition on recorded future controls: retrospective prediction,
not prospective planning. Development is reused, not untouched test. Ball
measurements cover a checked below-brick region; other positions remain untested.

| Observed reconstruction | H16 | H0 |
|---|---:|---:|
| Fitting ball detections |2/220|38/220|
| Fitting ball within1 resized pixel |1/220|28/220|
| Development ball detections |0/112|7/112|
| Development ball within1 pixel |0/112|5/112|
| Fitting/development paddle within1 pixel |94.14%/88.28%|99.22%/99.22%|
| Fitting/development brick balanced error |.001239/.190935|.000503/.168641|

H0 has0 fitting/1 development below-brick detection where the reference detector
has no supported ball; that includes ambiguous/out-of-region targets and is not
a full absence-calibration test. Observed RGB MSE is.00005592 fitting/.00073784
development. Fine ball localization still fails on most examples.

| Dense-development horizon | Supported balls | H0 within1 pixel |
|---|---:|---:|
|1|109|6 (5.50%)|
|4|109|2 (1.83%)|
|8|107|2 (1.87%)|
|16|114|2 (1.75%)|
|32|95|1 (1.05%)|

H16 has zero at these horizons. The gain is consistent with interference from
the extra image term in this recipe; it does not isolate mean-image pressure
from other optimization/gradient-budget effects. Earlier short deterministic
fitting gains from that term remain valid capacity evidence.

At supported ball-core pixels, H0 mean red intensity is.14978 fitting/.03929
development versus actual.4394. H16 is.04027/.01510. Yet61.90% of H0 development
core pixels remain below.01. This is improved contrast with a large residual
fitting/development gap, not a precise reconstructed state.

Frozen fitting-only global linear readouts on123 development RAM references give
CNN/posterior ball errors8.547/9.359 resized pixels, versus12.804/11.281 for H16.
Posterior backward-displacement error changes2.135→2.007 pixels/decision. Readouts
use256 fitting samples, fitting-lane cross-validation and one latent draw.
Failure does not prove information absent.

Dense-development positive-event Brier errors are.99329/.99851/.76866/.99998/
.99995 at1/4/8/16/32, on2/4/4/1/2 events. Sparse positive rewards remain mostly
missed; small samples do not establish calibration. One-step ending Brier improves
.37246→.25713 on22 fitting endings and.79882→.38831 on10 development endings.
Ending-only slices are outcome-conditioned, not population calibration.

## Additional CPU diagnostics

A conditional faint-image readout takes the largest red-excess peak in the same
region, with a local3×3 centroid and no brightness threshold. It always returns
a coordinate, even for noise, so it does not count detections or replace the
scored gate. On112 supported development images, mean position error is23.824
for MSE H16,10.774 for BCE H16 and5.330 for H0; within1 pixel is0/9/21. This
reveals some faint localized signal while leaving most targets inaccurate.

A shared local CNN-token readout fits a65-coefficient cell classifier and130-
coefficient offset regressor, with fitting-only lane validation. Conditional on
a valid RAM ball reference, H0 selects the correct16-pixel cell in60.16% of123
development states and gets13 within1 pixel, versus23.58%/2 for BCE H16. Mean
error remains8.700 pixels. Presence/absence is untested. Artificial tokens built
from true position make the same readout accurate on123/123, checking arithmetic
and layout, not Kindle competence. These diagnostic labels never train the model.

A fitting-only allocation calculation motivates the next experiment. Generic
weights `1+100*abs(current-previous RGB)`, normalized per image to mean1, assign
mean red weight40.29 at checked ball-core pixels. On the completed H0 fitting
outputs, the core's share of absolute output-logit gradient rises2.78%→31.18%.
The calculation uses image geometry only to measure allocation; the proposed
weights themselves use every RGB pixel without object/RAM labels. It is not a
measurement of upstream parameter gradients or proof of benefit. Target-dependent
attention weights are not calibrated generative likelihood. Resets/clip starts
must have uniform weights. The copied H16 scope label in the first H0 diagnostic
is corrected separately, preserving its original JSON/log and numeric arrays.

## Identity and closure

Root `/mnt/data/kindle-breakout-world-no-prior-image-20261107`.
Execution36a5d3e0 reuses native535de479, SHA256
`adfc2f9ec8e9174b56628ab8c390516ee6acee31dc4151418208bcfdf7f59c54`,
at `/mnt/data/kindle-breakout-world-bce-20261107/runtime`.61 CPU checks pass; no
native rebuild or repeated primitive qualification. The new H0 preflight passes
304 pins, matched204 tensors/18 arrays, one finite nine-loss update,204 frozen
tensors and18 exact trained-restore arrays.

The selected learner/restore queue completes. All305 pins,8,192 finite updates,
source samples, exposure counters and source archive pass. All68 logical
parameters change;204 parameter/moment tensors remain exact through frozen
evaluation. Restore reproduces18 arrays exactly on eight development origins.
No GPU fault, numerical failure, host recovery or repeated GPU stage is reported.

`review.json`, `analysis-receipt.json`, `analysis/comparison.json`, frozen JSON/
NPZ, true/predicted panels, global/spatial/faint readouts and allocation receipts
remain outside Git. Source archive `source-36a5d3e0.tar.gz` SHA256:
`2960d85642650a097ceed088224515409b0a092fec16bbdae3d59980a52ef357`.
All phases are closed; no unchanged extension or automatic successor is selected.
Earlier original failures/CPU repairs remain in their linked roots. Untouched
tests, prospective branches, calibrated stochastic outcomes, late-game/tunnel/
second-wall coverage and fresh confirmation remain unfinished. Keep the goal
active and unproven; the next motion-attention experiment requires its own
implementation, loss/derivative/restore qualification and fixed declaration.
