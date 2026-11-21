# Fixed-weight Breakout gradient-noise and loss-interference diagnosis

The [corrected-gradient comparison](../results/2026-11-07-breakout-world-score-function.md)
passes its conditional derivative checks but fails learning. Select a focused
measurement before another recipe, architecture, budget or seed. This is not a
learner, collector, emulator evaluation or backend optimization project. The
complete world goal and all missing late-game/held-out requirements stay open.
Root `/mnt/data/kindle-breakout-world-gradient-noise-20261107`.

## Fixed inputs and interventions

Use only the retained coverage final8,192 checkpoint's68 logical world parameters
(8,852,675 values), same RGB64/12M/B8/T32/Auto configuration. Preserve the original
checkpoint and204 tensors; load only model weights into raw training sessions.
Do not configure an optimizer or update parameters. Verify all68 parameters
bitwise after each batch and save before/after logical snapshots for CPU review.

Choose exactly coverage plan indices0,2047,8191, independent of model outcomes.
Reuse its two fitting corpora and standard4/2/1/1 plan. Export only cached native-
resized RGB, submitted actions, external reward and natural termination. No RAM,
object, phase or policy label enters the graph. Report each batch's actual label
counts and locations; these three batches do not establish population behavior.

Compare the unchanged observed-motion objective, ST energy and conditional-score-
function energy at the same weights. Hold posterior uniforms fixed across modes
and replicas. Vary only the separate predictive stream, including prior/head
samples and encoded target draws, matching old/new energy streams per replica.
Sixteen predictive seeds are fixed before execution. No best-seed or batch selection.

Per batch execute observed control twice; each predictive mode16 times plus a
repeat of replica0. Total108 forward/backward steps, no optimizer step. Raw
per-parameter gradients remain outside Git, with source/shapes/offsets and loss/
input fingerprints. Preserve repeat controls to measure execution noise.

Shared component outputs1..8 must match control and all13 ST/score-function
forward outputs must match pairwise at1e-5 relative/absolute tolerance. Output0
is the total, so predictive totals must instead equal control total plus100
times energy. This distinction was corrected during source review before GPU
execution; the first draft incorrectly called all first-nine outputs shared. Common
and auxiliary input hashes must match as declared. Repeated inputs must reproduce
gradients within1e-5 relative L2 plus1e-5 absolute L2, per parameter. Stop on real
GPU faults/nonfinite values; no retry/reset/reload/reboot. Known warnings alone
are not a gate. Use the existing host guard, one900s worker after CPU preparation,
source/executable/data pins and live-process review. No previous worker repeats.

## Analysis fixed before results

Subtract observed-control gradients to isolate each added predictive contribution.
Report per-parameter and encoder/posterior/core/prior/decoder/reward/continuation
families: norm of the16-replica mean, trace sample variance, sampling-error norm
sqrt(variance/16), and cosine with control. Report both the naive mean norm and
sqrt(max(0,mean_norm²-variance/16)); the latter is a noisy debiased signal-norm
estimate, not proof of zero signal when truncated. Compare paired ST/SF decoder
gradients and the repeat-control floor before interpreting sampler noise.

On CPU only, simulate auxiliary multipliers0,.001,.01,.1,1 on the same exported
gradients. Apply the pinned whole-parameter AGC formula(.3,pmin.001), then LaProp
using the retained anchor's moments and next step8193. Record normalized fresh
gradient and hypothetical full-update direction separately. Use F64 reference
arithmetic with declared F32 configuration values; this is not bitwise GPU replay
or evidence of improved training. Preserve original scale1 results. No new scalar,
loss, runtime or learner is selected by this diagnostic alone.

Compare mean transformed gradients with transformation of the finite-replica mean
and report stochastic spread. This can expose sensitivity to clipping/RMS and
objective interference. It cannot separate all optimizer causes, estimate a true
population gradient, qualify calibrated forecasts or establish sufficient capacity.

The earlier `adam_m` first-update comparison includes LaProp RMS normalization
before momentum. Its near-equal norms are not raw-gradient or clipping-saturation
evidence. Preserve that original audit; clarify interpretation rather than relabel
it as a raw-gradient measurement. Pin the actual423d373 backend sources, not a
neighboring checkout. No package rebuild/port or repeated primitive qualification.
