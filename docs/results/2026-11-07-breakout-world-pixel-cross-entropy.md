# Pixel cross-entropy does not resolve broader Breakout ball reconstruction

The [fixed comparison](../experiments/2026-11-07-breakout-world-pixel-cross-entropy.md)
completes8,192 updates and exact frozen restore. **Development ball reconstruction
remains0/112**, matching the squared-error control's failure. Cross-entropy
raises brightness at some true ball pixels and modestly improves coarse latent
readouts, but misses the90% observed-reconstruction diagnostic gate by a wide
margin. Do not extend this learner or call the world model qualified.

The next selected ablation removes the extra16-step prior-image loss while
retaining observed reconstruction and RSSM KL/dynamics learning. It uses the
existing native cross-entropy runtime; no capacity increase or new collector
is justified by this result. Actor/critic and the five-game campaign stay parked.

## Matched experiment

The only selected training-objective change from the
[control](2026-11-07-breakout-world-model-corpus.md) is squared pixel error versus
logits-based soft-RGB cross-entropy, for both observed reconstruction and16-step
prior images. Keep the8,852,675-parameter CNN/RSSM/decoder, RGB64,12M preset,
B8/T32,Auto,seed41001,4e-4 rates, reconstruction coefficient100,KL1/.1/free-nat1,
and the full8,192-update uniform source-sampling plan. Initial204 tensors,
evaluation origins and source samples match exactly. No actor/critic, imitation,
replay-value or intrinsic objectives learn.

Fitting data are16,384 actions/65,503 emulator frames; development is8,192/
32,755 on separate seed families. Eight environments use sticky.25,repeat4 and
full18 actions. The same retained corpus is reused; no collection repeats.
Both learners replay2,097,152 arrivals, covering all16,414 fitting arrivals.
Replayed exposures are not new environment actions. Both see2,649 terminal
rows,1,334 prior terminal rows,52,753 positive-reward rows and2,656 natural reset
rows. Future reset masks affect losses only; terminal arrivals remain scored.

Native training totals **765.084s (12m45.084s)** versus759.246s, a0.77% increase.
The whole training/evaluation/checkpoint loop totals1,003.581s versus1,032.918s;
this includes host work and exports and is not a kernel-performance comparison.
Training-loss values have different units/irreducible soft-target entropy, so
they are not a shared competence measure. Keeping coefficient100 also changes
gradient scale and relative KL weighting; this is not an isolated test of
sigmoid saturation alone.

## Frozen physical results

Primary dense exports use256 fitting and128 development origins, one fixed
latent draw. All source arrays/labels/origins match the control. Periodic
natural/contact/ending curves use four fixed draws and1/4/8/16/32 decisions.
These forecasts condition on recorded future controller commands: retrospective
prediction, not a prospective action/planning test. Draws are not independent
episodes or learners. Ball measurements conservatively cover below the bricks;
unsupported positions remain untested.

| Final observed reconstruction | Squared error | Cross-entropy |
|---|---:|---:|
| Fitting ball detections |0/220|2/220|
| Fitting ball within1 resized pixel |0/220|1/220|
| Development ball detections/within1 pixel |0/112|0/112|
| Fitting paddle within1 pixel |94.53%|94.14%|
| Development paddle within1 pixel |92.97%|88.28%|
| Fitting brick balanced error |.001007|.001239|
| Development brick balanced error |.199349|.190935|

Dense-development ball detections also remain zero at every forecast horizon.
Improvements to scene brightness or coarse readouts do not repair the
missing-object failure. No gameplay assessment is appropriate: no policy learns.

At true ball-core pixels, mean decoded red intensity rises.000323→.040271 in
fitting and.000293→.015100 in development, against true intensity about.4394.
Yet80.95% of development core pixels remain below.01. This changes the observed
output behavior without establishing accurate localization or an upstream cause.

On123 valid development RAM references, frozen fitting-only linear ball-position
readouts change mean error13.269→12.804 resized pixels for CNN features and
13.323→11.281 for posterior features. Posterior readout within1 pixel changes
0/123→2/123; backward displacement error2.227→2.135 pixels/decision. These are
small fitting datasets (256 examples versus1024/2560 features), one latent draw,
and analysis-only labels. Readout failure does not prove information absent.

The posterior-trained readout applied to prior states has one-step development
ball error11.609 versus2.507 persistence/.283 constant velocity, on the same121
finite-context origins. Controls use true context RAM, so they are privileged
diagnostic references, not deployable pixel policies. Longer-horizon control
failures cannot qualify the learned model. Use common finite supports throughout.

Positive-reward Brier errors on dense-development positive events are
.99772/.93924/.99564/.99780 at1/8/16/32 decisions, on2/4/1/2 events. Event counts
are too small for reliability claims and the observed positives are mostly
missed. Separate one-step ending Brier changes.48991→.37246 on22 fitting
endings and.81482→.79882 on10 development endings. This ending-only slice is
outcome-conditioned, not population calibration. Natural dense endpoints have
no endings, so their small aggregate terminal errors cannot establish competence.

Artifacts under the root below include `analysis/comparison.json`,
`learning-curves.png`, `development-forecasts.png`, `readouts.json`,
`output-gradient-arithmetic.json`, all original JSON/NPZ and frozen checkpoints.

## Coverage and label audit while learning

A CPU-only audit hashes each reused source file and counts the available states:

| Exposure | Fitting | Development |
|---|---:|---:|
| Life losses |122|70|
| Paddle-height reversal candidates |368|168|
| Positive-reward arrivals |408|189|
| Maximum completed/censored episode score |56|34|
| Ball arrivals above top brick row |0|0|
| Actions at cumulative score≥216 or≥432 |0|0|

Native-image brick occupancy ranges76–108 fitting/83–108 development. All18
commands occur globally, but command17 has no fitting example after a ball-
absent state; commands7/11/15 have none in development. Ball-absent is not an
exact serving label. Reversals are candidates rather than a collision oracle.
These are exposure counts, not learned competence. Late-game/tunnel/second-wall
and complete state/action coverage remain missing. Early states already contain
the visible balls used by the failed reconstruction check.

An independent native-pooled color-majority measurement over each6×8 brick cell
finds17 fitting/10 development disagreements with the resized center-color
instrument, out of1,772,712/886,680 cell measurements. All27 are false presence
in the resized instrument. Native-image removal counts408/189 match positive
reward-arrival counts and have zero apparent additions; resized counts425/199
removals and17/10 additions include these brief artifacts. Both are image
measurements, not emulator-state truth. This limits precise collision/removal
timing claims. Scored metrics are unchanged for the comparison; the20% model
brick balanced error is much larger than this instrument discrepancy.

`coverage.json`, `coverage-review.json` and `brick-label-review.json` retain the
counts, source hashes and examples. The first coverage script fails on an
unflattened6×18 array; its source/log remain beside the corrected CPU result.
No GPU or emulator is constructed by these analyses, and labels never train
the world model.

## Qualification and retained failures

Root `/mnt/data/kindle-breakout-world-bce-20261107`.
Execution source878f66be, unchanged native build535de479, SHA256
`adfc2f9ec8e9174b56628ab8c390516ee6acee31dc4151418208bcfdf7f59c54`.
Runtime is the exact matching `root/runtime`; no rebuild is needed for the
selected no-prior-image ablation.34 CPU world tests,60 packaged Python tests
and strict workspace/bindings Clippy pass. Native F64 loss/derivatives,12-update
mask/restore fixture and full-size matched-init/legacy-restore/one-update worker
all exit0.

The initial qualification queue **fails its declared count** (17 expected,
16 reported). Preserve that failure. A separate CPU repair verifies304 pins and
all18 saved arrays including horizons/locations; no GPU phase repeats. Future
runners explicitly compare18 arrays. Original CPU bias-shape, F32-reciprocal
tolerance and installed-package import failures remain retained separately
from their corrected checks.

The selected learner/restore queue completes. Final frozen exports preserve all
204 parameter/moment tensors exactly; all68 logical parameters change from
initialization. Restore reproduces18 arrays exactly on eight development origins.
The initial closure audit uses a nonexistent local corpus path despite reuse;
its source/log and already-created source archive remain. A separate CPU repair
checks the correct corpus,306 pins, all8,192 finite updates, exact sampler,
finite/frozen checkpoints and archived source. Do not relabel the failed audit
or qualification queue passed. No learner, evaluation or collector repeats.

Source archive `source-878f66be.tar.gz` SHA256:
`51ae332e3597139e3987265508f652cf6113147b4d9a90edf1260dee5be1e8b3`.
No real GPU fault, numerical failure or host recovery is reported. No GPU job
is active at closure. The world-model goal remains active and unproven.

## Next isolated question

An independently sampled prior-image loss minimizes expected error against one
realized future. For uncertain outcomes, both squared error and pixel
cross-entropy can favor conditional mean images over separated physical futures.
For a two-outcome toy image with equally likely left/right ball positions,
independent pointwise squared error is minimized by two half-intensity balls.
Posterior-conditioned reconstruction has the realized image available to its
recognition model and can instead associate the latent with that realization.
The shared decoder and latent alignment permit interaction between these losses.

This is a reason to test interference, **not proof that it caused this failure**.
The next selected change is prior-image horizon16→0, with observed cross-entropy,
RSSM KL/dynamics, reward/continuation and all data/initialization/evaluation fixed.
Zero forecast-image loss does not remove transition learning or open-loop tests.
Qualify one full-size update/restore and nine-loss bookkeeping with the retained
runtime, then declare a separately bounded comparison. No capacity increase,
new data, unchanged extension or actor/critic resume is selected. Passing an
observed reconstruction diagnostic would still leave forecasting, prospective
branches, untouched tests, late-game coverage and fresh confirmation unfinished.
