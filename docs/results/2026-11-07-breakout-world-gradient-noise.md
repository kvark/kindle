# Breakout: real predictive-gradient noise at fixed world weights

The corrected conditional score-function estimator has much greater sampling
variance than the earlier straight-through estimator at the same frozen model.
Across three fixed development batches, prior variance increases550–893×,
reward variance1,250–11,699× and continuation variance57–207×. Exact repeated
inputs reproduce every gradient bitwise. This is evidence of estimator noise
and optimizer sensitivity, not a GPU fault or a complete explanation of the
[failed learner](2026-11-07-breakout-world-score-function.md).

The [declared diagnostic](../experiments/2026-11-07-breakout-world-gradient-noise.md)
is **closed**. No learner, collector, emulator, new forecast or optimizer step
ran. The world-model goal remains unproven; no successor is selected here.

## Measurement

Root `/mnt/data/kindle-breakout-world-gradient-noise-20261107`. Source
`7c66bf127465d8d3201cf180c20aec31dee230b7`; test executable SHA256
`636bf7e5f9e40ca05cd57bd86c486c42ec5ca1853317f5f321bac035412a4537`.
This adds a test-only measurement module, with no production graph/API/package
change. Retained training runtimes keep their original identities.

Use the [coverage](2026-11-07-breakout-world-coverage.md) checkpoint at8,192
updates,68 logical parameters/8,852,675 F32 values,RGB64/12M/B8/T32/Auto.
Select fitting-plan indices0,2047,8191 before inspecting outcomes. Positive
reward/terminal/reset/valid row counts are6/1/9/247,6/1/9/247,7/2/10/246.
Only cached RGB,actions,external rewards and natural termination enter the graph.

At each batch, compare observed-motion loss twice with16 predictive draws plus
one repeated draw under each energy estimator. Posterior uniforms stay fixed;
prior/head/encoded-target streams match between estimators. Total108 raw
forward/backward steps. Sessions have no configured optimizer and their step
counters remain0. All68 parameters match the anchor after each batch; six
before/after snapshots contain408 exact logical tensors. The original204-tensor
anchor, including its optimizer state, remains untouched.

All shared component outputs1..8 match control, and all13 predictive reports
match between estimators within declared tolerance. Total0 equals observed
total plus100×energy. The protocol's initial all-first-nine wording was corrected
before GPU execution because total0 contains the additional loss. Auxiliary and
common input hashes match as declared. Paired decoder-gradient relative
differences are4.99e-7–6.14e-7; repeated-input differences are exactly0 throughout.

## Raw gradient evidence

Subtract the observed-control gradient from each predictive gradient. Values
below concern that additional gradient, conditional on the fixed posterior
draw. Noise is the square root of trace sample variance across16 replicas;
mean error is noise/4. Neither is an uncertainty interval for the whole dataset.

| Fixed batch | Observed prior gradient norm | Corrected auxiliary mean norm | Single-draw noise norm | Mean sampling-error norm |
|---|---:|---:|---:|---:|
|0|7.121|109.431|393.874|98.468|
|2047|7.302|118.097|459.508|114.877|
|8191|8.010|114.342|433.319|108.330|

Even the16-draw prior mean is poorly resolved. Debiased signal-norm estimates
are27.39–47.74, versus single-draw noise393.87–459.51: estimated ratios.060–.121.
The earlier estimator's corresponding ratios are2.125–2.739. These estimates
have finite-sample uncertainty and compare different estimators, not two proven
population gradients. Truncating an estimate at0 does not establish absent signal.

The recurrent core also becomes noisier: corrected auxiliary noise192–239
versus28–37 earlier. Encoder/posterior noise increases too. The decoder remains
nearly identical between paired estimators. The large decoder norm dominates
whole-model summaries, so a whole-model noise statistic would obscure the
much smaller prior/reward/continuation networks.

Reward-head corrected auxiliary noise is12.50–25.99 against observed-gradient
norms.042–.075. Reward auxiliary mean/control cosines are.144,−.261,−.306;
the earlier estimator also conflicts on the latter two batches. Continuation
corrected means are poorly resolved in two batches. This does not establish
that every useful predictive gradient should align with reconstruction or head
supervision, but it rules out treating these gradients as small perturbations.

## Optimizer sensitivity

CPU F64 references apply the pinned whole-parameter AGC formula and LaProp to
observed gradient plus auxiliary multipliers0,.001,.01,.1,1. They use F32
configuration values, the coverage anchor's moments and next step8,193. These
are hypothetical local updates at this checkpoint, not actual training or
bitwise shader replay, including its subnormal-handling details.

At multiplier1, corrected prior normalized-fresh-gradient means have control
cosines.077–.110 and noise norms3,117–3,171, versus mean norms967–1,042.
Transforming the16-replica mean instead of averaging transformed replicas
changes the vector substantially: gap norms3,289–3,541. Actual optimizer
nonlinearity therefore matters; averaging raw gradients is not equivalent to
averaging the resulting updates. All scales and families are retained in
`gradient-review.json`, including per-parameter AGC factors.

The hypothetical full momentum updates retain prior control cosines.878–.885;
past moments buffer the fresh signal. Fresh-gradient and full-update directions
must not be conflated. Smaller auxiliary multipliers reduce perturbation but do
not improve the auxiliary estimator's intrinsic signal/noise ratio or establish
better forecasts. No multiplier or learning recipe is selected by this audit.

The earlier `first-update-gradient-review.json` compared `adam_m` tensors.
LaProp normalizes by RMS **before** momentum; those tensors were not raw
gradients, and nearly equal norms did not prove AGC saturation. Preserve that
original descriptive audit with this clarification. The present analysis pins
the actual Meganeura423d373 shaders/runtime, not a neighboring checkout.

## Validation and limits

Native measurement43.282s; CPU analysis77.774s.50 CPU world tests,23 GPU tests
ignored in that CPU run, strict workspace/all-target Clippy and five CPU optimizer
reference checks pass. Preserve the initial Option-size compilation error and
the corrected fixture's replay-value assertion failure in their original logs;
the final fixture clears behavior objectives and passes. No previous worker reran.

Native startup rechecks28 data and21 execution pins. CPU closure rechecks those
plus123 analysis pins, including all108 raw gradient files. One host guard and
its nine sealed-file audit pass, with no recovery, unfinished child, numerical
failure or active Kindle worker at closure. `closure-review.json` records this.

Three fixed fitting batches at one trained coverage checkpoint cannot identify
the cause of an entire fresh learner's regression. Posterior straight-through
BPTT remains, and even16 samples poorly resolve some corrected means. This
diagnostic does not establish capacity sufficiency, calibrated futures, improved
learning or a successful variance-reduction method. It gives a concrete reason
to investigate estimator variance and objective coupling before adding capacity
or training longer. Precise1/4/8/16/32 forecasts, reward calibration, brick memory,
late-game/tunnel/second-wall data, untouched tests and fresh confirmation remain
unfinished. Actor/critic and CDP efficiency comparisons stay parked.
