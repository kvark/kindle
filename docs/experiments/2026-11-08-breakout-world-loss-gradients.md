# Breakout: measure the existing world-loss gradient balance

The pixel-residual learner is closed/failed, including fitting data. Earlier
event diagnosis explicitly did not measure shared-gradient interference. The
previous gradient-noise experiment compared two predictive-energy estimators;
it did not separate the current motion RGB, KL and event objectives. Select a
new fixed-weight measurement before another architecture or training change.

Use the retained v13 suffix-event final at8,192 updates, not its extension:
RGB64/12M/B8/T32/H16/full BPTT/Auto,80 logical weights/8,871,955 scalars,
240 saved weights/moments. Spatial fusion on; image feedback, residual copying,
main causal-prior supervision, actor/critic and all other auxiliaries off.
No checkpoint, runtime, public API, training graph or learner behavior changes.
Only a new opt-in Rust test selects a diagnostic scalar as output0; all other
production outputs and inputs remain. No optimizer is configured.

Reuse the already retained fitting-plan batches at indices0,2047,8191 from the
original gradient-noise preparation. Verify their original hashes and exact
full-plan identity. No data collection, old forecast, completed native fixture
or old analysis repeats. These are fitting batches, not held-out episodes.

For four fixed posterior/suffix noise seed pairs per batch, compute gradients of:
observed RGB (unscaled), configured KL, configured observed reward/continuation,
future RGB (unscaled), configured future reward/continuation, and the original
joint loss. Main/future RGB coefficient is100, KL1/.1, event coefficients1.
Repeat joint draw0 once per batch:75 forward/backward steps total. Input hashes
must match across modes for each condition; unchanged reports must agree.
The scalar totals must match the declared decomposition. Export all80 raw
gradients, representing genuinely absent parameter gradients as exact zeros.

CPU checks cover component selection/value/derivative identities with unequal
coefficients, actual tiny suffix-graph inputs and unique names, production reset/
reward/discount/suffix-mask construction, and separate seed streams. Reuse
previous sampling, fusion, CE/BCE and optimizer-reference qualification. Build
only the new test executable; do not rebuild or overlay a Python runtime.

After CPU/source review separately declare one900s AMD host-guarded measurement.
Load every logical weight from the anchor and preserve it bitwise throughout;
optimizer age must remain0. Keep twelve before/after logical-weight snapshots
(six modes), all gradient buffers, losses and input hashes. The original240-tensor
checkpoint, including moments, remains untouched. No images, games or policy are
queried for a new frozen assessment. Stop on real faults/nonfinite data; no host
recovery, NVML or automatic successor.

Review gradients by encoder, posterior, recurrent core, prior, decoder, reward
and continuation families. Check the native joint gradient against the weighted
component sum, normalized residual<=2e-4 relative to the sum of weighted norms
in every family/condition. Repeated joint gradients must be bitwise identical.
Report every condition, pairwise direction/cancellation and four-draw variation;
do not infer population significance or a causal physical-error fraction.

Use the previously qualified F64 whole-parameter AGC/LaProp reference, with the
same anchor moments and hypothetical next age8,193, to compare only the actual
RGB coefficient100 and one predeclared alternative1 (both main and future).
This is local optimizer sensitivity, not a sweep, actual update, calibrated
native optimizer replay or evidence that coefficient1 learns better. Keep all
per-parameter/family results; whole-model norms can hide small important heads.
No coefficient or future learner is selected by this declaration.

Full physics, event timing/calibration, persistent brick memory, late/tunnel/
second-wall coverage, untouched families and fresh learner confirmation remain
required. This diagnostic cannot complete the world-model goal or change any
existing gate. Root `/mnt/data/kindle-breakout-world-loss-gradients-20261108`.
No GPU allocation is active or declared at this protocol's creation.
