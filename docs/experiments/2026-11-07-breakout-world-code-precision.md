# Breakout: separate visual bias from variation across code samples

The observation-only code experiment fails retention: it detects105/112 observed
development balls but localizes only65/112 within one pixel. Before selecting
another learner, determine whether sampled-code variation or shared position
bias accounts for most of its measured localization error.

Select CPU analysis only. Root
`/mnt/data/kindle-breakout-world-code-precision-20261107`. No model/emulator import,
query, update, GPU allocation, new readout fit or acceptance-gate change.

Use all six final original periodic reconstruction exports (fitting/development,
natural/contacts/ending), all four fixed posterior draws. Current true RGB and
prediction arrays are already retained. Native qualification and source establish
that this new posterior/decoder do not use h; with current RGB fixed, these model
outputs vary with the sampled code. Pixel-detector response can also be nonlinear;
do not call its measured variance an isolated physical latent-noise quantity.
File order is fitting-natural, fitting-contacts, fitting-endings,
development-natural, development-contacts, development-endings; preserve row order.
Load the pinned pure NumPy instrument file directly, without importing the Kindle
package or native extension.

Use the existing conservative ball instrument. Report all visible-target/draw
detection and within-one-pixel counts, including missing predictions. For origins
detected in all four draws, compute 2D errors delta_d and verify exactly, in F64:

`mean(||delta||²) = ||mean(delta)||² + mean(||delta-mean(delta)||²)`.

Also report the finite-draw mean-square bias estimate subtracting centered
variance/(D-1), D=4; retain negative estimates. This is not a confidence interval.
Do not impute missing detections, choose closest draws or score the averaged
position as a deployable predictor. The all-detected subset is outcome-selected;
keep its support and errors separate from full-cohort competence.

Retain original cohort tables and a deduplicated view by split/current-RGB hash,
selecting the first row in fixed file/row order before inspecting prediction
outcomes. Count original source origins and unique images; do not treat repeated
frames/draws as independent episodes. Verify source joins to original selection
tables and exact current-image hashes. Pin input/source/instrument bytes and check
the decomposition against fixed nonzero-bias and zero-mean analytic examples.

For triage only, if both splits have at least16 unique visible all-detected images,
mean-square bias share >=2/3 favors investigating shared visual error before
sampling suppression; <=1/3 favors investigating code variation. Otherwise report
mixed or insufficient evidence. These thresholds never replace world-model gates
or automatically allocate training. Neither result identifies encoder aliasing,
capacity, optimizer or KL coupling as an isolated cause.

Preserve failures; do not rerun any completed learner, assessment or collector.
Precise causal forecasts, event calibration, brick memory, late/tunnel/second-wall
coverage, untouched families and fresh confirmation remain required. Actor/critic
and CDP comparisons stay parked; the full world-model goal remains unproven.
