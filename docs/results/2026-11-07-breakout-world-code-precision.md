# Observation-code localization error has both shared bias and draw variation

The [CPU-only decomposition](../experiments/2026-11-07-breakout-world-code-precision.md)
is closed. On deduplicated images detected in all four draws, empirical shared
position-bias shares are59.29% fitting/57.76% development, versus40.71%/42.24%
centered variation. Neither reaches the predeclared2/3 dominance threshold;
triage is **mixed**, not a diagnosis of an isolated encoder or sampling cause.
The failed observation-code learner stays closed; full world competence is unproven.

## Fixed evidence and support

Use all six original final periodic reconstruction exports, in fixed fitting/
development natural/contacts/endings order, all four fixed draws. All164 rows
join their original selected lane/current frame exactly; all96 published detection/
accuracy checks match. Hash current RGB before examining predictions and retain
the first occurrence per split. There are100 fitting/50 development unique images;
14 repeated fitting images are removed. Related images/draws are not independent
episodes or untouched tests. Current image is fixed; this new posterior/decoder
has no h input, but the pixel detector is nonlinear and its variability is not an
isolated physical latent-noise measure.

|Original cohort|Visible origins|Detected supported draws|Within1 pixel|All4 detected origins|
|---|---:|---:|---:|---:|
|fitting/natural|20|74/80|42/80|16|
|fitting/contacts|16|63/64|57/64|15|
|fitting/endings|0|0/0|0/0|0|
|development/natural|26|101/104|56/104|23|
|development/contacts|8|32/32|30/32|8|
|development/endings|0|0/0|0/0|0|

Deduplicated fitting/development have33/34 visible images,125/132 and133/136
supported detections,87/132 and86/136 within-one-pixel draws. Full accuracy is
65.91%/63.24%; missing predictions remain in these denominators. The decomposition
uses only28/31 images detected in all four draws,84.85%/91.18% of visible images.
This is outcome-selected support; do not generalize its conditional error split
to missing predictions, all game states or dynamics forecasts.

## Exact empirical decomposition

For each supported image and draw, delta is predicted minus true2D ball center.
Verify mean(||delta||²)=||mean(delta)||²+mean(||delta-mean(delta)||²), in resized
pixel². No average prediction is scored as a deployable predictor and no closest
sample is chosen. Finite-draw correction subtracts centered variance/3 from the
mean-square bias estimate; negative estimates are retained.

|Deduplicated, all detected|Fitting28 images|Development31 images|
|---|---:|---:|
|Mean total squared error|1.281573|1.427628|
|Squared mean-error component|0.759798|0.824591|
|Centered component|0.521775|0.603037|
|Empirical mean-error share|59.29%|57.76%|
|Finite-draw corrected mean-error share|45.72%|43.68%|
|Negative per-image corrected estimates|11/28|5/31|

Both split supports exceed16, but neither empirical share reaches2/3 or falls
below1/3. Corrected shares45.72%/43.68% likewise do not identify one dominant
source. Four fixed samples give no confidence interval. This does not establish
encoder aliasing, insufficient capacity, optimizer/KL coupling or a useful remedy.

## Execution and next experiment

Root `/mnt/data/kindle-breakout-world-code-precision-20261107` retains49 input pins,
all164 row records,source/RGB hashes,cohort/deduplicated tables and exact scripts.
All supported-row algebra identities and two analytic fixtures (MSE4/bias0/
variance4 and MSE29/bias25/variance4) pass; the negative zero-mean correction is
preserved. Execution0.617s CPU, no model/native/emulator import,
query, fit, optimizer, learning or GPU allocation. The existing pure NumPy
instrument is loaded directly and its canonical byte identity is pinned.

The encoder prefix is byte-identical to qualified stride-two source22c262ea.
Select a separate [strided/lower-mixing recipe](../experiments/2026-11-07-breakout-world-code-stride.md)
using existing flags, not a new loss: stride-two encoder and unimix.001 under the
observation-only code. This is a practical combined hypothesis, not implied by
mixed triage or an isolated repair. Earlier whole-agent stride/mixing failures
remain negative; no completed old stage repeats. Composition qualification must
precede separately declared learning; no automatic successor or gate change.

Precise causal forecasts, calibrated events, brick memory, late/tunnel/second-wall
coverage, untouched families and fresh confirmation remain unfinished. Actor/
critic and CDP comparisons stay parked; full goal remains active.
