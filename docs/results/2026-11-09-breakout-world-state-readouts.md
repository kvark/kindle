# Frozen local-world states: current position becomes readable, forecasts do not

The CPU-only diagnostic is complete. Posterior-history training makes current
ball position substantially more accessible to the tested encoder and RSSM
readouts. It does not yield a precise, transferable prior-state readout or
repair the failed RGB forecasts. The retained prior-suffix model has better
RGB forecasts despite very weak results from these partial-state probes.
No learned model changes; the full world-model goal remains unproven.

[Protocol](../experiments/2026-11-09-breakout-world-state-readouts.md).
Root `/x/Temp/kindle-world-state-readouts-20261109`.

## What is being read

Three existing1M/RGB64 frozen cases: shared actual initial, prior-suffix
small64 final16,384 and posterior-history RGB100 final16,384. Six dense exports,
256 fitting/128 development origins, first saved draw, no new inference.
Both trained recipes and all original failed/positive RGB scores stay unchanged.

**The640-value h/z vector is only part of the local world state.** The decoder
also reads24,576 two-image-history values and the incoming command. These probes
omit that image history. Encoder256 is a separate current-image reference.
Weak h/z probes cannot establish information absence or condemn the full state.
The older [state/history interventions](2026-11-08-breakout-world-state-history.md)
already found strong image-history effects in different12M models; that result
does not isolate the cause in these new1M finals.

## Current coordinates

Development support is112 visible true balls. Counts are within one RGB64 pixel;
RMSE is Euclidean pixels. Targets use the unchanged RGB detector, not native
position/RAM labels. Coordinate heads have no presence output.

|Frozen case|Encoder affine /RBF hits|Encoder RBF RMSE|h/z affine /RBF hits|h/z RBF RMSE|Original RGB hits|
|---|---:|---:|---:|---:|---:|
|Actual initial|3 /1|13.721|0 /0|16.848|0|
|Prior-suffix final|0 /1|14.339|1 /1|13.546|103|
|Posterior-history final|38 /55|2.019|11 /24|3.090|108|

The new recipe's current coordinate accessibility changes markedly relative to
both initial and prior-suffix states. This does not establish an isolated training
mechanism: objectives, imagined-event losses, native source and wrappers differ.
The prior-suffix RGB decoder's103/112 is much better than either h/z probe,
consistent with the omitted image-history path being relevant. That comparison
does not causally demonstrate that the decoder ignores its latent state.

## Posterior-to-prior transfer and horizon-specific probes

Observed-posterior probes are fitted once and transferred unchanged to prior
states. Separately, horizon-specific probes are fitted to that horizon's saved
fitting prior features and targets. Both use fitting-only stream-grouped
selection; no development target tunes a parameter. Results below show the RBF
family for the posterior-history final; all affine results remain in the JSON.

|Horizon|Transferred RMSE|Horizon-specific RMSE|Horizon-specific hits /support|Original RGB hits /same support|
|---|---:|---:|---:|---:|
|1|3.662|3.935|27/109|99/109|
|4|5.884|5.954|4/109|45/109|
|8|8.316|9.360|4/107|19/107|
|16|14.807|14.197|3/114|5/114|
|32|19.043|14.757|1/95|0/95|

Horizon-specific fitting does not supply a precise short/medium-horizon rescue.
The1/95 at32 is a conditional coordinate coincidence, not a replacement gate,
presence prediction or learned-world improvement. The affine horizon-specific
head is also weak:13/109,2/109,5/107,1/114,2/95. Failed transfer alone would not
prove information loss, and these additional restricted heads do not prove it
either. Prior draws are stochastic; point error against one recorded future is
not a distributional calibration test.

Readout overfitting/generalization limits are substantial. The new RBF head's
fitting resubstitution hits at1/4/8 are173/177,177/180,176/176, versus development
27/109,4/109,4/107. At8,fitting RMSE is.01 pixel versus9.36 development pixels.
These interpolating fitting scores use future coordinate labels to fit each
head; they do not establish that the original world forecasts fitting futures
correctly. The original RGB fitting eight-step score remains41/176. Current
encoder RBF similarly fits163/163 but reaches55/112 development; the current
h/z RBF fits163/163 but reaches24/112. Few fitting rows relative to256/640
features and reused development distributions materially limit the probes.

## Method and integrity

Fitting-only affine and RBF kernel ridge, five fixed alphas1e-4/1e-3/.01/.1/1,
four source-stream-separated fitting folds. Feature normalization, target means
and unsupervised RBF bandwidth are recomputed inside each fold. Alpha selection
uses pooled fitting CV squared distance; both families and every horizon/case
are retained without development selection. Some choices hit grid boundaries;
the declared grid is not expanded after results. No coordinate clipping.

Labels exist only for the conservative true-ball detector's visible support,
with reset-valid suffix masks. All ten physical fields match across cases,
all feature shapes/values are finite and h-then-one-hot-z layout checks pass.
All36 original image-baseline counts are independently reproduced on the exact
same support, with missing image detections failing. Raw target masks, all144
fitting/development metric rows,42 fits/CV choices/folds, per-origin predictions,
27,648 per-origin CSV rows with conditional errors, and the reviewed four-panel figure remain
outside Git. Unsupported states are retained as unscored targets, not fabricated
coordinates. No above-brick visibility or full-game coverage is added.

Three synthetic checks pass: affine dual/primal equivalence, stream-fold
separation and RBF constant-target behavior. One120s-capped CPU analysis finishes
in1.918s, verifying30 immutable input/source/receipt pins; plotting does not
refit probes. Vulkan is disabled and native model/emulator imports are blocked.
No original source, checkpoint, learner, decoder or completed score is changed.
No analysis failure, retry, native query, emulator action or optimizer step.
Declaration sourceb9c6bd9a; scripts, raw outputs and closed manifests remain.

The diagnostic narrows a readout hypothesis; it does not identify a trainable
fix or a unique cause. Better current-state coding under posterior-history
training coexists with worse free RGB ball forecasts. Full-state image-history
use, recurrent transition learning, decoder behavior and probe generalization
remain distinct questions. No new native intervention, learner, collector,
assessment, extension or automatic successor is selected here. Accurate physics/
events, persistent brick memory, full-game coverage, untouched families and
fresh confirmation remain required; actor/critic, CDP andfive-game work stay parked.
