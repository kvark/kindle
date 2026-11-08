# Breakout: fitting launches expose averaged futures

The frozen corrective model still gives imperfect launch forecasts on the new
examples it trained on. More importantly, the retained real futures can differ
substantially for the **same past images and command**. A single sampled model
frame can contain faint copies of multiple ball outcomes, nearly unchanged
across four draws. This supports investigating the stochastic training objective
before another capacity or data change. It does not isolate every failure's cause.

[Protocol](../experiments/2026-11-08-breakout-world-launch-fitting.md).
Root `/mnt/data/kindle-breakout-world-launch-fitting-20261108`.
The [corrective trial](2026-11-08-breakout-world-launch-corrective.md) remains failed.

## Fitting predictions, not a replacement development assessment

Use only corrective final20,480, exact v18 native/source/runner7f797a2b,
SHA38680c73, size1M/RGB64, Auto. All288 retained launch branches enter36 groups:
18 commands × two source repetitions × eight parent lanes. Observe arrivals16–31,
forecast32 submitted commands, use four fixed latent draws and five endpoints.
H16 was supervised;17–32 extrapolate. RGB came from the original one GPU resize.
No future pixel/outcome enters a prediction. No other model, old forecast or
completed diagnostic is queried; no optimizer, gradient or collector runs.

Pooling nine FIRE commands, both source futures and four model draws:

|Decisions|Supported visible truth pairs|Model detected|Within1 resized pixel|
|---|---:|---:|---:|
|1|576|532|370|
|4|576|496|204|
|8|576|496|173|
|16|0|untested|untested|
|32|576|156|0|

All nine non-FIRE commands have zero false-ball detections at every endpoint:
576 absent-truth pairs per endpoint. At16 the conservative below-brick detector
has no FIRE truth support; this does not mean the physical ball is absent.
Every command/horizon/draw, paddle measurement and ordinary reward/ending report
is retained. Missing model detections remain failures on supported truth.

Eight source seeds provide only four recent observed image histories. Repeated
conditions, source futures and draws are related, not independent episodes.
These point counts cannot by themselves diagnose fitting quality for stochastic
outcomes: a correct sampled future need not match a particular realized future.

## Real conditional variation versus model samples

A separately pinned, post-hoc CPU support analysis compares the two source
futures within each of72 FIRE-command/parent conditions. Both share the exact
observed prefix and command. Model predictions are byte-identical across these
two source repetitions when the model seed is held fixed, confirming that their
different future images did not enter inference.

|Decisions|Two real outcomes agree within1px /72|Mean real separation, px|Median maximum model-draw separation, px|
|---|---:|---:|---:|
|1|36/72|6.949|.0326|
|4|36/72|8.257|.2544|
|8|45/72|9.627|2.1205|
|32|36/72|11.618|9.7672|

The model-separation statistic includes only conditions with all four model
balls detected:63/60/60/11 conditions respectively. Unsupported conditions remain
explicit. Two real outcomes and four model draws cannot establish the population
distribution, globally prove mode collapse or supply a new retention threshold.

The fixed `conditional-fire-preview.png` uses FIRE1/parent0, both real futures
and all four model draws at1/8/32. At the first step the two real ball locations
are widely separated; the generated frames show faint multiple ball images.
Later frames spread/fade. No best case or matching draw is selected. The ball
instrument can average disconnected red fragments into one centroid; a returned
center is not proof of one coherent ball.

Image-only energy with all paths alive is .121910/.232072/.282567/.953123 at
1/4/8/32. The privileged empirical two-label energies are
.027145/.032253/.037606/.045383. These reuse the unchanged canonical energy code
and sealed scalar-math receipt, without rerunning old fixtures or model inputs.
The empirical reference uses future labels and is not a population optimum.
These image-only scores ignore model ending errors and are not world gates.

## Integrity and next decision

Native inference completes in31.309066s with390 declared pins. All240 source/
before/after tensors, metadata and ages remain identical; zero training updates,
emulator actions or frames occur. The allocation terminates successfully and its
existing nine-file host guard passes.

The original CPU report fails while serializing NumPy coordinate arrays after
the numerical joins/checks. Its source, log and partial JSON remain. A separately
pinned CPU-only repair converts those coordinate containers to lists and writes
new output names; the original failure is not relabeled. It completes the missing
evidence in2.855s:2,016 exact source-field joins,36 finite exports, all pixel metrics
reconstructed and5,760 retained cases. No native inference is repeated. The
post-hoc spread analysis is separately declared and takes .118s. Closure seals502
pins in .340s, preserving the original failed review and separate repair. A
supplementary source-identity check verifies126 exact input/prediction/event
array pairs and the canonical math helper against its sealed fixture dependency.

Next inspect the exact sampled-prior pixel loss and its incentives under multiple
valid futures. This evidence is consistent with averaging, but does not establish
that changing one loss will repair physics, event heads or full-game memory.
Any implementation/learner needs a separate declaration. Existing completed
energy/score-function attempts remain retained; do not repeat them unchanged.
Precise physics, events/calibration, persistent brick memory, late/tunnel/second-
wall/above-top coverage, untouched families and fresh confirmation remain open.
