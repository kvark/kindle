# Breakout world-only learning on a broader corpus

The fixed 8,192-update pixel-world learner completes, but **does not qualify the
world model**. It learns the paddle and fitting brick patterns while omitting
the ball, including when reconstructing its observed context. This is a failure
before actor/critic integration. The previous short-clip capacity result does
not carry over to this broader dataset with the tested recipe.

The user-selected world-model goal remains active. The five-game campaign stays
parked. Do not extend this learner or repeat its completed stages. The next
selected investigation is an optional logits-based pixel cross-entropy loss,
initially in the standalone world path. Its implementation and GPU experiment
are not part of this completed result.

## Data and training

An input audit finds only commands0/1/3/4 in the previous corpora, leaving14 of
the18 action inputs uncovered. A separately selected collector keeps the fixed
pixel controller and adds random blocks of4–12 commands drawn from all18 IDs.
This collects new coverage, rather than restarting a completed collector.

| Split | Aggregate actions | Executed emulator frames | Game-overs | Minimum examples/action |
|---|---:|---:|---:|---:|
| Fitting |16,384|65,503|22|105|
| Development |8,192|32,755|10|44|

Eight environments use full18 actions, repeat4 and sticky probability.25.
Fitting seeds are151001+1009/lane; development seeds191001+1009/lane. The
splits do not share source seeds. All18 input counts being nonzero does not
establish adequate coverage of each action in every physical state. Serving,
late-game/tunnel/second-wall states and prospective alternative-action branches
remain unqualified. Development has been examined; it is not untouched test.

Training uses the native8,852,675-parameter CNN/categorical RSSM/RGB decoder,
external reward and continuation, with no actor/critic, replay-value, imitation
or intrinsic objectives. RGB64,12M preset,B8/T32/full BPTT,Auto, rates4e-4,
reconstruction100, dynamics KL1, representation KL.1 and free-nat1 remain fixed.
The decoder uses a sigmoid followed by summed squared pixel error. Direct
prior-image training covers16 imagined steps after16 observed arrivals, with
the same reconstruction coefficient and decoder.

A precomputed uniform sampler includes reset-crossing32-arrival windows.
Clipped starts retain genuine incoming reward/terminal targets. Natural resets
clear observed history; prior-image losses stop at the first future reset
without giving boundary labels to imagined state. Terminal arrivals remain
scored. All16,414 fitting arrivals are sampled. The8,192 updates replay2,097,152
arrivals; these are not additional environment actions. Recounted exposures
match the declaration:2,649 terminal rows,1,334 prior terminal rows,52,753
positive-reward rows and2,656 natural reset rows.

Native training calls total **759.246s (12m39.246s)**; the training/evaluation/
checkpoint loop totals **1,032.918s (17m12.918s)**. The difference includes
Python preparation, transfers, frozen exports and serialization; it is not a
GPU-kernel profile. No new environment actions occur during fitting.

## Frozen image and event results

Curves at0/1024/.../8192 updates use32 fixed natural origins/split, first contact
candidates from eight streams and all eligible game-over origins. Forecasts
cover1/4/8/16/32 decisions with four fixed latent draws. They condition on
recorded future controller actions: **retrospective prediction, not prospective
planning**. Repeated draws are not independent episodes or learners.

The final dense exports use256 fitting and128 development origins, one fixed
draw. Ball detection conservatively covers the below-brick region; omitted
positions are untested. Paddle errors use its checked image row, and bricks use
6×18 image-only occupancy cells. RAM does not enter world inputs or losses.

| Final dense observed reconstruction | Fitting | Development |
|---|---:|---:|
| Visible balls reconstructed |**0/220**|**0/112**|
| Paddle within1 resized pixel |94.53%|92.97%|
| Brick balanced error |0.00101|0.19935|

Ball detections are also zero at every scored forecast horizon in these dense
exports and across all four draws in the eight contact candidates/split.
Observed reconstruction failure prevents attributing the result solely to
accumulated transition error or stochastic action execution.

On the32 natural origins, first-draw observed RGB MSE falls from.173255 to
.00009065 in fitting and.172301 to.00091387 in development.
The low image error does not establish physical fidelity. Final eight-step
MSE is.00018134 fitting/.00118829 development, versus persistence
.00068831/.00074465. Fitting image error can beat copying while the ball is
absent; development is worse than copying. Brick balanced error also exposes
a substantial fitting/development gap hidden by full-frame averages.

At dense-development horizons1/8/16/32, positive-reward Brier errors on positive
events are.99262/.96093/.99999/.89833, based on2/4/1/2 events respectively.
Small counts do not establish calibration, but these forecasts plainly miss
the observed positive events. Natural dense endpoints contain no game-overs,
so their small termination errors are uninformative about endings. The separate
one-step ending slice has Brier error.48991 on22 fitting endings and.81482 on10
development endings; an always-nonterminal prediction scores1 on this slice.
The ending-only slice is outcome-conditioned, not population calibration.

Artifacts: `analysis/learning-curves.png`, `analysis/fitting-forecasts.png`,
`analysis/development-forecasts.png`, all frozen JSON/NPZ under
`train/evaluations` and `train/dense-final` in the root below.

## Frozen feature readouts

Fit linear ridge readouts only on observed CNN/posterior features from the256
fitting exports. Choose penalties from.0001/.001/.01/.1/1/10 using fitting-only
four-fold validation grouped by source lane. Freeze each readout and apply it
to128 development observed states and the same model's imagined states.
No realized-future labels train a prior-state readout. Ball position, paddle
position and backward decision displacement use analysis-only RAM references,
scaled to64-pixel units. Ball-absent labels and invalid/jumping displacements
are excluded. This is one latent draw and a small sample relative to feature
width1024/2560; failure is not proof of absent information.

Observed development ball-position mean error is13.269 pixels for CNN features
and13.323 for posterior features, versus16.721 for a fitting-mean control, on123
valid references. Neither places a ball within1 pixel. Posterior displacement
error is2.227 pixels/decision. These readouts do not rescue a precise usable
state behind the missing-ball decoder.

Use common finite context-position/displacement support for each forecast and
its controls. Persistence and constant velocity below use true context RAM,
so they are privileged diagnostic controls, not deployable pixel policies.

| Horizon | Common origins | Posterior-trained prior readout | Persistence | Constant velocity |
|---|---:|---:|---:|---:|
|1|121|13.216|2.507|0.283|
|4|120|13.505|9.468|2.675|
|8|116|14.129|17.460|10.594|
|16|115|12.804|24.812|38.782|
|32|107|14.145|24.927|84.908|

Values are mean Euclidean errors in resized-pixel units. Long-horizon controls
are weak because they do not model contacts; beating them there does not
qualify learned dynamics. The original CPU report compared different supports
when a ball was absent at context but present later. It is preserved under
`analysis/initial-readout-review`; `readout-comparison-repair.json` records the
CPU-only correction. No GPU prediction or emulator collection was repeated.

## Output-gradient candidate

At conservative red ball-core pixels in the dense observed images, true red
intensity averages.43946 fitting/.43938 development, but decoded intensity
averages.000323/.000293. Every supported core prediction is below.01.
Thus the decoder is confidently painting black where the ball should be.

For sigmoid output `p` and target `y`, squared-error logit derivative is
`2*(p-y)*p*(1-p)`. Soft-target cross-entropy has derivative `p-y`. At these
saved outputs the median magnitude ratio is2,655× fitting/2,514× development,
before any global coefficient. The mean absolute MSE logit derivative is
.000286/.000253, versus.43914/.43909 for cross-entropy. The checked core pixels
account for31.37% of fitting squared error but3.13% of development error.

This is **analytical output-level arithmetic**, not a measurement of parameter
gradients or proof of the optimization cause. It motivates a controlled
logits-based cross-entropy experiment, with soft RGB targets and unchanged data,
architecture, initialization, sampler and frozen evaluations. This changes
gradient magnitudes and the relative weighting against KL; an improvement
would not isolate saturation from all other objective effects. No labels or
emulator state need enter training. Keep the default squared-error path and
restore compatibility, qualify native loss/derivatives and frozen restore,
then declare a bounded comparison. Do not grow capacity or resume actor/critic
as a substitute for resolving this reconstruction failure.

## Qualification and identity

Root: `/mnt/data/kindle-breakout-world-corpus-20261107`.
Execution source `2774e4f24e842ac323cbfa0713fa2c6103002903`; native build source
`49c99fb1b9e4eff9eec43dadade3dabda7892736` (subsequent example-runner changes
do not alter packaged native/Python-library source). Native SHA256:
`1b9d0dc6d2667de13b84dbce9b3e089d5fba20c1113c31e40a2469b8140576b4`.
Runtime is `root/runtime`, with its exact matching runners/configs.

Thirty-two CPU world tests (18 GPU tests ignored),60 selected Python tests and
strict workspace/bindings Clippy pass. The initial loop-style Clippy failure
and corrected check remain preserved. Label validation preserves the original
ball-contaminated paddle reference and its corrected row/component measurement.
The corrected paddle reference supports1024/1024 images, mean discrepancy.116,
maximum.609 resized pixels. Original constructor/import failures from earlier
world-model stages remain in their original roots.

All four guarded stages complete: reset-mask fixture, all18 collector, learner
and frozen restore. The fixture tests partial/all-zero masks, unchanged prior
loss under altered masked future images, feature readback and clipped-start
terminal targets. Restore reproduces16 saved arrays exactly on eight development
origins. Final frozen evaluation leaves all204 parameter/moment tensors exact;
all68 logical parameters change from initialization. The CPU closure review
verifies278 pins, all8,192 finite update rows, exact source sampling plan,
corpus hashes/counters and source archive. No real GPU fault, numerical failure
or host recovery is reported. Hardware performance qualification is not claimed.

`review.json` and `analysis` retain these receipts. Archived execution source
`source-2774e4f2.tar.gz` SHA256:
`a2db0b135d0210bc3805642070c1c1499d6d3b45f6a2f29f212531ee3585693d`.
No GPU job is active after closure. Untouched test, prospective branches,
late-game coverage and fresh confirmation remain unfinished. Pixel supervision
made this failure externally visible; it does not establish pixel superiority
over CDP or attribute the earlier coupled agent's errors to this mechanism.
