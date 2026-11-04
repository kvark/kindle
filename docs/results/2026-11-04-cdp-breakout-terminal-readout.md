# Saved Breakout states contain linearly accessible ending information

A fixed class-balanced linear readout identifies **20 of24 game endings** on
held-out lanes using the retained model's saved posterior states. It incorrectly
labels **10 of96 continuing life losses** as endings. Balanced MSE is **.131485**,
below the saved head's.323868 and the balanced constant's.248501. Every lane's
balanced MSE improves; the paired difference−.192383 has95% lane interval
**[−.251523,−.130600]**. The prespecified primary diagnostic screen passes.

This establishes usable linear ending information on this fixed cohort. It does
not establish a production readout, population calibration or gameplay benefit.
The saved head already ranks endings well: its posterior AUC **.952691** is
higher than the readout's.917752. The head's high continuation values and the
balanced readout's increased false positives are a fitting/decision tradeoff,
not evidence that the latent state lacks all relevant information.

Retain the original scale1 recipe and70.750 sampled/143.667 greedy, all0/24
two-wall. No new learner, loss, sampler, inference correction or replacement
head is selected. The original864/90%-two-wall and five-game goal remain unfinished.

## Fixed method and held-out results

The [protocol](../experiments/2026-11-04-cdp-breakout-terminal-readout.md) uses
only the existing120 arrivals:24 game endings and96 intermediate life losses.
Each ofeight folds holds out one complete lane, with105 training and15 test
rows. The ridge penalty is fixed at1.0; preprocessing uses training-only means
and deviations. Training gives both classes equal total weight and fits an
unregularized intercept. No hyperparameter search, new model execution, emulator
replay or checkpoint update occurs.

Targets are0 at game endings and gamma=.9969969988 at continuing life losses.
Reported predictions are clipped to[0,1]; raw predictions, coefficients and all
preprocessing are retained. A prediction is classified as terminal below gamma/2
(.4984984994), with ties continuing. This differs from the earlier descriptive
.5 cutoff: the retained head has4 terminal detections here versus5 there.

| Fixed view | Readout balanced MSE | Saved-head balanced MSE | Readout ending recall | Continuing recall | Readout /head AUC | Paired balanced-MSE interval |
|---|---:|---:|---:|---:|---:|---|
|**Retained posterior, primary**|.131485|.323868|20/24|86/96|.917752 /.952691|[−.251523,−.130600]|
|Retained one-step prior|.160340|.456268|16/24|78/96|.858724 /.791667|[−.329746,−.253159]|
|Continuation10 posterior|.132829|.305758|18/24|84/96|.912543 /.883681|[−.240995,−.108489]|
|Continuation10 one-step prior|.169972|.446810|17/24|79/96|.838976 /.639323|[−.316766,−.234686]|

The constant gamma/2 baseline has balanced MSE.248501 and AUC.5; constant gamma
has balanced MSE.497002. All four fixed views pass the same descriptive screen,
without switching the primary or selecting a decoder. The intervals use5,000
lane-bootstrap resamples, seed20261104. Saved heads use each model's previously
native-qualified F64 readout on its own features; heads and latent coordinates
are never exchanged between models.

For the retained posterior, terminal MSE improves.644558→.155112 while
continuing-life MSE worsens.003178→.107857. Balanced accuracy rises58.33%→86.46%,
but the saved head has zero false positives on the96 continuing arrivals, versus
ten for the readout. These tradeoffs matter: balanced classification at selected
life losses is not the distribution used for imagined continuation discounts.

## What this narrows

Both posterior and one-step prior states retain usable ending information for a
simple readout. The existing posterior head also has strong rank discrimination,
despite assigning high continuation to most endings. Absence of all ending
information is therefore unsupported on this cohort. Readout fitting and output
use remain concrete questions; this result does not isolate the effects of
rare replay examples, objective weighting, shared gradients or stochastic states.

The outcome-conditioned cohort contains20% game endings, unlike full replay or
normal gameplay. Equal class weighting deliberately changes the fitting target
distribution. Its raw outputs must not be treated as calibrated continuation
probabilities for acting. Cross-validation holds out lanes, but reuses one
development policy and a small selected cohort; it is not independent training
confirmation. No new policy, gameplay score or causal architecture improvement
is claimed.

## Validation and closure

Seventeen CPU tests pass, including a separate primal ridge solution, intercept
and constant-feature cases, training-only preprocessing, class/tie metrics,
malformed inputs, exact real-data joins and the command boundary before fitting.
Native-model and emulator imports are explicitly blocked. All886 preparation /
887 execution pins remain unchanged.

The single analysis takes **4.368s** on one CPU core at nice10, completing
**04:36:18.195 UTC November4**, source`b8f670c8`, process1107426. Its wrapper
takes4.469s. Independent **4.242s CPU closure** completes **04:37:28.142**, verifies
all32 saved folds without refitting, reproduces predictions/metrics/intervals,
and checks weighted normal equations to maximum residual **2.010e-14**. All19
final artifact hashes verify. No preparation, fit or closure failure, repair or
completed-stage repeat occurs.

Root `/x/Temp/kindle-breakout-terminal-readout-20261104` is closed. No native
model, emulator, GPU work or learner update occurs; original captures and
checkpoints remain unchanged. Do not repeat this fit or closure. No automatic
successor is selected; check live state before future work.

| Artifact | SHA256 |
|---|---|
|Declaration|`4a505f38e164bf2498a18050614f5db83dd02f647bbb893e9c6602e2444a9ced`|
|Readout result|`412091245560166190fb2c2bb00a041344f223362dec9c84c1e2220a364cc01b`|
|Held-out predictions|`b8eaf1e7926e42ffdae3243f7542167e9a5edc030345ad054fcf8724357c3eb6`|
|Saved folds|`e9dedce49d5229b23ecb04feef45c2ce3b78664832af8245621c88ada78e03cb`|
|Completion|`d82ddbdc5228a8e4bb925fef7bc762c17d23e78e7dadb2144dee63ee69d34859`|
|Independent closure|`45053fb72396e2c08d287ca289e7258c74dcc19d6a52baa47ea752d69307fded`|
