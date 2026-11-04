# Terminal mixture improves game-ending predictions but worsens intermediate-life predictions

The completed terminal-mixture learner recognizes the selected game endings much
better, while making larger continuation errors at intermediate life losses.
On the same 24 observed terminal arrivals, mean continuation falls
**.759636→.369873** against target zero; squared error falls **.644558→.252775**.
The paired difference is **−.391783**, with lane-bootstrap95% interval
**[−.506700,−.274697]**. All eight lane means improve;19 individual errors fall
and five rise. Predictions below .5 increase from5/24 to17/24.

This does not reverse the [completed gameplay result](2026-11-04-cdp-breakout-terminal-mixture.md):
greedy143.667→85.250 and sampled70.750→65.708 in roughly21 minutes, all0/24
two-wall successes. Retain the existing sampler and continuation scale1. The
original864/90%-two-wall and five-game goal remain unfinished.

## Forecasts and observed arrivals show a tradeoff

The [fixed protocol](../experiments/2026-11-04-cdp-breakout-terminal-mixture-arrivals.md)
reuses the control's completed outputs and120 fixed endpoints:24 game endings
and96 intermediate life losses, equally distributed over eight lanes. The new
candidate alone re-observes43,424 recorded actions/173,660 frames. It uses the
recorded actions, with exact pixel, transition, reset and frame parity. No
candidate-based endpoint selection or repeated control capture occurs.

Prior means one-step prediction before seeing the arrival pixels; posterior
means the feature after observing those pixels, captured before reset. Terminal
target is zero; intermediate-life target is F32(1−1/333)=.996996999.

| Endpoint and state | Control mean | Candidate mean | Control MSE | Candidate MSE | Below .5, control /candidate |
|---|---:|---:|---:|---:|---:|
|Game ending, prior|.952850|.512441|.911006|.372422|0/24 /13/24|
|Game ending, posterior, primary|.759636|.369873|.644558|.252775|5/24 /17/24|
|Intermediate life loss, prior|.979477|.753504|.001530|.170491|0/96 /21/96|
|Intermediate life loss, posterior|.979466|.901959|.003178|.059561|0/96 /7/96|

Terminal prior MSE improves by−.538584, interval[−.598028,−.480791], with all
eight lane means improving. Intermediate-life prior MSE worsens by+.168962,
interval[.106436,.241616], with all eight lane means worsening. Its posterior
MSE worsens by+.056383, interval[.024276,.091056], with seven of eight lane
means worsening. The fixed5,000-resample bootstrap uses seed20261104; intervals
describe the reused eight-lane cohort, not training-seed reliability.

Within the candidate, observing the terminal arrival lowers mean continuation
.512441→.369873 and MSE.372422→.252775. That paired MSE difference−.119647 has
interval[−.200520,−.048021]. Observations also correct part of the intermediate-
life forecast error: mean continuation rises.753504→.901959 toward.996997.
Both posterior comparisons still retain substantial error.

## What this changes in the diagnosis

The higher terminal exposure is accompanied by better terminal predictions on
this fixed cohort. The head can respond to the changed training distribution;
the negative gameplay result cannot be summarized as no improvement in ending
recognition. It is accompanied by more false ending indications at intermediate
life losses, particularly before observing the arrival. This motivates studying
how the model distinguishes a lost life from the end of the game before another
learner is selected.

The subsequent [saved-output ranking check](2026-11-04-cdp-breakout-terminal-ranking.md)
finds posterior AUC.952691→.897569 with paired interval[−.141927,.008257], and261
reversed terminal/continuing pair orders. The interval does not establish lower
discrimination, but the changed orders mean a strict monotone output shift alone
cannot fully describe the observed tradeoff. No correction is selected.

These selected life-loss states are not ordinary ongoing play or a population
calibration set. A below-.5 count is a diagnostic threshold, not a validated acting
rule. The sampler changes all training objectives' data distribution without
importance correction. Native sources, learned representations, state draws and
experience differ from the control. The result does not isolate a head, memory,
sampling or actor mechanism, nor prove that the forecast tradeoff caused the
gameplay regression. No new threshold, loss, mixture setting, learner or automatic
successor is selected by this diagnostic.

## Execution and closure

Use candidate native2b5a3eea/SHA5ce1b443 from
`/x/Temp/kindle-terminal-replay-20261104/runtime`, unchanged from its completed
learner. Control uses447fe50/SHA35ad0db. The exact configurations differ only by
terminal_replay_fraction=.25 and serialization of the existing `imagined` replay-
value bootstrap default; continuation remains1. Native graph sources are pinned
for both versions. No build, overlay, new learning or checkpoint change occurs.

Sixteen CPU tests pass, including the actual entrypoint stopped before native-
agent/emulator construction, exact config/native rejection and paired synthetic
outcomes. The2.375s preflight verifies634 control/343 candidate input pins and
all fixed endpoint joins. All1,052 execution pins remain unchanged.

One guarded capture takes **51.327s**, with zero updates and all357 tensors
finite and invariant. All120 source/repeated queries preserve live features,
actor probabilities and counters. Full input/output pixels and transitions,
per-stream hashes, reset counts and173,660 frames match. The five-tensor F64 head
reference uses each model's own features and weights. Candidate prior outputs
match native with maximum absolute error **2.104e-7**, below1e-4; deterministic
prior/posterior features match exactly and categorical groups remain one-hot.
Posterior outputs are offline reference estimates, not native posterior readbacks.

The single-core/nice10 analysis takes.329s. MainPID1117438/invocation
`1cd73c9d5b3f4a208e47725ac2d29622`, source`6dda986d`, launches05:52:05 UTC
November4 and succeeds **05:53:02**. Independent4.265s CPU closure completes
**05:53:33**, recounting head arithmetic, all paired metrics and tensors and
verifying52 final artifacts plus the original service identity. Host guard passes
without recovery. No failure, repair, cap change or completed-stage repeat occurs.

Root `/x/Temp/kindle-terminal-mixture-arrivals-20261104` is closed. No learner or
GPU allocation remains active. Do not repeat the capture, analysis or closure;
check live state before future work. Retained gameplay and the full five-game
goal are unchanged.

| Artifact | SHA256 |
|---|---|
|Declaration|`f33c4dae25c1db5b67e36db7475bec609d888b222cb3932624fef1c50c24f72a`|
|Candidate capture|`d23c397e5acfaaf1ce786a0c45ecce87bd416fe6a3bff4f4264ce8f62b6f66cc`|
|Paired analysis|`3f251cc0afa272d792d4142cc214690e426221bfcc7f5b48e659004d2afe2121`|
|Main completion|`6dea189e0256230c1064b168d1be823a089a08268739ac662855791a75ceeb5d`|
|Independent closure|`c42f6fc02588c382d1cc48970f3dddf2ee457269ebbd1668d9c11f7576409b31`|
