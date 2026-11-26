# Breakout still predicts continuation one step before recorded endings

Feeding in the actual intervening observations lowers the mean predicted
**terminal-transition continuation from .996255 to .952850** on the same 24
recorded game endings. All 24 squared errors decrease, but the actual terminal
target is zero. The weakness therefore persists even in a one-step forecast
from an observed state; it is not confined to a long open-loop rollout.

The [fixed diagnostic](../experiments/2026-11-04-cdp-breakout-ending-feedback.md)
completes 43,424 source actions /173,660 emulator frames and 1,080 queries in
**52.878 seconds**, with zero updates and all 357 tensors unchanged. Retained
gameplay stays 70.750 sampled /143.667 greedy in 21m28s, with zero of 24 two-wall
successes. The original 864/90%-two-wall and five-game goal remain unfinished.

## Same ending, more observed context

Reuse all 120 recorded nine-action windows from the [rescue-forecast test](2026-11-04-cdp-breakout-rescue-forecasts.md).
The model initially sees the same scalar posterior history. At each subsequent
outgoing state it receives the actual new observation and forecasts the
remaining recorded actions through the loss. The original nine-step forecast
is reused, with an exact one-step prefix and actor-probability check at all 120
origins. No teacher branches, original captures or collectors repeat.

| Endpoint continuation on 24 game-overs | Saved nine-step forecast | One step after real observations |
|---|---:|---:|
| Mean prediction |.996255|.952850|
| Actual target |0|0|
| Mean squared error |.992532|.911006|
| Mean discounted endpoint value |28.702084|14.451461|

The paired one-step-minus-nine-step squared-error difference is **−.081526**,
with a fixed lane-bootstrap 95% interval **[−.112610,−.053525]**. All eight lane
means and all 24 individual differences decrease; none ties within 1e-6. This
interval describes the fixed eight-lane cohort, not independent learners or
population calibration. Endpoint continuation is one native prediction at the
last transition, distinct from the product of all continuation predictions.

| Remaining forecast steps | Mean endpoint continuation,24 game-overs | Mean endpoint continuation,96 intermediate-life losses |
|---|---:|---:|
|9|.996255|.997175|
|8|.994118|.997205|
|7|.993765|.996840|
|6|.993957|.996564|
|5|.992005|.996385|
|4|.994584|.996203|
|3|.992008|.995731|
|2|.979082|.989527|
|1|.952850|.979477|

The intermediate-life target remains the native F32 gamma **.996997**, because
losing a life does not terminate the game. Their endpoint squared error rises
from .000000426 to .001529616 as actual observations arrive. The improvement
on selected game endings therefore does not establish a general improvement
in continuation prediction. No life-loss target or episode boundary changes.

Multiplying the refreshed one-step predictions gives mean .906718 on game-over
windows, versus .971526 for the saved open-loop product and zero actual survival.
On intermediate-life windows it gives .953951, versus .975353 saved and .973295
actual discounted survival. The refreshed product uses future observations;
it is not a prediction available at the original start. The remaining predicted
endpoint value is descriptive, not a measured alternative-policy return.

## Interpretation

Actual observations provide useful information, especially in the final two
decisions, yet the one-step prior still predicts high continuation at the
recorded terminal arrival. This narrows the investigation beyond accumulated
long-rollout error. It does not distinguish the transition model, continuation
head, learned representation, terminal-example coverage or optimization as
the cause. In particular, the diagnostic has not measured the continuation
head on the actual arrival's posterior state or its class-specific training fit.

The windows were selected because a loss occurred. A realized stochastic ending
need not have been certain beforehand, and these results do not establish
population calibration. Observations, rollout length and later diagnostic
categorical draws change together. Initial parity is exact, but later draws
are not paired between horizons. Scalar re-observation still differs from the
original vector actor's posterior draws. No new loss, native feature, learner,
coefficient sweep or automatic successor is selected by this result.

## Validation and closure

Native447fe50/SHA35ad0db, the qualified corrective runtime and retained
replay-value1 checkpoint remain unchanged. A private copy of the earlier probe
changes only query selection and accounting; the original source and all prior
inputs remain untouched. The copy receipt pins both sources. Eighteen CPU
checks pass, covering fixed windows, variable forecast lengths, prefix parity,
terminal/nonterminal targets, discount products, malformed inputs, real saved
joins and the actual capture entrypoint stopped before native construction.

All 553 preparation / 554 execution pins pass, including 506 prior inputs and 43
prior final artifacts. Capture verifies every input/output pixel, action,
reward, frame, boundary, reset and full trajectory hash. All 120 original prefix
predictions and actor probabilities match exactly. All 120 repeated endpoint
queries reproduce; every query leaves live features, probabilities and counters
unchanged. All 357 tensors remain finite and byte-identical, with the correct
43,424-action counter advance and zero updates. Host guards pass without recovery.

The 52.878s is capture wall time, including scalar re-observation, native
forecasts and export, not isolated GPU kernel time. CPU analysis takes .198s.
Source `714bb790`, original mainPID1101025/invocation`c782fcb701a646e7991130fabef5e24e`
launches 02:53:33 UTC November4 and succeeds **02:54:31.038**. Independent
**3.434s CPU closure** completes **02:56:00.622**, verifies the original service
and journal, independently recounts all 1,080 queries and horizon metrics,
reproduces the bootstrap interval and checks all 357 tensors again.

All 39 final artifact hashes verify. Preparation, capture, analysis and original
collection pass without a failure, repair or repeated stage. The earlier
rescue-probe manifest failure remains preserved in its own closed root.

Root `/x/Temp/kindle-breakout-ending-feedback-20261104` is closed. No Kindle
learner or GPU allocation remains active. Do not restart the capture or
collectors; check live state before future work. Keep the original game gates.

| Artifact | SHA256 |
|---|---|
| Declaration |`ecd4cf394ed76463b6ec1b5d34b72d2d327da8b6256ed420f329f5900c031d5a`|
| Frozen capture |`7245e2c41ebb7da99f706dc8f785eb2640d1a74b2a0905789cdb558922f6709d`|
| Analysis |`80cc93f8d8876f9d5374820291883fcfc1968a52dea290f61311b365a3d56a0f`|
| Main completion |`bbf3ffc49136b7d08a1a7babb5d48c3d9a7d052709777bbd3831b9c66f59920c`|
| Independent terminal summary |`8d4acb3fd0137c29fc716074ebfe6b7986f2d5575ee91a6ffdec26fdba34955b`|
