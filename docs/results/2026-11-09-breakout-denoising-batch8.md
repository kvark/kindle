# Smaller batches improve fitting, but not recursive prediction

At the same65,536 training-target exposures, B8 with8,192 optimizer updates
improves actual-history one-step ball matches to **178/174/176/173 of198**
(87.4–89.9%), versus B32/2,048 updates at140/149/140/140 (70.7–75.3%).
Ordinary16-step rollouts fall to **33/31/29/28 of110**, versus35/38/34/33.
Both recipes pass **zero of24 fixed gates**. Retain B32/reward32 as the ordinary
rollout control; keep the B8 checkpoint as distinct fitting evidence.

| Measure | B32/2,048 | B8/8,192 |
|---|---:|---:|
| Training targets |65,536|65,536|
| Ordinary matches, pooled reused draws |140/440|121/440|
| Actual-history one-step matches, pooled reused draws |569/792|701/792|
| Native learn-call interval |161.420s|218.921s|
| Training-loop interval, including milestone outputs |169.299s|228.929s|

The loop costs 35.22% more.
These intervals include host dispatch/synchronization on a shared host and do not
isolate kernel speed. Four draws share one learner and selected fitting clips;
they are not independent training seeds or held-out tests.

The early curve is promising but does not persist in the ordinary aggregate:
at32,672 exposures B8 matches25/30/27/28 versus16/13/11/13. Final ordinary
first-step matches7/7/6/7 of8 versus6/6/5/6, while eight-step matches2/1/0/1
versus2/6/4/3. Sixteen-step matches1 each versus0/1/0/0. Keep every horizon;
these mixed changes do not establish accurate long forecasts.

All detected one-step balls are within one pixel:20/24/22/25 of198 supported
cases have no detectable ball, with zero ambiguous masks. Conditional median
position error is .043–.058 pixels and motion projection1.000 on165–170 pairs.
Those conditional values exclude missing detections. Ordinary predictions have
76/77/78/81 missing balls of110; only29–34 are detectable, and28–33 are correct.
Improved fitting on actual histories does not resolve prediction using generated
histories. This comparison does not isolate why that gap persists.

Observed noisy-image denoising matches7/7/7/8 of8, versus6/6/7/6; this task uses
a noisy actual image and is not a causal forecast. At ordinary horizon16/draw0,
RGB MSE improves .001309→.000977 and paddle matches3/8→4/8, but four false
brick removals appear versus zero. The cached curve JSONs retain all paddle,
brick, changed-pixel, ball and false-detection metrics at every horizon/draw.
No static-wall score establishes removed-brick memory.

Ordinary zero-reward RMSE is .000005694 versus .000031793. Teacher reward RMSE
regresses .0314885→.0348437; the one causal positive is approximately .992675
for1. Continuation ending Brier increases from approximately1e-10 to9.018e-6
on teacher outputs, though no alarms occur. All evaluated labels are nonterminal;
there is one distinct causal positive repeated across draws. Neither calibration
nor ending prediction is proved by this outcome-selected slice.

Only train_batch_size changes32→8. CPU checks verify the same224 source arrivals,
65,536 per-source exposures,588 positive exposures and32,768 forced-max-noise
targets. Each old group becomes four consecutive eight-lane micro-updates. Actual
four-frame prefixes,RGB64,seed41001,learning rate.001,residual/noise formulation,
loss,reward support andthree-stage sampler remain unchanged. Averaging,gradient
variance,Adam age andRNG interleaving change together; noise is not paired.
No target coordinates or emulator state enter training. No game actions/frames.

The fresh native constructor matches all135 initial tensors and20 initial model
arrays exactly to the cached control. All8,192 loss rows are finite and account
for one update/eight targets each;all45 parameters change. All135 frozen tensors
and20 restored model arrays match exactly;config andages are checked at every
milestone. Both original guarded jobs finish successfully with18 sealed guard
files;no native job or audit is repeated. No source/build/runtime change or
manually perturbed checkpoint is used. Native60b9b672/source8dbf9ab5 is reused.
Root `/mnt/data/kindle-breakout-denoising-batch8-20261109`, invocation
`c3c807c9794f488a9040c245f4b61870`.

The archive contains1,376 endpoint rows,all event/pixel/motion metrics,complete
curves,checkpoints and16 fixed lane0/draw0 lossless frames comparing recorded,
actual-history one-step andgenerated-history images. No detector threshold or
best-case substitution. Full physics/events,natural calibration,changed-brick
memory,late/tunnel/second-wall/above-top,untouched families andfresh confirmation
remain unproven. No automatic successor is selected. The broader world-model
goal stays active;actor/critic,CDP comparison andfive-game work stayparked.

Closure verifies 215 pins,one successful terminal service and18 sealed guard files.
