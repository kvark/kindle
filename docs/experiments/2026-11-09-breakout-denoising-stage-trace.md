# Frozen native denoising-stage inspection

The [half-max-noise learner](../results/2026-11-09-breakout-denoising-half-max-noise.md)
has11/17/16/14 correct one-step balls of198,but most predictions miss orcopy
previous positions. Select a new frozen diagnostic of this retained checkpoint:
inspect whether useful spatial estimates appear early andare lost later,orare
absent throughout. No learner,extension or closed assessment isrestarted.
Root `/mnt/data/kindle-breakout-denoising-stage-trace-20261109`.

Add a test-only native collector,withno production code/API/config/network/loss/
sampler change. Build its release executable andrun16 existing focused CPU
checks;the three GPU checks remain ignored until explicitly selected. Verify
production denoiser/network/Gaussian/runtime/CNN source bytes match the retained
compiled checkpoint source. No new wheel orbackend primitive. All computation
uses native Meganeura+Blade;Python prepares bytes andanalyzes outputs only.

The new diagnostic explicitly restores the current half-max-noise v4 checkpoint,
not other historical models. Forall27 one-step origins5..31,alleight recorded
lanes andallfour draws83001/83003/83009/83023,retain the clean-image denoising
estimates atallthree actual Euler stages. Give the trace helper only the observed
prefix andone incoming command per lane;expected outputs/target images are not
arguments. Keep the real sampler's residual coordinates,noise/RNG androunding.
Make108 traces/324 direct denoiser calls plusfour production predict calls for
instrumentation parity,not training or counterfactual intervention.

Decode only this selected checkpoint's cached teacher andorigin16 outputs for
required exact parity. Every final batch image must match the cached teacher
(108 batch arrays/864 RGB images),andall20 origin16 model arrays must match.
The production calls useonlyone future action,comparing cached first-step arrays.
Require model/optimizer age2,048 unchanged andall135 saved tensors exactly frozen.
If parity fails,preserve original failure/partial outputs andwithhold physical
interpretation;no automatic rerun,repair by weakening tolerance or new allocation.

Pin source/executable/input/case/checkpoint/cache/host receipts before a single
300s host-guarded job. Inspect live process identity;no timeout implies a stopped
job. Stop on real GPU/numerical faults;no reset/reload/NVML. Terminal success and
sealed guard audit precede CPU analysis. No emulator action/frame,collector,
actor/critic,CDP comparison or old learner/query queue restart. Release only own
disposable build cache after retaining executable/source/evidence.

Analyze every draw/lane/origin/stage:visible/coherent/missing ball candidates,
errors to next andprevious positions,movement projection,stage-to-stage creation/
loss ofnext-position matches,andseparate positive/negative motion-region pixel
signals. Coordinates/colors are analysis only anddo not enter the model.
Intermediate clean denoiser estimates can be conditional means rather than
samples;their point scores are descriptive,not alternative forecasts orgates.
Do not select the best stage orreplace the served sampler. Changed inputs
between stages prevent unique causal attribution to sigma alone. No automatic
training orsampler successor follows this diagnostic.

All original fitting/physical/event/calibration/brick-memory andlate/tunnel/
second-wall/above-top/untouched-family/fresh-confirmation requirements remain.
Full oracle lacks above-top endpoints;this frozen diagnostic cannot prove the
world model. The full goal remains active andunproven.

The [completed result](../results/2026-11-09-breakout-denoising-stage-trace.md)
passes exact parity/freeze andfinds weak motion already atstage1,creation/loss
ofcorrect positions atstage2,andno match changes atstage3. New-ball RGB-change
projection isabout10–11%,versus55% old-location erasure. A CPU residual-scale
statistic suggests a separate preconditioning hypothesis,not a proven cause.
The diagnostic isclosed;no learner orsampler successor isselected.
