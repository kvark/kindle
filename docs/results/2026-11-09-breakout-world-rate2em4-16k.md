# Fresh half-rate16k plain H16 world learning

Fixed final16,384 passes **0/3 original dense development gates**. The screen fails; do not adopt this recipe or allocate its conditional branch assessment.

The full Breakout world-model goal remains active and unproven. This tests a lower rate and larger optimization budget together; it does not isolate either change or demonstrate mastery.

[Protocol](../experiments/2026-11-09-breakout-world-rate2em4-16k.md).
Artifacts: `/x/Temp/kindle-world-rate2em4-16k-20261109`.

## Fixed final RGB predictions

First fixed draw, within one RGB64 pixel; missing/diffuse balls fail. Supports vary by horizon and related origins are not independent episodes. No best checkpoint/draw/tolerance or coordinate substitution.

|Split/model|Observed|1|4|8|16|32|
|---|---:|---:|---:|---:|---:|---:|
|Development: Actual initial|0/112|0/109|0/109|0/107|0/114|0/95|
|Development: Retained H0|108/112|94/109|33/109|14/107|3/114|0/95|
|Development: Original H16 8k|99/112|84/109|47/109|25/107|10/114|4/95|
|Development: Plain RGB1 8k|85/112|69/109|25/109|11/107|9/114|2/95|
|Development: Half-rate H16 16k|94/112|71/109|35/109|23/107|15/114|4/95|
|Fitting: Actual initial|0/163|0/177|0/180|0/176|0/171|0/179|
|Fitting: Retained H0|159/163|153/177|73/180|34/176|14/171|5/179|
|Fitting: Original H16 8k|154/163|130/177|86/180|61/176|21/171|19/179|
|Fitting: Plain RGB1 8k|126/163|121/177|63/180|49/176|28/171|15/179|
|Fitting: Half-rate H16 16k|148/163|129/177|93/180|71/176|43/171|25/179|

Original observed≥90%,one-step≥89/109,eight-step≥28/107 gates: `{'observed_ball': False, 'one_step_ball': False, 'eight_step_ball': False}`.

All384 dense origins/targets/commands/resets/outcomes match the original controls.204 actual-initial tensors match original H16 and the qualified default-off state;36 dense initial arrays match the retained sampled-inference control. All8 posterior/prior feature arrays retain one-hot categorical blocks. No old model query or collector repeats.

## New recipe and why it was selected

Original H16 periodic eight-step fitting15/120→27/120 and development8/132→16/132 improve from6,144→8,192 updates. Those pooled four-draw periodic cohorts differ from the dense-final cohort and do not prove convergence, a sufficient budget or an isolated architecture limit. The original curve files and read-only selection summary are pinned.

Initialize fresh with base/encoder/dynamics2e-4 instead of4e-4 and a fixed16,384 instead of8,192 updates. All other effective configuration matches original H16: RGB64/12M/B8/T32/H16/full BPTT/Auto,41001,RGB100,KL1/.1/free-nat1,mixing.01,observed reward/continuation1,categorical samples/ST gradients. No mean/Gaussian/spatial/position/color mode,new loss,architecture,rate sweep,schedule or restored extension.68 weights/8,852,675 scalars/204 tensors/11 reports,legacyv2.

The exact original8,192-row4/2/1/1 plan is replayed twice, with unchanged ordering.16 fitting/eight development streams;4,194,304 replayed arrivals,1,951,882 valid suffix targets,19,140 suffix terminal rows,32,016 positive suffix rewards,21,818 observed terminals,68,880 observed positive rows,151,616 context resets and20,976 source resets. Unique exposed arrivals remain32,849/32,850. This is additional optimization, not new experience or independent support. Reused seed/development families are not fresh confirmation. No new emulator actions/frames,collection,actor/critic,CDP or pretrained encoder.

## Raw training losses before weights

Fixed last1,024-update means. Both new plan cycles match original8,192 locations exactly, including the last1,024. Learned weights/samples/optimizer histories and budgets differ; raw KL is not mutual information, physical calibration or an isolated cause.

|Model|Observed RGB|Future RGB|Dynamics KL|Representation KL|Reward|Continuation|
|---|---:|---:|---:|---:|---:|---:|
|original_h16_rgb100|2807.802277|2654.737930|42.260927|42.260927|0.021793|0.026590|
|half_rate16k|2795.869875|2624.070828|41.308761|41.308761|0.018315|0.023278|

## Development detection, paddle and brick diagnostics

First fixed draw. Detection is conditioned on truth visibility; false detections concern truth-absent rows. Brick scores do not establish persistent memory through absence/occlusion. All other horizons, fitting metrics and event outputs remain retained.

|Model|Stage|Within1px|Detected /truth-visible|False balls|Paddle within1px /visible|Brick balanced error|Removed-brick recall|
|---|---|---:|---:|---:|---:|---:|---:|
|Retained H0|observed|108/112|111/112|0|126/128|0.2240|unscored|
|Retained H0|1|94/109|108/109|4|95/128|0.2229|1.0000|
|Retained H0|8|14/107|99/107|8|52/126|0.2289|0.3478|
|Retained H0|32|0/95|77/95|15|33/118|0.2574|0.2371|
|Original H16 8k|observed|99/112|109/112|1|111/128|0.2782|unscored|
|Original H16 8k|1|84/109|105/109|7|66/128|0.2792|0.6667|
|Original H16 8k|8|25/107|77/107|4|48/126|0.2725|0.5217|
|Original H16 8k|32|4/95|27/95|6|34/118|0.2943|0.2371|
|Half-rate H16 16k|observed|94/112|108/112|0|111/128|0.2540|unscored|
|Half-rate H16 16k|1|71/109|100/109|6|71/128|0.2646|0.6667|
|Half-rate H16 16k|8|23/107|78/107|4|51/126|0.2526|0.4348|
|Half-rate H16 16k|32|4/95|27/95|8|37/118|0.2709|0.2990|

## Original contact-family diagnostic

These eight clips are reused diagnostics, not untouched families or fitting windows of this broad learner. The old contact-only learner trained directly on them and matched110/110 on four draws; it is not a matched-budget control.

|Draw|Matches /110 supported|
|---|---:|
|0|26/110|
|1|25/110|
|2|27/110|
|3|24/110|

## Validation, execution and limits

All16,384 eleven-report rows are finite;weighted totals, both plan cycles, source/reset/valid masks and68 changed weights pass.204 own initial/frozen tensors,18 development/32 contact restore arrays,58 finite exports,allv2 ages/configs and8 one-hot state arrays pass. 178 execution/199 CPU analysis pins verified. Two terminal successful jobs,two guards/18 sealed files;no native retry or repair.

Use qualified nativeaffedefa/SHA256cc6022e36fbb0c3d145058fe85f5fee66c38172e5e54335bfc051cdcbcf07494 and its exact matching runner/runtime. Existing primitive/derivative/mask/restore evidence applies to the exposed rate/budget settings;no rebuild,backend qualification or overlay. Initial preparation lacked the qualified runner PYTHONPATH;its original traceback and successful correct-environment invocation remain before any GPU work. The separate live CPU initial review verifies204 original/frozen tensors,36 initial arrays and all three actual rates. No new build cache.

Learning calls 1508.740s; training/checkpoint/forecast loop 1671.213s. Original H16 calls 765.720s/loop 1105.997s at half the update budget. Current curves every2,048 differ from old1,024;loop excludes initial construction/dense exports,final contact diagnostic,separate restore andCPU analysis. These costs are not isolated backend speed or rate-effect measurements.

Retain0/2,048/4,096/final16,384 checkpoints,four-draw curves0/2,048/4,096/6,144/8,192/10,240/12,288/14,336/16,384,dense raw arrays/features,comparison figure andfixed lane0/draw0 lossless16-frame contact clip. Checkpoints omit replay/live belief/full RNG;any later extension would be a warm start. Invocation27c35e3ef3044f8f91b2d30a2ae8fd9c.

The prior strict forward contract remains failed, despite exact matched-sample recurrent paths;its original numerical/categorical differences and separate repairs are not erased. Earlier RGB1/mean/color and other learners remain closed. Keep all original supports,detectors,tolerances,regressions and failures.

Full accurate physics/action responses,natural reward/life/game-over calibration,persistent removed-brick memory,late/tunnel/second-wall/above-top coverage,untouched families andfresh learner confirmation remain required. No implementation/fitting/intermediate score completes the goal. No extension,collector orautomatic successor is selected. Actor/critic,CDP comparison andfive-game work stay parked;earlier jobs stay closed/fullres protocol-failed.

Closure verifies400 pins,terminal successful native queue/observer and18 sealed guard files. Broad0/3;conditional assessment remains unallocated.
