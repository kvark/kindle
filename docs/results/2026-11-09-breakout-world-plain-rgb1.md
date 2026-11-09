# Plain H16 world with RGB coefficient1

Fixed final8,192 passes **0/3 original dense development gates**. The screen fails; do not adopt this recipe or allocate its conditional branch assessment.

The full Breakout world-model goal remains active and unproven. Raw loss reductions cannot replace accurate physical forecasts, natural event calibration, persistent memory or full-game coverage.

[Protocol](../experiments/2026-11-09-breakout-world-plain-rgb1.md).
Artifacts: `/x/Temp/kindle-world-plain-rgb1-20261109`.

## Fixed final RGB predictions

First fixed draw, within one resized pixel. Missing or diffuse balls fail. Supports vary byhorizon; related origins are not independent episodes. No bestcheckpoint/draw/tolerance selection orcoordinate substitution.

|Split/model|Observed|1|4|8|16|32|
|---|---:|---:|---:|---:|---:|---:|
|Development: Actual initial|0/112|0/109|0/109|0/107|0/114|0/95|
|Development: Retained H0|108/112|94/109|33/109|14/107|3/114|0/95|
|Development: Original H16 RGB100|99/112|84/109|47/109|25/107|10/114|4/95|
|Development: Position1,000|85/112|73/109|35/109|15/107|5/114|5/95|
|Development: Position10,000|77/112|70/109|27/109|14/107|6/114|0/95|
|Development: Direct color attention|0/112|0/109|0/109|0/107|0/114|0/95|
|Development: Mean-latent world|106/112|61/109|28/109|18/107|9/114|1/95|
|Development: Plain H16 RGB1|85/112|69/109|25/109|11/107|9/114|2/95|
|Fitting: Actual initial|0/163|0/177|0/180|0/176|0/171|0/179|
|Fitting: Retained H0|159/163|153/177|73/180|34/176|14/171|5/179|
|Fitting: Original H16 RGB100|154/163|130/177|86/180|61/176|21/171|19/179|
|Fitting: Position1,000|134/163|119/177|50/180|34/176|12/171|15/179|
|Fitting: Position10,000|118/163|107/177|57/180|28/176|13/171|7/179|
|Fitting: Direct color attention|0/163|1/177|0/180|1/176|0/171|0/179|
|Fitting: Mean-latent world|155/163|102/177|63/180|42/176|17/171|12/179|
|Fitting: Plain H16 RGB1|126/163|121/177|63/180|49/176|28/171|15/179|

Original observed≥90%,one-step≥89/109,eight-step≥28/107 gates: `{'observed_ball': False, 'one_step_ball': False, 'eight_step_ball': False}`.

All384 initial/final dense origins, targets, commands, resets and outcomes match the controls. All204 initial tensors match original H16 and the qualified zero-step state;36 dense initial arrays match the retained sampled-inference control. The only effective recipe change from originalH16 is reconstruction coefficient100→1. All8 sampled posterior/prior feature arrays retain one-hot categorical blocks. No old model query orrecollection occurs.

## Raw training losses before weights

Mean over the fixed last1,024 updates,with all8,192 sampled locations matching the original H16 control. Learned states, samples and optimizer histories differ. Raw KL is not mutual information orphysical calibration; a lower value cannot prove collapse, absence of information or a causal mechanism. The independently weighted total is checked on every new row.

|Model|Observed RGB|Future RGB|Dynamics KL|Representation KL|Reward|Continuation|
|---|---:|---:|---:|---:|---:|---:|
|original_h16_rgb100|2807.802277|2654.737930|42.260927|42.260927|0.021793|0.026590|
|plain_rgb1|2807.545014|2654.974749|14.474038|14.474038|0.021652|0.025828|

## Development detection, paddle and brick diagnostics

First fixed draw. Detection is conditioned on true ball visibility; false detections concern true-absent rows. Brick scores do not prove memory through absence/occlusion. All other horizons, fitting metrics andevent outputs remain retained.

|Model|Stage|Within1px|Detected /truth-visible|False ball detections|Paddle within1px /visible|Brick balanced error|Removed-brick recall|
|---|---|---:|---:|---:|---:|---:|---:|
|Retained H0|observed|108/112|111/112|0|126/128|0.2240|unscored|
|Retained H0|1|94/109|108/109|4|95/128|0.2229|1.0000|
|Retained H0|8|14/107|99/107|8|52/126|0.2289|0.3478|
|Retained H0|32|0/95|77/95|15|33/118|0.2574|0.2371|
|Original H16 RGB100|observed|99/112|109/112|1|111/128|0.2782|unscored|
|Original H16 RGB100|1|84/109|105/109|7|66/128|0.2792|0.6667|
|Original H16 RGB100|8|25/107|77/107|4|48/126|0.2725|0.5217|
|Original H16 RGB100|32|4/95|27/95|6|34/118|0.2943|0.2371|
|Plain H16 RGB1|observed|85/112|109/112|3|101/128|0.2788|unscored|
|Plain H16 RGB1|1|69/109|102/109|8|80/128|0.2818|0.3333|
|Plain H16 RGB1|8|11/107|58/107|6|45/126|0.2742|0.5652|
|Plain H16 RGB1|32|2/95|20/95|3|23/118|0.2887|0.2371|

## Original contact-family diagnostic

These eight clips are reused diagnostics,not this learner fitting data oruntouched generalization. The original contact learner trained directly on them and matched110/110 on four draws;it is not a matched-budget control.

|Draw|Matches /110 supported|
|---|---:|
|0|25/110|
|1|18/110|
|2|20/110|
|3|14/110|

## Recipe, provenance and execution

Fresh8,192/RGB64/12M/B8/T32/H16/full BPTT/Auto,41001,rates4e-4,mixing.01,KL1/.1/freeNat1,observed reward/continuation1,normal categorical samples andstraight-through gradients. Change only observed/future RGB coefficient100→1 relative to original plain H16. No mean state,spatial decoder,auxiliary head,color attention ornew loss formula.68 weights/8,852,675 scalars/204 saved tensors/11 reports,legacyv2. Reused seed/development families are not fresh confirmation or a reliability estimate.

Earlier RGB1 fusion/suffix-event,local H0/transport andauxiliary-position failures remain closed. They are not this plain-model recipe. Native sources differ from originalH16; exact initialization/inference compatibility does not prove every trained tensor equivalent. This comparison does not diagnose image-gradient dominance or an isolated physical cause.

Exact16 fitting/eight development streams and4/2/1/1 plan:2,097,152 replayed arrivals,975,941 valid suffix targets,9,570 suffix terminals,16,008 positive suffix rewards.32,849/32,850 unique fitting arrivals,34,440 observed positives,10,909 terminals,75,808 context resets and10,488 source resets. No new emulator actions/frames,collection,actor/critic,CDP orpretrained encoder.

All8,192 eleven-report rows are finite;RGB1 total/source/reset identities pass. All68 weights change.204 own initial/frozen tensors,18 development/32 contact restore arrays,34 finite exports,allv2 ages/configs and8 one-hot state arrays pass.150 declaration/163 CPU analysis pins verified,two terminal successful jobs/two guards/18 sealed files. No native retry orrepair.

Reuse qualified default-off nativeaffedefa/SHA256cc6022e36fbb0c3d145058fe85f5fee66c38172e5e54335bfc051cdcbcf07494 andmatching runner. Its default-off path matches204 old initial tensors and32 cached model arrays. Existing motion-loss/derivative/world evidence applies to the exposed coefficient; no rebuild ornew primitive qualification. The mean-mode qualifier update is not resumed. Its original fixture Clippy failure andseparate corrected checks remain;811,446,272-byte build cache was already released. The baseline initial reference is anchored through the predeclared qualifier/contact closure chain.

Learning calls 759.598s; training/checkpoint/forecast loop 867.901s. OriginalH16 calls 765.720s/loop 1105.997s. Current evaluation every2048 differs from old1024; loop excludes initial construction/dense exports,final contact diagnostic,separate restore andCPU analysis. These wrapper times do not isolate kernel speed orlearning efficiency.

Retain0/2048/4096/final8192 checkpoints,allfour-draw curves0/2048/4096/6144/8192,dense raw arrays/features,comparison figure andfixed lane0/draw0 lossless16-frame contact clip. Checkpoints omit replay/live belief/full RNG;later extension would be a warm start. Invocation988b5bab40524ac3b532ea1f5d5d359b. No new build cache.

Full precise physics,natural reward/life/game-over calibration,persistent removed-brick memory,late/tunnel/second-wall/above-top,untouched families andfresh confirmation remain required. No implementation/fitting/intermediate score completes the goal. No extension,collector orautomatic coefficient sweep/successor. Actor/critic,CDP comparison andfive-game work stay parked;earlier jobs stayclosed/fullres protocol-failed.

Closure verifies330 pins,terminal successful native queue/observer and18 sealed guard files. Broad0/3;conditional assessment remains unallocated.
