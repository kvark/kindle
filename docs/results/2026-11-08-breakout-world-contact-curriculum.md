# Breakout: contact-first curriculum on the broader corpus

The selected phase is complete and **fails retention**. Full world-model competence remains unproven. Keep categorical H0 as the broad reference. No unchanged extension, repeated stage, sweep or automatic successor is selected. Actor/critic and CDP comparisons stay parked.

[Protocol](../experiments/2026-11-08-breakout-world-contact-curriculum.md).
Root `/mnt/data/kindle-breakout-world-contact-curriculum-20261108`.

## Staged recipe and comparison limits

The fixed-contact world fits all110 supported1..16 future ball positions on each of four draws, but broad-data H16 forecasts remain weak. This experiment starts from that precise contact checkpoint, then trains all weights on the broader corpus. It tests a practical data curriculum, not an isolated explanation of earlier failures.

Warm-start exactly the contact checkpoint at age2048, preserving all204 weights/moments and optimizer age. Add6144 broad updates to final8192. Reuse native85dbe150/SHA5b112866, RGB64/size12M/B8/T32/H16/full BPTT/Auto, seed41001, rates4e-4, categorical mixing.01, observed/future motion RGB cross-entropy100, KL1/.1/free-nat1 and observed reward/continuation1.68 logical/8,852,675 scalar weights/204 tensors. No rebuild, new loss, frozen subset, spatial decoder, image feedback, object/RAM training targets, pretrained encoder, actor or critic.

Restore supplies no fresh-training override. Saved architecture/objective/geometry and exact canonical F32 configuration stay. Checkpoints omit replay, live belief and full RNG; this is a warm start, not uninterrupted learning, fresh initialization or independent confirmation. Pretraining/data order/family exposure/optimizer history change together; no isolated curriculum effect follows.

Use the first6144 original4/2/1/1 plan rows:four original-fit uniform,two supplement uniform,one waiting,one terminal window. The same16 broad fitting/eight development streams and original selections stay. Pretraining uses eight different retained fitting source streams. All pretraining/broad fitting seed families are disjoint from development, but the development families and learner seed are reused. No new collection/emulator actions/frames. RGB remains the already qualified native max pool plus single GPU resize.

| Exposure | Fresh H16 control | Staged total |
|---|---:|---:|
| Total optimizer updates |8192|2048 contact +6144 broad|
| Replayed arrivals |2097152|2097152|
| Valid future targets |975941|994470|
| Future terminals |9570|7151|
| Positive future rewards |16008|12034|
| Observed terminal rows |10909|8152|
| Observed positive reward rows |34440|30014|

The broad phase samples32845 of32850 available arrival IDs,732326 valid suffix targets,7151 suffix terminals and12034 future positive rewards. Contact pretraining contributes256 unique source arrivals and262144 replayed future targets, with zero future reward/terminal positives. These are replay/source-ID counts, not independent experiences or matched family exposure. Equal total updates do not make this a matched-experience comparison.

## Recorded ball precision

Within one resized RGB64 pixel, first fixed draw, varying visible support. Missing predictions fail. Related origins are not independent episodes. Actual initial here means the inherited contact checkpoint evaluated on the broader corpus.

| Split/model | Observed | 1 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| Development: Warm contact initial, age2048 |0/112|1/109|0/109|1/107|0/114|0/95|
| Development: Retained H0 |108/112|94/109|33/109|14/107|3/114|0/95|
| Development: Fresh categorical H16 |99/112|84/109|47/109|25/107|10/114|4/95|
| Development: Contact-first curriculum |86/112|66/109|23/109|12/107|11/114|3/95|
| Fitting: Warm contact initial, age2048 |0/163|0/177|0/180|0/176|0/171|0/179|
| Fitting: Retained H0 |159/163|153/177|73/180|34/176|14/171|5/179|
| Fitting: Fresh categorical H16 |154/163|130/177|86/180|61/176|21/171|19/179|
| Fitting: Contact-first curriculum |138/163|112/177|53/180|24/176|20/171|9/179|

Observed>=90%,one>=89/109,eight>=28/107 gates stay.384 matched dense warm-initial/final origins and all source arrays match the retained controls. Perception, forecasts, rewards and brick errors stay separate; neither reconstructions nor declining losses establish a complete predictive state.

## Retention of original contact fitting

Before broad learning, all204 inherited tensors and32 cached contact prediction arrays match exactly. The same four draws/110 supported positions are evaluated after broad learning;18 positions per draw remain unsupported. This retention diagnostic does not replace the original full-corpus gates.

| Draw | Before broad phase | After broad phase |
|---|---:|---:|
| 0 |110/110|26/110|
| 1 |110/110|17/110|
| 2 |110/110|20/110|
| 3 |110/110|20/110|

Every1..16 endpoint, missing ball, observed/paddle/brick metric and raw event/feature array is retained. Fixed trajectory fitting can be memorized. This before/after comparison does not isolate forgetting, action causality or stochastic calibration.

## Frozen causal forecast assessment

Only the new final receives84 separately declared oracle groups,history16 primary/32 secondary,eight draws,all1/4/8/16/32 endpoints. Cached oracle/H0/fresh-H16 arrays are reused without old-model queries or collector repeats. Forecasts use only past frames and future commands. The primary score is category-balanced survival-weighted16/32 ball-feature energy; lower is better, threshold.12712666431732942. It scores a decoded-feature mixture, not full image likelihood or population calibration.

| Model | Primary energy | Freeflight8 NOOP/LEFT/RIGHT/REVERSE detections, each of64 | Ending16 Brier | Ending32 Brier | RIGHT16 ending; oracle0 | Non-ending >5%/304; maximum |
|---|---:|---|---:|---:|---:|---|
| Retained H0 |0.169502219|64/63/63/61|0.003345|0.144566|0.110494|0/304; 0.019069|
| Fresh categorical H16 |0.311976907|64/60/41/51|0.002422|0.189225|0.082001|23/304; 0.438399|
| Contact-first curriculum |0.322806611|53/59/53/48|0.053711|0.273752|0.131147|15/304; 0.075324|

Primary improvement versus H0: -90.444% (required25%).

Every freeflight command has identical oracle ball paths through8; each command needs>=90% visible balls. All model/oracle draw pairs are scored; no best match. Point/detection metrics omit survival weighting and cannot replace the primary score. The invalid historical FIRE pair-match gate is not reused.

| Intermediate gate | Result |
|---|---|
| observed_ball |fail|
| one_step_ball |fail|
| eight_step_ball |fail|
| serve_noop |pass|
| ending_brier |fail|
| ending16_brier |fail|
| ending16_right |fail|
| nonending_false_alarms |pass|
| mean_ball_energy_reduction |fail|
| freeflight_each_plan_presence_at8 |fail|

`supplementary-metrics.json` retains every individual non-ending alarm>5% and every action/horizon comparison. The non-ending gate concerns category means; individual failures remain visible. Regressed cells (ball/paddle/reward Brier/ending Brier/alive-only brick Brier):

- Versus Retained H0: 130 of210/158 of210/150 of210/179 of210/47 of209.
- Versus Fresh categorical H16: 136 of210/88 of210/105 of210/161 of210/35 of209.

Related cells/draws are not independent tests, significance evidence or seed reliability. H0 has no suffix objective. Fresh H16 shares the objective/native but has different family exposure and optimization history.

## Identity, execution and cost

Repository/source snapshot `3ce77ff840a1f5f06b2ea4ad7e186ad51a2f66f0`; exact experimental wrapper/config and runtime are pinned. Native/source `85dbe150aed1e87a42a3089ee612fc6ef2d88f86`, SHA256 `5b1128667e6436b9865c31dedcb50e22318a43250e884046b155e51dc955ad82`. Existing qualified motion loss/derivatives/masks/native restore are reused without repeating primitive fixtures or rebuilding.

CPU preparation/import closure passes with GPU loader disabled and no native/emulator constructor. Independent plan/reset/outcome/mask recount, original selection identity and source-family separation pass. Learning verifies243 declared pins/156 CPU inputs, all6144 finite eleven-loss rows/weighted totals/masks,68 changed weights,204 inherited initial/frozen tensors,384 dense warm-initial/final origins and46 finite training exports. Age2048/4096/6144/8192 metadata remains exact. Final restore reproduces18 development arrays plus32 contact draw-arrays exactly, with204 unchanged tensors.

New native training 565.070s; new train/evaluation/checkpoint loop 683.255s. Inherited contact native 186.289s/loop 191.500s; total native 751.359s/loop 874.755s. Loop timers exclude construction,dense-initial queries and pinning; the contact diagnostic also has initial exports outside its loop. These distinct single-run wrapper costs are not isolated kernel benchmarks or seed reliability.
Observed train worker/polling span 710.012s.
Observed restore worker/polling span 11.255s.

New frozen assessment 166.236s; CPU comparison 38.094s; learning audit 6.299s. Three successful guards/27 sealed files pass. No new native retry, numerical fault, host recovery or NVML. The inherited contact experiment's failed pre-import launcher and separate CPU repair remain preserved, not relabeled. No new build cache is created; old runtime/source/useful checkpoints remain.

Comparison/learning plots and three full causal APNGs use cached arrays only, fixed lane0/draw0 rather than best matches. Four-draw curve mean/min–max are not confidence intervals. Curriculum broad curves start at inherited age2048; initial contact training is not plotted as broad-corpus learning. Clip display resizing changes no model input or metric, and decoded frames are checked losslessly.

Precise all-horizon physical forecasts,event timing/calibration,persistent removed-brick memory,late/tunnel/second-wall coverage,untouched families and fresh confirmation remain required. The retained oracle has no above-top-brick endpoint and its detector supports a restricted region. Intermediate gates cannot complete the full goal. No unchanged extension, repeated learner/collector, sweep or automatic successor is selected.
