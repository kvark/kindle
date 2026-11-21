# Breakout: imagined event supervision without image feedback

The fresh experiment is complete and **fails retention**. Full world-model competence remains unproven. Keep categorical H0 as the inspectable reference. No unchanged extension, repeated stage, sweep or automatic successor is selected. Actor/critic and CDP comparisons remain parked.

[Protocol](../experiments/2026-11-08-breakout-world-suffix-events.md).
Root `/mnt/data/kindle-breakout-world-suffix-events-20261108`.
Source/native/runner `a8efffc6c7b13cf98616199f336e5f985a44a912`; native SHA256 `f40555b56883aa804300ab702fb2b1f23e62c240abbb73f42c6e3e98c196a943`.

## What this tests

The preceding nonlinear fusion model improves some recorded ball forecasts but loses event recognition and predicts many false endings during rollout. The earlier image-feedback experiment adds future event losses while also changing recurrent conditioning. This experiment adds event supervision to the ordinary prior-state suffix without generated-image re-encoding.

Explicit standalone `motion_event_rollout` requires categorical fusion RGB and a positive suffix. Each imagined action advances the same RSSM prior used for its RGB prediction, reward and continuation. Generated RGB shifts only two-image history; it is not re-encoded and does not correct recurrent belief. Future RGB, event labels and reset masks are stopped targets/weights only, never state inputs. Terminal arrivals stay valid; mask after reset crossings. Event gradients reach the RSSM and heads without traversing the decoder, while RGB gradients retain full BPTT through generated-image history.

No architecture or parameter change:80 logical weights/8,871,955 scalars/240 tensors, RGB64/12M, RSSM recurrent2048/stochastic512/encoder1024 and feature2560 plus24,576 image-history scalars. All240 initial tensors and18 initial forecast arrays exactly match cached fusion v12. Reused initialization/seed is not an independent learner confirmation. No new labels, pretrained encoder, teacher, actor/critic, collection or backend primitive.

v13 binds the new objective. Defaults and original v1-v12 paths stay. Old native rejects v13 before GPU allocation; incompatible architecture, zero suffix and training phases are rejected. Legacy feature exports still omit image history.

## Training and data

Fresh8,192 updates from0, RGB64/12M/B8/T32/H16/full BPTT/Auto,seed41001,rates4e-4,mixing.01. Keep observed/future motion RGB cross-entropy100, KL1/.1/free-nat1 and observed reward/continuation1. Add future two-hot reward CE1 and soft-label continuation BCE1 on causal prior states. Average each over full BxH, not valid count. No class reweighting or scalar calibration. Reports expand11→13 and the total-loss identity includes both new heads.

Original sixteen fitting/eight development streams and full 8,192-row 4/2/1/1 sampling plan stay. Recount 975,941 valid suffix targets, 9,570 future terminals and 16,008 future positive-reward targets; 10,909 observed terminal rows, 34,440 observed positive-reward rows, 75,808 context resets and 10,488 source resets.32,849 unique arrivals among32,850 available arrivals are sampled. Replayed rows are not new environment interactions. No new collection, emulator actions/frames, teacher, policy or critic training occurs.

Reused seed/families are development. This is neither untouched generalization nor fresh confirmation. Initial/2048/4096/final checkpoints and all periodic draws are retained; only the fixed final is assessed. No best checkpoint/draw replaces the endpoint.

## Recorded ball precision

Within one resized RGB64 pixel, first fixed draw, varying visible-ball support. Missing predicted balls fail. Related origins are not independent episodes.

| Split/model | Observed | 1 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| Development: Suffix-event actual initial |0/112|0/109|0/109|0/107|0/114|0/95|
| Development: Retained categorical H0 |108/112|94/109|33/109|14/107|3/114|0/95|
| Development: Categorical H16 |99/112|84/109|47/109|25/107|10/114|4/95|
| Development: Fusion H16 |93/112|95/109|62/109|34/107|15/114|4/95|
| Development: Suffix-event H16 |98/112|97/109|58/109|31/107|13/114|6/95|
| Fitting: Suffix-event actual initial |0/163|0/177|0/180|0/176|0/171|0/179|
| Fitting: Retained categorical H0 |159/163|153/177|73/180|34/176|14/171|5/179|
| Fitting: Categorical H16 |154/163|130/177|86/180|61/176|21/171|19/179|
| Fitting: Fusion H16 |141/163|150/177|96/180|72/176|38/171|23/179|
| Fitting: Suffix-event H16 |140/163|148/177|99/180|71/176|30/171|20/179|

Even on fitting streams, eight-action precision is 71/176. This does not establish successful fitting of short forecasts on training streams; held-out generalization alone cannot account for the measured gap. It does not isolate a particular representational, optimization or rollout cause.

Observed>=90%, one>=89/109 and eight>=28/107 gates stay. 384 dense actual-initial/final origins, 20 source-array identities, 240 own initial/dense-frozen tensors and 18 initial-cohort arrays pass. Accurate observed rendering alone is not forecasting competence.

## Frozen causal forecast assessment

Only the new final receives 84 separately declared frozen groups, history 16 primary/32 secondary, eight draws and 1/4/8/16/32 endpoints. Reuse cached emulator oracle/H0/categorical H16/fusion arrays; no old-model queries or collector repeats. Feed past images and planned actions, with no true future corrections. The primary score is category-balanced survival-weighted ball-feature energy at 16/32; lower is better, unchanged threshold .12712666431732942. It scores a finite decoded-feature mixture, not full image likelihood or calibrated population futures.

| Model | Primary energy | Freeflight8 NOOP/LEFT/RIGHT/REVERSE detections, each of64 | Ending16 Brier | Ending32 Brier | RIGHT16 ending; oracle0 | Non-ending >5%/304; maximum |
|---|---:|---|---:|---:|---:|---|
| Retained categorical H0 |0.169502219|64/63/63/61|0.003345|0.144566|0.110494|0/304; 0.019069|
| Categorical H16 |0.311976907|64/60/41/51|0.002422|0.189225|0.082001|23/304; 0.438399|
| Fusion H16 |0.345425368|64/64/56/64|0.077296|0.483186|0.064732|276/304; 0.990268|
| Suffix-event H16 |0.283365170|64/64/56/64|0.032212|0.226666|0.205177|239/304; 0.999482|

Primary improvement versus H0: -67.175% (required 25%).

Every freeflight command has the same oracle ball path through 8; each command needs>=90% detections. Pair accuracy omits survival weighting and cannot replace the primary score. Sticky-action randomness means a single future picture is not a complete forecast. The previous unattainable FIRE pair gate remains invalid and is not reused.

| Intermediate gate | Result |
|---|---|
| observed_ball |fail|
| one_step_ball |pass|
| eight_step_ball |pass|
| serve_noop |pass|
| ending_brier |fail|
| ending16_brier |fail|
| ending16_right |fail|
| nonending_false_alarms |fail|
| mean_ball_energy_reduction |fail|
| freeflight_each_plan_presence_at8 |fail|

`supplementary-metrics.json` retains every individual non-ending false alarm>5% and every per-action/horizon cell versus all three controls. The non-ending gate concerns category means; individual alarms remain explicit. Regressed cells (ball/paddle/reward Brier/ending Brier/alive-only brick Brier):

- Versus Retained categorical H0: 106 of210/180 of210/202 of210/169 of210/17 of209.
- Versus Categorical H16: 120 of210/157 of210/196 of210/171 of210/7 of209.
- Versus Fusion H16: 83 of210/119 of210/126 of210/94 of210/9 of209.

Related cells are not independent tests, significance evidence or seed reliability. H0 has no suffix loss; categorical H16 and fusion H16 have motion-RGB suffix supervision but not this additional event-supervised suffix. No isolated causal attribution follows.

## Execution, cost and preserved failures

105 affected world Rust CPU/224 packaged Python CPU checks and strict core/binding Clippy pass. First native qualification: 1217 pins/33.925s. New future-target/image/mask perturbations leave states and RGB unchanged, event losses respond,zero masks zero all suffix losses,and lane isolation passes. All240 cached fusion initial tensors/18 initial arrays,240 own initial/frozen tensors,18 initial/trained restores,77 required changed weights,13 finite reports/total/mask and old-native v13 rejection pass. A separate CPU comparison of saved first-update reports finds all ten original components identical and the new total exactly equal to the old total plus future CE/BCE under sequential F32 addition. No native query repeats. The event-loss function body matches the earlier qualified implementation exactly; its completed native math fixture is reused by source/receipt identity,not repeated.

Learning: 1225 pins/170 CPU inputs,8,192 finite thirteen-report rows,total identity,all80 changed weights,full source plan/exposures,240 exact initial/frozen tensors,384 dense origins,58 finite exports and18 exact final restore arrays pass. Checkpoints omit replay,live belief/image history and full RNG; a restored extension would be a warm start and is not selected.

Native training calls 1166.306s; measured training/evaluation loop 1352.208s. Loop excludes native construction,dense-initial queries and pinning. Qualification,restore,frozen assessment,CPU analysis and build/packaging are separate. No inherited learning cost is hidden.
Observed train worker/polling span: 1374.535s.
Observed restore worker/polling span: 11.255s.

Frozen assessment 213.416s; CPU comparison 54.055s; learning audit 8.938s. Four successful guards/36 sealed files pass; no native-stage retry,host recovery,NVML or unfinished child. Disposable 865,583,382-byte build cache released; wheel,runtime,CPU executable,source,logs and useful checkpoints retained.

Preserve `cpu-world.log`/`cpu-fixture-first.rs`: the new fixture used integer assignments for two optional microbatch values and failed compilation. `cpu-fixture-repair.json` records their Some-value correction; source review also corrected the fixture count to38 raw graph outputs rather than13 API reports before tests executed. The corrected CPU run passes105/33 ignored. No production math change or native retry repairs that fixture.

The first build-receipt construction incorrectly expected224 checks after only the169 world-prefix subset and stopped at its assertion before writing a receipt. Preserve `packaging-count-review.json`; the missing55 disjoint corpus-module tests then pass without repeating the169 successful tests. Combined224 packaged checks pass. No packaged test fails and native/package bytes are unchanged by this accounting completion.

Retained H0 used529.515s native/796.900s loop; categorical H16 used765.720s/1105.997s; fusion used1142.861s/1317.429s. These single-run timings are not isolated kernel benchmarks. No additional environment actions/frames are hidden in these offline learning costs. Added event-head training work must justify its cost through useful frozen predictions.

Four-model comparison/learning figures and three full causal APNGs use retained arrays only, fixed lane 0/draw 0 rather than favorable matches. Native oracle 160x210 and RGB64 predictions share display aspect only. Clips decode losslessly; plotting versions are retained. No input/metric preprocessing change or new model/emulator query occurs.

This tests one joint-world recipe. Precise all-horizon physics,event timing/calibration,persistent removed-brick memory,late/tunnel/second-wall coverage,untouched families and fresh seed confirmation remain required. Existing oracle has no above-top-brick endpoint and its detector supports a restricted region. Intermediate gates cannot complete the full goal. No unchanged extension,sweep or automatic successor is selected.
