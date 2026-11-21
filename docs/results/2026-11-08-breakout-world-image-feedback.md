# Breakout: self-generated image feedback and future event supervision

The fresh experiment is complete and **fails retention**. Full world-model competence remains unproven. Keep categorical H0 as the inspectable reference. No unchanged extension, repeated stage, sweep or automatic successor is selected. Actor/critic and CDP comparisons remain parked.

[Protocol](../experiments/2026-11-08-breakout-world-image-feedback.md).
Root `/mnt/data/kindle-breakout-world-image-feedback-20261108`.
Source/native/runner `ea90ee963be4485c3aa18c280b7e4cbe2c5742db`; native SHA256 `9d3d8536b558657a9e7714c15139ea1faadb4a0e69c81fc00cb22c7f3c80bbf9`.

## What this tests

The preceding spatial decoder feeds two-image memory into RGB decoding but does not feed its generated images into recurrent belief or reward/continuation heads. This joint experiment couples generated-image feedback with direct future event supervision. Recurrence, sampling, gradient paths and future losses change together; the result cannot isolate their individual effects.

Default-off `world_image_feedback` requires the categorical standalone RGB spatial decoder and explicit `motion_image_feedback`. For each imagined action, advance GRU once, sample the prior, decode RGB, encode only that generated image, then sample a posterior at the same already-advanced recurrent h. Event heads and the next GRU use this self-observed belief. Shift generated RGB into two-image history. Full BPTT passes through decoder, generated-image CNN, posterior and recurrence. No second time advance, actual future RGB, outcome or reset label enters belief.

Keep RGB64/12M, recurrent2048/stochastic512/encoder1024, all spatial layers and 77 logical/8,857,723 scalar weights/231 checkpoint tensors. No new parameters. All 231 spatial actual-initial tensors match exactly; 201 common tensors/67 weights match H0. These are matched fresh initializations, not warm starts or independent learner seeds. No pretrained encoder, object/RAM/phase training targets or backend primitive is added.

Full world state includes 24,576 image-history scalars plus RSSM feature2560. Legacy `prior_features` now exports self-observed imagined features, rather than the raw prior, and still omits image history. Encoder/posterior diagnostic semantics stay. An extra feedback inference session shares existing parameters but adds activation/buffer/work costs; weight count is not peak GPU memory.

## Training and data

Fresh 8,192 updates from0, RGB64/12M/B8/T32/H16/full BPTT/Auto, seed41001 and rates4e-4. Keep observed motion RGB cross-entropy 100, KL 1/.1/free-nat 1, reward/continuation 1, and future motion RGB 100. Add future two-hot reward CE 1 and discounted continuation soft-label BCE 1 on self-observed imagined features. Targets and masks are stopped; preserve terminal arrivals and mask after reset crossings. Average over full BxH, not valid count. Reports expand from 11 to 13 and total-loss identity includes both future heads.

Original sixteen fitting/eight development streams and full 8,192-row 4/2/1/1 sampling plan stay. Recount 975,941 valid suffix targets, 9,570 future terminals and 16,008 future positive-reward targets; 10,909 observed terminal rows, 34,440 observed positive-reward rows, 75,808 context resets and 10,488 source resets.32,849 unique arrivals among32,850 available arrivals are sampled. Replayed rows are not new environment interactions. No new collection, emulator actions/frames, teacher, policy or critic training occurs.

Reused seed/families are development. This is neither untouched generalization nor fresh confirmation. Initial/2048/4096/final checkpoints and all periodic draws are retained; only the fixed final is assessed. No best checkpoint/draw replaces the endpoint.

## Recorded ball precision

Within one resized RGB64 pixel, first fixed draw, varying visible-ball support. Missing predicted balls fail. Related origins are not independent episodes.

| Split/model | Observed | 1 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| Development: Feedback actual initial |0/112|0/109|0/109|0/107|0/114|0/95|
| Development: Retained categorical H0 |108/112|94/109|33/109|14/107|3/114|0/95|
| Development: Categorical H16 |99/112|84/109|47/109|25/107|10/114|4/95|
| Development: Spatial H16 |87/112|81/109|46/109|24/107|5/114|2/95|
| Development: Image-feedback H16 |90/112|87/109|44/109|26/107|17/114|7/95|
| Fitting: Feedback actual initial |0/163|0/177|0/180|0/176|0/171|0/179|
| Fitting: Retained categorical H0 |159/163|153/177|73/180|34/176|14/171|5/179|
| Fitting: Categorical H16 |154/163|130/177|86/180|61/176|21/171|19/179|
| Fitting: Spatial H16 |137/163|131/177|76/180|69/176|31/171|5/179|
| Fitting: Image-feedback H16 |140/163|120/177|85/180|73/176|41/171|17/179|

Even on fitting streams, eight-action precision is 73/176. This does not establish successful fitting of short forecasts on training streams; held-out generalization alone cannot account for the measured gap. It does not isolate a particular representational, optimization or rollout cause.

Observed>=90%, one>=89/109 and eight>=28/107 gates stay. 384 dense actual-initial/final origins, 20 source-array identities, 231 own initial/dense-frozen tensors and 18 initial-cohort arrays pass. Accurate observed rendering alone is not forecasting competence.

## Frozen causal forecast assessment

Only the new final receives 84 separately declared frozen groups, history 16 primary/32 secondary, eight draws and 1/4/8/16/32 endpoints. Reuse cached emulator oracle/H0/categorical H16/spatial arrays; no old-model queries or collector repeats. Feed past images and planned actions, with no true future corrections. The primary score is category-balanced survival-weighted ball-feature energy at 16/32; lower is better, unchanged threshold .12712666431732942. It scores a finite decoded-feature mixture, not full image likelihood or calibrated population futures.

| Model | Primary energy | Freeflight8 NOOP/LEFT/RIGHT/REVERSE detections, each of64 | Ending16 Brier | Ending32 Brier | RIGHT16 ending; oracle0 | Non-ending >5%/304; maximum |
|---|---:|---|---:|---:|---:|---|
| Retained categorical H0 |0.169502219|64/63/63/61|0.003345|0.144566|0.110494|0/304; 0.019069|
| Categorical H16 |0.311976907|64/60/41/51|0.002422|0.189225|0.082001|23/304; 0.438399|
| Spatial H16 |0.478844057|64/63/57/61|0.012795|0.230475|0.033346|187/304; 0.968467|
| Image-feedback H16 |0.256859267|56/48/41/64|0.038424|0.241261|0.157460|99/304; 0.658299|

Primary improvement versus H0: -51.537% (required 25%).

Every freeflight command has the same oracle ball path through 8; each command needs>=90% detections. Pair accuracy omits survival weighting and cannot replace the primary score. Sticky-action randomness means a single future picture is not a complete forecast. The previous unattainable FIRE pair gate remains invalid and is not reused.

| Intermediate gate | Result |
|---|---|
| observed_ball |fail|
| one_step_ball |fail|
| eight_step_ball |fail|
| serve_noop |pass|
| ending_brier |fail|
| ending16_brier |fail|
| ending16_right |fail|
| nonending_false_alarms |fail|
| mean_ball_energy_reduction |fail|
| freeflight_each_plan_presence_at8 |fail|

`supplementary-metrics.json` retains every individual non-ending false alarm>5% and every per-action/horizon cell versus all three controls. The non-ending gate concerns category means; individual alarms remain explicit. Regressed cells (ball/paddle/reward Brier/ending Brier/alive-only brick Brier):

- Versus Retained categorical H0: 104 of210/172 of210/182 of210/162 of210/26 of209.
- Versus Categorical H16: 117 of210/113 of210/169 of210/157 of210/16 of209.
- Versus Spatial H16: 92 of210/110 of210/154 of210/79 of210/25 of209.

Related cells are not independent tests, significance evidence or seed reliability. H0 has no suffix loss; categorical H16 and spatial H16 have motion-RGB suffix supervision but not this generated-image feedback/future event objective. No isolated causal attribution follows.

## Execution, cost and preserved failures

99 affected Rust CPU/135 packaged Python/strict core-binding Clippy checks pass. First native qualification: 1196 pins/36.543s. Future CE/BCE F64 values/analytic derivatives, stopped targets/zero masks, fixed-h generated-image effects/lane isolation, all 231 spatial initial tensors/201 common H0 tensors,231 own initial/frozen tensors, 18 initial/trained restore arrays,74 changed first-update weights, 13 finite reports/totals/masks,cached spatial prefix/first RGB and old-native v11 rejection pass.

Learning: 1203 pins/168 CPU inputs, 8,192 finite 13-report rows, total identity, all 77 changed weights,full source plan/exposures, 231 exact initial/frozen tensors, 384 dense origins, 58 finite exports and 18 exact final restore arrays pass. v11 persists both architecture flags/objective and rejects incompatible headers/phases/whole-agent paths. Default v1-v10 paths stay. Checkpoints omit replay,live beliefs/image history/full RNG; a restored extension would be a warm start and is not selected.

Native training calls 1232.062s; measured training/evaluation loop 1412.747s. Loop excludes native construction,dense-initial queries and pinning. Qualification,restore,frozen assessment,CPU analysis and build/packaging are separate. No inherited learning cost is hidden.
Observed train worker/polling span: 1435.499s.
Observed restore worker/polling span: 12.276s.

Frozen assessment 209.398s; CPU comparison 52.389s; learning audit 8.232s. Four successful guards/36 sealed files pass; no native-stage retry,host recovery,NVML or unfinished child. Disposable 835,754,676-byte build cache released; wheel,runtime,CPU executable,source,logs and useful checkpoints retained.

Preserve the first CPU fixture failure in `cpu-feedback.log`/`geometry-fixture-first.rs`: the fixture expected 36 outputs and omitted two existing posterior debug outputs. Separate `geometry-fixture-repair.json` and corrected CPU log pass with 38 outputs. No production math change or repeated native qualification fixes this fixture. Previous experiments and their failures stay closed.

Retained H0 used 529.515s native/796.900s loop; categorical H16 used 765.720s/1105.997s; spatial H16 used 1021.332s/1186.625s. Distinct single-run runtimes/wrappers are not isolated kernel benchmarks or a learning-per-hour benefit. Generated-image re-encoding and extra graph state add work despite unchanged parameter count.

Four-model comparison/learning figures and three full causal APNGs use retained arrays only, fixed lane 0/draw 0 rather than favorable matches. Native oracle 160x210 and RGB64 predictions share display aspect only. Clips decode losslessly; plotting versions are retained. No input/metric preprocessing change or new model/emulator query occurs.

This tests one joint-world recipe. Precise all-horizon physics,event timing/calibration,persistent removed-brick memory,late/tunnel/second-wall coverage,untouched families and fresh seed confirmation remain required. Existing oracle has no above-top-brick endpoint and its detector supports a restricted region. Intermediate gates cannot complete the full goal. No unchanged extension,sweep or automatic successor is selected.
