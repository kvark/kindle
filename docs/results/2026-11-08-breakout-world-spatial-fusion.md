# Breakout: nonlinear fusion of recurrent and image-history features

The fresh experiment is complete and **fails retention**. Full world-model competence remains unproven. Keep categorical H0 as the inspectable reference. No unchanged extension, repeated stage, sweep or automatic successor is selected. Actor/critic and CDP comparisons remain parked.

[Protocol](../experiments/2026-11-08-breakout-world-spatial-fusion.md).
Root `/mnt/data/kindle-breakout-world-spatial-fusion-20261108`.
Source/native/runner `7e1408c975914d95ce1644992331f5d27b92d692`; native SHA256 `0a6e1bd2cffa97908e8bd803d51269e46bcc7c1320e20beca10e13e4c1a43eb4`.

## What this tests

The preceding spatial decoder concatenates independently computed global and history fields immediately before one linear RGB output convolution. This experiment inserts a nonlinear hidden layer after their concatenation. It permits learned joint interactions; qualification establishes that the implementation produces them, not that missing interactions caused previous forecast failures.

Default-off `rgb_spatial_fusion` requires standalone categorical RGB with `rgb_spatial_decoder`, without image feedback. At12M, concatenate32 global and8 history channels, apply5x5 stride1 convolution+bias, per-pixel channel RMS normalization and SiLU to16 channels, then5x5 RGB output16→3. The original output40→3 narrows. Net+14,232 scalars:80 logical weights/8,871,955 scalars/240 checkpoint tensors. The effective history field grows17→21 pixels. Capacity, output width, initialization, receptive field and gradients change together; no isolated causal conclusion follows.

RSSM recurrent2048/stochastic512/encoder1024 and feature2560 stay. Full world state also includes24,576 image-history scalars, omitted from legacy feature exports. Each imagined step advances the categorical RSSM, decodes from current belief and two causal images, then shifts its generated RGB into history. Full BPTT crosses generated-image history. No true future RGB, outcome or reset label enters imagined state; future truth is only a stopped loss target/mask. Reward/continuation heads remain RSSM-only. No new backend operation, pretrained encoder, object/RAM/phase label, actor or critic is added.

228 common initial tensors/76 weights match spatial v10, and201/67 match H0. All240 own initial tensors restore exactly. These are matched fresh initializations, not warm starts or independent learner seeds. Sixteen cached non-image initial arrays match spatial v10 exactly; RGB changes as expected. v12 persists the fusion architecture, defaults/v1-v11 remain, and an old native rejects v12 before GPU allocation.

## Training and data

Fresh8,192 updates from0, RGB64/12M/B8/T32/H16/full BPTT/Auto, seed41001 and all rates4e-4, latent mixing.01. Keep observed motion RGB cross-entropy100, KL1/.1/free-nat1, observed reward/continuation1 and future motion RGB100. No generated-image re-encoding or future event loss. All eleven reports and total-loss identity remain. Targets/masks are stopped; terminal arrivals remain valid and reset-crossing suffixes are masked. Average over full BxH, not valid count.

Original sixteen fitting/eight development streams and full 8,192-row 4/2/1/1 sampling plan stay. Recount 975,941 valid suffix targets, 9,570 future terminals and 16,008 future positive-reward targets; 10,909 observed terminal rows, 34,440 observed positive-reward rows, 75,808 context resets and 10,488 source resets.32,849 unique arrivals among32,850 available arrivals are sampled. Replayed rows are not new environment interactions. No new collection, emulator actions/frames, teacher, policy or critic training occurs.

Reused seed/families are development. This is neither untouched generalization nor fresh confirmation. Initial/2048/4096/final checkpoints and all periodic draws are retained; only the fixed final is assessed. No best checkpoint/draw replaces the endpoint.

## Recorded ball precision

Within one resized RGB64 pixel, first fixed draw, varying visible-ball support. Missing predicted balls fail. Related origins are not independent episodes.

| Split/model | Observed | 1 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| Development: Fusion actual initial |0/112|0/109|0/109|0/107|0/114|0/95|
| Development: Retained categorical H0 |108/112|94/109|33/109|14/107|3/114|0/95|
| Development: Categorical H16 |99/112|84/109|47/109|25/107|10/114|4/95|
| Development: Spatial H16 |87/112|81/109|46/109|24/107|5/114|2/95|
| Development: Nonlinear fusion H16 |93/112|95/109|62/109|34/107|15/114|4/95|
| Fitting: Fusion actual initial |0/163|0/177|0/180|0/176|0/171|0/179|
| Fitting: Retained categorical H0 |159/163|153/177|73/180|34/176|14/171|5/179|
| Fitting: Categorical H16 |154/163|130/177|86/180|61/176|21/171|19/179|
| Fitting: Spatial H16 |137/163|131/177|76/180|69/176|31/171|5/179|
| Fitting: Nonlinear fusion H16 |141/163|150/177|96/180|72/176|38/171|23/179|

Even on fitting streams, eight-action precision is 72/176. This does not establish successful fitting of short forecasts on training streams; held-out generalization alone cannot account for the measured gap. It does not isolate a particular representational, optimization or rollout cause.

Observed>=90%, one>=89/109 and eight>=28/107 gates stay. 384 dense actual-initial/final origins, 20 source-array identities, 240 own initial/dense-frozen tensors and 18 initial-cohort arrays pass. Accurate observed rendering alone is not forecasting competence.

## Frozen causal forecast assessment

Only the new final receives 84 separately declared frozen groups, history 16 primary/32 secondary, eight draws and 1/4/8/16/32 endpoints. Reuse cached emulator oracle/H0/categorical H16/spatial arrays; no old-model queries or collector repeats. Feed past images and planned actions, with no true future corrections. The primary score is category-balanced survival-weighted ball-feature energy at 16/32; lower is better, unchanged threshold .12712666431732942. It scores a finite decoded-feature mixture, not full image likelihood or calibrated population futures.

| Model | Primary energy | Freeflight8 NOOP/LEFT/RIGHT/REVERSE detections, each of64 | Ending16 Brier | Ending32 Brier | RIGHT16 ending; oracle0 | Non-ending >5%/304; maximum |
|---|---:|---|---:|---:|---:|---|
| Retained categorical H0 |0.169502219|64/63/63/61|0.003345|0.144566|0.110494|0/304; 0.019069|
| Categorical H16 |0.311976907|64/60/41/51|0.002422|0.189225|0.082001|23/304; 0.438399|
| Spatial H16 |0.478844057|64/63/57/61|0.012795|0.230475|0.033346|187/304; 0.968467|
| Nonlinear fusion H16 |0.345425368|64/64/56/64|0.077296|0.483186|0.064732|276/304; 0.990268|

Primary improvement versus H0: -103.788% (required 25%).

Every freeflight command has the same oracle ball path through 8; each command needs>=90% detections. Pair accuracy omits survival weighting and cannot replace the primary score. Sticky-action randomness means a single future picture is not a complete forecast. The previous unattainable FIRE pair gate remains invalid and is not reused.

| Intermediate gate | Result |
|---|---|
| observed_ball |fail|
| one_step_ball |pass|
| eight_step_ball |pass|
| serve_noop |pass|
| ending_brier |fail|
| ending16_brier |fail|
| ending16_right |pass|
| nonending_false_alarms |fail|
| mean_ball_energy_reduction |fail|
| freeflight_each_plan_presence_at8 |fail|

`supplementary-metrics.json` retains every individual non-ending false alarm>5% and every per-action/horizon cell versus all three controls. The non-ending gate concerns category means; individual alarms remain explicit. Regressed cells (ball/paddle/reward Brier/ending Brier/alive-only brick Brier):

- Versus Retained categorical H0: 123 of210/166 of210/204 of210/167 of210/29 of209.
- Versus Categorical H16: 120 of210/128 of210/195 of210/167 of210/22 of209.
- Versus Spatial H16: 116 of210/132 of210/162 of210/112 of210/23 of209.

Related cells are not independent tests, significance evidence or seed reliability. H0 has no suffix loss; categorical H16 and spatial H16 have motion-RGB suffix supervision but not this nonlinear spatial fusion. No isolated causal attribution follows.

## Execution, cost and preserved failures

104 unique affected Rust CPU checks,208 packaged Python CPU checks with GPU loader disabled,and strict core/binding Clippy pass. First native qualification: 1212 pins/27.200s. F64 finite differences cover the composed40-input/16-output fusion; native values/raw derivatives and frozen parameters pass. The actual decoder has joint pre-sigmoid logit interaction and lane isolation.228 spatial/201 H0 common initial tensors,240 own initial/frozen tensors,18 initial/trained restore arrays,77 changed first-update weights including all three fusion parameters,eleven finite reports/totals/masks and old-native v12 rejection pass.

Learning: 1219 pins/170 CPU inputs,8,192 finite eleven-report rows,total identity,all80 changed weights,full source plan/exposures,240 exact initial/frozen tensors,384 dense origins,58 finite exports and18 exact final restore arrays pass. v12 binds fusion and rejects incompatible headers/whole-agent paths. Checkpoints omit replay,live belief/image history/full RNG; a restored extension would be a warm start and is not selected.

Native training calls 1142.861s; measured training/evaluation loop 1317.429s. Loop excludes native construction,dense-initial queries and pinning. Qualification,restore,frozen assessment,CPU analysis and build/packaging are separate. No inherited learning cost is hidden.
Observed train worker/polling span: 1337.234s.
Observed restore worker/polling span: 9.211s.

Frozen assessment 211.880s; CPU comparison 54.104s; learning audit 8.547s. Four successful guards/36 sealed files pass; no native-stage retry,host recovery,NVML or unfinished child. Disposable 865,550,538-byte build cache released; wheel,runtime,CPU executable,source,logs and useful checkpoints retained.

Three original CPU failures remain preserved. `cpu-fusion.log` and `geometry-fixture-first.rs` expected8,882,355 weights using the wrong preset CNN depth; the count-only correction and full-channel finite-difference fixture pass. `clippy.log` and `fusion-inline-first.rs` preserve the enum-layout warning; boxing only the optional CPU descriptor fixes it without changing the GPU graph. `packaged-cpu.log` and `python-fixture-first.py` preserve207 passes/one failure because incomplete tiny metadata reached full-BPTT validation first; separate standalone/full-BPTT fixture repair passes208 tests. Native/package/runner remain7e1408c9; corrected Python-test-only source is6d9cd8b1. No native qualification or successful old stage is repeated.

Retained H0 used 529.515s native/796.900s loop; categorical H16 used 765.720s/1105.997s; spatial H16 used 1021.332s/1186.625s. Distinct single-run runtimes/wrappers are not isolated kernel benchmarks or a learning-per-hour benefit. The new hidden convolution adds work; useful quality must justify its cost.

Four-model comparison/learning figures and three full causal APNGs use retained arrays only, fixed lane 0/draw 0 rather than favorable matches. Native oracle 160x210 and RGB64 predictions share display aspect only. Clips decode losslessly; plotting versions are retained. No input/metric preprocessing change or new model/emulator query occurs.

This tests one joint-world recipe. Precise all-horizon physics,event timing/calibration,persistent removed-brick memory,late/tunnel/second-wall coverage,untouched families and fresh seed confirmation remain required. Existing oracle has no above-top-brick endpoint and its detector supports a restricted region. Intermediate gates cannot complete the full goal. No unchanged extension,sweep or automatic successor is selected.
