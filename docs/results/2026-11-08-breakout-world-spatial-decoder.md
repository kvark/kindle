# Breakout: decoder with causal spatial image history

The fresh experiment is complete and **fails retention**. Full world-model competence remains unproven. Keep categorical H0 as retained reference; no unchanged extension, repeat, sweep or automatic successor. Actor/critic and CDP comparisons remain parked.

[Protocol](../experiments/2026-11-08-breakout-world-spatial-decoder.md).
Root `/mnt/data/kindle-breakout-world-spatial-decoder-20261108`.
Source/native/runner `0a50d6119103ade877d34e79a8462505241a3f21`; native SHA256 `0228e6257ab109b57e69d6752a66dbdb08fc323edf9bdb80d68e4c929e533c34`.

## Architecture, state and selected recipe

Optional default-off `rgb_spatial_decoder` keeps the categorical RSSM and adds three5x5 stride1 ConvNorm layers,width8,from the two preceding RGB images to the final decoder projection. History receptive field17 pixels. This tests direct spatial memory/local motion alongside the global latent path; a bottleneck cause or an isolated benefit is not established.

RGB64/12M recurrent2048/stochastic512/encoder1024 and existing decoder layers stay. Add5,048 weights:77 logical/8,857,723 scalars/231 checkpoint tensors.67 unchanged-shape initialized weights/201 weight-moment tensors match H0 actual initial exactly; the expanded RGB output weight and nine new spatial parameters differ. The new actual-initial and own restored controls are retained. No warm start,teacher,pretraining,object/RAM/phase labels or new backend primitive.

The full world state includes24,576 image scalars in addition to RSSM feature2560. Legacy encoder/posterior/prior feature readbacks omit image history; do not interpret them as the full belief. Past images,capacity,output geometry,initialization,work and gradient paths change together. Event heads still read RSSM only; image quality cannot substitute for reward/event calibration. Parameter counts exclude activation/staging/driver memory and are not measured peak GPU memory.

Posterior reconstruction reads two earlier images only. Observed resets zero history; an intervening earlier reset zeros the older image. Forecasts seed from the last two observed images,zeroing the older across a last-observed reset. Each future step shifts in its own decoded image,with full BPTT. Future images and outcome/reset labels never correct or clear imagined history. Future images/motion weights are stopped targets; reset crossings are loss-masked while terminal arrivals stay. Inference uses the same shift. No future teacher forcing.

Fresh8,192 updates,RGB64/12M/B8/T32/full BPTT/Auto,H16 motion RGB cross-entropy100 plus observed RGB100,KL1/.1/free-nat1,reward/continuation1,seed41001,rates4e-4. Original sixteen fitting/eight development streams and complete8,192-row4/2/1/1 source plan stay:975,941 valid suffix targets/9,570 suffix terminals,10,909 observed terminals,34,440 positive reward rows and32,849/32,850 unique arrivals. No new emulator actions/frames or data. Replayed rows are not new experience. Reused seed/families are development,not fresh confirmation or untouched generalization.

## Recorded ball precision

Within one RGB64 pixel,first fixed draw,varying visible support. Missing predicted balls fail; related origins are not independent episodes.

| Split/model | Observed | 1 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| Development: Spatial actual initial |0/112|0/109|0/109|0/107|0/114|0/95|
| Development: Retained categorical H0 |108/112|94/109|33/109|14/107|3/114|0/95|
| Development: Categorical H16 |99/112|84/109|47/109|25/107|10/114|4/95|
| Development: Spatial H16 |87/112|81/109|46/109|24/107|5/114|2/95|
| Fitting: Spatial actual initial |0/163|0/177|0/180|0/176|0/171|0/179|
| Fitting: Retained categorical H0 |159/163|153/177|73/180|34/176|14/171|5/179|
| Fitting: Categorical H16 |154/163|130/177|86/180|61/176|21/171|19/179|
| Fitting: Spatial H16 |137/163|131/177|76/180|69/176|31/171|5/179|

Observed>=90%,one>=89/109,eight>=28/107 gates stay.384 matched dense actual-initial/final origins,20 identity-array checks,231 own initial/dense-frozen tensors and18 own initial cohort arrays pass. Observed rendering alone is not predictive competence.

## Fixed-action physical and event assessment

Only the new final receives84 separately declared frozen groups,history16 primary/32 secondary,eight draws,all1/4/8/16/32 endpoints. Original oracle/control arrays are reused; no old-model query or collector rerun. Primary category-balanced survival-weighted16/32 observable ball energy is lower-is-better; unchanged threshold.12712666431732942. It does not itself certify calibrated futures.

| Model | Primary energy | Freeflight8 NOOP/LEFT/RIGHT/REVERSE detections, each of64 | Ending16 Brier | Ending32 Brier | RIGHT16 ending; oracle0 | Non-ending >5%/304; maximum |
|---|---:|---|---:|---:|---:|---|
| Retained categorical H0 |0.169502219|64/63/63/61|0.003345|0.144566|0.110494|0/304; 0.019069|
| Categorical H16 |0.311976907|64/60/41/51|0.002422|0.189225|0.082001|23/304; 0.438399|
| Spatial H16 |0.478844057|64/63/57/61|0.012795|0.230475|0.033346|187/304; 0.968467|

Relative primary improvement versus H0: -182.500% (required25%).

All oracle freeflight commands have identical ball paths through8; every command needs>=90% visible balls. Related pair precision omits survival weighting and cannot replace the primary statistic. Every history/horizon/event/reward/brick cell and per-origin probability remains retained.

| Intermediate gate | Result |
|---|---|
| observed_ball |fail|
| one_step_ball |fail|
| eight_step_ball |fail|
| serve_noop |pass|
| ending_brier |fail|
| ending16_brier |fail|
| ending16_right |pass|
| nonending_false_alarms |fail|
| mean_ball_energy_reduction |fail|
| freeflight_each_plan_presence_at8 |fail|

`supplementary-metrics.json` retains every individual non-ending false alarm>5% and every action/horizon cell versus H0/categorical H16. The non-ending gate applies to category means; individual alarms remain explicit. Regressed cells (ball/paddle/reward Brier/ending Brier/alive-only brick Brier):

- Versus Retained categorical H0: 113 of210/164 of210/184 of210/171 of210/32 of209.
- Versus Categorical H16: 114 of210/117 of210/108 of210/176 of210/21 of209.

Related cells are not significance tests or seed reliability. H0 lacks the suffix loss; categorical H16 shares the selected objective and full plan. Image memory,decoder geometry,initialization and gradients differ. No isolated causal attribution follows.

## Execution, cost and preserved failures

100 affected Rust CPU/118 packaged Python/strict core-binding Clippy pass. First native qualifier:1189 pins/25.687s. History layout/reset values and derivatives match F64 references; fixed-latent perturbations of either history frame alter decoding only in their lane.201 common initial tensors/67 weights,231 own initial/frozen tensors,18 initial/trained restore arrays,74 first-update changed weights across required families/spatial layers,finite11-report totals/masks and old-native v10 rejection pass. Existing Conv5/RMS/upsample/categorical primitive receipts are reused.

Learning:1196 pins/168 CPU inputs,8,192 finite11-report rows,all77 changed weights,full source plan/exposures,231 exact initial/frozen tensors,384 dense origins,58 finite exports and18 exact final restore arrays pass. v10 persists architecture; incompatible headers/objectives/phases/whole-agent entry points reject. Defaults and old architectures stay. Checkpoints omit replay/live belief including image history/full RNG; any restored extension would be a warm start and is not selected.

Native learning calls 1021.332s; train/evaluation loop 1186.625s. No inherited learning cost. Qualification,restore,frozen assessment,CPU analyses,build/packaging are separate; the loop excludes construction,dense-initial queries and pinning.
Observed train worker/polling span:1204.658s.
Observed restore worker/polling span:9.212s.

Frozen assessment 192.280s; CPU comparison 38.940s; learning audit 7.795s. Four successful guards/36 sealed files pass; no native stage failure/retry,host recovery,NVML or unfinished child. Disposable 835,554,056-byte build cache released; wheel,runtime,CPU executable,source and useful checkpoints retained.

Preserve first Rust fixture literal syntax error (`history-fixture-first.rs`/`syntax-fixture-repair.json`). Preserve `cpu-spatial.log`/`analytic-fixture-first.rs`: its1e-12 analytic gradient tolerance was below the1.24e-9 rounding from the graph's F32 mean reciprocal. The separate1e-8 tolerance repair passes; production math is unchanged. Native mechanism fixtures additionally retain unused parameters as readback outputs without changing first-loss gradients. Earlier experiments/failures remain closed.

Retained categorical H16 used765.720s native/1105.997s loop; H0 used529.515s native/796.900s loop with no suffix objective. Single-run costs from distinct runtimes/wrappers are not isolated kernel benchmarks or a learning-per-hour benefit. Extra image state and activation work matter despite the small weight increment.

Comparison/learning figures and three full causal APNGs use cached arrays only,fixed lane0/draw0 rather than best matches. Native oracle160x210 and RGB64 models share display aspect only. Decoded clip frames are checked losslessly; plotting environment is retained. No input/metric preprocessing change or new model/emulator query.

This tests one spatial-memory recipe. Precise all-horizon physics,event timing/calibration,persistent removed-brick memory,late/tunnel/second-wall coverage,untouched families and fresh confirmation remain necessary. Existing oracle has no above-top-brick endpoint. Intermediate gates cannot complete the full goal. No unchanged extension,sweep or automatic successor.
