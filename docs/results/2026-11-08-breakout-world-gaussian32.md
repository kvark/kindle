# Breakout: compact Gaussian32 world reference

The fresh experiment is complete and **fails retention**. Full world-model competence remains unproven. Keep categorical H0 as retained control; no extension, repeat, sweep or automatic successor. Actor/critic and CDP comparisons remain parked.

[Protocol](../experiments/2026-11-08-breakout-world-gaussian32.md).
Root `/mnt/data/kindle-breakout-world-gaussian32-20261108` retains its selection tag; native qualification/learning occurred November8.
Source/native/runner `40828aef088e868e8f01e45487938facf39e79f5`; native SHA256 `657d0ba47d869ec9c22c64aefb28707cd0a77c3d9e87d7fb7ad3b6d593fadf06`.

## Architecture and selected recipe

Optional default-off `world_gaussian` replaces categorical straight-through states with32 continuous diagonal-Gaussian scalars (`world_gaussian_dim:32`). Prior/posterior predict mean and raw scale; std=.1+sigmoid(raw), bounded[.1,1.1], means unbounded. Native reparameterized samples use independent standard-normal draws. Exact Gaussian KL is summed over32 scalars, with stopped-q dynamics1, stopped-p representation.1 and per-state free-nat1. Categorical mixing is unused. This changes family, capacity, KL geometry, scale parameterization, sampling and gradients; no isolated estimator or architecture cause is established.

CNN/GRU/encoder1024/RGB decoder/reward-continuation layers stay. Compact stochastic width32 plus recurrent2048 gives feature2080. Prior/posterior output64 mean/raw-scale values, interleaved per dimension.68 logical/8,008,003 scalar weights/204 checkpoint tensors,1,107,840 weights fewer than Gaussian512.60 common initialized weights/180 weight-moment tensors match H0 actual initial exactly; eight parameter shapes differ. New actual-initial/own restored controls are retained. Capacity, noise dimension, projection geometry and initialization change together; no isolated noise cause or equal-information-capacity claim. No teacher, warm initialization, pretrained encoder, object/RAM/phase labels or backend primitive.

Fresh8192 updates, RGB64/12M/B8/T32/full BPTT/Auto, H16 existing motion RGB cross-entropy100 plus observed RGB100, KL1/.1/free-nat1 and reward/continuation1, seed41001, rates4e-4. Original sixteen fitting/eight development streams and complete8192-row4/2/1/1 source plan stay.975,941 valid suffix targets/9,570 suffix terminals,10,909 observed terminals,34,440 positive-reward rows and32,849/32,850 unique arrivals match. No new emulator actions/frames or collection. Replayed rows are not new experience.

Future samples use only past observed state and submitted future actions. Future images/motion weights are stopped targets, not prediction inputs; retain terminal arrivals, mask reset crossings and average over full Bx16. Normal draws are fixed for a gradient evaluation. This remains pixel-supervised world-only learning; no actor/critic. The reused seed/families are development, not fresh confirmation or untouched generalization.

## Recorded ball precision

Within one RGB64 pixel, first fixed draw, varying visible support. Related origins are not independent episodes.

| Split/model | Observed | 1 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| Development: Gaussian32 actual initial |0/112|0/109|0/109|0/107|0/114|0/95|
| Development: Retained categorical H0 |108/112|94/109|33/109|14/107|3/114|0/95|
| Development: Categorical H16 |99/112|84/109|47/109|25/107|10/114|4/95|
| Development: Gaussian512 H16 |105/112|76/109|29/109|16/107|10/114|2/95|
| Development: Gaussian32 H16 |103/112|68/109|39/109|13/107|6/114|1/95|
| Fitting: Gaussian32 actual initial |0/163|0/177|0/180|0/176|0/171|0/179|
| Fitting: Retained categorical H0 |159/163|153/177|73/180|34/176|14/171|5/179|
| Fitting: Categorical H16 |154/163|130/177|86/180|61/176|21/171|19/179|
| Fitting: Gaussian512 H16 |154/163|120/177|68/180|45/176|14/171|6/179|
| Fitting: Gaussian32 H16 |153/163|112/177|78/180|48/176|23/171|12/179|

Observed >=90%, one >=89/109 and eight >=28/107 gates stay.384 matched dense-initial/final origins,20 identity-array checks,204 own-initial/dense-frozen tensors and18 own initial cohort arrays pass. Observed rendering alone is not predictive competence.

## Fixed-action physical and event assessment

Only the new final receives84 separately declared frozen groups, history16 primary/32 secondary,eight draws,all1/4/8/16/32 endpoints. Original oracle/control arrays are reused; no old-model query or collector rerun. Lower primary category-balanced survival-weighted16/32 observable ball energy is better; unchanged threshold.12712666431732942. It does not itself certify calibrated futures.

| Model | Primary energy | Freeflight8 NOOP/LEFT/RIGHT/REVERSE detections, each of64 | Ending16 Brier | Ending32 Brier | RIGHT16 ending; oracle0 | Non-ending >5%/304; maximum |
|---|---:|---|---:|---:|---:|---|
| Retained categorical H0 |0.169502219|64/63/63/61|0.003345|0.144566|0.110494|0/304; 0.019069|
| Categorical H16 |0.311976907|64/60/41/51|0.002422|0.189225|0.082001|23/304; 0.438399|
| Gaussian512 H16 |0.557270519|40/40/40/47|0.122647|0.153687|0.659267|8/304; 0.055299|
| Gaussian32 H16 |0.409349797|48/48/49/48|0.101748|0.035072|0.524479|0/304; 0.032928|

Relative primary improvement versus H0: -141.501% (required25%).

All oracle freeflight commands have identical ball paths through8; every command needs >=90% visible balls. Related model/oracle pair precision omits survival weighting and cannot replace the primary statistic. All histories/horizons/events/rewards/brick cells and per-origin probabilities remain in the reports.

| Intermediate gate | Result |
|---|---|
| observed_ball |pass|
| one_step_ball |fail|
| eight_step_ball |fail|
| serve_noop |pass|
| ending_brier |pass|
| ending16_brier |fail|
| ending16_right |fail|
| nonending_false_alarms |pass|
| mean_ball_energy_reduction |fail|
| freeflight_each_plan_presence_at8 |fail|

`supplementary-metrics.json` retains every individual non-ending false alarm >5% and every action/horizon cell against H0, categorical H16 and Gaussian512. The non-ending gate applies to category means; individual cases remain reported even when means pass. Regressed cells (ball/paddle/reward Brier/ending Brier/alive-only brick Brier):

- Versus Retained categorical H0: 129 of210/132 of210/156 of210/88 of210/52 of209.
- Versus Categorical H16: 124 of210/93 of210/102 of210/81 of210/32 of209.
- Versus Gaussian512 H16: 93 of210/104 of210/153 of210/45 of210/37 of209.

These related cells are not significance tests or seed reliability. H0 lacks the suffix loss; categorical H16 and Gaussian512 share the selected RGB objective and full source plan. Family, latent width, eight parameter shapes and initialization differ. Do not compare Gaussian and categorical KL magnitudes as a common competence measure.

## Execution, cost and preserved limits

96 affected Rust CPU/100 packaged Python checks and strict core/binding Clippy pass after a preserved dependency-build storage failure and cache relocation. First native qualifier: 1180 input pins/25.514s. Gaussian sample/KL values and derivatives match F64,with both stop sides/free-nat floor,explicit sample elements and frozen math parameters.180 common initial tensors/60 weights,204 own-initial tensors,18 initial/trained restore arrays,65 first-update changed weights across expected families,both mean/scale projection changes,continuous samples,204 frozen tensors and old-native v9 rejection with GPU loader disabled pass. Unchanged CNN/GRU/resize primitives are reused.

Learning: 1187 pins/168 CPU inputs,8192 finite11-report rows,all68 changed parameters,full plan/exposures,204 exact initial/frozen tensors,384 dense-initial origins,58 finite exports and18 final restore arrays pass. v9 persists explicit architecture/config/geometry;old headers cannot disguise it,unsupported objectives/phases and whole-agent entry points reject the standalone variant. Default categorical paths stay unchanged. Checkpoints omit replay/live games/full RNG; a restored extension would be a warm start and is not selected.

New native learning calls 734.664s;train/evaluation loop 881.548s. There is no inherited training cost. Qualification,restore,frozen assessment,CPU analyses,build/packaging are separate;the loop excludes construction,dense-initial queries and pinning.
Observed train worker/polling span: 899.227s.
Observed restore worker/polling span: 9.212s.

Frozen assessment 168.671s;CPU comparison 50.490s;learning audit 7.621s. Four successful guards/36 sealed files pass;no host recovery,NVML or unfinished child. No native stage failure/retry. Student parameter counts exclude activation/staging/driver memory and are not measured peak memory. Disposable 1,008,055,634-byte build cache released;wheel,runtime,CPU executable,source and useful checkpoints retained.

The first dependency build failed with system-disk ENOSPC before tests. Preserve cpu-gaussian.log and cpu-build-storage-failure.json; the separately repaired build on /mnt/data passes. Production math is unchanged. All earlier failures/results remain closed and preserved.

The retained categorical H16 run used765.720s native/1105.997s loop; Gaussian512 used810.106s native/959.732s loop. These are single-run observations from distinct runtimes/wrappers,not isolated kernel benchmarks or a learning-per-hour benefit.

Comparison/learning figures and three full causal APNGs use cached arrays only, fixed lane0/draw0 rather than best matches. Native oracle160x210 and RGB64 models share display aspect only;no input/metric preprocessing change. Decoded frames are checked losslessly;plotting environment is recorded. No new model/emulator query for these artifacts.

This tests one32-dimensional bounded-scale Gaussian recipe;it does not rule out other continuous models or identify a unique representation,gradient,capacity,scale or data cause. Precise all-horizon physics,event timing/calibration,persistent brick memory,late/tunnel/second-wall coverage,untouched families and fresh confirmation remain necessary. This oracle has no above-top-brick endpoint. Full goal remains active and unproven;no unchanged extension,sweep or automatic successor.
