# Breakout: causal prior world-model training

The bounded warm continuation is complete and **fails the declared retention gates**. Full world-model competence remains unproven. The goal remains active; no successor allocation is selected. Actor/critic and CDP comparisons remain parked. Keep categorical H0 as the broad inspectable reference.

[Protocol](../experiments/2026-11-08-breakout-world-clean-prior.md).
Artifacts: `/mnt/data/kindle-breakout-world-clean-prior-20261108/resume`.

## What we are trying to learn

A useful world model must predict the consequences of actions from past observations: ball motion and collisions, paddle response, removed bricks, rewards and life/game endings. Rendering the current image accurately is insufficient. This experiment replaces main-pass posterior reconstruction with causal RGB/reward/continuation forecasts at every valid observed transition, while retaining the H16 free-rollout losses. The posterior has already consumed the current image; the causal prior has not.

Use explicit `motion_clean_prior`, native v14. Main readouts use h_t and sampled prior z_t before q_t consumes the current image. The real-image posterior teacher still conditions subsequent transitions and learns through KL and later transitions. Keep the two-frame decoder history and full BPTT through generated-image history; no generated-RGB re-encoding. Mask first/reset arrivals without incoming state and retain genuine terminal arrivals. No extra parameters, decoder pass, labels, pretrained encoder, teacher, collection, actor or critic. Conditioning, gradients, reset weighting and RNG sampling change together; this is not an isolated frequency test.

RGB64/12M/B8/T32/H16/full BPTT/Auto, seed41001, rates4e-4, mixing.01. World-only model80 logical weights/8,871,955 scalars/240 checkpoint tensors including moments. Main and suffix motion RGB CE100/reward CE1/continuation soft-label BCE1; KL1/.1/free-nat1. Fourteen finite reports include two fractions excluded from the total. Native learning/inference remain on Meganeura + Blade.

## Interrupted lineage and data

The original learner was paused by the user after2,998 reported updates, retaining weights only at2,048. Its interrupted/unsealed training guard, logs and pause receipt remain unchanged. After user resumption, restore2,048 and run6,144 new updates using exactly original plan rows2,048:8,192. All240 weights and optimizer moments are retained; RNG restarts. This is a warm start, neither uninterrupted learning nor an independent learner. The950 unsaved updates remain spent compute, not retained lineage.

The original actual-initial dense384-origin baseline and periodic0/1,024/2,048 arrays are reused. New evaluations begin3,072. Retain initial2,048/new4,096/6,144/final8,192 and the original initial0/checkpoint2,048. Only final8,192 is assessed; no best-checkpoint or best-draw selection.

Original16 fitting/eight development streams, unchanged4/2/1/1 plan, no new environment actions or emulator frames. Retained-lineage2,097,152 replay arrivals cover32,849 unique among32,850 available. Clean causal targets2,021,344 include10,826 terminal/33,188 positive arrivals; suffix975,941 include9,570 terminal/16,008 positive arrivals. These are repeated exposures, not independent samples. RAM/object annotations select or score examples only and never enter the training model.

## Recorded prediction quality

Within one resized RGB64 pixel; missing predicted balls fail. First fixed draw, varying visible-ball support, related reused origins. Observed means posterior rendering; numeric horizons are open-loop action forecasts.

| Split/model | Observed | 1 action | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| Development: Actual initial |0/112|0/109|0/109|0/107|0/114|0/95|
| Development: Retained H0 |108/112|94/109|33/109|14/107|3/114|0/95|
| Development: Spatial fusion |93/112|95/109|62/109|34/107|15/114|4/95|
| Development: Suffix events |98/112|97/109|58/109|31/107|13/114|6/95|
| Development: Causal prior (warm) |93/112|90/109|38/109|27/107|8/114|3/95|
| Fitting: Actual initial |0/163|0/177|0/180|0/176|0/171|0/179|
| Fitting: Retained H0 |159/163|153/177|73/180|34/176|14/171|5/179|
| Fitting: Spatial fusion |141/163|150/177|96/180|72/176|38/171|23/179|
| Fitting: Suffix events |140/163|148/177|99/180|71/176|30/171|20/179|
| Fitting: Causal prior (warm) |134/163|139/177|75/180|47/176|16/171|14/179|

Eight-action precision is 47/176 on fitting and 27/107 on development streams. These measurements do not isolate representation, optimization, data coverage or rollout error. Reused development is not untouched generalization or fresh confirmation.

At eight actions on development, 42 of107 supported balls are not detected in the rendered prediction, and 38 more are detected beyond one resized pixel. Paddle precision is 45/126. Overall brick-cell accuracy is 99.765%, but removed-brick recall is only 4/23. Static image accuracy conceals the changing parts of the world. These detector-supported measurements do not establish why optimization or memory fails.

## Action interventions and events

Only the new final receives84 frozen groups, histories16 primary/32 secondary, eight draws and1/4/8/16/32 endpoints. Reuse cached emulator oracle/H0/fusion/suffix-event arrays. No true future image correction or old-model query. The emulator uses sticky actions, so score forecast distributions as well as fixed draws. Primary score averages survival-weighted ball-feature energy equally over seven categories and horizons16/32, with unchanged threshold.12712666431732942 (25% below H0). This finite feature score is not full-image likelihood or certified population calibration.

| Model | Primary energy ↓ | Freeflight8 detections NOOP/LEFT/RIGHT/REVERSE, each /64 | Ending16 Brier | Ending32 Brier | RIGHT16 ending (oracle0) | Non-ending alarms >5%, /304 |
|---|---:|---|---:|---:|---:|---:|
| Retained H0 |0.169502219|64/63/63/61|0.003345|0.144566|0.110494|0|
| Spatial fusion |0.345425368|64/64/56/64|0.077296|0.483186|0.064732|276|
| Suffix events |0.283365170|64/64/56/64|0.032212|0.226666|0.205177|239|
| Causal prior (warm) |0.452156264|64/64/56/32|0.095987|0.189247|0.562829|235|

Primary change versus H0: -166.755% reduction (required25%). Independent event gates prevent a model from improving survival-weighted geometry simply by predicting premature death. All individual false alarms and per-action/horizon ball/paddle/reward/ending/brick regressions remain in `supplementary-metrics.json`.

| Unchanged intermediate gate | Result |
|---|---|
| observed_ball |fail|
| one_step_ball |pass|
| eight_step_ball |fail|
| serve_noop |pass|
| ending_brier |fail|
| ending16_brier |fail|
| ending16_right |fail|
| nonending_false_alarms |fail|
| mean_ball_energy_reduction |fail|
| freeflight_each_plan_presence_at8 |fail|

The non-ending gate concerns category means; individual alarms stay explicit. Freeflight commands share the same oracle ball trajectory through8; each must retain at least90% detected balls. The previously invalid FIRE pair gate is not reused. Related pairs/cells are not independent tests or seed reliability.

## Cost, verification and preserved failures

New native training calls 875.363s; new measured training/evaluation loop 999.845s. Retained-lineage native cost 1166.406s; actual logged spent native cost 1301.933s, including135.527s from950 discarded reports. The last interrupted in-flight call is not measured, so logged spent cost is not exact total hardware occupancy. Loop excludes construction, pinning and original dense-initial queries. Qualification, restore, assessment and analysis are separate.

Frozen assessment 208.883s; CPU learning audit 15.458s; comparison 56.534s. Original native qualification 35.760s; repaired checkpoint restore13.049s. Single-run timings do not isolate kernels or establish efficiency at matched useful quality.

Original108 world Rust CPU/241 packaged Python CPU checks and strict core/binding Clippy remain passed. Original1,222-pin native qualification proves causal readout/target independence, past-image lane isolation, zero mask behavior, finite full-shape update and legacyv14 rejection. New learning 1431 pins, 127 CPU audit inputs; all8,192 retained-lineage14-report rows, totals and validity/exposure counts pass. All80 trainable weights change after resume;240 initial tensors match the paused checkpoint,240 final/frozen tensors are exact,18 final restore arrays match,384 actual-initial/final origins align and58 exports are finite. No completed qualification or original assessment is repeated.

The first frozen-assessment CPU summary failed on a wrong metric field name (`clean` instead of the saved `events`). Its script, log and observer failure remain preserved. A separately declared CPU reader repair leaves all model/emulator outputs and gates unchanged; no GPU stage repeats.

The first resumed restore failed before forecasting because it compared differing decimal JSON representations of the same float32 config. A separate CPU repair proves exact float32 identity and compares the getter to its cached getter representation; the repaired native check matches18 cached arrays and240 unchanged tensors. Original failed scripts/guard remain preserved. Separately retain the original user-interrupted unsealed guard. Five successful guards/45 sealed files (original qualification, repaired resumed restore, resumed learning, final restore, frozen assessment) coexist with these two non-passing attempts. No host recovery/NVML or unfinished child.

Native/source `35c090d0b3a53555f5b306911d2fa5d5a6e10e4d`, SHA256 `d60f2888dff1a91cfe126edcbb3a4948dadb5282c94d249154d532d79bb50728`; resumed runner `61bf7acb3fb71d767ab1b1c1be552360817be70b`. Native/runtime is unchanged. Released original865,662,743-byte build cache remains absent; wheel/runtime/CPU executable/source/logs/useful checkpoints retained.

Figures and three causal clips use only retained arrays. Fixed lane0/draw0, not favorable matches; native oracle and RGB64 predictions share display aspect only. APNG frames decode losslessly. Images do not change model inputs or metrics.

## What remains before calling this a good world model

Accurate all-horizon ball and paddle physics; correct collision/reward/life-loss timing and event calibration; persistent removed-brick memory; late-game/tunnel/second-wall coverage; untouched trajectory families; and a fresh learner confirmation. The existing oracle has no above-top-brick endpoints and its detector supports a restricted region. Intermediate gates do not prove a perfect world. Policy/critic work and a later CDP efficiency comparison should use demonstrated predictive competence as their starting point.

No unchanged extension, sweep or automatic successor is selected. Original failed learners and collectors remain closed.
