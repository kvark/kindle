# Broad world learning with position weight 10,000

Fixed final 8,192 passes **0/3 original dense development gates**. The screen fails; retain the H0 reference and do not allocate the conditional oracle assessment.

The Breakout world-model goal remains active and unproven. Coordinate accuracy cannot replace ordinary RGB physics, event calibration or full-game coverage.

[Protocol](../experiments/2026-11-09-breakout-world-centroid10000-learning.md).
Artifacts: `/x/Temp/kindle-world-centroid10000-learning-20261109`.

## Fixed final RGB forecasts

First fixed draw, within one resized pixel. Missing or diffuse balls fail. Supports vary across horizons; related origins are not independent episodes. No checkpoint, draw or tolerance selection.

|Split/model|Observed|1|4|8|16|32|
|---|---:|---:|---:|---:|---:|---:|
|Development: Actual initial|0/112|0/109|0/109|0/107|0/114|0/95|
|Development: Retained H0|108/112|94/109|33/109|14/107|3/114|0/95|
|Development: Original H16|99/112|84/109|47/109|25/107|10/114|4/95|
|Development: Position1,000|85/112|73/109|35/109|15/107|5/114|5/95|
|Development: Position10,000|77/112|70/109|27/109|14/107|6/114|0/95|
|Fitting: Actual initial|0/163|0/177|0/180|0/176|0/171|0/179|
|Fitting: Retained H0|159/163|153/177|73/180|34/176|14/171|5/179|
|Fitting: Original H16|154/163|130/177|86/180|61/176|21/171|19/179|
|Fitting: Position1,000|134/163|119/177|50/180|34/176|12/171|15/179|
|Fitting: Position10,000|118/163|107/177|57/180|28/176|13/171|7/179|

Original observed ≥90%, one-step ≥89/109 and eight-step ≥28/107 gates: `{'observed_ball': False, 'one_step_ball': False, 'eight_step_ball': False}`.

All 384 actual-initial/final dense origins, RGB targets, commands, resets and outcomes match the retained controls. Actual-initial model arrays also match the weight1,000 control exactly. All horizon, paddle, brick, RGB, changed-pixel and event metrics remain in the retained exports. No old model was queried again.

## Position readout and contact diagnostic

The CPU F64 affine calculation uses saved F32 state features and the trained [2560,2] head matrix. Native affine parity and observed-target alignment were verified separately at the earlier checkpoint; these exact new rows and weights were not queried through a native head. The head has no presence output and is scored only where the true ball is visible. It cannot establish RGB accuracy or information absence.

|Final split|State|Within1px /support|Euclidean RMSE (pixels)|
|---|---|---:|---:|
|fitting|observed 0|44/163|1.775|
|fitting|prior 1|47/177|2.672|
|fitting|prior 4|32/180|2.897|
|fitting|prior 8|22/176|4.909|
|fitting|prior 16|16/171|7.472|
|fitting|prior 32|7/179|11.453|
|development|observed 0|40/112|1.648|
|development|prior 1|27/109|1.648|
|development|prior 4|30/109|4.433|
|development|prior 8|15/107|6.335|
|development|prior 16|7/114|9.244|
|development|prior 32|0/95|12.973|

All initial/final errors, medians, tails and out-of-image coordinates remain in head-review.json and head-endpoints.csv. Slight differences between the pixel-derived centroid target and the fixed scoring detector, and unsupported extra mass-valid labels, remain in the qualification evidence.

The eight original contact clips are reused diagnostics, not this learner’s fitting data or untouched generalization. The original contact learner trained on them and matched110/110 at all four draws; that is not a matched-budget control.

|Contact draw|Matches /110|
|---|---:|
|0|19/110|
|1|25/110|
|2|16/110|
|3|21/110|

## What changed and what was verified

Only the auxiliary coefficient changes from1,000 to10,000 relative to the closed RGB1/position1,000 run. Configuration equality after that one substitution, all210 initial tensors and every initial dense array pass. Same seed41001, RGB64/12M/B8/T32/H16/full BPTT/Auto, rates4e-4, mixing.01, KL1/.1/free-nat1, observed reward/continuation1, RGB1, shared linear head,70 logical weights/8,857,797 scalars/210 tensors. A single reused development seed does not establish reliability. The earlier frozen gradient measurement justified this scale test but did not predict an Adam/AGC update or establish a cause of the failure.

Targets retain red excess over1.5×max(other channels), region[4/64,28/64,60/64,56/64], normalized pixel centers and mass>1e-6. This is game-specific supervision derived from RGB, with no object or emulator-state input. Targets and masks are detached; future targets do not correct prior states. Ordinary physics and event gates stay unchanged.

Exact original16 fitting/eight development streams and4/2/1/1 plan:2,097,152 replayed arrivals,975,941 valid suffix targets,9,570 suffix terminals,16,008 positive suffix rewards.32,849/32,850 unique fitting arrivals,34,440 observed positives,10,909 terminals,75,808 context resets and10,488 source resets. No new emulator experience, actor, critic, CDP or collection.

All8,192 fifteen-report rows are finite and reconstruct total loss with coefficient10,000; source/reset/valid masks pass. All70 weights change.210 initial/own frozen tensors,18 development/32 contact restore arrays,34 finite forecast exports and v22 metadata/ages pass.149 native declaration/149 CPU analysis pins verified; two terminal successful native jobs/two guards/18 sealed files. No native retry or repair.

The copied launcher retained a stale top-level centroid_scale:1000 summary in train/declaration.json. The pinned protocol, input config, nested declaration config, native getter, all checkpoint configs and every reported total loss use10,000. Original source and summary remain unchanged; declaration-label-review-v1.json is a separately pinned CPU clarification. This bookkeeping defect did not change the executed objective and was not repaired by restarting training.

Qualification source449361ae/native SHA256ee44ec16b509c601d6beefafeb03630aa9be9a3681e632a444dbd67b2b788712 and matching runner are reused, without rebuild or overlay.139 Rust CPU/strict core+binding Clippy/19 Python checks and masked native derivatives belong to that earlier qualification; its original fixture failures and separate corrected checks remain.

Native learning calls 768.912s; training/checkpoint/forecast loop 874.419s. Initial construction/dense exports, final contact diagnostic, separate restore and CPU analysis are outside that loop. The weight1,000 control uses the same export cadence; other older controls differ. These wrapper intervals do not isolate GPU kernel speed.

Retain0/2048/4096/final8192 checkpoints, four-draw curves0/2048/4096/6144/8192, all dense arrays/features and a fixed lane0/draw0 lossless16-frame contact clip. Checkpoints omit replay/live belief/full RNG; later restores would be warm starts. Invocation0ff931bf85e449cf9ddf5fdc32083d07. No build cache was created.

Full precise physics, naturally calibrated rewards/life/game-over timing, persistent removed-brick memory, late/tunnel/second-wall/above-top coverage, untouched families and fresh confirmation remain required. Passing an intermediate gate cannot complete the goal. No extension, collector or automatic successor is selected. Actor/critic, CDP comparison and five-game work stay parked; earlier jobs stay closed and fullres stays protocol-failed.

Closure verifies326 pins,terminal successful native queue/observer and18 sealed guard files. Broad0/3;conditional assessment remains unallocated.
