# Breakout: fixed full observation filter with direct future RGB supervision

The phase is complete and **fails retention**. Full world-model competence remains unproven. Keep the coverage H0 control; no repeat, extension, sweep or automatic successor is selected. Actor/critic and CDP comparisons stay parked.

[Protocol](../experiments/2026-11-07-breakout-world-fixed-image.md).
Root `/mnt/data/kindle-breakout-world-fixed-image-20261107`.
Native/source/runner `b4d4f8f259caea9e062ae785f18f619faf96d5db`; native SHA256 `9402c44d3b1c025d1ba121fa1f2d4cb51ea90cfea5087fac1070df4c0d7d4f53`.

## Intervention

The preceding fixed-target phase improves some forecasts but fails eight-step precision, per-command presence, ending32 and primary energy. Earlier frozen-visual H16 RGB training leaves recurrent observation conditioning changing. This tests the missing combination: a fully immutable observation filter with direct RGB forecast supervision through a fixed decoder. Neither completed learner is repeated or extended.

Warm-start retained H0 final8192, preserving all204 weights/moments and optimizer age. The independent H0 filter, including recurrence/encoder/posterior, stays fixed.36 visual weights stay fixed;32 forecast dynamics/reward/continuation weights learn. Replace auxiliary H16 latent KL1 plus h energy100 with existing H16 motion-weighted RGB cross-entropy100. Base observed RGB100, one-step KL1/.1/free-nat1 and reward/continuation1 remain. Derivatives through fixed visual functions remain; their72 optimizer moments may change.

RGB64/12M/B8/T32/full BPTT/Auto, mixing.01, seed41001, rates4e-4,16 original fitting/eight development streams and first4096 original4/2/1/1 plan rows stay.4096 new updates end12288. No new emulator actions/frames, data, object/RAM/phase labels, pretrained encoder, larger model, actor/critic, analytic loss or backend primitive.

The fixed filter supplies image-informed training states, with per-lane sampling seeds solely from seed/optimizer age. Realizations vary; repeated explicit seeds stay exact across updates/restore. Forecasts start at observed index15 and use only future submitted actions/prior samples. Future images/motion weights are stopped targets, never prediction inputs. Terminal arrivals remain; later reset episodes are masked; average over full Bx16. Pixel cross-entropy is not a calibrated joint distribution objective; repeated-sample physical and event gates remain mandatory.

Warm initialization, objective/sample/gradient structure and observation-filter separation limit isolated causal claims. This is reused development, not untouched generalization, an independent seed, uninterrupted learning or a matched fresh-budget comparison.

## Recorded ball precision

Within one RGB64 pixel, first fixed draw, varying visible support; related origins are not independent episodes.

| Split/model | Observed | 1 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| Development: H0 |108/112|94/109|33/109|14/107|3/114|0/95|
| Development: Fixed latent/h targets |108/112|98/109|43/109|17/107|10/114|4/95|
| Development: Earlier frozen visual RGB |108/112|94/109|51/109|24/107|4/114|0/95|
| Development: Fixed full-filter RGB |108/112|94/109|47/109|17/107|3/114|0/95|
| Fitting: H0 |159/163|153/177|73/180|34/176|14/171|5/179|
| Fitting: Fixed latent/h targets |159/163|156/177|82/180|53/176|40/171|20/179|
| Fitting: Earlier frozen visual RGB |163/163|150/177|77/180|40/176|18/171|10/179|
| Fitting: Fixed full-filter RGB |159/163|147/177|73/180|39/176|15/171|1/179|

Observed/one/eight gates remain >=90%, >=89/109 and >=28/107. Matched384 dense-initial/final origins,108 complete cached initial cohort arrays,24 exact observed RGB/encoder/posterior arrays,204 inherited/frozen tensors and34 finite exports pass. The full filter and observed readouts do not change. Accurate observed rendering does not establish predictive dynamics.

## Same-origin fixed-action forecasts

Only the new final receives84 frozen groups, history16 primary/32 secondary,eight draws,all1/4/8/16/32 endpoints. Original oracle/control arrays are reused; no old-model assessment or collector reruns. The primary category-balanced survival-weighted16/32 ball statistic is lower-is-better; unchanged threshold.12712666431732942. It and internal losses do not certify calibrated physical futures.

| Model | Primary16/32 ball energy |
|---|---:|
| H0 |0.169502219|
| Fixed latent/h targets |0.141405433|
| Earlier frozen visual RGB |0.204741015|
| Fixed full-filter RGB |0.303067140|

Relative improvement versus H0: -78.798% (required25%).

| Model | Freeflight8 NOOP/LEFT/RIGHT/REVERSE detections, each of64 | Ending16 Brier | Ending32 Brier | RIGHT16 ending; oracle0 | Non-ending >5%/304; maximum |
|---|---|---:|---:|---:|---|
| H0 |64/63/63/61|0.003345|0.144566|0.110494|0/304; 0.019069|
| Fixed latent/h targets |63/62/48/64|0.000253|0.334714|0.024970|0/304; 0.032650|
| Earlier frozen visual RGB |49/49/43/46|0.011062|0.179440|0.177495|0/304; 0.045316|
| Fixed full-filter RGB |64/52/64/64|0.000001|0.451774|0.001589|18/304; 0.109944|

All retained oracle commands have identical freeflight ball paths through8; every command must retain >=90% visible balls. Individual end probabilities, survival, rewards and bricks remain in the raw per-plan reports. Related model/oracle pairs are not independent episodes or a replacement for the primary statistic.

| Intermediate gate | Result |
|---|---|
| observed_ball |pass|
| one_step_ball |pass|
| eight_step_ball |fail|
| serve_noop |pass|
| ending_brier |fail|
| ending16_brier |pass|
| ending16_right |pass|
| nonending_false_alarms |pass|
| mean_ball_energy_reduction |fail|
| freeflight_each_plan_presence_at8 |fail|

`supplementary-metrics.json` retains every action/horizon cell against all three controls and every individual non-ending false alarm >5%. Regressed cells (ball/paddle/reward Brier/ending Brier/alive-only brick Brier):

- Versus H0: 92 of210/145 of210/88 of210/23 of210/31 of209.
- Versus Fixed latent/h targets: 101 of210/137 of210/183 of210/12 of210/28 of209.
- Versus Earlier frozen visual RGB: 82 of210/117 of210/148 of210/32 of210/41 of209.

The non-ending gate applies to category means; individual cases above5% remain reported even when that gate passes. All per-action regressions stay visible; aggregate gains cannot replace event/presence gates. The controls differ in objective/filter conditioning and cost; these related cells are not significance tests or seed reliability.

## Execution and cost

85 affected world CPU/97 packaged Python checks and strict core/binding Clippy pass. The first native qualifier verifies462 input pins in19.872s:204 exact inherited tensors,18 cached initial/18 trained restore arrays,two finite11-report updates,32 active changed/36 visual fixed weights,204 frozen inference tensors,exact GPU anchor/targets/observed outputs after updates and restore,old-native v7 rejection with GPU loader disabled. Unchanged filter sampling/observe source bytes are verified; prefix/lane and primitive receipts are reused rather than rerun.

Learning verifies469 pins/123 CPU inputs:4096 finite11-report rows,32 active changed/36 visual fixed weights,72 allowed visual moment changes,original full plan/exposures,18 final restore arrays,teacher SHA/GPU identity at initial/intermediate10240/final12288.488,356 suffix targets,4,755 suffix terminals,5,419 observed terminals,17,425 positive-reward rows and32,833/32,850 unique arrivals match. v7 persists scope/start/objective; teacher SHA checked before GPU restore; ordinary construction requires explicit source phase.

New native learning calls455.465s and train/evaluation loop532.665s. Inherited H0 adds529.515s/796.900s: sums984.980s/1329.565s. The loop excludes construction,dense-initial queries and pinning; worker/polling spans are retained below. Qualification,restore,frozen assessment,CPU/build/packaging are separate. No learning-per-hour or matched-budget claim.
Observed train worker/polling span: 550.646s.
Observed restore worker/polling span: 9.212s.

Frozen84-group assessment: 150.138s; CPU comparison47.351s; learning audit6.085s. All four successful host guards/36 sealed files pass,with no host recovery,NVML or unfinished child. No native stage failure/retry occurs in this experiment; earlier experiments' failures remain preserved and are not relabeled.

Student68 logical/8,852,675 scalar weights/204 tensors is not total footprint. Extra immutable filter39 logical/6,203,136 scalar copies(24,812,544 bytes),session allocation35,799,168 bytes(excludes driver/staging,not peak or exact system increment),embedded106,261,452-byte full anchor including original moments. Its forward passes/host transfers are included. Disposable834,978,639-byte build cache released; wheel,runtime,CPU executable,source archive and useful checkpoints retained.

## Interpretation and remaining requirements

This comparison tests direct RGB forecast supervision with a fully fixed observation filter. Failure cannot be attributed solely to changing observation targets; passing any subset of gates cannot establish the full world-model goal. The results do not identify a unique encoder,sampling,gradient-estimator,dynamics or data cause.

Comparison/learning figures and three full causal APNGs use cached arrays only, fixed lane0/draw0 rather than a best match. Native oracle160x210 and RGB64 predictions are resized for display only; decoded clip frames are checked losslessly. No new model/emulator queries; offline plotting environment is recorded.

Precise all-horizon physics,event timing/calibration,persistent removed-brick memory,late/tunnel/second-wall coverage,untouched families and fresh confirmation remain required. This oracle has no above-top-brick endpoint. Keep the original goal active and unproven; no unchanged extension,sweep or automatic successor.
