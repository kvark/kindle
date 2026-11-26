# Breakout: fixed observation filter with future-event supervision

The repaired4,096-update phase **fails retention**. Full world-model competence remains unproven. Keep H0 as the inspectable reference. No unchanged extension, sweep or automatic successor is selected. Actor/critic and CDP comparisons remain parked.

[Protocol](../experiments/2026-11-08-breakout-world-fixed-events.md).
Active/completed phase root `/mnt/data/kindle-breakout-world-fixed-events-20261108/runner-repair`; parent retains native qualification and the failed first launcher.
Native/source `249786f13fc96fd8b624cc93e11900ffe72a4ab7`, SHA256 `ab645a5b68e013b0ffd5a199599137ee6f5b4c9f9b624e6c863ef2f9295bbc15`; repaired Python runner `84de0f5ad84c587bb93bcd75a3014749df2a1745`. The repair changes Python validation only; no native rebuild or second native qualification.

## Intervention and scope

Add future reward/continuation supervision to the two existing recurrent rollout draws of the original fixed-filter phase. Each action advances h, samples ordinary prior z, and reads existing heads from causal h/z. Average both draws and full BxH using unchanged stopped-target two-hot reward CE and soft-label continuation BCE, both scales1. Retain terminal arrivals; mask the first future reset and later rows. Reuse the recurrent mask without duplicate input names. Only the final prior samples are newly needed for these event readouts; preceding recurrent transition paths stay. Future labels/masks never enter predicted states.

New explicit `motion_fixed_teacher_events` and `start_fixed_teacher_events_phase`; v16 binds objective/scope and reports16 values:original14 plus future reward/continuation. Ordinary construction rejects before GPU. Preserve defaults and old paths. No new parameters,RGB suffix loss,pretrained encoder,object/RAM training target,teacher policy,collector,actor/critic or backend primitive. Surrogate categorical ST derivatives remain biased.

Use the same retained H0 final8192 as the original phase, not its trained12288 final. Preserve its204 weights/moments and optimizer age. Keep an independently allocated immutable H0 observation filter and36 fixed student visual weights;32 dynamics/head weights learn. Decoder/observed coordinates stay fixed while future dynamics change. Teacher h/z from future observations are stopped training targets only. Predicted suffixes start from the observed prefix and receive no true future correction. Filter target draws use independent declared seeds at each optimizer age. Frozen means fixed functions, not an unchanged random draw.

Student68 logical/8,852,675 scalars/204 tensors plus extra filter39/6,203,136 scalars (24,812,544 parameter bytes);student is not the total footprint. Reported filter session allocation35,799,168 bytes excludes driver/staging and is not peak memory. The embedded106,261,452-byte H0 anchor includes full weights/moments. Target readbacks and extra forward work are included in phase cost.

## Training and retained initial evidence

Warm phase8192→12288,4,096 new updates,RGB64/12M/B8/T32/H16/full BPTT/Auto,seed41001,rates4e-4,mixing.01. Keep base RGB100,KL1/.1/free-nat1,latent overshooting1,recurrent energy100 and all event scales1. Same first4096 rows of the original4/2/1/1 plan,16 fitting/eight development streams. Reuse sealed counts:32,850 available/32,833 sampled unique arrivals,5,419 observed terminals,4,755 suffix terminals,17,425 observed positive-reward rows,37,878 context resets,5,227 source resets and488,356 valid suffix rows. No new environment actions/frames or data collection.

The first launcher completed initial/dense predictions and then failed a Python assertion after one native update returned. No report or advanced checkpoint was saved. Preserve that discarded update and the failed guard/observer; its exact native duration/loss values are unmeasured. Two new actual CPU training-loop cases reproduce the validity-at10 mistake; the new layout stores validity at13. The Python-only membership repair applies the existing fixed-filter validity/energy/weighted-total checks to the new objective. Four fresh/warm legacy/new cases pass.

The separately declared repair starts again from H0 with fresh phase RNG,4096 updates. Copy/hash the completed initial/dense arrays and initial-dense-frozen checkpoint; skip those queries. All204 fresh initial tensors match H0. First16 logged reports exactly match native qualification. This is not an uninterrupted continuation or an independent seed. Keep the failed original guard/launcher failed. The one discarded update remains spent cost.

Keep initial/intermediate10240/final and every periodic draw,384 matched dense origins,all16 reports and exact final restore. Only fixed final12288 is assessed; no best checkpoint/draw. Reused seed/families are development, not untouched generalization or fresh confirmation.

## Recorded ball precision

Within one resized RGB64 pixel, first fixed draw; missing balls fail and visible support varies. Related origins are not independent episodes.

| Split/model | Observed | 1 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| Development: Retained H0 |108/112|94/109|33/109|14/107|3/114|0/95|
| Development: Fixed filter |108/112|98/109|43/109|17/107|10/114|4/95|
| Development: Joint suffix events |98/112|97/109|58/109|31/107|13/114|6/95|
| Development: Fixed filter + future events |108/112|83/109|45/109|22/107|6/114|1/95|
| Fitting: Retained H0 |159/163|153/177|73/180|34/176|14/171|5/179|
| Fitting: Fixed filter |159/163|156/177|82/180|53/176|40/171|20/179|
| Fitting: Joint suffix events |140/163|148/177|99/180|71/176|30/171|20/179|
| Fitting: Fixed filter + future events |159/163|155/177|85/180|44/176|27/171|16/179|

Observed>=90%,one>=89/109,eight>=28/107 gates stay. Observed RGB/encoder/posterior outputs stay exact because the full observation filter/visual weights are fixed; this does not make imagined dynamics accurate or prove information absent. All24 fixed-filter visual arrays,108 initial cohort arrays,384 dense origins and20 source-field identities pass. Fitting-stream forecast errors remain explicit.

## Frozen action-conditioned forecasts

84 separately declared new-final-only groups,history16 primary/32 secondary,eight draws,endpoints1/4/8/16/32. Reuse cached oracle,H0,original fixed-filter and joint suffix-event arrays; no old-model query or collector repeat. Only past images/actions and planned commands enter forecasts. The primary is category-balanced survival-weighted16/32 ball-feature energy, lower better, unchanged threshold.12712666431732942. It scores a finite decoded-feature mixture, not full image likelihood or certified population futures.

| Model | Primary energy | Freeflight8 NOOP/LEFT/RIGHT/REVERSE, each of64 | Ending16 Brier | Ending32 Brier | RIGHT16 ending; oracle0 | Nonending>5%/304;maximum |
|---|---:|---|---:|---:|---:|---|
| Retained H0 |0.169502219|64/63/63/61|0.003345|0.144566|0.110494|0/304;0.019069|
| Fixed filter |0.141405433|63/62/48/64|0.000253|0.334714|0.024970|0/304;0.032650|
| Joint suffix events |0.283365170|64/64/56/64|0.032212|0.226666|0.205177|239/304;0.999482|
| Fixed filter + future events |0.166163803|61/62/60/62|0.094185|0.120330|0.438322|0/304;0.035056|

Primary improvement versus H0:1.970% (required25%).

Each freeflight command has the same oracle ball path through8; require>=90% model presence on every command. Point/detection metrics omit survival weighting and cannot replace the primary statistic. Sticky actions make a single future picture insufficient. Unsupported/above-board endpoints remain explicit; the invalid FIRE pair gate is not reused.

| Intermediate gate | Result |
|---|---|
| observed_ball |pass|
| one_step_ball |fail|
| eight_step_ball |fail|
| serve_noop |pass|
| ending_brier |fail|
| ending16_brier |fail|
| ending16_right |fail|
| nonending_false_alarms |pass|
| mean_ball_energy_reduction |fail|
| freeflight_each_plan_presence_at8 |pass|

`supplementary-metrics.json` retains every individual nonending false alarm>5% and every per-action/horizon cell. Nonending gates use category means; individual alarms remain visible. Regressed ball/paddle/reward-Brier/ending-Brier/alive-only-brick-Brier cells:

- Versus Retained H0:81 of210/101 of210/110 of210/79 of210/18 of209.
- Versus Fixed filter:111 of210/97 of210/170 of210/59 of210/22 of209.
- Versus Joint suffix events:72 of210/40 of210/3 of210/30 of210/42 of209.

Related cells are not independent tests, significance evidence or seed reliability. The original/new fixed-filter phases use the same anchor,budget,config and data order; new event losses and extra final prior draws alter gradients,optimizer history and later rollout noise. Joint suffix events also have different architecture/objectives/initialization/budget. No isolated error cause or broad recipe reliability is established.

## Correctness, cost and preserved failures

Native qualification1241 pins/33.158s passes first native execution:204 inherited tensors,18 cached initial arrays,original13 raw components,16 finite reports/new total,32 changed/36 fixed weights,18 exact trained restore arrays,filter/anchor/GPU/prefix/lane invariance,new target/mask/lane controls and old-native v16 rejection. A separate saved-report CPU check verifies all original14 within declared rounding tolerance and exact sequential-F32 old total plus new head terms. Rounded scalar totals alone do not prove small-head gradients; CPU graph connectivity and new native causal controls are separate evidence.

Learning1377 pins/123 CPU inputs passes4,096 finite16-report rows,energy/weighted-total identities,all32 changed active weights/36 exact visual weights,full plan/exposures,204 own initial/frozen tensors,34 finite exports and18 exact final restore arrays. Teacher digest/GPU readbacks stay exact at every saved checkpoint. All72 frozen visual optimizer moments change as allowed; their weights remain exact. Checkpoints omit live belief/replay/games/full RNG; no uninterrupted-life claim.

117 distinct affected Rust CPU/224 original packaged Python checks and strict core/test/binding Clippy pass. Four Python repair training-loop checks overlap older cases and are not an independent added total. Preserve the original Rust fixture formatting failure and first CPU116-pass/one-failure target-collection omission with separate production-gate repair/targeted pass. No116-test or old native-primitive rerun. Native/source249786f1 is unchanged by Python repair84de0f5a.

| Timing scope | H0 source | Original fixed phase | New repaired phase |
|---|---:|---:|---:|
| Native learning calls |529.515s|388.861s|395.123s|
| Training/evaluation loop |796.900s|459.954s|457.361s|

Known source+new native calls924.637s, plus one discarded unsaved update with unmeasured native duration. Original failed worker/polling span34.725s includes completed initial queries and that update; do not treat it as native time or omit it. New loop excludes construction,initial-output copies,pinning and the completed initial queries reused from the first attempt. Old/new loop scopes differ in initial evaluation; no isolated speed claim. Qualification,builds,restore,assessment and CPU work are separate.
Frozen assessment140.149s;learning CPU audit9.126s;branch CPU comparison49.393s. These single runs share a host with other work and are not kernel benchmarks.

Four successful guards/36 sealed files cover qualification,repaired training,restore and assessment. The original failed training guard stays failed;27 sealed files/no unsealed files pass integrity audit. No automatic recovery,reset,NVML or unfinished child. Retain all failure receipts/reports/checkpoints; no native requalification or repeated initial assessment repairs the launcher. Release836,276,083 counted build-cache bytes after packaging;source,wheel,runtime and native test executable remain.

Three-model figures and three lossless causal APNGs use saved arrays at fixed lane0/draw0,not favorable matches. Native oracle160x210 and decoded RGB64 share display aspect only. No input/metric preprocessing or new model/emulator query. Learning curves show mean/min–max over four draws,not confidence intervals;H0 is displayed unchanged with no extra training.

Full precise physics,event timing/calibration,persistent removed-brick memory,late/tunnel/second-wall coverage,untouched families and fresh confirmation remain required. Existing oracle has no above-top-brick endpoints and its detector supports a restricted region. Intermediate gates cannot complete the world-model goal. No unchanged extension,sweep or automatic successor is selected.
