# Breakout: observed and future RGB loss weight 1

The fresh 8,192-update experiment **fails retention**. Full world-model competence remains unproven. Keep H0 as the inspectable reference. No unchanged extension, sweep or automatic successor is selected. Actor/critic and CDP comparisons remain parked.

[Protocol](../experiments/2026-11-08-breakout-world-rgb1.md).
Root `/mnt/data/kindle-breakout-world-rgb1-20261108`.
Reused native/source/runner `a8efffc6c7b13cf98616199f336e5f985a44a912`, native SHA256 `f40555b56883aa804300ab702fb2b1f23e62c240abbb73f42c6e3e98c196a943`.

## What changed

The preceding fixed-weight diagnostic found future-RGB gradients 22–87 times KL in the prior and 336–884 times KL in the core, with frequent observed/future image-gradient opposition in the encoder. This local evidence justified testing one different weight; it did not prove a physical cause or useful coefficient.

Change only `loss_scales.reconstruction` from100 to1 relative to v13 suffix-event training. This changes observed and imagined RGB cross-entropy together. Keep configured KL dynamics1/representation.1/free-nat1 and observed/future reward and continuation1. No architecture, representation, loss formula, normalization, teacher, label, data or backend change. Rescaling the total loss is not evidence of better physics.

Fresh seed41001,8,192 updates from0,RGB64/12M/B8/T32/H16/full BPTT/Auto,rates4e-4,mixing.01; fusion on, image feedback and residual off.80 logical weights/8,871,955 scalars/240 tensors. Same exact v13 native/runner without a new build or repeated native derivative fixture. A fresh optimizer history develops under the new relative gradients; fixed-checkpoint sensitivity does not predict it exactly.

Use the original sixteen fitting/eight development streams and complete4/2/1/1 plan. Reuse the sealed plan/exposure receipt rather than recollect/recount:2,097,152 replayed arrivals,32,849 unique among32,850 available;10,909 observed terminal rows,34,440 observed positive-reward rows,75,808 context resets,10,488 source resets;975,941 valid suffix rows/9,570 terminals/16,008 positive rewards. No new environment actions or emulator frames. Reused development seed/families are not held-out confirmation. Keep initial/2048/4096/final weights and every periodic draw; only the fixed final is assessed.

## Raw loss tradeoff

Over the fixed final1,024 updates7169–8192, mean raw posterior/prior KL falls25.759→1.477. Mean observed RGB CE changes2800.701→2814.295, and future RGB CE2664.432→2670.997. Event fitting losses are retained in loss-tradeoff.json. All8,192 saved location rows match; no new model query. Different learned states/samples/optimizer histories remain. Lower divergence and nearly unchanged aggregate image loss do not certify physical precision or prove mutual information, information absence, posterior collapse or a causal error mechanism.

## Recorded ball precision

Within one resized RGB64 pixel, first fixed draw; missing balls fail and support varies by horizon. Related origins are not independent episodes.

| Split/model | Observed | 1 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| Development: RGB1 actual initial |0/112|0/109|0/109|0/107|0/114|0/95|
| Development: Retained H0 |108/112|94/109|33/109|14/107|3/114|0/95|
| Development: RGB100 suffix events |98/112|97/109|58/109|31/107|13/114|6/95|
| Development: RGB1 suffix events |91/112|85/109|48/109|25/107|9/114|6/95|
| Fitting: RGB1 actual initial |0/163|0/177|0/180|0/176|0/171|0/179|
| Fitting: Retained H0 |159/163|153/177|73/180|34/176|14/171|5/179|
| Fitting: RGB100 suffix events |140/163|148/177|99/180|71/176|30/171|20/179|
| Fitting: RGB1 suffix events |134/163|135/177|79/180|40/176|18/171|4/179|

Fitting-stream eight-action accuracy also falls71/176→40/176. The measured deficit is not confined to development generalization; this does not isolate why fitting is weak.

All original gates stay: observed>=90%, one>=89/109, eight>=28/107. Accurate observed reconstruction cannot replace open-loop physics. 384 dense actual-initial/final origins and all source-array identities stay; every snapshot/draw is retained.

## Frozen action-conditioned forecasts

84 separately declared final-only frozen groups; history16 primary/32 secondary,eight draws,endpoints1/4/8/16/32. Feed past images/actions and future commands without true future correction. Reuse cached oracle/H0/motion/fusion/v13 outputs; no old model or emulator query. The primary category-balanced survival-weighted16/32 ball-feature energy scores a finite mixture of decoded observables, not full image likelihood or certified future probabilities. Lower is better; unchanged threshold.12712666431732942.

| Model | Primary energy | Freeflight8 NOOP/LEFT/RIGHT/REVERSE, each of64 | Ending16 Brier | Ending32 Brier | RIGHT16 ending; oracle0 | Nonending >5%/304; maximum |
|---|---:|---|---:|---:|---:|---|
| Retained H0 |0.169502219|64/63/63/61|0.003345|0.144566|0.110494|0/304;0.019069|
| Categorical H16 |0.311976907|64/60/41/51|0.002422|0.189225|0.082001|23/304;0.438399|
| Fusion H16 |0.345425368|64/64/56/64|0.077296|0.483186|0.064732|276/304;0.990268|
| RGB100 suffix events |0.283365170|64/64/56/64|0.032212|0.226666|0.205177|239/304;0.999482|
| RGB1 suffix events |0.477756727|64/61/24/64|0.044787|0.219824|0.273831|175/304;0.988374|

Primary improvement versus H0: -181.859% (required25%).

Every freeflight command has the same oracle ball path through8; each must retain>=90% ball presence. Pair/detection metrics omit survival weighting and cannot replace the primary score. Sticky actions make one future picture insufficient. Keep all missing-ball failures, unsupported endpoints and event false alarms; the invalid FIRE pair gate remains invalid.

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

`supplementary-metrics.json` retains every individual nonending false alarm>5% and all per-action/horizon reward/paddle/brick regressions. Nonending gates use category means; individual alarms remain explicit. Regressed ball/paddle/reward-Brier/ending-Brier/alive-only-brick-Brier cells:

- Versus Retained H0: 146 of210/189 of210/205 of210/165 of210/38 of209.
- Versus Categorical H16: 146 of210/145 of210/197 of210/170 of210/27 of209.
- Versus Fusion H16: 115 of210/96 of210/130 of210/70 of210/23 of209.
- Versus RGB100 suffix events: 131 of210/79 of210/143 of210/78 of210/39 of209.

These related reused cells are not independent tests, significance evidence or seed reliability. This compares one changed scalar while its full training trajectory differs; gradient dominance/conflict does not become a causal physical-error attribution.

## Verification and cost

One-update preflight passes143 pins/29.604s:240 exact cached v13 initial tensors,18 exact initial prediction arrays,all12 raw reports identical to cached v13,the correct new total,77 changed weights including every required family,and18 exact arrays each for initial/trained restore. Frozen tensors and all13 finite reports pass. Reuse the original causal/derivative/package receipts unchanged; no old model query, new build or repeated primitive qualification. The cached100-weight total1,668,465.25 becomes16,707.16796875 with identical raw reports; that arithmetic change itself is not learning.

Fresh learning passes153 pins/170 CPU inputs:8,192 finite13-report rows,configured weighted-total identity,all80 weights changed,full plan/exposures,240 exact own initial/frozen tensors,384 dense origins,58 finite exported files and18 exact final restore arrays. Checkpoints omit live belief/replay/games/full RNG. No warm extension is selected.

Native training calls1155.480s; measured training/evaluation loop1329.204s. Native construction,initial-dense inference,pinning,preflight,restore and assessment are separate. Frozen assessment200.933s; learning CPU audit8.354s; branch CPU comparison68.814s. These are single runs on a host with other work, not isolated kernel-speed comparisons. RGB100 native/loop1166.306/1352.208s; H0 529.515/796.900s are retained reference costs.

Four successful host guards/36 sealed files; no native retry, recovery, NVML or unfinished child. No new build cache was created. Runtime/runner/source receipts, initial/useful checkpoints, all raw reports/arrays and fixed causal clips remain retained. Original v13 and gradient-diagnostic failures remain preserved at their source roots; this setting-only experiment does not relabel or repair those stages.

Three-model comparison/learning figures and three causal APNGs use fixed lane0/draw0 from retained arrays, never favorable matches. The oracle160x210 and predicted RGB64 share display aspect only. Lossless clip checks pass. No model-input or metric preprocessing changes. Curves show mean/min–max of four draws, not confidence intervals.

Full precise all-horizon physics,event timing/calibration,persistent removed-brick memory,late/tunnel/second-wall coverage,untouched families and fresh confirmation remain required. The existing oracle has no above-top-brick endpoint and its detector supports a restricted region. Intermediate gates cannot prove the whole world model. Goal stays active and unproven; no unchanged extension,sweep or automatic successor.
