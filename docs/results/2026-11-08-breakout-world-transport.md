# Breakout: learned pixel transport with an RGB fallback

The fresh experiment is complete and **fails retention**. Full world-model competence remains unproven. Retain H0 as the inspectable reference. Actor/critic and CDP comparisons remain parked; no unchanged extension,sweep or automatic successor is selected.

[Protocol](../experiments/2026-11-08-breakout-world-transport.md).
Root `/mnt/data/kindle-breakout-world-transport-20261108`.
Source/native/runner `80dc8c761ba099fa162072f076d845aee25718d6`; native SHA256 `013ae7a7153f78d373e4e880fb4d4da32184f86e5b2cfcd301a30a39a8a51740`.

## Recorded precision

Within one RGB64 pixel,first fixed draw,with varying visible-ball support. Missing balls fail. Related reused development origins are not independent episodes. Actual-initial rows use the new decoder before learning.

| Split/model | Observed | 1 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| Development: Transport actual initial |0/112|0/109|0/109|0/107|0/114|0/95|
| Development: Retained H0 |108/112|94/109|33/109|14/107|3/114|0/95|
| Development: Fusion H16 |93/112|95/109|62/109|34/107|15/114|4/95|
| Development: Suffix events H16 |98/112|97/109|58/109|31/107|13/114|6/95|
| Development: Pixel transport H16 |94/112|102/109|50/109|31/107|16/114|1/95|
| Fitting: Transport actual initial |0/163|0/177|0/180|0/176|0/171|0/179|
| Fitting: Retained H0 |159/163|153/177|73/180|34/176|14/171|5/179|
| Fitting: Fusion H16 |141/163|150/177|96/180|72/176|38/171|23/179|
| Fitting: Suffix events H16 |140/163|148/177|99/180|71/176|30/171|20/179|
| Fitting: Pixel transport H16 |142/163|145/177|93/180|54/176|23/171|19/179|

Fitting eight-step precision is 54/176. Fitting-stream error is not explained by held-out generalization alone; this does not isolate a representation,decoder,dynamics,optimizer or sampling cause.

## Fixed-action forecasts and events

Only the new final checkpoint receives84 frozen groups,history16 primary/32 secondary,eight fixed draws,all1/4/8/16/32 endpoints. Cached oracle,H0,motion/fusion/suffix-event arrays are reused without old-model queries or new collection. The primary category-balanced survival-weighted16/32 ball-feature energy scores a finite decoded-feature mixture; it is not full image likelihood or a population calibration proof. Threshold stays.12712666431732942.

| Model | Primary energy | Freeflight8 NOOP/LEFT/RIGHT/REVERSE,each /64 | Ending16 /32 Brier | RIGHT16 risk;oracle0 | Nonending >5% /304;maximum |
|---|---:|---|---|---:|---|
| Retained H0 |0.169502219|64/63/63/61|0.003345/0.144566|0.110494|0/304;0.019069|
| Fusion H16 |0.345425368|64/64/56/64|0.077296/0.483186|0.064732|276/304;0.990268|
| Suffix events H16 |0.283365170|64/64/56/64|0.032212/0.226666|0.205177|239/304;0.999482|
| Pixel transport H16 |0.200277257|64/64/64/64|0.017587/0.337100|0.068080|183/304;0.558192|

Primary error is18.156% higher than H0;the required result is25% lower. Every command shares oracle freeflight ball paths through8,but needs its own>=90% presence. Pair accuracy is support-conditioned and omits survival weighting; it cannot replace the primary score. The invalid FIRE pair gate stays retired.

Transport improves primary energy over fusion and suffix events,and restores freeflight ball presence for every command. It still trails H0. Compared with suffix events,ending32 Brier worsens.226666→.337100. Keeping a ball visible is not precise physics,and short recorded gains do not establish correct long distributions.

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
| freeflight_each_plan_presence_at8 |pass|

`supplementary-metrics.json` retains every action/horizon cell,individual false alarm>5% and all regressions. Category means define the nonending gate; individual alarms remain explicit. Regressed cells,ball/paddle/reward Brier/ending Brier/alive-only brick Brier:

- Versus Retained H0: 116 of210/164 of210/204 of210/126 of210/24 of209.
- Versus motion_rollout: 116 of210/134 of210/197 of210/129 of210/14 of209.
- Versus Fusion H16: 84 of210/90 of210/112 of210/44 of210/14 of209.
- Versus Suffix events H16: 90 of210/72 of210/116 of210/29 of210/28 of209.

These related cells are not independent significance tests or seed reliability.

## Intervention,training and validation

Default-off `rgb_spatial_transport` adds a1x1 head on the existing16-channel fusion map:17 horizontal and17 vertical offset logits,plus one blend logit. Offsets−8..8 have per-pixel softmax weights shared across RGB. Zero-padded horizontal mixing then vertical mixing transports the latest causal image; sigmoid blending uses the ordinary RGB head as fallback. Convert probabilities to logits after clipping1e−6..1−1e−6. The separable order is explicit; this is not a joint289-offset distribution. Generated images shift history with full BPTT,but never re-encode or correct RSSM belief.

Adds595 weights:82 logical/8,872,550 scalars/246 checkpoint tensors. Feature2560 and two-image history24,576 stay. Keep categorical RSSM and RSSM-only reward/continuation heads. New flag/objective/scope bind v17; defaults and v1-v16 remain. No backend primitive,pretrained encoder,object/RAM training labels,teacher,actor or critic change. The ordinary named initializer adds no flow/blend bonuses.

Fresh8,192 updates from0. RGB64/12M/B8/T32/H16/full BPTT/Auto,seed41001,rates4e-4,mixing.01,observed/future motion RGB100,KL1/.1/free-nat1,observed/future event scales1. Thirteen reports and total-loss identity stay. Reuse the original sixteen fitting/eight development streams and full4/2/1/1 plan with sealed exposures; no new recount or environment action/frame.975,941 valid suffix targets,9,570 future terminals,16,008 future positive rewards and32,849 unique of32,850 available arrivals. Data/seed reuse is development,not untouched or fresh confirmation.

Both new parameters and all required families change in the full-shape preflight. All240 common v13 initial tensors and16 nonimage arrays match cached originals exactly; initial non-RGB report indices2/3/4/5/6/7/8/10/11/12 also match exactly. RGB changes at initialization are intended. No old model is queried again. Both new parameters retain ordinary named initialization; decoder composition and gradients change together,so this is not isolated causal attribution.

122 distinct affected Rust CPU checks,249 isolated packaged Python checks and strict core/test/binding Clippy pass. Native composition qualification:1247 pins/115.674s. F64/native values/derivatives,boundaries,lane isolation,finite extremes and zero optimizer age pass. One full-shape update has13 finite reports/correct masks/total,246 finite/frozen tensors,and18 exact initial/trained restore arrays. Old native rejects v17 before GPU. Existing unchanged primitive receipts are reused.

Learning review:1254 input/170 CPU pins,all8,192 finite13-report rows,all82 changed weights,full plan/exposures,246 exact own initial/dense-frozen/final-frozen tensors,384 dense actual-initial/final origins,58 finite exports and18 exact final restore arrays pass. Initial/2048/4096/final and all periodic draws remain; no best checkpoint replaces final. Checkpoints omit replay,live belief/history and full RNG. Four successful host guards/36 sealed files pass,with no reset,recovery,NVML or unfinished child.

Preserve the first CPU fixture compilation failure:its parameter vector inferred unsized str/[f32] types. `cpu-fixture-repair.json` records the explicit Vec<(String,Vec<f32>)> repair;production transport math is unchanged. Corrected3 transport plus119 world CPU checks pass;37 native tests remain ignored in those filters.

Preserve the invalid first2048 preview:the new display script treated uint8 truth/context arrays as centered floats. `preview-failure-review.json` marks that artifact invalid for visual comparison. A separate CPU-only `preview-repair/` corrects truth display and preserves original arrays,training,source metrics and first artifact. Its fixed first origin/draw is a display diagnostic,not an accuracy estimate. No model or environment query is repeated.

Native training calls:1562.699s;train/evaluation loop:1734.273s. Loop excludes construction,dense-initial queries and pinning. Frozen assessment:220.168s;CPU learning audit:9.057s;CPU branch comparison:74.807s. Qualification,restore,build/package and display costs remain separate. Retained H0 used529.515s native/796.900s loop;suffix events1166.306s/1352.208s. These single runs are not isolated throughput claims;other host GPU work may coexist. No inherited training cost or new environment interactions are hidden. Disposable 836,444,204 counted build-cache bytes released;source,wheel,runtime,executable,weights and arrays remain.

`comparison.png` and `learning-curves.png` show fixed metrics/draw ranges. Three full causal APNGs use fixed lane0/oracle0/model0,not favorable matches. Native oracle160x210 and RGB64 share display aspect only;clips are decoded losslessly and analysis versions retained. No input/metric preprocessing changes.

Precise physics,event timing/calibration,persistent removed-brick memory,late/tunnel/second-wall coverage,untouched families and fresh confirmation remain required. Existing oracle has no above-top-brick endpoints and the detector supports a restricted region. Intermediate gates cannot complete the full goal. No unchanged extension,sweep,completed-stage rerun or automatic successor is selected.
