# Breakout: local action-conditioned image prediction

The fresh experiment is complete and **fails retention**. Full world-model competence remains unproven. Keep H0 as the inspectable reference. Actor/critic and CDP comparisons remain parked. No unchanged extension, sweep or automatic successor is selected.

[Protocol](../experiments/2026-11-08-breakout-world-local-action.md).
Root `/mnt/data/kindle-breakout-world-local-action-20261108`.
Source/native/runner `7f797a2babebec97edf4a801e4c92e590283ad2e`; native SHA256 `38680c739929c82c8e1e5a4f15bfa1277ba32170e9c7d14034c2d9c3f75ba323`.

## Recorded precision

Ball centers within one RGB64 pixel, first fixed draw, with varying visible-ball support. Missing balls fail. Related origins are not independent episodes. Actual-initial rows use the new architecture before training.

| Split/model | Observed | 1 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| Development: Local-action actual initial |0/112|0/109|0/109|0/107|0/114|0/95|
| Development: Retained H0 |108/112|94/109|33/109|14/107|3/114|0/95|
| Development: Fusion H16 |93/112|95/109|62/109|34/107|15/114|4/95|
| Development: Suffix events H16 |98/112|97/109|58/109|31/107|13/114|6/95|
| Development: Pixel transport H16 |94/112|102/109|50/109|31/107|16/114|1/95|
| Development: Local action H16 |98/112|101/109|63/109|43/107|29/114|11/95|
| Fitting: Local-action actual initial |0/163|0/177|0/180|0/176|0/171|0/179|
| Fitting: Retained H0 |159/163|153/177|73/180|34/176|14/171|5/179|
| Fitting: Fusion H16 |141/163|150/177|96/180|72/176|38/171|23/179|
| Fitting: Suffix events H16 |140/163|148/177|99/180|71/176|30/171|20/179|
| Fitting: Pixel transport H16 |142/163|145/177|93/180|54/176|23/171|19/179|
| Fitting: Local action H16 |142/163|150/177|105/180|79/176|43/171|30/179|

Fitting eight-step precision is 79/176. Fitting-stream error cannot be explained by held-out generalization alone. These metrics do not isolate a representation, decoder, dynamics, optimizer or sampling cause.

## Fixed-action forecasts and events

Only the new final checkpoint receives84 frozen groups: history16 primary/32 secondary, eight fixed draws and1/4/8/16/32 endpoints. Cached oracle and earlier model arrays are reused, without old-model queries or new collection. The primary category-balanced, survival-weighted16/32 ball-feature energy scores a finite decoded-feature mixture. It is not full image likelihood or natural-frequency calibration. The unchanged threshold is.12712666431732942.

| Model | Primary energy | Freeflight8 NOOP/LEFT/RIGHT/REVERSE,each /64 | Ending16 /32 Brier | RIGHT16 risk;oracle0 | Nonending >5% /304;maximum |
|---|---:|---|---|---:|---|
| Retained H0 |0.169502219|64/63/63/61|0.003345/0.144566|0.110494|0/304;0.019069|
| Fusion H16 |0.345425368|64/64/56/64|0.077296/0.483186|0.064732|276/304;0.990268|
| Suffix events H16 |0.283365170|64/64/56/64|0.032212/0.226666|0.205177|239/304;0.999482|
| Pixel transport H16 |0.200277257|64/64/64/64|0.017587/0.337100|0.068080|183/304;0.558192|
| Local action H16 |0.240565683|64/64/64/64|0.094168/0.208852|0.507002|244/304;0.982117|

Primary error is 41.925% higher than H0; the required result is25% lower. Every command shares the same oracle freeflight ball paths through8, but each command must independently retain>=90% presence. Presence is not position accuracy. Related-pair precision omits survival weighting and cannot replace the primary score. The invalid FIRE pair gate stays retired.

Recorded and fixed-action ball-position gains coexist with paddle/reward/ending regressions. The primary score also includes predicted survival; the point metrics omit it. These different scores cannot isolate a causal mechanism. Structurally, event heads read only RSSM state while image predictions also use local image history. Image/event coherence is therefore an investigation target, not an established explanation or a replacement retention gate.

On common supported related model/oracle pairs, all-pair ball-position accuracy at1/4/8/16/32 is 70.66/45.08/22.45/20.22/5.90% for H0 and 70.23/66.71/55.11/73.22/17.77% for local action. These pairs share origins and omit survival weighting; they are not independent episodes, calibrated distribution scores or substitute gates.

| Intermediate gate | Result |
|---|---|
| observed_ball |fail|
| one_step_ball |pass|
| eight_step_ball |pass|
| serve_noop |pass|
| ending_brier |fail|
| ending16_brier |fail|
| ending16_right |fail|
| nonending_false_alarms |fail|
| mean_ball_energy_reduction |fail|
| freeflight_each_plan_presence_at8 |pass|

`supplementary-metrics.json` preserves every action/horizon cell and every individual nonending alarm>5%. Category means define the nonending gate. Regressed cells, ball/paddle/reward Brier/ending Brier/alive-only brick Brier:

- Versus Retained H0: 84 of210/194 of210/201 of210/195 of210/19 of209.
- Versus Categorical H16: 102 of210/154 of210/197 of210/192 of210/12 of209.
- Versus Fusion H16: 82 of210/120 of210/125 of210/140 of210/18 of209.
- Versus Suffix events H16: 85 of210/96 of210/119 of210/139 of210/23 of209.
- Versus Pixel transport H16: 93 of210/137 of210/108 of210/182 of210/20 of209.

Related cells are not independent significance tests or seed reliability. Full raw per-origin probabilities and both histories remain retained.

## Intervention and training

Default-off `rgb_spatial_action` appends incoming one-hot action planes after the six channels of two earlier RGB images. Each action scalar is repeated spatially. The three motion ConvNorm layers widen8→16; fusion input widens and its output stays16. The rest of the decoder, categorical RSSM, CNN encoder and RSSM-only reward/continuation heads stay. There is no transport, residual logit path or generated-image re-encoding.

Observed reconstruction receives the masked incoming action for that arrival and only two earlier images. Reset action planes are zero. Each imagined step receives its submitted transition action and shifts its generated RGB into history, with full BPTT. Future RGB/events/reset labels are stopped targets or masks only; they never correct predictions. Future actions cannot affect earlier outputs. Local conditioning, capacity, initialization, work and gradients change together; this is not an isolated action-path or global-state causal diagnosis.

Adds21,248 scalars relative to v13:80 logical/8,893,203 scalars/240 checkpoint tensors. Ten parameter shapes change;70 weights/210 common initial tensors match cached v13 exactly. Feature2560 plus24,576 image-history scalars stay. Feature exports omit image history, so they are not the full world belief. The16 initial nonimage arrays and ten non-RGB initial report components match cached v13 exactly. First logged training components match all13 qualifier components. No old model is queried again.

v18 binds the new flag, objective, scope and geometry; defaults and v1-v17 paths remain. Old native rejects before GPU allocation. No backend primitive, pretrained encoder, object/RAM training label, teacher, actor/critic change or new collection occurs.

Fresh8,192 updates from0; RGB64/12M/B8/T32/H16/full BPTT/Auto,seed41001,rates4e-4,mixing.01. Keep observed/future motion RGB100,KL1/.1/free-nat1 and all observed/future event scales1. Thirteen reports and total-loss identity stay. Use the original16 fitting/eight development streams and sealed full4/2/1/1 plan, without recounting:975,941 valid suffix targets,9,570 suffix terminals,16,008 future positive rewards and32,849 unique of32,850 available arrivals. No new environment actions or frames. Reused data/seed are development, not untouched or fresh confirmation.

## Validation and costs

First CPU runs pass140 affected Rust checks;273 isolated packaged Python checks and strict core/test/binding Clippy pass. Native qualification passes 1250 pins/31.256s:fixed-state/history action response, F64/native values and gradients, lane/reset controls, future target/mask causality, one full-shape finite13-component update with correct masks/total,240 finite/frozen tensors and18 initial/trained restore arrays. The selected native fixtures have zero optimizer updates. Existing unchanged primitive receipts are reused.

Learning review passes 1257 input/170 CPU pins,8,192 finite thirteen-component rows,all80 changed weights,full plan/exposures,240 exact own initial/dense-frozen/final-frozen tensors,384 dense actual-initial/final origins,58 finite exports and18 exact final restore arrays. Initial/2048/4096/final and all periodic draws remain;only the fixed final receives the frozen assessment. Checkpoints omit replay,live belief/history and full RNG;no restored extension is selected.

Native learning calls:1389.476s;train/evaluation loop:1568.062s. Loop excludes construction,dense-initial queries and pinning. Frozen assessment:200.553s;CPU learning review:8.609s;CPU branch comparison:165.306s. Qualification,restore,build/package and display costs are separate. H0 used529.515s native/796.900s loop;v13 used1166.306s/1352.208s;transport1562.699s/1734.273s. These single runs are not isolated throughput claims;other host GPU work may coexist.

Four successful guards/36 sealed files pass. No native failure/retry, reset,recovery,NVML or unfinished child. Disposable 836,613,824 counted build-cache bytes are released;source,wheel,runtime,executable,checkpoints and arrays remain. Earlier protocol failures and separate repairs stay in their linked reports and roots.

`comparison.png`, `learning-curves.png` and three full causal APNGs use retained arrays only. Clips fix lane0/oracle0/model0, rather than favorable matches. Native oracle160x210 and RGB64 share display aspect only; clip frames decode losslessly and analysis versions are retained. No model-input/metric preprocessing changes or new queries.

Precise physics,event timing/calibration,persistent removed-brick memory,late/tunnel/second-wall coverage,untouched families and fresh confirmation remain required. Existing oracle has no above-top-brick endpoint and its detector supports a restricted region. Intermediate gates cannot complete the full goal. No unchanged extension,sweep,completed-stage repeat or automatic successor is selected.

Worker/polling spans (including that worker’s pin checks, construction and cleanup; not measured GPU-active time): qualification 31.684s, train 1588.188s, restore 11.254s, assessment 291.083s. Queue-declaration hashing, source build, CPU reviews and display work are separate. The native-call and loop timings above must not be presented as the full end-to-end experiment cost.
