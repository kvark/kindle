# Breakout: copy the latest image and learn its correction

The fixed experiment is complete and **fails retention**. Full world-model competence remains unproven. Keep H0 as the broad inspectable reference. No further allocation is selected; actor/critic and CDP comparisons stay parked.

[Protocol](../experiments/2026-11-08-breakout-world-pixel-residual.md).
Artifacts: `/mnt/data/kindle-breakout-world-pixel-residual-20261108`.

## Hypothesis and change

The preceding causal-prior learner failed physics and event gates. Its eight-action development predictions miss42/107 supported balls and misplace38 more; brick-cell accuracy99.765% conceals only4/23 newly removed bricks detected. Source review finds no new training/inference alignment mismatch. The spatial decoder rebuilds every pixel through learned features, without a direct persistence path. This motivates a copy path; it does not establish that its absence caused those errors.

Optional `rgb_spatial_residual`, default off, adds the latest causal image log odds to the existing decoder output: `logits = correction + log(p) - log(1-p)`, with `p = clamp(latest_RGB + .5, 1/255, 254/255)`. Zero correction copies within one intensity level. Learn motion, new content and removal through the correction. Gradients pass through generated image history and the copy path; no future target enters predictions. First/reset histories retain the existing centered-zero convention.

Use the v13 suffix-event recipe with posterior observed RGB/reward/continuation and KL, plus H16 free RGB/event losses. The failed v14 main causal readout objective is not carried forward. Same RGB64/12M/B8/T32/full BPTT/Auto, seed41001, rates4e-4, mixing.01, main/future RGB100 and reward/continuation1, KL1/.1/free-nat1.80 logical trainable weights/8,871,955 scalar weights/240 checkpoint tensors including optimizer moments. No new heads, parameters, labels, encoder, collection, actor/critic or backend primitive. No generated-image re-encoding. This changes outputs, initialization, parameter interpretation and gradient routes together; not an isolated memory test.

v15 binds the optional decoder/objective/geometry. Defaults and v1-v14 semantics stay; incompatible headers/objectives fail before GPU. All240 parent actual-initial tensors and16 cached non-image arrays match the v13 parent. RGB differs by construction; no parent-identical RGB claim. These identities do not make reused initialization an independent seed.

## Data and fitting

Fresh8,192 updates from0; original full4/2/1/1 plan,16 fitting/eight development streams. No new environment actions or emulator frames.2,097,152 replay arrivals cover32,849 unique stored slots among32,850 available, with10,909 terminal/34,440 positive-reward arrivals. H16 has975,941 valid suffix targets/9,570 terminal/16,008 positive-reward targets. Counts are exposures, not independent samples. RAM/object annotations select or score examples only, never model inputs or training labels.

Actual initial,2,048,4,096 and final8,192 checkpoints and all periodic draws remain. Dense actual-initial/final evaluation covers384 matched origins. Only fixed final is assessed; no best checkpoint, draw or easier threshold. Seed/families are reused development, not untouched generalization or fresh confirmation.

## Ball prediction

Within one resized RGB64 pixel, first fixed draw; missing predicted balls fail. Observed is posterior reconstruction, numeric horizons are action forecasts with no true future image correction. Support varies and related origins are not independent trials.

| Split/model | Observed | 1 action | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| Development: Own actual initial |0/112|1/109|1/109|0/107|0/114|0/95|
| Development: Retained H0 |108/112|94/109|33/109|14/107|3/114|0/95|
| Development: Fusion H16 |93/112|95/109|62/109|34/107|15/114|4/95|
| Development: Suffix events H16 |98/112|97/109|58/109|31/107|13/114|6/95|
| Development: Copy + correction H16 |92/112|83/109|45/109|22/107|3/114|3/95|
| Fitting: Own actual initial |0/163|0/177|0/180|0/176|0/171|0/179|
| Fitting: Retained H0 |159/163|153/177|73/180|34/176|14/171|5/179|
| Fitting: Fusion H16 |141/163|150/177|96/180|72/176|38/171|23/179|
| Fitting: Suffix events H16 |140/163|148/177|99/180|71/176|30/171|20/179|
| Fitting: Copy + correction H16 |129/163|125/177|72/180|39/176|12/171|7/179|

## Changed objects and paddle

Whole-image or brick-cell accuracy can be dominated by static content. The following are detector-supported development measurements on the same fixed origins, with missing paddles/balls kept in their accuracy denominators. Removed-brick recall is conditional on available measured changes, not a persistent-memory proof.

| Model/horizon | Missing balls / visible | Paddle within1px / visible | Brick-cell accuracy | Newly removed bricks detected / supported |
|---|---:|---:|---:|---:|
| Retained H0 / 8 |8/107|52/126|95.201%|8/23|
| Retained H0 / 32 |18/95|33/118|94.594%|23/97|
| Fusion H16 / 8 |29/107|46/126|99.728%|8/23|
| Fusion H16 / 32 |62/95|31/118|98.635%|14/97|
| Suffix events H16 / 8 |28/107|43/126|99.787%|7/23|
| Suffix events H16 / 32 |60/95|30/118|98.760%|11/97|
| Copy + correction H16 / 8 |49/107|45/126|99.757%|8/23|
| Copy + correction H16 / 32 |71/95|28/118|98.972%|18/97|

## Same-state action interventions and event predictions

Only the new final receives84 frozen groups, histories16 primary/32 secondary, eight draws, horizons1/4/8/16/32. Reuse cached emulator oracle/H0/fusion/suffix-event arrays; no old-model query or collector repeat. Sticky actions permit several valid futures. Primary score is category-balanced survival-weighted ball-feature energy averaged over16/32, lower is better, unchanged threshold.12712666431732942. This finite decoded-feature score is not full-image likelihood or certified population calibration.

| Model | Primary energy | Freeflight8 detections NOOP/LEFT/RIGHT/REVERSE, each /64 | Ending16 Brier | Ending32 Brier | RIGHT16 ending (oracle0) | Non-ending >5%, /304 |
|---|---:|---|---:|---:|---:|---:|
| Retained H0 |0.169502219|64/63/63/61|0.003345|0.144566|0.110494|0|
| Fusion H16 |0.345425368|64/64/56/64|0.077296|0.483186|0.064732|276|
| Suffix events H16 |0.283365170|64/64/56/64|0.032212|0.226666|0.205177|239|
| Copy + correction H16 |0.519581523|57/18/64/64|0.011151|0.349354|0.028120|213|

Primary error versus H0: 206.534% increase; required25% reduction.

Every freeflight plan shares the same supported oracle ball path through8; each requires>=90% detections. Presence alone does not qualify location. Independent ending gates prevent premature-death predictions from gaming survival-weighted geometry. The non-ending gate concerns category means; every individual alarm>5% and all per-action/horizon ball/paddle/reward/ending/brick regressions remain in `supplementary-metrics.json`.

| Unchanged intermediate gate | Result |
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

Related model/oracle pairs and metric cells are not independent episodes, significance evidence or seed reliability. Existing invalid FIRE pair gate is not reused.

## Execution, evidence and failures

Native/source/runner `36c2b6c853e6531e2caf2e412be3c2e0453b8700`, native SHA256 `e18a94efdf2ab52e0c03505c34000aaf568555a009c90879e69a7bde48c46779`. Seven new Rust CPU and ten affected legacy header checks pass; strict core/test/binding Clippy passes. The first Rust fixture compilation failed on test-only typing/import mistakes; its log remains. First packaged Python run249 passes/one wrong-message fixture failure; only that failed case is repaired and rerun, giving250 distinct passing cases. The installed native did not change for the Python fixture repair.

The original qualification declaration checked the v13 source against a v14 file with a forwarding wrapper and stopped. Removing that wrapper gives the exact old suffix body; the repaired declaration checks whole files against the qualified v14 parent. Preserve the first declaration script and CPU repair receipt. A premature launcher attempted to read the absent queue, exited before creating any GPU guard/model, and remains a separate failed service/journal. No failed native qualification or learner is reused.

First actual native qualification passes182 pins/32.006s: new copy/correction F64 values/derivatives, actual production decoder zero-correction copying/lane isolation,240 parent initial tensors,16 cached non-image arrays,77 required changed weights,13 finite reports/total/valid fraction,240 frozen tensors,18 initial/trained restore arrays and old-native v15 rejection before GPU. Existing sampling/fusion/event/motion math is reused without completed native fixtures repeated.

Learning passes190 pins/170 CPU audit inputs:8,192 finite13-report rows, total and suffix fraction identity, full source plan/exposures, all80 changed weights,240 exact own-initial/frozen tensors,18 own initial-cohort arrays,384 matched dense origins,58 finite exports and18 exact final restore arrays. Four successful guards/36 sealed files pass, no host recovery/NVML or unfinished child.

Native training calls1157.190s; measured train/evaluation loop1335.676s. Loop excludes construction,dense-initial queries and pinning. Frozen assessment206.575s; learning CPU audit15.730s; branch CPU comparison91.409s. Build, failed preparation, qualification, restoration, figures and audits are separate costs. Single-run timings do not establish isolated kernel performance or efficiency at matched useful quality.

Released2,939,816,086 counted build-cache bytes after packaging. Runtime,wheel,native test executable,source,logs and useful checkpoints remain. Figures and three complete causal APNGs use saved arrays only, fixed lane0/draw0 rather than favorable matches; display resizing changes no model input or metric and clip frames decode losslessly.

## Remaining full goal

All-horizon causal ball/paddle physics, collision/reward/life-loss timing and calibration, persistent removed-brick memory, late-game/tunnel/second-wall coverage, untouched families and fresh learner confirmation remain required. The existing oracle has no above-top-brick endpoints and uses a restricted detector. A copy path, local fitting or intermediate gates alone cannot complete the world-model goal. No unchanged extension,sweep or automatic successor is selected.
