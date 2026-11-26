# Breakout: smaller RGB64 world

The fixed learner and final frozen assessment are complete.
This capacity/budget recipe fails retention. Keep H0 as the broad reference.
Actor/critic and CDP comparison stay parked. No extension, sweep, collector or
successor is selected. [Protocol](../experiments/2026-11-08-breakout-world-small64.md).
Root `/mnt/data/kindle-breakout-world-small64-20261108`.

## Capacity and budget

Use the existing 1M configuration with the v18 local-action decoder at unchanged
RGB64. Actual world: **603,915 scalar weights**, 80 logical parameters and 240
checkpoint tensors. The 12M local-action world has 8,893,203 weights. Local motion
width 16 and direct command planes stay. CNN depth, recurrent width, categorical
classes and feature dimensions shrink together. This is not an isolated cause.

Fresh **16,384** updates, B8/T32/H16/full BPTT32, Auto, seed41001, rates 4e-4,
mixing .01, RGB100, KL1/.1/free-nat1 and event scales 1. Reuse the original 16 fitting/
eight development streams and exact 8,192-row plan twice in one learner. Weights,
optimizer age and RNG do not restart between cycles. There are 1,951,882 valid
suffix targets, 19,140 terminal and 32,016 positive suffix targets; 32,849 unique of
32,850 arrivals. Sealed exposures scale without recounting. No new environment
actions/frames, collection, camera path, pretrained encoder or object/RAM labels.
Learning/inference remain native on Meganeura+Blade. Capacity and optimizer
budget change together; no matched-compute or reliability claim.

## Frozen ball forecasts

Same 256 fitting/128 development origins, actual initial and fixed final, first
fixed draw. Counts are visible balls within one resized RGB64 pixel. Missing or
ambiguous balls fail. These reused related windows are development evidence.

|Model|Observed|1|4|8|16|32|
|---|---:|---:|---:|---:|---:|---:|
|Actual initial|0/112|0/109|0/109|0/107|0/114|0/95|
|12M H0 reference|108/112|94/109|33/109|14/107|3/114|0/95|
|12M local action|98/112|101/109|63/109|43/107|29/114|11/95|
|12M action + transport|103/112|100/109|65/109|43/107|22/114|8/95|
|1M local action|103/112|102/109|69/109|53/107|22/114|8/95|
|1M fitting cohort|146/163|154/177|107/180|84/176|37/171|20/179|

Other draws, fitting targets, paddle/reward/ending/brick metrics and raw arrays
remain retained. Periodic curves do not select an earlier checkpoint or draw.

## Chosen-action forecasts and events

Only the new fixed final receives 84 groups: history 16 primary/32 secondary,
eight draws and 1/4/8/16/32 endpoints against the unchanged cached oracle. Primary
balanced survival-weighted16/32 ball energy is **0.172365704612**,
versus H0 .169502219090, local action .240565683475 and required <=.127126664317.
This is 1.689% worse than H0.

For the chosen-action cohort, supported ball-pair accuracy also regresses
against 12M local action. These scores omit predicted survival and cannot
replace the joint score. Recorded and chosen-action cohorts differ.

|Model|8|16|32|
|---|---:|---:|---:|
|H0 reference|22.5%|20.2%|5.9%|
|12M local action|55.1%|73.2%|17.8%|
|1M local action|47.7%|62.5%|2.6%|

Ending16/32 Brier: 0.026988374280/
0.279206664698. RIGHT16 false-ending risk:
0.252475605456, against oracle zero. Primary
individual nonending alarms above 5%: 265/304,
maximum 0.957569239534. Category-mean gates and individual
alarms remain distinct. Freeflight8 presence NOOP/LEFT/RIGHT/REVERSE:
64/64/64/64 of 64 each; detection does not establish position.

|Original intermediate requirement|Result|
|---|---|
|observed ball|Pass|
|one step ball|Pass|
|eight step ball|Pass|
|serve noop|Pass|
|ending brier|Fail|
|ending16 brier|Fail|
|ending16 right|Fail|
|nonending false alarms|Fail|
|mean ball energy reduction|Fail|
|freeflight each plan presence at8|Pass|

All per-action/horizon regressions and individual alarms remain for **both**
histories. Primary regressions versus H0:

|Metric|Regressed/scored cells|
|---|---:|
|ball energy|97/210|
|paddle energy|194/210|
|reward brier|204/210|
|ending brier|191/210|
|brick brier|16/209|

Point-pair metrics omit survival and do not represent independent episodes.
All oracle outcomes through 16 are alive; lower Brier there measures false-alarm
suppression, not terminal recognition or natural-frequency calibration.

## Cost, integrity and scope

|Learning cost|12M local action|1M local action|
|---|---:|---:|
|Fresh updates|8,192|16,384|
|Native learning calls|1389.475798s|2203.087522s|
|Training/evaluation loop|1568.062112s|2369.265223s|

Loop timings exclude construction, initial dense exports and preparation.
Preflight 29.151661s; new final assessment
176.103903s; CPU comparison
16.543625s. Other host GPU work was present. Parameter counts
and separate shared-host timers do not certify an isolated hardware speedup.

Native/source/runner7f797a2b, SHA256
`38680c739929c82c8e1e5a4f15bfa1277ba32170e9c7d14034c2d9c3f75ba323`.
No Rust change, build or new primitive. Preflight passes 137 pins, 77 first-update
changes/required families, 13 finite components/mask/total, 240 own initial/frozen
tensors and 18 initial/trained restores. The fresh learner uses 159 finalized
input pins and matches the first qualified report exactly. All 16,384 finite
rows, 80 changed weights, continuous ages/two-cycle plan, exact exposures/four
milestone configurations, 384 dense actual-initial/final origins, 58 finite
exports, 240 initial/frozen tensors and 18 final restore arrays pass.

Cached-control aggregation reproduces 1,680 scalar fields within absolute 1e-13
and validates 480 point rows. Only new-model RGB is rechecked; old control scores
and point summaries are reused. H0 truth arrays supply the shared oracle join.
The first plot-schema check fails on H0's older timer format; its source/failure
remain. Separate `step` fallback validates both 8,192-row control clocks and exact
final native totals, without repeating a learner or forecast. This CPU repair
does not relabel prior protocol failures.

Closure verifies 1,358 pins and four successful guards/36 sealed files,
with all three allocation services terminal. Auto on RX7900XT/RADV, no driver
reset/recovery/NVML. No new build cache exists. Runtime, source receipts, all
useful checkpoints/logs/arrays, fixed comparison/learning figures and three
lossless new-model/oracle causal clips remain. Every decoded clip frame matches;
no favorable-case or draw selection. Original 141-pin preallocation declaration
remains beside the finalized 159-pin declaration with own initial references.

The [pixel-generator review](2026-11-08-breakout-world-pixel-generators-review.md)
contains research hypotheses, not a selected successor. Full physical precision,
event timing/calibration, persistent brick memory, late/tunnel/second-wall states,
untouched families and fresh confirmation remain required. Current oracle has
no above-top-brick endpoint. Even passing these intermediate gates could not
complete the full world-model or five-game goal.
