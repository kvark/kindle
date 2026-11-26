# Paired-action data regresses Breakout world learning

The [paired-data experiment](../experiments/2026-11-07-breakout-world-paired-actions.md)
is **closed and fails retention**. One fresh 8,192-update world learner regresses
observed and recorded forecasts, fails the new per-action presence gate, and
worsens long-horizon distribution and ending errors. Keep the coverage control;
do not extend or repeat this recipe. The full world-model goal remains unproven.

This tests the combination of new [matched fitting branches](2026-11-07-breakout-world-paired-actions-data.md)
and changed sampling, motivated by the [action-gap audit](2026-11-07-breakout-world-action-gap.md).
It does not establish that paired data is useless in general or isolate a causal
architecture/optimizer defect. Early fixed-action point tracking improves on a
different cohort, so do not describe every measured quantity as regressing.
Actor/critic and CDP comparisons remain parked.

## Intervention and exact comparison

Reuse native source `85dbe150` / SHA256
`5b1128667e6436b9865c31dedcb50e22318a43250e884046b155e51dc955ad82`
and the motion-rollout matching runtime. Python runner source is
`e751a8838d941fd21af426b2b3565c2889c1eef9`. No native rebuild or new loss.
Keep RGB64/12M/B8/T32/H16/Auto, initialization seed 41001, rates 4e-4, motion
cross-entropy, loss scales and initial tensor identity. The 448 new fitting
branches and sampler replace the prior coverage-only data mixture together;
volume, source weighting and within-batch correlations change. This is not an
isolated action-confounding estimate or fresh confirmation seed.

Four lanes share one fitting parent/repeat index and window start, two sample
original fitting streams, one samples supplement waiting, and one cycles a
seeded terminal-pool permutation including all 84 paired terminal streams.
Half the updates place the parent at the end of observation where branch lengths
allow; the other half sample valid common cuts. Short branches clamp the common
cut earlier. All targets remain real; the first arrival is a context reset,
while incoming source actions and actual terminal/reward targets are preserved.
Repeat0 shares parent RNG across plans; repeat1 uses evolving per-plan RNG.
A repeat index does not imply common RNG or independent experience.

The fixed plan replays 2,097,152 stored arrivals over 464 streams / 60,254 stored
arrivals, with 51,698 distinct stream/index slots selected. That last count
includes repeated shared prefixes across branch files and is not unique emulator
experience. Collection added only 13,068 new actions / 52,122 frames; no emulator
work is part of this learner. Paired lanes expose 488,428 future and 560,148
shared-prefix arrivals, including 436,680 valid intervention suffix targets.
All 112 parent/repeat groups are selected 56–94 times. Exposure totals:

|Item|Declared count|
|---|---:|
|Valid H16 suffix targets|989,057|
|Observed terminal targets|10,892|
|Valid suffix terminal targets|10,562|
|Positive reward rows|32,687|
|Natural source resets|7,872|
|Context and natural resets|73,282|

Keep periodic and dense fitting evaluation on the original sixteen streams;
development uses its original eight streams. Both original selection tables
match the control. All 448 paired fitting origins get a separate one-draw
post-training diagnostic. Unavailable tails are unscored; this is fitting evidence,
not an added held-out gate. Its time is outside the established main loop timer.
Original observed/one/eight-step, 16/32 energy, serving/ending/false-alarm gates
remain unchanged, plus >=90% rendered ball presence in each retained freeflight
plan at eight decisions. Presence alone does not qualify location/calibration.

## Original cohorts and paired fitting

All original source locations, images, targets, masks, references and geometry
match the retained controls. Counts below use the first fixed model draw, with
one-pixel tolerance only where the target ball is supported. They are related
recorded states, not independent episodes or calibrated distribution scores.

|Measure|Coverage fit|Motion fit|Paired fit, original cohort|Coverage dev|Motion dev|Paired dev|
|---|---:|---:|---:|---:|---:|---:|
|Observed|159/163|154/163|119/163|108/112|99/112|76/112|
|One-step|153/177|130/177|88/177|94/109|84/109|46/109|
|Four-step|73/180|86/180|47/180|33/109|47/109|8/109|
|Eight-step|34/176|61/176|22/176|14/107|25/107|5/107|
|Sixteen-step|14/171|21/171|11/171|3/114|10/114|3/114|
|Thirty-two-step|5/179|19/179|4/179|0/95|4/95|0/95|

All three original observed/one/eight-step gates fail. Development observed brick
balanced error is .273918 versus .223954 control/.278206 motion; original fitting
is .113784 versus .008525/.098104. Lower full-image error or a static-looking
background would not qualify object motion or brick memory.

The separate new paired fitting export has 448 related origins and one fixed
model draw: observed 352/376, one-step 326/374, four-step 221/338, eight-step
76/251, sixteen-step 39/213, thirty-two-step 0/102. The freeflight-origin subset
has observed/one-step 56/64, four-step 10/56, eight-step 4/64, sixteen-step 0/56
and thirty-two-step 0/30. Only 30/64 supported balls are detected at eight steps;
14/56 at sixteen and 0/30 at thirty-two. These futures can encounter collisions;
the starting category is not an entire-path physics label. Duplicate prefixes,
RNG repeats, varying visibility and unavailable tails limit these measurements.
Failure on fitting states means this is not solely a held-out-generalization
problem; it does not identify an isolated cause.

## Frozen fixed-action forecasts

All 84 new-checkpoint groups complete, with original oracle arrays identical.
The same raw conservative feature functions and first/full endpoint identities
pass. Primary history16 energy equally weights seven categories and horizons
16/32; history32 remains secondary. Pooled point metrics below instead use all
supported model/oracle pairs and omit predicted survival weights. Their mixes and
support differ from dense recorded cohorts; no best matching draw is selected.

|Fixed-plan ball measurement|Coverage|Motion rollout|Paired data|
|---|---:|---:|---:|
|1-step pair within 1px|70.66%|72.82%|79.72%|
|4-step pair within 1px|45.08%|52.04%|36.95%|
|8-step pair within 1px|22.45%|27.51%|6.88%|
|16-step pair within 1px|20.22%|39.63%|4.94%|
|32-step pair within 1px|5.90%|0.29%|0.00%|
|Primary 16/32 ball energy|.169502|.311977|.349371|
|Ending32 Brier|.144566|.189225|.378178|
|Non-ending origin/plans above 5% ending probability|0/304|23/304|134/304|
|Largest non-ending probability|1.91%|43.84%|96.24%|

Primary energy is 106.12% worse than the retained control (11.99% worse than
motion rollout), not the required 25% gain. One-step point accuracy improves on
the fixed-plan cohort but four/eight/sixteen/thirty-two-step tracking regresses.
Ending16 Brier .0000000224 and RIGHT16 mean probability .0000146 pass; ending32,
non-ending false alarms and long forecasts fail. NOOP serving false detections
remain 0/64. Retain every category, action, support count and both histories in
`paired-branch-comparison.json`; unsupported ball regions are not validated by
low energy. The retained oracle still lacks above-top-brick endpoints.

The original eight freeflight histories have identical oracle ball paths across
four plans through eight decisions. Model ball detections there are:

|Plan, 64 model images each|Coverage|Motion rollout|Paired data|
|---|---:|---:|---:|
|NOOP|64|64|8|
|LEFT|63|60|32|
|RIGHT|63|41|32|
|REVERSE|61|51|43|

All new plans fail the >=90% presence gate. Unwanted observable action sensitivity
persists. Mean squared action contrast of normalized presence/x/y at8 is .0467342;
paired finite-draw correction gives .0448652 versus .0004065 control/.0218740
motion. These are descriptive feature units, not causal fractions or calibrated
latent existence probabilities. Faint decoder output can affect detection, but
presence alone would not rescue position, event or distribution failures.

## Execution and retained artifacts

Native learning calls take 769.167s versus 765.720s motion (+.45%); the established
training/evaluation/export loop takes 1,128.611s versus 1,105.997s (+2.04%). That
loop excludes construction and the extra 448-origin fitting diagnostic. The
separate frozen 84-group assessment takes 299.786s. These are application timers,
not isolated kernel benchmarks or matched experience/hour scores.

Ten new sampler/fourteen existing CPU tests pass. The 27.956s one-update preflight
checks 517 pins, 204 exact initial/frozen tensors, eighteen initial/restore arrays,
eleven finite losses, 65 changed logical parameters and exact suffix-mask/total
identities. Actual fresh learning/restore review checks all 524 pins, all 8,192
finite loss rows, all source/count/mask/plan identities, 68 changed parameters,
204 initial/frozen tensors, 108 original initial evaluation arrays, 57 finite
export files and eighteen restore arrays. Paired fitting diagnostics preserve
204 frozen tensors separately. No raw-gradient proof is claimed.

The frozen assessment checks 857 input pins, 204 unchanged tensors and all 84
groups; CPU comparison pins 1,078 inputs and checks 420 oracle fields and 504
cached features. The three current guards audit 27 sealed pins with no unfinished
children, host recovery or unsealed files. Earlier collection/preflight guards
and original failed-launch receipts remain intact. CPU learning review 8.115s,
paired fitting review .976s and frozen comparison 39.691s complete first try.
The new learner/assessment has no repaired or repeated native stage.

Root `/mnt/data/kindle-breakout-world-paired-actions-20261107`. The original
parent NaN/112-file CPU repair and base-interpreter NumPy launch failure stay
preserved; they occurred before this separately declared learning/assessment.
No old collector, learner, assessment or failed fixture was rerun.

`paired-comparison.png` and `paired-learning-curves.png` show matched metrics and
all original periodic controls. Three fixed-case four-panel causal APNGs compare
native oracle, coverage, motion and paired data using cached full32 arrays:
`paired-freeflight-noop-causal.png`, `paired-contact-left-causal.png`,
`paired-life_loss-noop-causal.png`. Fixed lane0, oracle0/model0/history16, all
32 decisions /128 original frames each. All 96 decoded frames match exactly;
display-only nearest scaling preserves physical aspect and never changes model
input/metrics. The 3.232s CPU clip stage checks 14 pins and creates no new model/
emulator query or experience. Posters and both plots were visually inspected.

No GPU worker or learner remains active. Keep useful checkpoints and raw outputs
outside Git; no disposable native build cache was created. All original quality
failures remain failures. No selected checkpoint or favorable draw replaces the
final 8,192 endpoint, and no unchanged extension or scalar/seed sweep follows.

The next selected [CPU-only diagnosis](../experiments/2026-11-07-breakout-world-linear-reference.md)
compares errors on exactly linear, visibly supported reference segments, using
past-RGB point controls. That can separate familiar simple-motion errors from
collision/visibility cases without another model query; it cannot prove a causal
architecture defect or replace full-world requirements. No new GPU experiment
is selected. Full physics, event calibration, brick memory, late/tunnel/second-wall
coverage, untouched families and fresh confirmation remain open.
