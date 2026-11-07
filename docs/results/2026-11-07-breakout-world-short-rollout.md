# Four-step direct image rollout: fixed learning result

The [bounded experiment](../experiments/2026-11-07-breakout-world-short-rollout.md)
is **closed and fails retention**. Full world-model competence remains unproven;
actor/critic and CDP comparisons stay parked. Keep the coverage control.
No unchanged extension, repeat, sweep or automatic successor is selected.

## Fixed intervention

Shorten existing direct motion-weighted RGB prediction H16→4. Use the retained
recurrent-conditioned posterior and h/z decoder, stride-two off, categorical
mixing.01, RGB64/12M/B8/T32/full BPTT/Auto, seed41001, rates4e-4, observed RGB100,
one-step KL1/.1/free-nat1 and observed reward/continuation1. No new loss formula,
Rust/backend change, rebuild, data, pretrained encoder or object/RAM/phase labels.
All68 logical/8,852,675 scalar weights/204 weight-and-moment tensors remain.

From observed index27, roll four prior transitions under submitted actions.
Future images enter detached cross-entropy targets/motion weights only; no image
feedback. Reset and later suffix rows are masked out; terminal arrivals remain.
The first motion pair is image27→28. Average existing future image loss across
full Bx4, then scale100. Shortening the horizon also changes prefix length,
suffix positions, exposures and categorical RNG consumption. It does not isolate
a variance mechanism. Straight-through gradients and conditional-mean pressure
remain; target-dependent motion weighting is not a calibrated likelihood.

Fresh8192 updates, original sixteen fitting/eight development streams, all8192
original4/2/1/1 source locations. Non-suffix counts match:10,909 observed terminals,
34,440 positive rewards,10,488 source resets,75,808 context resets,32,849 unique
stored slots of32,850. H4 has256,201 valid suffix targets and2,382 suffix terminals.
No new emulator actions/frames. Reused seed/families are development, not a fresh
confirmation. Initial/2048/4096/final checkpoints and all periodic draws remain;
no favorable checkpoint is selected.

## Recorded forecasts

First fixed draw, within one resized pixel on supported visible balls. Missing
predictions count as errors. Supports differ across horizons; related origins
are not independent episodes or a stochastic calibration test.

|Measure / future decisions|Actual initial dev|Retained H0 dev|Direct image H16 dev|New H4 dev|New H4 fit|
|---|---:|---:|---:|---:|---:|
|Observed|0/112|108/112|99/112|91/112|136/163|
|1|0/109|94/109|84/109|73/109|105/177|
|4|0/109|33/109|47/109|21/109|44/180|
|8|0/107|14/107|25/107|5/107|9/176|
|16|0/114|3/114|10/114|4/114|0/171|
|32|0/95|0/95|4/95|0/95|1/179|

Matched dense-initial and final exports cover256 fitting/128 development origins,
with identical source RGB,locations,targets and masks. All204 weights/moments
stay exact during dense-initial inference and match the qualified fresh initial
and retained control. Original initial development forecasts also match exactly.
This establishes actual-initial gains, not generalization or sufficient physics.
Fitting forecasts are also weak:9/176 at eight and0/171 at sixteen. The failure
is not solely a development generalization gap; its cause remains unresolved.

Observed development detects111/112 balls;
detected mean/p95 errors 0.589/
1.369px. Brick balanced error
0.298194, versus H0
0.223954.
Every horizon,paddle/brick/error/false-detection and reward/event metric remains
in `learning-review.json`; observed state quality does not prove removal memory.

## Same-origin alternative actions

Only the new final checkpoint is queried:84 groups,history16(primary)/32(secondary),
eight draws,all1/4/8/16/32 endpoints. All original oracle fields,feature recounts
and full-first-draw endpoint identities match. Old control arrays are reused;
no old-model query or repeated collection. Repeatedly examined families are not
untouched tests and do not establish population calibration.

Primary seven-equally-weighted-category16/32 survival-weighted ball energy:
H4 **0.646172**, H0 0.169502, H16 0.311977.
Relative reduction against H0: -281.22%; unchanged
threshold.127127. Supported point/detection metrics omit survival weighting and
cannot replace the distribution score. Freeflight8 NOOP/LEFT/RIGHT/REVERSE
ball detections 18/25/42/35 of64 per plan, versus90% each; oracle ball paths agree
across these plans through8.

Ending16/32 Brier 0.000002/
0.494911; RIGHT16 false ending probability
0.002054. Non-ending origin/plans above5% ending
probability:13/304, maximum
9.50%. The separate category-mean gate below
does not erase individual false alarms. RIGHT32 ending probability averages
4.45%
against oracle100%. Correctly avoiding early endings at16 does not establish
correct ending timing when the actual later ending is missed. Ending32 reward-event Brier
0.408619; it is not a full reward-calibration test.

|Unchanged intermediate gate|Result|
|---|---|
|Observed ball precision|Fail|
|One-step ball precision|Fail|
|Eight-step ball precision|Fail|
|NOOP serve false balls|Pass|
|Ending Brier at32|Fail|
|Ending Brier at16|Pass|
|RIGHT ending probability at16|Pass|
|Non-ending category mean false alarms|Pass|
|Primary distribution error|Fail|
|Ball presence on every freeflight plan|Fail|

All action/history/horizon/category/reward/paddle/brick regressions and supported
pair counts remain in `branch-comparison.json`. Missing support is untested.
Precise physical trajectories,event timing/calibration,persistent removed-brick
memory,late/tunnel/second-wall coverage,untouched families and fresh confirmation
remain necessary for the full goal. The retained oracle has no above-top-brick
endpoints. Passing intermediate gates would not complete these requirements.

## Execution and cost

Root `/mnt/data/kindle-breakout-world-short-rollout-20261107`. Exact reused native
85dbe150/SHA256
`5b1128667e6436b9865c31dedcb50e22318a43250e884046b155e51dc955ad82`,
matching API runner606720ca. No native rebuild/cache. All36 affected Python CPU
checks pass; existing native primitives/derivatives and legacy restore evidence
are reused without rerunning old fixtures or models.

First qualification139 pins/15.354s passes204 exact initial
tensors,18 original initial arrays,18 initial/trained restore arrays,persisted H4,
finite11-report/mask/total,65 expected first-update changed parameters and204
frozen tensors. Learning147 pins,8192 finite reports,all68 changed parameters,
full plan/exposures,384 matched dense-initial origins,204 exact initial/frozen
tensors and18 final restore arrays pass. All four successful host guards/36
sealed files pass with AMD/Auto; no host recovery or unfinished child.

Native learning calls **579.490s**, versus H0
529.515s/H16
765.720s. Main train/evaluation loop
728.258s excludes dense-initial queries. Observed guard worker
span 742.800s includes construction,dense-initial,
main loop,pin verification and polling; components are not separately isolated.
Frozen branches 166.754s are outside learning timers. These are
single-run wall observations, not kernel benchmarks or seed reliability.

`comparison.png`,`learning-curves.png` and three causal APNGs use retained arrays
only,fixed first origins/draws,no best-match selection and display-only resizing.
Original failures and closed negative experiments remain intact.
