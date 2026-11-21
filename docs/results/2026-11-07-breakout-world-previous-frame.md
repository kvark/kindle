# Previous-frame world encoder: fixed learning result

The [bounded architectural experiment](../experiments/2026-11-07-breakout-world-previous-frame.md)
is **closed and fails retention**.
Full world-model competence remains unproven; actor/critic and CDP comparisons
stay parked. Keep the coverage control.
No unchanged extension,repeat,sweep or automatic successor is selected.

## Causal motion input, with current RGB targets

Optional `rgb_previous_frame` gives the encoder previous/current resized images
from the same lane, oldest first. Native Rust duplicates current RGB at clipped
starts and reset actions. RGB3 replay, public arguments,current-image targets,
RGB3 decoder and one GPU resize stay unchanged. No preceding-episode/future image
or object/RAM/phase label. Only the first convolution widens3→6,adding2,400
weights:68 logical/8,855,075 scalar weights/204 weight-and-moment tensors. Encoder
output1024 and RSSM feature2560 remain. Version5 persists mode; default is off.

The full RSSM already sees image history, so this adds no information overall.
It tests a shorter path for learning motion, not an established missing signal
or isolated physical-error cause. The earlier [whole-agent CDP pair comparison](2026-11-03-cdp-rgb-frame-pair.md)
remains negative/closed:greedy114/sample39.208 versus143.667/70.750 with13.13%
more time. This isolated pixel objective/replay is different; no old stage repeats.

Keep retained H0 motion-weighted observed RGB100,KL1/.1/free-nat1,reward/continuation1,
RGB64/12M/B8/T32/full BPTT/Auto,mixing.01,seed41001 and rates4e-4. No direct future-
image auxiliary,actor/critic,pretrained encoder,new loss formula or backend primitive.
Native frame selection/copy adds host work; time must be measured.

Fresh8192 updates, original sixteen fitting/eight development streams and all8192
original4/2/1/1 source locations. Exposures match H0:10,909 terminals,34,440 positive
rewards,10,488 source resets,75,808 context resets,32,849 unique stored slots of
32,850; suffix targets/terminals0. No new emulator actions/frames. New convolution
geometry changes initialization; shared seed is not matched-architecture control
or an independent confirmation. Initial/2048/4096/final checkpoints and all fixed
periodic draws remain; no favorable checkpoint is selected.

## Recorded fitting and development forecasts

First fixed draw,within one resized pixel on supported visible balls. Missing
predictions stay in the denominator. Varying horizon support and related origins
are not independent episodes or a calibrated stochastic distribution score.

|Measure / future decisions|Own actual initial dev|Retained H0 dev|Motion image H16 dev|Previous-frame H0 dev|Previous-frame H0 fit|
|---|---:|---:|---:|---:|---:|
|Observed|0/112|108/112|99/112|108/112|160/163|
|1|0/109|94/109|84/109|90/109|150/177|
|4|0/109|33/109|47/109|35/109|81/180|
|8|0/107|14/107|25/107|10/107|47/176|
|16|0/114|3/114|10/114|7/114|16/171|
|32|0/95|0/95|4/95|3/95|7/179|

Matched dense-initial/final exports cover256 fitting/128 development origins,
with identical source RGB,locations,targets and masks. All204 own initial tensors
match qualification and stay exact through dense-initial inference;18 initial
original-cohort arrays also match. This establishes actual-initial gains, not
held-out reliability or sufficient causal physics. Geometry differs from H0,so
no cross-architecture initial tensor/prediction equality is claimed.
Fitting eight-step tracking improves34→47/176 while development falls14→10/107.
This does not isolate an encoder or optimization cause, and neither result is
sufficient physical competence.

Observed development detects111/112 balls;
detected mean/p95 error 0.331/0.745px.
Observed brick balanced error 0.193399,versus H0
0.223954. Every
horizon,paddle/brick/error/false-detection and reward/event metric remains in
`learning-review.json`; observed accuracy does not establish removed-brick memory.

## Alternative actions and events

Only the new final checkpoint is assessed:84 groups,history16(primary)/32(secondary),
eight fixed draws,all1/4/8/16/32 endpoints. Original oracle arrays,feature recounts
and full-first-draw endpoint identities match. Reuse old control arrays; no old-
model assessment or repeat collection. These repeatedly examined families are
development,not untouched generalization or population calibration.

Primary seven-equally-weighted-category16/32 survival-weighted ball energy:
new **0.189612**, H0 0.169502, motion H16 0.311977.
Relative reduction versus H0:-11.86%; unchanged
threshold.127127. Point accuracy/detection omit survival weighting and cannot
replace this score. Freeflight8 NOOP/LEFT/RIGHT/REVERSE detections 47/62/62/59 of64
per plan,versus90% each. Oracle ball paths agree across those plans through8.

Ending16/32 Brier 0.017841/0.073287;
RIGHT16 false ending probability 0.223745,RIGHT32
ending probability 54.37%
versus oracle100%. Non-ending origin/plans above5% ending probability:
0/304,max 4.15%.
The category-mean gate below does not erase individual false alarms. Ending32
reward-event Brier 0.352369; not a full reward-
calibration test. All actions,histories,horizons,paddle,brick and reward regressions
and supported pair counts remain in `branch-comparison.json`.

|Unchanged intermediate gate|Result|
|---|---|
|Observed ball precision|Pass|
|One-step ball precision|Pass|
|Eight-step ball precision|Fail|
|NOOP serve false balls|Pass|
|Ending Brier at32|Pass|
|Ending Brier at16|Fail|
|RIGHT ending probability at16|Fail|
|Non-ending category mean false alarms|Pass|
|Primary distribution error|Fail|
|Ball presence on every freeflight plan|Fail|

Missing support is untested. Precise physical trajectories,event timing/calibration,
persistent brick memory,late/tunnel/second-wall coverage,untouched families and
fresh confirmation remain required for the full goal. The retained oracle has
no above-top-brick endpoints. Passing intermediate gates cannot complete these.

## Qualification, source and cost

Root `/mnt/data/kindle-breakout-world-previous-frame-20261107`; native/source/runner
70da4788,SHA256
`96adefd251ec813bff450f660acdd52be0432734d8688abe4ac35692cc0b03f9`.
All60 affected Rust/89 packaged Python checks and core/binding strict Clippy pass.
Preserve first float-syntax/import compilation errors and flat-weight-shape fixture
failure. Corrections change no learning result. Existing convolution/concat/resize
primitive evidence is reused without repeating native primitive fixtures.

First456-pin native composition qualification passes in17.391s:
204 own-initial restored tensors,18 initial/trained restore arrays,65 expected
changed parameters,204 frozen tensors,previous-frame encoder influence,lane
independence,all8 reset-output arrays identical,and18 legacy cached arrays.
Old native rejects v5 with its GPU loader disabled. Legacy204 tensors stay exact.
This establishes native execution/causality,not learned motion competence.

Learning466 pins,8192 finite nine-report losses,all68 changed parameters,full
plan/exposures,384 matched dense-initial origins,58 finite exports,204 exact own
initial/frozen tensors and18 final restore arrays pass. All four successful host
guards/36 sealed files pass on AMD/Auto,no recovery/NVML or unfinished child.
The918-pin new-final declaration is verified.
Source members are copied byte-for-byte from the pinned Git archive; original
qualification declarations remain unchanged. Preserve wheel/runtime/CPU executable;
834,734,615 counted bytes of disposable cache are released after qualification.

Native learning calls **538.013s**,versus H0 529.515s. Main
train/evaluation loop 674.426s excludes dense-initial queries.
Observed guard worker span 688.647s includes construction,
dense-initial,main loop,pin checks and polling; components are not separately
isolated. Frozen branches 126.066s lie outside learning timers.
These single-run observations are not isolated kernel benchmarks or seed reliability.

`comparison.png`,`learning-curves.png` and three causal APNGs render retained arrays
only,fixed first origins/draws,no best-match selection and display-only resizing.
All earlier negative results/failures remain closed and preserved.
