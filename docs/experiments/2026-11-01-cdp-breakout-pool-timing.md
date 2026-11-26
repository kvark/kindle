# Separate Breakout pooled-observation timing from response residuals

The retained teacher's contact diagnostic finds larger paddle than landing
errors. Four- and six-feature mean-response refits both fail their fixed
held-out MAE screens. Before changing another controller or loss,use existing
raw-frame detections to measure what two-frame pooling contributes to the
observed positions and one-step residuals. This is CPU-only work within the
existing five-game goal;the original Qbert R32 learner continues unchanged.

Use exactly the retained225.083-point teacher's first three natural episodes
per eight lanes. Join the52,072-action source log,208,254 raw centroid rows and
matching frame annotations by lane,episode and cumulative executed raw frame.
These are already verified reconstructions;do not replay the emulator. Exclude
each episode's initial observation from raw alignment because its reset image
was not recorded. Do not carry tracks across resets or confuse cumulative
frame counts with episode-relative counts.

For each selected decision with visible pooled paddle,compare its recorded
center against the most recent raw component center. A component is accepted
only when present,not cropped,12 or16 pixels wide,and matching the original raw
centroid within1e-9. Retain all exclusions by reason. Where both latest raw
frames pass and have equal width/row,also compare against their midpoint and
report the last-frame paddle displacement. These pixel proxies are not exact
emulator collision state. Midpoint agreement can identify pooling geometry on
accepted rows without proving that delayed observations cause failed control.

On consecutive same-episode decisions,reproduce the original all/near-descending
pooled next-position error counts,MAE,median and95th percentile exactly within
1e-10. On shared rows with accepted next raw component,compare the unchanged
first-step prediction against both next pooled and latest raw positions. Save
the exact residual decomposition:prediction-minus-raw equals prediction-minus-
pooled plus pooled-minus-raw. Components can cancel;do not interpret the target
change as the fraction of control error explained by pooling.

Report count,bias,MAE,RMSE,median/95th-percentile absolute error,maximum and
exact-match counts for all accepted rows,each lane,near descending(projected
and ball y≥150),and projected impacts within one action(0<steps≤1). Preserve
all aligned rows and exclusions. No fit,correction coefficient,threshold search,
new teacher action,gameplay test,native import,GPU job or learner update occurs.
The response model was calibrated on pooled positions;substituting current raw
positions would change its input and target semantics. This diagnostic cannot
select such a change or establish a Kindle representation limit.

Six CPU fixtures cover arithmetic/decomposition,raw/pair rejection conditions,
equal-width pooling geometry,global/episode timestamp alignment and phase
boundaries. Verify exact source hashes and the original active Qbert identity
and490 pins before/after work. Run on one low-priority CPU core under60s;retain
the entire co-host time without correcting Qbert training cost. Preserve any
failure;do not repeat completed replay,fit or GPU stages.

Root `/x/Temp/kindle-breakout-pool-timing-20261101`. Active original Qbert
PID792387/invocation460e0220583948f4b088b5dc7c92956a,service
`kindle-qbert-h5-r32-greedy-200k-20261101`. No policy change,learner or automatic
successor is selected by this analysis.

[Contact evidence](../results/2026-11-01-cdp-breakout-teacher-contact-errors.md),
[rejected history fit](../results/2026-11-01-cdp-breakout-response-history.md).

CPU preflight passes in2.568s with all six fixtures and source hashes. Original
Qbert identity and490 input pins remain unchanged before/after;no native or
emulator import occurs. Preserve this co-host time;do not repeat qualification.
