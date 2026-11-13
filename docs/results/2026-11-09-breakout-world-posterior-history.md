# Breakout: posterior-conditioned history training fails its early-budget check

The new objective completes a fresh2,048-update learner, exact restore and
final-only frozen assessment, but fails nine of ten original intermediate gates.
It does not establish faster useful learning. On the matched2,048-update natural
development origins, one-step ball accuracy falls23/27→18/27 versus H0; both
have zero eight-step matches. The full world-model goal remains unproven.

[Protocol](../experiments/2026-11-09-breakout-world-posterior-history.md).
Root `/mnt/data/kindle-breakout-world-posterior-history-20261109`. Keep this as a
negative bounded development result, not a claim that all later budgets or
posterior-conditioned methods fail. No extension, sweep or automatic successor
is selected. The earlier H0 references remain retained.

## What was trained

Explicit `motion_posterior_history`, checkpoint v21. Ordinary observed RGB/KL/
reward/continuation losses remain. The extra16-arrival suffix decodes the actual
image-informed posterior at each arrival, but feeds previous decoded images into
its two-image history, seeded once from the real prefix. Derivatives flow through
generated history; reset masks clear it. Existing motion weights and suffix
support masks remain. No independent prior future is scored against an unrelated
realized image. This addresses that specific outcome-assignment issue without
guaranteeing a useful prior or coherent sampled images.

Inference is unchanged: only prior samples, past images and future commands.
No future real image corrects an evaluation trajectory. No new weight, primitive,
actor, critic, pretrained encoder or object/RAM training label. Default and old
checkpoint objectives retain their behavior; v21 is explicit and restore rejects
legacy disguises and changed objectives.

Source/runner64f5489f, native SHA256
`aa12d0507897b09d69d8d41cd7a1cd010f183ee47ffb99d1217b10a901956c16`.
Size1M/RGB64/B8/T32/full BPTT32,603,915 weights/80 logical/240 saved tensors,
Auto, seed41001, rates4e-4, RGB1, KL1/.1/free-nat1, observed events1, mixing.01.
The extra reconstruction changes both gradient paths and total loss balance;
this experiment does not isolate its coefficient from its recurrence.

The first2,048 rows of the broad corrective plan cover496 retained streams and
62,666 source arrivals,56,189 unique arrivals. This new prefix is counted once:
2,575 observed terminals,2,394 supported suffix terminals,5,552 positive targets,
18,636 context resets and2,294 source resets. The legacy counter name
prior_terminal_rows means posterior suffix support here. No new game actions or
emulator frames. Source data and development families are reused, not untouched
confirmation. The same first16 natural source streams supply dense fitting data.

## Matched-budget comparison and dense forecasts

The H0 control retains a cached2,048-update evaluation on the same fixed natural
origins. Initial240 tensors, configuration, first replay prefix and first ordinary
unweighted losses match. Model updates differ because of the added objective;
checkpoint/evaluation cadence and measured wrapper work also differ. No old model
is queried. The32-origin curves are distinct from the final dense128-origin
assessment and do not replace it.

| Matched2,048-update natural development | Observed |1|4|8|16|32|
|---|---:|---:|---:|---:|---:|---:|
| H0/RGB1 |23/26|23/27|2/26|0/27|0/27|0/25|
| Posterior history |18/26|18/27|1/26|0/27|0/27|0/25|

All fixed natural/contact/ending curves for both fitting and development are
retained in `learning-review.json`; no best origin, draw or checkpoint is chosen.

Dense final fractions count supported visible truth balls within one RGB64 pixel;
missing predictions fail. Fixed first draw. Actual initial counts are zero.

| Scope | Observed |1|4|8|16|32|
|---|---:|---:|---:|---:|---:|---:|
| Fitting, new2k |108/163|95/177|21/180|6/176|1/171|1/179|
| Development, new2k |78/112|57/109|7/109|2/107|0/114|0/95|

The earlier16k H0 dense result109/112 observed,107/109 one-step and15/107
eight-step is a different budget. It is retained for context, not used to claim
a matched-budget causal effect. New early losses and images do not establish
precise fitting, robust rollout or useful control.

## Original frozen gates also fail

One final checkpoint receives84 groups:42 plans, histories16 primary/32 secondary,
eight fixed draws, endpoints1/4/8/16/32. Six sealed control reports/summaries are
reused without old model queries, old predicted RGB reads or repeated completed
scalar/point tests. Physical truth and image-derived features match the retained
oracle. All action/horizon ball/paddle/reward/ending/brick regressions and every
nonending alarm remain in `supplementary-metrics.json`, at both histories.

Primary balanced, survival-weighted16/32 ball energy is.534384, above the original
.127127 gate and global H0 reference.169502. Other retained final controls include
small64.172366, warm corrective.229385,12M local-action.240566, action/transport
.387593 and local H0/RGB1.429470. Those checkpoints have different budgets/recipes;
they are fixed references, not matched2k learners.

New category energies: serve.319893, freeflight.622402, contact.388928,
brick.408661, wall.545167, life-loss.815846, ending.639791. Survival weighting
prevents interpreting category changes alone as better image physics. A detector
centroid of faint fragments is not proof of a coherent sampled ball.

Freeflight8 ball presence:8/64 NOOP,3/64 LEFT,13/64 RIGHT,11/64 reversal; all fail
90%. Serve NOOP has no false ball at8 and is the only passing gate. Ending16
Brier.021188, ending32.276176 and RIGHT16 risk.102788 fail. Nonending alarms
occur in278/304 cases, maximum probability.576858; category risks also fail.
Early all-alive slices measure false alarms, not genuine ending recognition.

Fixed first-draw/lane0 serve LEFTFIRE, freeflight NOOP and contact LEFT clips retain
96 lossless checked frames. The freeflight decision8 poster shows a diffuse red
streak rather than the real compact ball. These fixed examples illustrate the
failure; they do not establish a causal population decomposition.

## Execution and verification

All657 learner input/129 CPU-audit pins pass. All2,048 eleven-component rows are
finite with the correct combined total and support. The first actual update is
inside the budget: all ordinary unweighted components agree with H0 within1e-6,
77 weights change in required families,240 tensors are finite at age1, and initial
240 weights/18 arrays match the retained control. All80 logical weights change by
final. Initial/512/1024/final/final-frozen ages and v21 configurations agree;
final/frozen240 tensors and18 restored arrays are exact. All27 natural forecast
exports and84 full causal exports are finite. Assessment1,611 input/236 CPU pins;
three nine-file train/restore/assessment guards pass and both services are terminal.

Learner-call timer225.649s, loop290.849s. Matched H0 prefix call timer135.039s;
new observation is67.10% higher, but also includes first-state checking and two
extra checkpoint saves. Shared-host/wrapper observations do not isolate GPU
kernel speed. Frozen assessment182.831s; CPU learning review3.683s and score
review16.049s. No pause, warm start, host recovery or NVML polling.

The original CPU compile stops on a JSON fixture literal. After its repair, the
broad CPU run completes131 passed/3 failed/39 ignored before a proposed stop;
no process is signaled. Failures expose omitted v21 restore allowlists, an
incorrect fixture reward-head size and an incorrect loss-ancestry assertion for
the final display sigmoid. Original sources/logs remain. Focused corrections pass
three new tests and13 affected header tests; strict core Clippy and326 installed-
package CPU checks pass. Padded graph report slots and posterior traces remain
distinct from the11 scalar API reports. Qualified existing decoder/history/
motion-CE value/derivative receipts are reused; this is not a new independent
full-CNN F64 derivative qualification.

The first unqualified wheel is overwritten before retention; its source and
build log remain, and this limitation is explicit. An intermediate objective-
rejecting wheel and the separately built/installed measured candidate remain.
No original failed comparison is relabeled. This run does not erase earlier
source-specific preparation or precision failures.

Learning service invocation3637dcbe1a134963ac42283dd683a9ec; assessment invocation
a32f0a08f7be4335ac85c87ecad13081. Matching runtime/wheel/source/CPU executable,
checkpoints, outputs, corrections and logs remain. Disposable build caches total
3,206,658,485 bytes are released after packaging and reviewed learning.

## Implication

The previous refresh experiment establishes transient sensitivity to corrupted
image history. This new early-budget training test does not convert that
sensitivity into faster state reading, better long forecasts or reliable events.
It neither proves all later budgets fail nor justifies an unchanged extension.
Review the image generator's conditional distribution and sampling behavior
before another training proposal; outcome-conditioned recurrence alone is not
a measured repair. A different image generator remains a separately justified
architectural option, not selected or qualified here.

All original full-game requirements remain: precise physics/events, natural-
frequency calibration, persistent removed-brick memory, late/tunnel/second-wall/
above-top coverage, untouched families and fresh learner confirmation. Existing
oracle has no above-top-brick endpoint. The world-model and five-game goals stay
unfinished; actor/critic and CDP comparisons remain parked.
