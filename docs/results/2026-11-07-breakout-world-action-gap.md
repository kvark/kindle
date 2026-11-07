# Short tracking gains transfer; long rollouts lose objects and show unwanted action sensitivity

The CPU-only [action-gap audit](../experiments/2026-11-07-breakout-world-action-gap.md)
is closed. It narrows the previous diagnosis: **some eight/sixteen-step point
tracking improvements also occur on fixed action branches**. The earlier summary
compared recorded eight-step point accuracy with a different16/32 distribution
score. That did not establish that all short-term gains failed to transfer.

The [motion-rollout learner](2026-11-07-breakout-world-motion-rollout.md) still
fails its original retention gates. Thirty-two-step tracking collapses, object
detections disappear and event/observable-distribution errors remain worse. The
matched audit identifies an additional concrete error: paddle commands change
rendered ball predictions in verified freeflight cases whose oracle ball paths
are identical. No evaluation offset/layout mismatch is identified in the checked
source/data. This is not proof that every native input buffer was correct or
that training-data confounding causes the error. Full competence remains unproven.

## Same metric and horizons

Use all retained outputs for coverage/motion-rollout,42 plans,seven categories,
histories16/32,eight model draws and all1/4/8/16/32 horizons. Recompute conservative
ball/paddle features from cached RGB exactly. Compare every model/oracle pair,
with actual arrival validity and terminal arrival included. No nearest/best draw
is chosen. These are repeated, related pairs from33 distinct native histories,
not thousands of independent episodes. The table pools supported visible-ball
pairs; serving has more plans and support varies across horizons/categories.

|History16 visible-ball pair within1px|Coverage|Motion rollout|Supported pairs|
|---|---:|---:|---:|
|1 decision|70.66%|72.82%|15,360|
|4|45.08%|52.04%|16,896|
|8|22.45%|27.51%|11,976|
|16|20.22%|39.63%|9,232|
|32|5.90%|.29%|8,328|

The fixed first model/oracle draw separately gives53→54/187 at8,33→59/152 at16
and6→0/130 at32. Dense recorded development remains14→25/107,3→10/114,0→4/95.
Case mixes differ: do not compare absolute rates across these cohorts as an
isolated action effect. Per-category gains are mixed: at8,freeflight worsens
34.77%→23.05% and contact50.62%→26.39%,while brick14.84%→49.61%,wall37.93%→47.84%
and ending78.13%→95.83% improve. All first/all-pair counts and both histories stay
in `result.json`; unsupported cells remain untested.

These point measurements omit predicted survival weighting and are not calibrated
distribution scores. They do not replace the original equal-category16/32 energy,
event or retention gates. Reconstructed energy agrees with all140 published
model/history/category/horizon cells. The original84.05% energy regression and
23/304 non-ending false alarms remain failures. The distinction is between
partial short tracking gains and a reliable full world,not a repaired comparison.

## A verified action-no-effect slice

All eight freeflight parents qualify at1/4/8: every one of four plans/eight oracle
repeats has available nonterminal arrivals,finite identical native ball positions,
and identical supported RGB ball centers. Paddle commands differ; the ball paths
do not. The model receives only history/actions,not the emulator snapshot.

|Rendered ball detected at8,64 model images per plan|Coverage|Motion rollout|Oracle|
|---|---:|---:|---:|
|NOOP|64/64|64/64|64/64|
|LEFT|63/64|60/64|64/64|
|RIGHT|63/64|41/64|64/64|
|REVERSE|61/64|51/64|64/64|

This is observable output sensitivity,not a calibrated latent existence probability.
It can include faint/ambiguous decoder outputs. Eight model draws limit precision;
presence alone would not qualify a correctly located ball or a nonconstant model.

Mean squared action contrast of normalized[presence,x/64,y/64] features at8 is
.00120131 control/.02491886 new; paired finite-draw correction gives.00040651/
.02187400. Center contrasts across actions within each shared draw before the
correction. Negative estimates at other horizons are preserved. These internal
feature units are descriptive,not causal fractions or new acceptance gates.

## Training support is nonzero but uneven

Inspect the exact65,536 declared H16 suffixes.56,521 are complete before any
reset; they include4,989 freeflight-tagged starting states. All975,941 valid
suffix arrivals match the learning ledger. Counts below include sampler repeats
and overlapping windows,not independent histories.

|Literal16-command pattern|Complete suffixes|Freeflight starts|Last observed source_wait annotation|
|---|---:|---:|---:|
|NOOP0|5,069|96|4,941|
|RIGHT3|4,202|199|3,630|
|LEFT4|4,038|107|2,981|
|RIGHT4 then LEFT12|14|6|14|

Raw original fitting has only10 constant-NOOP16 windows and zero constant LEFT/
RIGHT16 or reverse windows. The supplement supplies most long-pattern coverage.
At32,original fitting and recorded development have none of those literal
patterns; supplement has1,139/1,064/1,066 constant NOOP/RIGHT/LEFT and four reverse
windows,with25/78/26/two freeflight starts. Raw32 behavior support is not direct
32-step supervision: the learner has16 future-image targets.

The wait annotation records a source phase and is imperfect; it is absent from
original/development files. Missing RAM references do not prove physical absence.
These are neither identical-state interventions nor a proof that every fixed
plan is out of distribution. They motivate matched fitting branches,not a claim
that coverage or action-feedback confounding explains all failures.

## Limits of the visible-ball instrument

A separately pinned post-hoc support audit retains finite-RAM/visible-RGB counts.
At32,1,041 of2,552 available oracle images have supported visible balls; brick32
has none. A lower ball-feature energy in such a cell does not validate accurate
ball trajectories there. At the first horizon, 192 finite reference locations are clearly outside
the conservative RGB ROI; at eight decisions, 183 are. Other finite references without a detection
are ambiguous and can include stale RAM state after loss. Do not replace image
presence with the assumption that every finite RAM reference is a live ball.
Full-field/late-game support remains a separate missing requirement.

## Full causal clips and execution

The saved oracle `first_native_sequence` and model `first_prediction` fields
already contain all32 arrivals. They support complete causal clips without a
new emulator/model query. Fixed lane0,oracle repeat0/model draw0,history16 cases:

|Case|Decisions / original frames|Artifact under root|
|---|---:|---|
|freeflight/NOOP|32 /128|`freeflight-noop-causal.png`|
|contact/LEFT|32 /128|`contact-left-causal.png`|
|life_loss/NOOP|32 /128|`life_loss-noop-causal.png`|

Lossless APNGs display at5fps,with decision/command and per-draw survival weights.
The oracle is native160×210 pooled RGB; RGB64 model outputs are expanded to the
same physical display aspect. Display resizing never enters model data/metrics.
Futures are independently sampled,not optimal matches. Posters use fixed step8.
All96 decoded animation frames match generated frames exactly. Original384
frames are visualized; **zero new emulator actions/frames** were executed.
`comparison.png` shows matched metrics and the action-no-effect slice.

Root `/mnt/data/kindle-breakout-world-action-gap-20261107`,source base
`b54a3699663dfca742b94615380337fc3e9532ea`.877 input hashes,644 historically verified
references,three exact qualified instrument functions,210 endpoint controls,
336 exact cached-feature controls and140 published energy cells pass. Main
analysis38.158s,coverage4.726s,clips5.154s; support audit is post-hoc and separately
pinned. Source/data checks cannot recover native argument buffers that were not
saved. Newly pinned raw outputs are current authoritative bytes; no claim that
older closures hashed every raw RGB array. No original failure is relabeled.
`closure-review.json` records CPU completion and unchanged failed quality gates.

The selected [paired fitting-data protocol](../experiments/2026-11-07-breakout-world-paired-actions.md)
prepares56 parents from fitting streams only,eight per category,56 distinct native
histories,with saved32-frame prefixes and original source transcripts. No development or oracle source enters parent selection; cross-split
identical-history exclusion was not separately tested. The first preparation writes all112 prefix/transcript
files but fails strict JSON on missing-coordinate NaNs. A CPU-only serialization
repair writes nulls,verifies112 unchanged outputs and all original predicates/
spacing/source values,without reselection or emulator work. Preserve that failure.
Its prepared replay bound is503,280 actions/2,013,120 frames,not executed counts.
The separately declared [collection](2026-11-07-breakout-world-paired-actions-data.md)
is complete: 13,068 new actions / 52,122 frames in 228.179s, with independent CPU
alignment/count review passed. Its first launcher failure occurred before
collector import and remains preserved; a separate CPU-only interpreter repair
ran the still-unstarted first collection. Nine CPU collector checks pass. No
learner is launched by this audit, and no GPU worker remains active. The full world-model goal remains
open,including precise all-horizon physics,events,brick memory,late/tunnel/second-
wall coverage,untouched families and fresh confirmation before CDP/behavior work.
