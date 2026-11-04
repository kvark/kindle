# Fresh200k Breakout comparison with representation KL scale0.01

Select one full comparison after the completed
[representation0.01 check](../results/2026-11-04-cdp-breakout-representation001.md).
Its8,192 actions/489 updates take47.300s; the padded200k projection21m10.262s
passes the30-minute cap. All706 pins,357 matched initial/invariant frozen
tensors,47 pre-update events,weighted world losses,full training replay and
original services pass. Preserve the earlier CPU fixture spelling failure and
repair. Do not repeat the check or transfer its experience.

Change only existing representation KL scale0.1→0.01. This reduces posterior
regularization against a stopped prior; dynamics KL1 separately fits the prior
to a stopped posterior. Both free-nat floors remain1. Aggregate retained KL
and older gradient measurements motivate the hypothesis without proving a
causal bottleneck. No new objective, parameter, native build or runtime overlay
is introduced. Optional prior-continuation,terminal-weight and H3 losses stay off.

Use native447fe50/SHA35ad0db,the corrective runtime and bounded runner73254c52.
Keep centered CDP12M64/Auto/N8/B16/T16/fullBPTT/micro16/H15/R16/context1,
base/encoder4e-5,dynamics4e-4,behavior8e-5,policy1/critic1/replay-value1,
continuation1,CDPfuture500,disagreement1,expert100/unit weights/world imitation
gradients,sampled collection/posteriors,actor-unimix.01,replay100k,full18/sticky.25/
repeat4/noops0,native pooled frames/one GPU64 resize and20% episode guidance/
cap1920. Teacher,all-state labels and actual-command feedback remain unchanged.

Start fresh learner1009 with the retained eight lane offsets. Train200k aggregate
actions/12,477 updates within1800s,first update376 and final residual credit.5.
Save actual initial,read-only100k/6,227 recovery and final checkpoints. Require
all357 initial arrays equal to bounded and retained controls,with47 matching
pre-update semantic events. Recovery retains weights/moments,not replay,games,
live beliefs or full RNG state; it is not an exact continuation checkpoint.

Evaluate final and actual initial with sampled and greedy actions,guidance and
labels off,seed4000001009. Each arm selects the first three natural episodes in
each of eight lanes,24 total,with zero updates/truncations and357 invariant
arrays. Keep600k-action/300s bounds and100k-frame episode limits. Retain excess
episodes and partial tails. Both initial semantic traces must match the control.

Require finite reports,correct cadence/label bounds,weighted world totals with
representation0.01,changed CNN/reward/continuation/actor/value,full training
pixel/action/label replay,frozen action/outcome/frame/boundary replays/videos,
source/native/ROM/host checks and curves. Record raw/floored KL and imitation
summaries on changing replay without treating them as fixed held-out measures.
Split exposure into teacher,native before cap and native after cap,reconcile
episodes/tails,and report score bands separately from verified wall successes.
Guided online returns are not competence.

Training replay cap900s; frozen replay/video500s each; main7200s and terminal
observer7500s. The independent100k observer verifies once on CPU. Keep observer
and other CPU co-host costs in actual training time without correction.

Primary evaluation is **greedy**,fixed before outcomes; sampled is secondary.
Compare retained1287.910495s training,greedy143.666667/initial2.208333 and
sampled70.750/initial1.416667. Report returns,actual training time,separate
construction,and own-initial-subtracted return per training minute. Use10,000
paired lane-bootstrap resamples,seed0,95% intervals. Favor the candidate only if
greedy return and gain/min both rise and the greedy gain/min interval excludes
zero. Check the original mean864/90%-two-wall gate directly. No checkpoint
selection,post-outcome primary switch or threshold relaxation.

Before launch,qualify helpers against bounded/retained data and the complete
comparison entry with synthetic outcomes and corrupted headers. Verify the
declared runner hash and representation-scale change. Exercise all five queued
commands before native construction,replay argv/manifest paths,recovery
schedule and cleared-invocation/null-journal handling. Preserve exact historical
source resolutions and earlier failures. Do not rewrite earlier declarations.

Full helper qualification preserves its initial exact-float sum assertion
failure: sequential and compensated totals differ by at most9.095e-13. A
1e-12 relative/absolute comparison repairs only that audit assertion. The
separate CPU suffix resumes unfinished checks; completed exposure/selection
fixtures and all GPU stages are not repeated. The original source and exit1
receipt remain pinned alongside the successful repair.

Root `/x/Temp/kindle-breakout-representation001-20261104/full`,service
`kindle-breakout-representation001-200k-20261104`. Verify an idle host before launch;
stop on real faults or nonfinite reports. No overlap,successful-stage repeat,
cap increase,sweep,second learner,extension or automatic successor. This is
one reused development seed/cohort; any promising result needs a separately
selected fresh confirmation. The five-game goal remains unfinished.
