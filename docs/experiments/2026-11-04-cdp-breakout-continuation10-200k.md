# Fresh200k Breakout comparison with continuation loss scale10

Select one full comparison after the completed
[continuation10 check](../results/2026-11-04-cdp-breakout-continuation10.md).
Its8,192 actions/489 updates take46.764s; the padded200k projection20m55.877s
passes the30-minute cap. All699 pins,357 matched initial/invariant frozen
tensors,47 pre-update events, weighted world losses, full training replay and
original services pass. Do not repeat the check or transfer its experience.

Change only existing world continuation BCE scale1→10. This weights terminal
and nonterminal rows and shared-world gradients; it is not class balancing or a
tenfold optimizer-step claim. Terminal-state errors motivate the test without
proving the cause. Intermediate life losses remain nonterminal. No H3, new
objective, parameter, native build or runtime overlay is introduced.

Use native447fe50/SHA35ad0db, the corrective runtime and bounded runner74af8941.
Keep centered CDP12M64/Auto/N8/B16/T16/fullBPTT/micro16/H15/R16/context1,
base/encoder4e-5, dynamics4e-4, behavior8e-5, policy1/critic1/replay-value1,
disagreement1, expert100/unit weights/world imitation gradients, sampled
collection/posteriors, actor-unimix.01, replay100k, full18/sticky.25/repeat4/noops0,
native pooled frames/one GPU64 resize and20% episode guidance/cap1920. Teacher,
all-state labels and actual-command feedback remain unchanged.

Start fresh learner1009 with the retained eight lane offsets. Train200k aggregate
actions/12,477 updates within1800s, first update376 and final residual credit.5.
Save actual initial, read-only100k/6,227 recovery and final checkpoints. Require
all357 initial arrays equal to bounded and retained controls, with47 matching
pre-update semantic events. Recovery retains weights/moments, not replay, games,
live beliefs or full RNG state; it is not an exact continuation checkpoint.

Evaluate final and actual initial with sampled and greedy actions, guidance and
labels off, seed4000001009. Each arm selects the first three natural episodes in
each of eight lanes,24 total, with zero updates/truncations and357 invariant
arrays. Keep600k-action/300s bounds and100k-frame episode limits. Retain excess
episodes and partial tails. Both initial semantic traces must match the control.

Require finite reports, correct cadence/label bounds, weighted world totals with
continuation10, changed CNN/reward/continuation/actor/value, full training
pixel/action/label replay, frozen action/outcome/frame/boundary replays/videos,
source/native/ROM/host checks and curves. Report continuation fitting summaries
without treating them as calibration. Split exposure into teacher, native before
cap and native after cap, reconcile episodes/tails, and report score bands
separately from verified wall successes. Guided online returns are not competence.

Training replay cap900s; frozen replay/video500s each; main7200s and terminal
observer7500s. The independent100k observer verifies once on CPU. Keep observer
and other CPU co-host costs in actual training time without correction.

Primary evaluation is **greedy**, fixed before outcomes; sampled is secondary.
Compare retained1287.910495s training, greedy143.666667/initial2.208333 and
sampled70.750/initial1.416667. Report returns, actual training time, separate
construction, and own-initial-subtracted return per training minute. Use10,000
paired lane-bootstrap resamples, seed0,95% intervals. Favor the candidate only if
greedy return and gain/min both rise and the greedy gain/min interval excludes
zero. Check the original mean864/90%-two-wall gate directly. No checkpoint
selection, post-outcome primary switch or threshold relaxation.

Before launch, qualify helpers against bounded/retained data and the complete
comparison entry with synthetic outcomes and corrupted headers. Verify the
declared runner hash and continuation-scale changes. Exercise all five queued
commands before native construction, replay argv/manifest paths, recovery
schedule and cleared-invocation/null-journal handling. Preserve exact historical
source resolutions and earlier failures. Do not rewrite earlier declarations.

Root `/x/Temp/kindle-breakout-continuation10-20261104/full`, service
`kindle-breakout-continuation10-200k-20261104`. Verify an idle host before launch;
stop on real faults or nonfinite reports. No overlap, successful-stage repeat,
cap increase, sweep, second learner, extension or automatic successor. This is
one reused development seed/cohort; any promising result needs a separately
selected fresh confirmation. The five-game goal remains unfinished.
