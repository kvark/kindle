# Fresh400k Breakout comparison at replay ratio8

The [bounded R8 check](../results/2026-11-01-cdp-breakout-r8.md) passes execution,
initial/frozen tensors,full guided replay,host guards and a conservative26m01s
cost projection. Its original terminal collector succeeds13:03:34 UTC November1;
completionSHA256`9cf15b6df62474915a1e2f694820c3d6348ca45861090da72c12a5673dc95bef`.
No Kindle job remains running at selection review. Deliberately select one fresh
400k-action learner after that evidence;the full harness must pass CPU
qualification before a declaration,queue or GPU allocation is created.

The retained200k/R16 control scores56.292 sampled and104.667 greedy in21m28s,
allzero of24 two-wall successes. More fitting work under H5/R24 lowered imitation
loss without improving play. This comparison asks whether twice the experience
at roughly the same update count improves useful frozen learning. The earlier
Qbert400k/R8 failure remains relevant;there is no prior evidence of a benefit
in this Breakout regime.

Keep centeredCDP/12M64/Auto/N8/B16/T16/H15,encoder/base/actor4e-5,dynamics4e-4,
warmup1000,actor-unimix.01,action-effects coefficient1,full18/sticky.25/repeat4,
native max-pooled pixels and one GPU resize. Keep expert100/unit action weights/
world gradients,all-state corrective labels,20% episode guidance/cap1920,
native447fe50,exact runner6c1b69fb and teacher. Change train ratio16→8 and
experience200k→400k together. Cadence,sampling,warmup duration and exposure also
change;this does not isolate data coverage. Learner1009/evaluation4000001009
are reused development seeds,not independent confirmation. No prior weights,
replay,belief,games,RNG or bounded-check experience are reused.

Train400,000 aggregate actions with a **35-minute cap**,12,489 updates,first
update376 and residual credit.25. Require recovery at100k/3,114,200k/6,239 and
300k/9,364 updates. Audit and preserve all four checkpoint files at each recovery
before the next overwrite,with matching metadata/hashes;do not restore or repeat
training to obtain them. Checkpoints omit replay,belief,games and RNG,so they
are not uninterrupted-training recovery. Require357 finite initial tensors
matching control/check,47 matching pre-update events,finite supervised reports,
changed CNN/reward/actor/value,exact config/cadence,expert row/weight accounting
and aligned raw world/actor losses. Require full training pixel/action/label
replay and host guards.

Assess final and actual-initial checkpoints with sampled and greedy actions.
Each arm has300s/600k-action caps and selects the first three natural episodes
per eight lanes=24. Guidance/labels are off;requirezero updates and selected
truncations,357 invariant tensors,paired initial semantic traces and complete
recorded-action/outcome/frame/reset replays/videos. Ordinary frozen logs lack
live pixel hashes;do not claim comparison against unavailable live images.
Preserve the90%-two-wall task gate. Sampled remains primary;report greedy
separately,absolute means and own-initial-subtracted return per training minute.
Do not switch primary mode or choose checkpoints after observing outcomes.

Main service cap5400s includes training,four frozen arms and CPU audit1800s;
full guided replay cap1200s. Postprocessing/collection limits6000s/6300s.
Observers monitor only the original process and never start or restart GPU
work. Collect curves,coverage,exposure,paired comparison and original terminal
states. Retain co-host timing without wall-time correction. Stop on real GPU
faults or nonfinite reports;no driver reset/recovery. Preserve original failures
if a separate CPU repair is needed;never repeat a completed GPU stage.

CPU qualification must exercise all five actual CLI paths,the intended config
and schedule changes,357 shapes,recovery audit/retention cases,observer parsing,
source paths and exact native/runner/teacher identity without constructing a
native agent or emulator. Reuse passed bounded qualification;do not rebuild.
No extension,cap increase,ratio retry,confirmation or automatic successor is
selected. Full frozen benefit remains unmeasured.

Prospective root `/x/Temp/kindle-breakout-r8-20261101/full`,trial
`breakout-12m-64-b16-r8-encoder4e5-expert100-400k-seed1009`,service
`kindle-breakout-r8-400k-20261101`. No full trial,queue or allocation exists at
this protocol's selection. Check current live state before declaration/launch.
