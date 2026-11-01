# Fresh Qbert H5/R32 comparison with greedy collection

Select one fresh200k-action comparison after the completed bounded check.
Its82.177s training/36m47s conservative projection passes45min;all479 pins,
357 initial/invariant tensors,47 pre-update events,full replay and guards pass.
Main and collector finish12:07:15/12:07:53 UTC November1. No other learner is
active at selection. Passing the short check establishes execution/cost only.

The current H5/R24 recipe passes15k/90% on8039 but misses on10007,despite both
greedy policies reaching the first pyramid at the same frame inall24 episodes.
Their later-play gap motivates testing extra replay work on10007. Earlier R32
with different rates/collection was mixed;this combination is unmeasured.
The hypothesis is that additional world/expert/behavior updates improve later
control at an acceptable cost,not that the phase diagnostic proves too few
updates caused the gap. This experiment belongs to the existing five-game goal.

Change only replay ratio24→32 as a learner setting. Keep centeredCDP/12M48/
Auto/N8/B16/T16/H5,encoder/base/actor4e-5,dynamics4e-4,greedy live collection,
sampled posteriors,actor-unimix0,disagreement1,expert100/all18 weights1/world
gradients,all-state corrective labels,20% episode continuation guidance capped
at1920,full18/sticky.25/repeat4 and one GPU resize to48 from native max-pooled
frames. Use the unchanged qualified447fe50 runtime,runneraba4300d,teacher and
guidance schedule. No build or production learning-code change.

Fix reused learner10007/evaluation4000010007. Start from scratch without old
weights,replay,beliefs,game state,RNG or bounded-check experience. Training must
complete200,000 actions/24,954 updates under2700s (45 minutes),first update376,
one update every8 subsequent actions andzero residual credit. Require100k/
12,454 recovery. R24→32 increases scheduled world/expert/imagined work by about
one third and changes cadence,sampling and warmup action duration together.
Reusing the failed seed/cohort is development selection,not independent seed
confirmation or an isolated mechanism test.

Assess the fixed final and actual-initial checkpoints with sampled and greedy
actions,first three natural episodes per eight lanes:24 per arm,zero guidance,
labels,updates and truncations. Retain all excess episodes and partial tails.
Each arm keeps300s/600k actions. **Greedy is primary** against the retained
10007's14,339.583;sampled control13,761.458 remains separately reported. The
original mean15,000/90%-first-pyramid gate is unchanged. Report both absolute
quality and own-initial-subtracted return per training minute;more updates do
not imply faster training or better learning per minute. A passing outcome
would be one development learner,not reliability.

Require357 control/bounded-matching initial tensors,47 matching pre-update
events,finite supervised reports,changed CNN/reward/actor/value,357 invariant
tensors per frozen arm,matching actual-initial semantic traces,full training
pixel/action/label replay,frozen outcome/frame/reset replays/videos,host guards,
curves,coverage,exposure and paired comparison. Ordinary frozen logs lack live
pixel hashes;do not claim unavailable live-image equality. Online guided return
and expert training loss are not unaided competence.

Full CPU qualification passes3.400s:five actual CLI paths,24 config/eight mode
cases,14 startup/four observer/ten recovery cases,50 paths,15 scripts,479 bounded
and442 control pins. Plot-environment import passes.085s. Two preparer wording
assertion failures precede successful generation and remain with exact sources;
no full files or GPU work were produced by those failed attempts. The bounded
invalid-seed preflight failure is also retained. No passing check is repeated.

Root `/x/Temp/kindle-qbert-h5-r32-greedy-20261101/full`,trial
`qbert-12m-48-b16-h5-r32-encoder4e5-greedy-expert100-200k-seed10007`,service
`kindle-qbert-h5-r32-greedy-200k-20261101`. Launch requires all qualified inputs,
a clean source commit and an idle host. Main service5400s,training replay1200s,
enclosing audit1800s;original progress,recovery,postprocess and terminal-collection
observers run independently. Preserve any failure;repair only unfinished CPU
work without repeating completed GPU stages. No retry,extension,cap increase,
checkpoint selection,ratio sweep,confirmation learner or automatic successor.

[Bounded result](../results/2026-11-01-cdp-qbert-h5-r32-greedy.md),
[retained control](../results/2026-11-01-cdp-qbert-h5-r24-seed10007.md),
[later-play diagnostic](../results/2026-11-01-cdp-qbert-h5-r24-phases.md).
