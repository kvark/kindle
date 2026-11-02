# Breakout 25M: fresh 200k-action capacity comparison

Separately select one full development learner after the
[bounded25M check](../results/2026-11-02-cdp-breakout-capacity25m.md) closes with
all380 pins,357 schema/finite/invariant tensors,full guided replay and guards
passing. Its conservative200k projection37m59.071s fits the predeclared45-minute
ceiling. Short training costs77.95% more than12M. This allocation measures
whether a larger model earns that cost through better unaided gameplay.

Change only the preset from12M to25M at native RGB64/Auto/N8/B16/T16/fullBPTT/
microbatch16/H15/R16. Keep encoder/base4e-5,behavior8e-5,dynamics4e-4,replay-value1
with imagined bootstrap,expert100/unit weights/world gradients/all-state
corrective labels,20% episode guidance/cap1920,actor-unimix.01,action-effects1,
replay100k and warmup1000. Preserve full18/sticky.25/repeat4/noops0 and native
pooled pixels with one GPU resize. Native447fe50/SHA35ad0db,runnercf4ea238 and the
retained teacher are byte-identical; no build or production change is needed.
Runtime `/x/Temp/kindle-breakout-corrective-20261028/runtime`.

The model expands CNN depth16→24,recurrent width2048→3072,hidden/head width
256→384 and categorical classes16→24 across32 variables. Parameters grow
16,334,353→36,608,401. Several components,initial values and subsequent states
change together. The experiment cannot isolate one component or establish an
architectural cause of weak control. Seed1009/evaluation4000001009 and the
eight lane offsets are reused development inputs,not independent confirmation.
Start fresh without transferred weights,replay,belief,games,RNG or experience.

Train **200,000 aggregate actions /12,477 updates**, first update376 and final
credit.5, under **2,700 seconds**. Save the100k/6,227 recovery checkpoint and
final200k checkpoint. The original recovery observer must audit and retain the
100k files before replacement. Match all357 actual-initial tensors to the bounded
25M check and its47 pre-update transitions. Do not require equality with12M's
different architecture. No intermediate evaluation or checkpoint selection.

Run the fixed final/actual-initial sampled and greedy assessments, sampled
primary,first three natural episodes per eight lanes=24 per arm. Each arm has
600k-action/300-second limits,zero updates,guidance,labels or truncations and
357 invariant tensors. Preserve the original90%-two-wall task gate. Report
scores against the retained12M control **70.750 sampled /143.667 greedy**, own
initials1.417/2.208,in21m27.910s. Candidate own initials may differ. Report both
absolute gameplay and initial-subtracted return per training minute; higher
capacity or lower training loss alone does not justify retaining the candidate.
Keep useful tradeoffs explicit and do not claim reliability from this reused seed.

Require exact config/source/native/counter checks,finite reports and checkpoints,
changed CNN/reward/actor/value,full training pixel/action/label replay,four frozen
recorded-action outcome replays/videos,host guards,curves,actual guidance exposure,
comparison and original terminal collection. Ordinary frozen logs lack live
pixel hashes; do not claim direct live-image equality for those captures.

Keep600s inner/630s outer guided replay,500s per frozen replay and1200s total
CPU audit. Main service cap5400s; postprocessing/progress6000s with5700s waits;
recovery3300s with3000s internal wait; collection6300s with6000s internal wait.
Observers cannot restart or extend training. Preserve CPU co-hosting time without
correcting measured training duration. Preflight must exercise actual commands,
configuration rejection,observer arguments,recovery retention and resolved input
paths before launch. Pin the complete declaration and protocol snapshot.

Check live state; no overlapping GPU jobs. Stop on real faults or nonfinite
values without driver recovery. Preserve failures; inspect completed artifacts
before a separately scoped CPU repair. No retry,cap increase,extension,capacity
sweep,confirmation,automatic successor or reuse of bounded-check experience.

Root `/x/Temp/kindle-breakout-capacity25m-20261102/full`,trial
`breakout-25m-64-b16-r16-replayvalue1-expert100-200k-seed1009`.
Main `kindle-breakout-capacity25m-200k-20261102`,with matching postprocess,
progress,recovery and collection services. The five-game goal remains unfinished.

CPU preparation passes in2.484s (2.585s wrapper): five actual CLI commands,
32 audit/comparison config cases,seven actual prelaunch command cases,14 startup
header cases,four observer argument cases,ten recovery config cases and recovery
metadata/retention checks. All380 bounded/338 control pins verify,357 bounded
initial tensors match the25M schema,and18 scripts compile. The selector's actual
input list resolves414 existing files plus its six explicitly deferred outputs;
44 source path checks also pass. No native/environment construction or GPU work
occurs. Source preparation and qualification have no failures or repeated stages.
