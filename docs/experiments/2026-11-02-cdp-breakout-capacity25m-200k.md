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

## Original learner launched

Main starts **10:47:21.939 UTC November2**, source`0e69b83`,PID933689/invocation
`d3457431b99a41aebd2ec054df3f4d7d`. DeclarationSHA1985ac2c pins420 inputs;
the actual prelaunch command check passes before the original service starts.
Postprocessing,progress,recovery and collection observers are active under their
original identities. Do not restart or overlap this learner.

The1.707s startup wrapper passes all420 pins,exact config,cadence and357 initial
tensors matching the bounded25M checkpoint. All47 pre-update transitions match;
163 finite supervised reports appear through2,984 actions. No12M initial-value
equality is claimed. Startup review is retained in the trial; frozen gameplay
is unmeasured and the fixed200k/45-minute budget remains unchanged.

Read-only recovery and success-only terminal readers are prepared under the
package root. The terminal reader's five score cases pass in.087s CPU/.105s
wrapper, preserving the original active main identity. Reader sources are
SHA9f5abf2d (recovery) /3bdfcaae (terminal),with a separate provenance receipt.
Run recovery confirmation only after the original100k audit and retention
complete; run the terminal reader only after all original services and full
comparison succeed. At startup neither reader has executed its final stage.
Retain CPU co-host durations without wall-time correction. No learner or GPU
failure or stage repeat occurs.

## Retained halfway checkpoint

The original recovery observer succeeds **11:05:29.736 UTC November2**, under
PID933835/invocation`b8e295c002c64f3eb4287187c5254b98`. Its100k/6,227 checkpoint
passes357 finite tensors and retains461,432,010 bytes across metadata,world,
behavior and slow-value files. The complete log prefix contains6,296 finite
reports through101,112 actions. All420 declaration pins remain unchanged.

The prepared read-only recovery confirmation runs once and passes
**11:06:26.188 UTC**, in1.492s CPU/1.510s wrapper. It verifies the original
observer's single audit payload,terminal success,all retained hashes and exact
metadata/configuration. Original main933689/d3457431b99a41aebd2ec054df3f4d7d
remains active before/after; no observer,learner,replay or GPU stage repeats.
Retained recovery omits replay,belief,games and RNG. No frozen competence is
measured. Keep CPU co-host duration without correcting training time.

Recovery reviewSHA41839203,retentionSHA9e8a40ce,metadataSHAd3e6d8ff; full hashes
and confirmation are in the trial's `recovery-confirmation.json`. Do not repeat
this reader. Terminal and saved-log coverage final stages remain pending original
full success; fixed200k/45-minute limits and all assessment gates stay unchanged.

## Saved-log coverage follow-up

Prepare one CPU-only coverage reader while the original learner runs. Execute
its final stage only after the original full comparison and terminal reader
succeed. Read this learner's training and four frozen logs once, using the
qualified cumulative-frame parser. Reuse the completed12M control from
`kindle-breakout-score-coverage-20261102/review.json`, `trials.candidate`,
without recounting its trajectories or repeating any emulator replay.

Keep pre-action score bins0–215,216–431,432–647,648–863 and864+. Reconcile
completed episodes,partial tails,resets,rewards and cumulative frame differences;
count actual teacher overrides and native actions separately. Frozen summaries
use the original first three natural episodes per lane; extras and tails remain
accounting evidence. Score fractions are not verified wall boundaries or
equivalent states, and exposure does not isolate a capacity/coverage mechanism.

Validate the actual25M configuration against12M,including rejection cases, and
check the parser's200k/12,477 counter assertion before outcomes. Retain the five
qualified boundary cases without repeating them. Pin the reader,helper,protocol,
prior evidence and active declaration; preserve original process identity and
all420 active pins before/after preparation. Use one low-priority CPU under60s,
without native/emulator imports,GPU work or updates. Record co-host duration
without correcting training time. Root is the existing25M package; output
`score-coverage-review.json`. No new learner,guidance change,sweep,extension or
automatic successor is selected by this diagnostic.

The first coverage preflight fails10:55:41.798 UTC in1.397s because its new AST
selector matches both training and frozen learner-counter assertions. No
coverage declaration or analysis output is created. Preserve the original
reader,generator,protocol snapshot and failed receipt. The learner remains the
original933689/d3457431b99a41aebd2ec054df3f4d7d.

A separate reader correction restricts that fixture selector to the assertion
containing both `run_step` and `learner_updates`; parsing,metrics,thresholds,
selection and actual counters remain unchanged. Qualify the corrected reader
against the same six config/three counter cases,without repeating old trajectory
analysis or boundary fixtures. Use `review_score_coverage_repaired.py` and
`score-coverage-repair-declaration.json`. The final stage remains pending original
full success and writes the new `score-coverage-review.json` once. Preserve the
CPU co-host cost and all original inputs; no learner/observer/GPU stage changes.

Corrected coverage qualification passes **10:57:05.358 UTC November2** in2.673s
(2.699s wrapper): six actual configuration cases,three actual counter cases,
five retained boundary fixtures without repetition,and all420 active pins.
Original main933689/d3457431b99a41aebd2ec054df3f4d7d stays active. Nine unchanged
reader functions match their original AST; parser,metrics and thresholds remain
byte-identical. Original failureSHA2e759ebe is retained; repaired readerSHAfa452cfc.
Do not repeat either preflight or run the failed reader. Final coverage analysis
remains unexecuted,conditional on original full success and terminal confirmation.

## Completed comparison and readers

The original learner completes200k actions/799,572 frames/12,477 updates in
36m08.795s. Frozen sampled49.958/greedy104.292 are below12M70.750/143.667 by
29.39%/27.41%, for68.40% more training time. Own initials1.458/1.583 differ
from12M as expected; all0/24 two-wall. Initial-subtracted return/min falls
58.46%/56.88%. Retain12M and the existing recipe; no extension,confirmation,
capacity sweep,retuning or automatic successor is selected.

All420 pins,100k retained recovery,357 bounded-matching initial/invariant frozen
tensors,47 pre-update transitions,finite reports,full replays/videos,guards,
curves,exposure and comparison pass. Main/post/collection succeed
11:28:52.929/11:29:26.592/11:29:35.005 UTC November2. Allfive original services
are terminal success. Terminal confirmation completes11:30:31.187 in3.019s;
repaired coverage completes11:30:39.622 in3.214s with55 diagnostic/420 full pins,
290,216 actions/1,160,239 frames reconciled and no repeated trajectory analysis.

Native216–431 actions fall1,262→307,with none at432+. These score fractions are
not verified walls or causal coverage evidence. Expert CE improves.561→.499
while frozen scores fall. All original coverage-preflight failure evidence is
preserved. No original learner,observer or GPU stage repeats; no original
full-pipeline failure or cap change occurs. CPU co-host costs remain uncorrected.

Recovery,terminal and coverage readers are complete; do not repeat them. No
Kindle allocation is active at closure; check live state. See the
[completed result](../results/2026-11-02-cdp-breakout-capacity25m.md) for exact
scores,limitations and artifact hashes. Breakout and the five-game goal remain
unfinished.
