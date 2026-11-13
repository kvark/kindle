# Fresh200k Breakout comparison with one-point pixel catch feedback

Select one full comparison after the completed
[bounded catch-reward check](../results/2026-11-04-cdp-breakout-catch-reward1.md).
Its8,192 actions/489 updates take47.580s; the padded200k projection21m17.777s
passes30 minutes. All405 current/338 retained pins,357 matching initial/invariant
frozen tensors,47 shared pre-update events,139 independently replayed catch
rewards and original services pass. Preserve both CPU preparation failures and
their separate repairs. Do not repeat the check or reuse its experience.

Change only the declared training reward: add+1 for each fixed causal pooled-pixel
catch. The qualified adapter,detector,extractor,audit and private runner25327009
remain unchanged. Raw game rewards,scores,frames,termination and frozen task
gates remain unchanged. This is an explicit game-specific training aid,not an
extra policy observation or a new native auxiliary loss. Nearer catch feedback
is a hypothesis; the prior life-loss penalty regression remains relevant and
does not establish a benefit for this different intervention.

Use native447fe50/SHA35ad0db and the qualified overlay at
`/x/Temp/kindle-breakout-catch-reward1-20261104/runtime`. Keep centered
CDP12M64/Auto/N8/B16/T16/fullBPTT/micro16/H15/R16/context1,base/encoder4e-5,
dynamics4e-4,behavior8e-5,policy1/value1/replay-value1,representation.1,
dynamicsKL1/free-nats1,continuation1,CDPfuture500,disagreement1,expert100/unit
weights/world imitation gradients,sampled collection/posteriors,actor-unimix.01,
replay100k,full18/sticky.25/repeat4/noops0,native pooled RGB/one GPU64 resize and
20% episode guidance/cap1920. Teacher,all-state labels and command feedback stay
exact. Other optional rewards,losses and boundaries remain off.

Start fresh learner1009 with retained lane offsets. Train200k aggregate actions/
12,477 updates within1800s,first update376 and final residual credit.5. Retain
actual initial,read-only100k/6,227 recovery and final checkpoints. Require all357
initial arrays equal to bounded and retained controls. The47 pre-update semantic
events must match the bounded aid run and share actions/raw outcomes with the
unshaped retained control; record permitted differences only in declared catch
fields,training rewards and stored extrinsic components. Recovery retains
weights/moments,not replay,games,beliefs or complete RNG state.

Evaluate final and actual initial with sampled and greedy actions,aid/guidance/
labels off,seed4000001009. Each arm selects the first three natural episodes in
each of eight lanes,24 total,with zero updates/truncations and357 invariant
arrays. Keep600k-action/300s bounds and100k-frame episode limits. Retain excess
episodes and partial tails. Both initial semantic traces must match the retained
control,allowing only already declared episode elapsed-time differences.

Require finite native reports,correct cadence/label bounds,unchanged native
configuration and teacher,changed CNN/reward/actor/value weights,full training
pixel/action/teacher-label/catch/reward replay,frozen outcome replays/videos,
source/native/ROM/host checks and raw-score curves. Reconcile all catch events,
raw rewards,shaped rewards and native stored extrinsics independently. Report
catch frequency and reward contribution; these are training exposure,not task
competence. Report teacher/native exposure and score bands separately from
verified wall successes. Guided online scores are not unaided competence.

Training replay cap900s;frozen replay/video500s each;main7200s and terminal
observer7500s. The independent100k observer verifies once on CPU. Retain all
observer/other CPU co-host cost in measured training time without correction.

Primary evaluation is **greedy**,fixed before outcomes;sampled is secondary.
Compare retained1287.910495s training,greedy143.666667/initial2.208333 and
sampled70.750/initial1.416667. Report raw returns,actual training time,separate
construction and own-initial-subtracted return per training minute. Use10,000
paired lane-bootstrap resamples,seed0,95% intervals. Favor the candidate only
if greedy return and gain/min both rise and the greedy gain/min interval excludes
zero. Check the original mean864/90%-two-wall gate directly. No checkpoint
selection,post-outcome primary switch or threshold relaxation.

Before launch,qualify full audit/comparison/exposure helpers against bounded and
retained data,including corrupted catch/reward declarations and native reward
inputs. Exercise all five exact queued commands before native construction,
replay argv/manifest paths,recovery cadence and terminal-state handling. Preserve
original sources/receipts; do not overwrite previous declarations or repeat
completed GPU stages. Carry forward the exact qualified runner and reward module.

Root `/x/Temp/kindle-breakout-catch-reward1-20261104/full`,service
`kindle-breakout-catch-reward1-200k-20261104`. Preparation is pending at selection;
no full learner is active yet. Verify idle live state before launch;stop on real
faults/nonfinite reports. No overlap,cap increase,sweep,extension,second learner
or automatic successor. One reused development learner/cohort cannot establish
broad reliability; any promising result needs a separately selected confirmation.
The full five-game goal remains unfinished.
