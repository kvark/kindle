# Retain encoder4e-5 with the current Breakout critic

The fresh encoder8e-5 comparison finishes200k actions in **21m13.443s**,
1.12% less training time than the retained encoder4e-5 recipe. Primary greedy
return falls **143.667→103.583 (−27.90%)**, and initial-subtracted greedy return
per training minute falls **27.52%**. Sampled return rises70.750→74.000, but
both paired intervals span zero. The locked improvement rule fails: retain
encoder4e-5, with no extension, confirmation, rate queue or automatic successor.
All final arms remain **0/24 two-wall successes**; the original864/90% gate and
five-game goal remain unfinished.

| Fixed 24-episode assessment | Retained encoder4e-5 | Candidate encoder8e-5 |
| --- | ---: | ---: |
| Greedy final / actual initial |143.667 /2.208|103.583 /2.208|
| Sampled final / actual initial |70.750 /1.417|74.000 /1.417|
| Two-wall successes, either final mode |0/24|0/24|
| Training time |21m27.910s|21m13.443s|
| Greedy gain per training minute |6.5901|4.7764|
| Sampled gain per training minute |3.2300|3.4199|

The greedy paired return difference is−40.083,95% lane interval[−88.001,23.253];
gain/min difference is−1.814[−4.055,1.152]. Sampled differences are+3.250
[−12.125,21.000] and+.190[−.534,1.026] per minute. These10,000 lane resamples,
seed0, use a repeatedly reused development cohort; they are not training-seed
reliability intervals. Neither reliable harm nor a sampled advantage is
established. The primary rule fails without needing a claim of reliable harm.

## What changed and what this means

Only encoder learning rate4e-5→8e-5 changes from the retained single-image
12M64/Auto/N8/B16/T16/fullBPTT/microbatch16/H15/R16/context1 recipe. Keep
base4e-5/dynamics4e-4/behavior8e-5, policy1/imagined-value1/replay-value1,
discount333/lambda.95, expert100/unit weights/world gradient, all-state corrective
labels,20% guided episodes/cap1920, sampled collection/posteriors, actor-unimix.01,
replay100k/warmup1000 and full18/sticky.25/repeat4/noops0. Native447fe50/SHA35ad0db,
runnercf4ea238 and teacher/schedule bytes are unchanged;16,334,353 trainable
parameters and initial values match. One GPU resize of native pooled RGB stays.

The earlier [encoder8e-5 result](2026-11-01-cdp-breakout-encoder8e5.md) improved
greedy play with behavior4e-5/replay-value.3. That improvement does not carry over
to this observed current-critic comparison. This does not isolate an interaction
mechanism or establish a universal encoder rate. Training/evaluation seeds1009/
4000001009 remain reused, and control wall time is historical. The small timing
reduction is not an isolated backend speedup. No previous weights, replay,
games, live belief, RNG or bounded-check experience enters the fresh learner.
[Fixed full protocol](../experiments/2026-11-04-cdp-breakout-value1-encoder8e5-200k.md).

Training completes200,000 actions/799,555 emulator frames/12,477 updates.
Learning occupies88.08% of measured training time; construction takes2.419s
separately. All75,262 memory snapshots pass2GiB, with minimum11.159GiB headroom.
This whole-run result supersedes the earlier single post-recovery observation.
The visually inspected three-panel curve shows guided online returns against
actions, frames and wall time; its caption reports sampled frozen74.000/1.417.
It does not turn the online curve into a frozen learning curve.

Final50k-action mask-corrected expert CE improves.560514→.548159, while primary
frozen play worsens. Actual89,562 overrides are44.781% of actions, versus
retained43.0585%;12,477 finite supervised reports contain2,989,421 repeated
labeled replay rows. The55 guided episodes average191.855, while202 unguided
training episodes average16.832; neither is the fixed frozen assessment.
Only1,313 native actions occur at cumulative scores216–431 (0.6565% of all
training); none occurs at432+. Maximum score including partial tails is422.
These score ranges are exposure counts, not verified wall events or causal
proof that coverage or imitation fit explains the remaining control gap.

## Completed verification and original closure

All120 current/338 retained inputs,100k/6227 recovery,357 finite initial tensors
matching control and bounded check,47 pre-update semantic events and357 invariant
frozen tensors per arm pass. Actual-initial sampled/greedy traces reproduce799/
794 semantic events; only finite episode wall times differ. All finite reports,
changed CNN/reward/actor/critic, full pixel/action/label training replay, frozen
outcome replays/videos, guards, curves, exposure, coverage and comparison pass.
The257-episode training replay takes108.634s. Ordinary frozen logs have no live
pixel hashes; no equality to unavailable live images is claimed.

Four frozen arms total89,424 actions/357,516 frames/89.473s, zero updates and
truncations. Each selects the first3 natural episodes per8 lanes=24; excess
natural episodes and partial tails remain retained. Sampled final/initial take
22.542/4.041s; greedy final/initial58.848/4.041s. No completed stage repeats.

MainPID1091232/invocation`0dd5518ac0c94b879959da7818f66c36` launches00:19:16.636
UTC November4/source6543be61. Training ends00:40:34.316. Original main succeeds
**00:45:07.541**, and terminal collection completes **00:45:34.600** in2.738s.
All three original services are terminal success;57 completion artifacts verify.
A separate.445s read-only summary (.514s wrapper) completes00:45:58.544,
checking original receipts,120 original/12 repair pins, artifact hashes and
report arithmetic without repeating training, evaluation, replay or plotting.
GPU descriptors are absent at closure. Check live state before future work.

Recovery100k/6227 passes00:30:02.503 in.319s. Independent verification passes
00:30:17.292 in.455s (.514s wrapper), with357 finite tensors, exact counters,
metadata and hashes,0555 directory/0444 files,120 unchanged inputs and the same
active learner identity. It makeszero model calls/updates. Recovery contains
weights and optimizer moments, not replay, games, beliefs or complete RNG.
The.375s startup audit and.013s post-recovery observation also co-host; these
and other CPU work remain in measured wall time without correction.

Root `/x/Temp/kindle-breakout-value1-encoder8e5-20261104/full` is closed. Do not
restart its learner, assessments, replay, observers, collector or summary.
Completion SHA`665f748a138bd89cba9844b52b395c9f84a47e0092ee2864110126126494ac88`;
terminal journal SHA`e39fdb383cce0c0a565c27f2377c2bddaa9df14b99068d69ce902f296dc5b439`.
The checkpoints, videos, curves, original failures and receipts remain outside Git.

## Preserved CPU preflight failure

Full helper qualification passes3.245s, including the actual comparison entry
and corrupted headers. The original queue preflight then fails before GPU work:
its wrong-rate CLI fixture correctly rejects at the configuration boundary after
a second mocked guidance object is constructed, while the copied assertion
expects one. Original source, declaration, stderr and exit1 remain unchanged.
Separate`preflight_repaired.py` expects two matching mocked objects and passes
all five actual commands, three wrong-command and four replay-input cases in
.525s (.620s wrapper), withzero real native agents/emulators/GPU descriptors.
No successful helper stage repeats. Separate repair declaration/review files
pin the failure and correction; the qualified launcher verifies them before
the sole GPU launch. No production source, command, cap or evaluation rule changes.
The successful full experiment does not erase this earlier CPU failure.

## Earlier bounded check

The [bounded protocol](../experiments/2026-11-04-cdp-breakout-value1-encoder8e5.md)
completes8192 actions/32,722 frames/489 updates in48.178s,1.02% above control.
Padded200k projection21m33.842s passes30min. Frozen1024/.920s haszero updates
and no natural episodes; this establishes execution and early cost only.
All56 current/338 retained inputs,357 finite matching initial/invariant frozen
tensors,47 pre-update events, cadence, changed weights, finite aligned reports,
full replay and guards pass. Training replay takes5.556s;117,058 repeated
labeled rows equal weight sum, maximum world/actor raw loss difference8e-7.
Actual3,733 overrides are45.5688%, not unaided competence.

CPU qualification passes5.511s (5.670s wrapper): two actual commands,33 rejection
cases,357 shapes and an actual collector fixture, withzero agent/emulator/GPU
construction; only native metadata is loaded. One historical replayer is
resolved from exact retained bytes, preserving old pins. MainPID1090844 launches
00:14:01/source507b773d and main/collector succeed00:15:09/00:15:13. No bounded
stage fails or repeats. Root`/x/Temp/kindle-breakout-value1-encoder8e5-20261104/check`
stays closed; no bounded experience transfers to the separately selected full run.
