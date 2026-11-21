# Initial-episode guidance: measured Breakout learning comparison

The [full200k comparison](#full-200k-comparison) fails retention.
Greedy114.000/sampled49.125 in21m28.245s
compare with143.667/70.750 in21m27.910s. Primary own-initial gain/min falls
20.993%. Retain the original20% schedule and143.667 greedy/70.750 sampled.
All declared full-run audits pass; the five-game goal remains unfinished.

The [declared schedule](../experiments/2026-11-06-cdp-breakout-initial-guidance.md)
forces each lane's first two natural episodes, retaining the1,920-action cap,
then resumes20% Bernoulli episode selection. Every episode consumes its normal
random draw. The existing pixel teacher, all-state labels/actual-action feedback,
RGB64 architecture, losses and learning rates remain.

CPU schedule, replay and CLI checks pass. Two CLI fixture attempts initially
omit the mocked passive-label method argument and action vocabulary; their logs,
sources and failed receipts remain. The corrected fixture plus affected block
tests pass41 checks in.13s. Earlier runs also pass the existing schedule, vector,
corrective and minimal-vocabulary checks; no production bug was found by those
two fixture failures. Actual train/restore commands reach mocked construction
boundaries with zero native/emulator attempts. Independent event recount tests
asynchronous phase exit, cap release and four forged-log rejections.

Source a92643db reuses native7704d25c with exact current Python wrappers and the
new guidance module/runner; no native build. Original256 source pins resolve,
with the two changed Python files resolved to their exact historical copies.
The retained teacher's three policy class ASTs match. Both original RGB96 native
fixture failures and their separate repairs remain unchanged.

One fresh seed1009 run completes **8,192 actions /32,768 emulator frames /
489 updates in46.584001s**, versus retained47.689684s. Construction takes2.540136s.
First update is376; final debt.5. A separate1,024-action/4,096-frame greedy
restore takes.872280s with zero updates and all357 tensors invariant.
All357 initial tensors match the retained RGB64 checkpoint exactly. Weighted
world/behavior losses, finite reports, schema/count16,334,353 and changed
CNN/reward/continuation/actor/critic parameters pass.

All eight lanes remain in their first episode:1,024 overrides each,100% of
training actions, no cap release and no phase exit. This short execution does
not exercise the live two-to-three boundary; the independent CPU checks do.
Full training replay verifies every action, pixel hash, label, reward, frame
and boundary. Frozen action/outcome replay and both host guards pass.

The padded200k projection is1,251.035s (20m51s), below the declared30-minute cap.
Minimum observed native headroom is17,988,628,480bytes; subtracting the entire
5,939,200,000-byte full replay leaves12,049,428,480bytes (11.22GiB), above2GiB.
These are feasibility projections, not a measured full-run speedup or competence.
Changed experience and historical timing prevent isolated performance attribution.

Controller1511679 and both original GPU children1511686/1511857 terminate with
exit0. Execution completes15:49:08.964 UTC November6; independent closure at
15:50:31.393 verifies302 unchanged pins and no Kindle service/render holder.
No GPU/audit stage fails or repeats. Root
/x/Temp/kindle-breakout-initial-guidance-20261106/bounded is closed.

The separately declared [fresh200k comparison](../experiments/2026-11-06-cdp-breakout-initial-guidance-200k.md)
tests whether improved initial experience earns more frozen return per minute.
It changes guidance timing, amount and state distribution together. No bounded
experience transfers. Retain143.667 greedy/70.750 sampled pending that evidence;
original864/90%-two-wall and the five-game goal remain unfinished.

Full comparison helpers, independent schedule recount, complete comparison entry,
plotting and negative loss/header fixtures pass in3.872s with no native/emulator
construction. Exact queued-command qualification and launch follow the separate
full protocol; no GPU allocation has started at this declaration.

## Full-run declaration and startup

All five queued commands pass before native construction in.381s. Execution
sourcef797ad78 starts15:55:04.816 UTC November6; original main PID1512125,
invocationcfe8a3d86199424d954c680e213badb9, collector1512128 and recovery
observer1512131. First live recount at15:56:19.964 verifies12,104 actions/733
updates,374 unchanged pins and all original live service identities. Startup
confirms357 initial tensors equal retained/own bounded and47 pre-update events
equal the own bounded trace. The first lane is in episode2; all others are in
episode1, with no cap release or phase exit yet. Full phase coverage,100k recovery
and frozen learning results remain pending; online guided returns are not
competence. Original services continue without any restart or stage repeat.

The independent15:58:45.001 recount verifies35,240 actions/2,179 updates and
all eight phase exits at aggregate actions[22,232,28,376,28,496,26,816,23,664,
30,344,25,232,29,912]. Initial-phase actions total26,884 and later-phase actions
8,356;257 actions have occurred after selected-episode caps. All374 pins and
original learner/collector/recovery identities pass. This confirms live schedule
coverage, not learning quality; frozen evaluation remains pending.

The100k/6,227 checkpoint passes the learner's observer at16:05:44.146 UTC and
the independent CPU verifier at16:06:05.370. The latter takes.351s, checks357
finite tensors/51,502,659 elements, exact schemas/config/counters and0555/0444
directory/file modes, and verifies374 pins with the original active main identity.
It executes zero model forwards or updates. Recovery observer1512131 terminates
successfully; the original learner and collector continue. Recovery retains
weights and moments, not replay, games, live beliefs or complete RNG state.


## Full 200k comparison

The [fixed protocol](../experiments/2026-11-06-cdp-breakout-initial-guidance-200k.md)
uses one fresh development learner, seed1009/evaluation4000001009, execution
sourcef797ad78/native7704d25c. No bounded experience transfers. The full allocation
fails retention; the original864/90%-two-wall gate
remains unmet.

| Measurement | Retained20% episodes | Initial two episodes, then20% |
|---|---:|---:|
| Actions / updates |200,000 /12,477|200,000 /12,477|
| Emulator frames |799,595|799,655|
| Training time |21m27.910s|21m28.245s|
| Greedy final / actual initial |143.667 /2.208|114.000 /2.208|
| Sampled final / actual initial |70.750 /1.417|49.125 /1.417|
| Greedy own-initial gain/min |6.590132|5.206697|
| Sampled own-initial gain/min |3.230038|2.222016|
| Final greedy / sampled two-wall successes |0/24 /0/24|0/24 /0/24|

Training time increases 0.026% and primary gain/min falls 20.993%.
The paired eight-lane bootstrap uses10,000 resamples, seed0:

| Candidate minus retained | Mean |95% interval |
|---|---:|---:|
| Greedy return |-29.666667|[-87.916667,25.666667]|
| Greedy gain/min |-1.383435|[-4.096855,1.194018]|
| Sampled return |-21.625000|[-37.333333,-7.123958]|
| Sampled gain/min |-1.008022|[-1.739736,-0.332526]|

These intervals describe the reused evaluation cohort, not training-seed
reliability. The primary interval spans zero; the lower point estimate is not
evidence of a reliable population regression. Four of eight greedy lane
differences are positive. The prespecified retention criterion still fails.
Guidance timing, quantity and state distribution change together;
this comparison does not isolate an early-experience mechanism.

### Schedule coverage and validation

All eight lanes leave the initial phase at aggregate actions
[22232, 28376, 28496, 26816, 23664, 30344, 25232, 29912]. Initial-phase actions total
26,884; later-phase actions total173,116. Actual overrides total103,444
(51.722%) versus retained43.0585%. Selected episodes reaching
their cap total16; actions after
those caps total1,233. Independent event recount and
full native-pixel/action/label/reward/frame/boundary replay agree. Teacher labels
remain available on every state; guided online returns are not frozen competence.

Training completes233 episodes plus eight partial tails. Maximum score including
tails is422;26 episodes/tails reach216 and none reach432. Pre-action score-band
counts are192,162 below216 and7,838 from216 to432, with no higher-score actions.
Of those7,838 actions,6,605 are teacher-controlled and1,233 native-controlled.
These are cumulative-score bands, not wall events. More initial guidance and
51.722% total overrides still leave higher-score training coverage absent.

All357 initial tensors equal retained and own bounded checkpoints;47 pre-update
events equal own bounded. Retained training actions intentionally differ.
Both frozen-initial semantic traces equal retained, including sampled/greedy
means1.417/2.208. Every final/initial frozen arm selects24 natural episodes with
zero updates/truncations and357 invariant tensors. Recorded-action replays,
task outcomes and videos pass; frozen logs do not contain live pixel hashes.

All12,477 finite reports, cadence/debt, weighted world/behavior loss accounting,
16,334,353 parameters, unchanged schemas/actor sharing and changed CNN/reward/
continuation/actor/critic weights pass. Training audit takes
110.806s. The100k/6,227 recovery retains357 finite tensors/
51,502,659 elements and read-only0555/0444 files; its independent verifier passes
at16:06:05.370 in0.351s with374 unchanged pins
and original live identity. It performs no model forward/update. Recovery omits
replay, games, live beliefs and complete RNG state.

### Runtime and closure

Training finishes at 16:16:37.276 UTC November 6. Recorded component times are:

| Component | Seconds |
|---|---:|
| Learning |1,144.169|
| Action selection |63.080|
| Environment |62.364|
| Observation |12.379|
| Resets |1.004|
| Checkpoint submission |0.572|

Construction is separate at 2.517386s. CPU observer time stays in the measured
training wall; component accounting does not isolate GPU utilization.
One read-only device snapshot at106,496 actions/full100k replay records
11,982,360,576 free VRAM bytes (11.159GiB); it is not a measured minimum.

Original main1512125/invocationcfe8a3d86199424d954c680e213badb9 completes
16:20:30.454; collector1512128 completes16:20:37.393.
Recovery observer1512131 is successful. Closure16:21:18.893
verifies original terminal services and all five GPU children,374 current/
338 retained pins and56 completion artifacts. No Kindle
service/render holder remains; no full GPU/audit stage fails or repeats.
The two incomplete CLI fixture attempts and prior native fixture failures remain
preserved with separate repairs.

Retain the original20% schedule and143.667 greedy/70.750 sampled. No extension, repeated stage,
coefficient/schedule sweep or automatic successor is selected. Useful checkpoints,
logs, replays and the reused native package remain outside Git. No build cache
was created for this Python-only schedule change. The five-game goal remains
unfinished.
