# Initial-episode guidance passes its bounded execution and cost check

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

## Full200k comparison running

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
