# Ascending-ball centering fails the expanded reference screen

The unchanged candidate scores **233.208 versus 225.083 (+3.61%)** over the
fixed24 natural episodes, with **four of eight lane means improving**. Both
controllers have **zero of24 two-wall successes**. It fails both the predeclared
10% mean-gain and five-improved-lane criteria. Retain the original teacher;
no fresh cohort, native integration, learner, target or velocity-window sweep
is selected. The positive eight-episode screen remains recorded below.

| Lane | Retained teacher mean | Ascending-centering mean |
|---|---:|---:|
|0|190.333|127.667|
|1|96.667|256.333|
|2|184.667|260.333|
|3|221.000|121.333|
|4|345.000|420.667|
|5|363.667|277.333|
|6|99.000|203.000|
|7|300.333|199.000|

The [expanded protocol](../experiments/2026-11-02-cdp-breakout-ascending-center-24.md)
keeps the policy and response-model bytes unchanged and selects the first three
natural episodes per lane. The eight earlier first episodes reproduce exactly,
including16,425 semantic events with input/output pixel hashes, actions, policy
debug information, rewards, frames, resets and outcomes. This is an overlapping,
heavily reused development cohort, not an independent confirmation. The native
teacher and original90%-two-wall task gate remain unchanged.

Collection covers **60,704 actions /242,767 frames in35.990s**. All selected
episodes terminate naturally; extra episodes and partial tails are excluded
from the score. Full recorded-action and exact live-pixel replay, recomputed
policy/debug checks and video pass in45.402s. The original retained24-episode
control is reused without rerunning it. Two new first-episode-prefix fixtures
pass; the13 completed policy fixtures are reused without repetition.

Preflight passes at06:03:46.172 UTC November2 in11.928s. The controller completes
at **06:06:24.338**, in87.435s, sourcef8e4931. All31 diagnostic inputs and1,046
active Qbert inputs remain unchanged, preserving original mainPID901816 and
invocation976a1ed5007f4ce8b343732f91c38cf9 before and after. No failed or repeated
stage, native module import, GPU work, learner update or RAM input occurs.
Retain the CPU co-host durations without correcting Qbert training time.

Root `/x/Temp/kindle-breakout-ascending-center-20261102/standard24` is complete.
Do not repeat its preflight, selection, collection or replay. This rejects this
fixed controller change under its development criterion; it does not isolate
Kindle's architectural limitation or prove that earlier positioning cannot help.

| Artifact | SHA256 |
|---|---|
| Declaration |`150b2ea3abb7b1a9cdb41b2a4d3748396f63c8f6214479e530bbe6aa66d03c2e`|
| Replay review |`3b2fdc1ec5debef542fe07789cc1fe6646a6d96c1339e6cba2487489883f30d8`|
| Comparison |`5207acd49bbd3d885687bf885dd0fe8276f4f2daf0ea77e8ca4e9beb1444c82d`|
| Completion |`973d8ca708a94580c2fe9fa051fd6d076fef7e6e14b96973911f7418e88c5426`|

## Earlier eight-episode screen

The fixed CPU candidate raises mean return **185.375→272.375 (+46.93%)**,
with **six of eight lanes improving**. Both controllers have **zero of eight
two-wall successes**. This passes the predeclared10%-mean/five-lane development
screen. It supports a separately selected24-episode assessment,not integration
into the native teacher or a new learner. The five-game goal remains unfinished.

| Lane | Retained teacher | Ascending centering |
|---|---:|---:|
|0|151|91|
|1|97|138|
|2|256|433|
|3|226|148|
|4|204|339|
|5|331|384|
|6|82|397|
|7|136|249|

The [fixed change](../experiments/2026-11-02-cdp-breakout-ascending-center.md)
sets the existing five-step planner's target to81.5 only when visible ball
motion satisfies−16≤dy<0 and|dx|≤16. Descending interception,missing-object
fallback,stationary or implausible motion,detector,response model,edge reset,
history and costs remain unchanged. The native teacher files are untouched.
The change alters positioning and subsequent visited states; the result does
not isolate a failure mechanism in either the teacher or Kindle.

All selected episodes terminate naturally. Collection covers22,712 actions /
90,835 emulator frames in14.397s; complete exact pixel/action/debug/reward/
frame/reset/task replay and video take17.725s. Thirteen pure CPU phase-boundary
and unchanged-branch fixtures pass. Preflight records5.999s and prints6.991s
including final hashing. All27 declared inputs and1,046 active Qbert inputs
remain unchanged. Original QbertPID901816/invocation
`976a1ed5007f4ce8b343732f91c38cf9` is preserved before/after.

The controller completes in38.005s at **06:00:50.246 UTC November2**,
sourceac3aacc. No failed or repeated stage,native module import,GPU work,learner
update or RAM input occurs. Keep CPU co-host durations without correcting Qbert
training time. The retained eight-episode control is reused without replay.

This is a hand-written controller on a heavily reused development cohort.
The first eight episodes alone do not establish independent reliability or
Kindle learning. The original90%-two-wall task gate is unchanged. No target,
velocity-window sweep,teacher integration or automatic successor is selected.

Root `/x/Temp/kindle-breakout-ascending-center-20261102` is complete. Do not
repeat its preflight,selection,collection or replay. A separate expanded
assessment must keep the exact policy bytes and disclose the overlapping cohort.

| Artifact | SHA256 |
|---|---|
| Declaration |`500897f03c0e38fb805bf1339a85dd080e627bc9f0d0e76980f945c6cda63a6b`|
| Replay review |`b679bf30f358e519d9a165112159506cc6f46024dda82d0078cd59ab7e2d7686`|
| Comparison |`b6356c3a58cca728bca3decdad9b8197651ee1f9ddb835e90cc48b5f1d30f108`|
| Completion |`e62be9e3fde251abbc8040fdc4ed525f19ae9390e682a785e3e1f9e5874b1629`|
