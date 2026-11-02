# Ascending-ball centering passes the eight-episode reference screen

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
