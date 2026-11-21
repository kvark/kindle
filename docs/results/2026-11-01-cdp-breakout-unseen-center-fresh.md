# Breakout centering fails the fixed screen on a fresh cohort

Keep the qualified native teacher. The unchanged centering reference scores
**212.042 versus209.833 (+1.05%)** on the predeclared fresh evaluation cohort,
with **four of eight** paired lane means improving and **zero of24 two-wall
successes** for both controllers. This fails the fixed10%-mean/five-lane screen.
The earlier19.88% development gain does not carry over at the declared threshold.
No teacher integration,learner,parameter retry or further cohort sweep follows.

This is a fixed hand-written reference comparison,not a learned Kindle policy
or independent learner confirmation. It narrows the evidence for this particular
centering change;it does not rule out useful earlier positioning or other control
strategies. The five-game goal remains unfinished.

## Fixed comparison and complete episode scores

The [protocol](../experiments/2026-11-01-cdp-breakout-unseen-center-fresh.md)
declares seed4000012011 before collection,with lane seeds
`(seed + lane * 1000003) mod 2^32`. Both policy/model copies are byte-identical
to their completed old-cohort assessments. Only the candidate's missing-ball/
visible-paddle branch centers at81.5 using the existing stopping estimate,
deadband4 and LEFTFIRE/RIGHTFIRE/FIRE. Visible-ball planning,detector,response
model,edge reset and all costs stay fixed. No native teacher bytes change.

Both arms use identical runner bytes,full18 actions,sticky.25,repeat4,zero reset
noops,native pooled RGB and the original100,000-frame episode cap. Select the
first three natural episodes per eight lanes;retain all extra episodes and tails.
Both complete24 selected natural episodes withzero truncations under the fixed
600k-action/180s collection/360s subprocess caps. The original90%-two-wall
task gate is unchanged and fails for both.

| Lane | Control returns | Centering returns | Control mean | Centering mean |
| --- | --- | --- | ---: | ---: |
| 0 |129,320,122|55,346,238|190.333|213.000|
| 1 |124,116,108|179,111,229|116.000|173.000|
| 2 |75,416,363|391,113,266|284.667|256.667|
| 3 |121,342,358|56,85,185|273.667|108.667|
| 4 |230,176,361|410,259,388|255.667|352.333|
| 5 |355,48,101|308,122,405|168.000|278.333|
| 6 |419,99,81|31,233,114|199.667|126.000|
| 7 |181,328,63|330,106,129|190.667|188.333|

The [old cohort](2026-11-01-cdp-breakout-unseen-center.md) improves225.083→269.833
overall,but its selected eight screening episodes account for the gain;the added
16 worsen244.938→238.250. Those stages are reused as evidence without replaying
them. The new cohort supplies new trajectories for both fixed controllers.

## Execution and evidence

CPU qualification passes14:20:37 UTC November1 in2.545s,checking exact policy/
model/runner changes,source protocol and ROM/package identity,prior receipts,
and the two changed seed fields. All29 qualification inputs pass;the eight
previously passed policy fixtures are reused. No emulator/native learner is
constructed during qualification. Its success receipt survives truncated tool
output;the preflight is not repeated and no failure is inferred from that output.

Selection14:24:07/source`b7edcf8` pins31 inputs. Sequential control/candidate
collection and full replay run on one low-priority CPU core:

| Arm | Actions | Emulator frames | Collection | Exact replay |
| --- | ---: | ---: | ---: | ---: |
| Control |46,056|184,200|27.779s|31.845s|
| Centering |54,208|216,799|32.716s|37.148s|

Full pixel hashes,requested actions,policy debug observations,rewards,resets,
task outcomes and retained video hashes pass for both arms. Total100,264 actions/
400,999 frames;controller135.931s completes **14:26:31 UTC November1**. All31
inputs and506 active Qbert learner inputs remain unchanged before/after the arms,
with originalPID803859/invocation`798ad208226e40a3a73fa9e43fec8066` still active.
Retain all CPU co-host time without correcting that learner's measured wall time.
No native/GPU work,learner updates,failed stage or repeated gameplay occurs.

Root`/x/Temp/kindle-breakout-unseen-center-fresh-20261101` retains both complete
episode ledgers,source differences,declarations,execution receipts,replays/videos
and comparison. Complete;do not repeat the preflight or either gameplay arm.

- DeclarationSHA256`5c75a3d1f994b05fddd9ccad3f72161a58c249e9e056d7d598479235e7d13d86`.
- Control review`6d57e9f1b8357e92bcdaa738f78fa70d1f9d61ad58397d99eac7c5771bdc268e`.
- Candidate review`7b222955f89ff10c5b08e9349d1b25fba41267682c5f1050809a8949350fddf9`.
- Comparison`765781edc625ff67f7e9494a51a45bf286b2244d4a99a63c4c4eba5d6acf62d5`.
- Completion`182eb637a089a117966c13994c02a33d44d0833e4c497586486e778b826457cb`.
