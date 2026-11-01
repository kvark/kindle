# Centering during unseen-ball intervals helps the reused reference cohort

The fixed CPU reference improves **185.375→333.000** on the eight screening
episodes,then **225.083→269.833 (+19.88%)** on the expanded24. Eight of eight
screening lanes and five of eight expanded lane means improve. Both versions
still havezero two-wall successes. The fixed development screen passes,but the
expanded cohort includes the selected eight episodes. The additional16 average
**244.938 control versus238.250 candidate**:the overall gain comes from the
original screening episodes,not improvement on the added episodes. This last
slice is descriptive and post hoc,not an independent confirmation test.

Keep the qualified native teacher unchanged. No teacher integration,new learner,
parameter retry or automatic successor is selected. The result supports further
investigation of earlier paddle positioning,without establishing a reliable
reference improvement or any gain in Kindle learning.

## Saved-log motivation and exact change

The completed CPU diagnostic reads the retained edge-reset eight-episode log.
All12,914 selected decisions/rewards reconcile to the episode ledger. The paddle
is always visible;the ball is unseen on1,114 decisions (8.63%),all requesting
FIRE. Mean paddle distance from center is42.176 source pixels during these
intervals,and697 decisions have distance above35. Classification from the last
valid ball observation yields839 upper-candidate actions,267 lower-candidate
actions andeight initial actions. These are proxies,not verified occlusions or
life boundaries. Requested actions do not identify actual sticky-action execution.
The1.558s analysis completes13:31:18 UTC November1,with315 active learner inputs
and original PID/invocation unchanged;no emulator,replay,native or GPU work.

The candidate changes only the missing-ball/visible-paddle branch of the private
CPU policy. It aims at center81.5 with the existing stopping-position estimate
`paddle_x + .75 * paddle_velocity` and deadband4,requesting RIGHTFIRE or LEFTFIRE
outside the deadband and FIRE inside. Missing-paddle fallback,detector,crop,
visible-ball planner,response weights,edge reset and planning costs stay fixed.
The branch can affect both occlusion and serve phases;their effects are not
isolated. No native teacher bytes change.

## Completed eight-episode screen

| Lane | Control | Centering candidate |
| --- | ---: | ---: |
| 0 | 151 | 327 |
| 1 | 97 | 172 |
| 2 | 256 | 456 |
| 3 | 226 | 351 |
| 4 | 204 | 406 |
| 5 | 331 | 383 |
| 6 | 82 | 340 |
| 7 | 136 | 229 |

All eight policy fixtures pass once;the2.853s preflight checks exact branch
changes without emulator construction. The screen records26,376 actions/
105,487 frames in16.591s and replays them in20.311s. Full pixel hashes,actions,
debug observations,rewards,resets and task outcomes reproduce;video is retained.
All26 declared inputs and315 then-active learner inputs remain unchanged.
Controller38.593s completes **13:36:45 UTC**,source460fb3a. No failed or repeated
gameplay stage,learner update or GPU work occurs. Retain CPU co-host timings
without correcting the simultaneously running R8 learner's measured wall time.

## Completed expanded assessment

The [separate24-episode selection](../experiments/2026-11-01-cdp-breakout-unseen-center-24.md)
occurs only after the R8 learner and all five original services close. It uses
the exact successful policy bytes,seed4000001009,three natural episodes per
eight lanes,full18 actions,sticky.25,repeat4,native pooled RGB and the original
100,000-frame episode cap. The fixed600k-action/180s collection and360s subprocess
caps remain unchanged. The retained control is not rerun.

| Lane | Control mean,three episodes | Candidate mean,three episodes |
| --- | ---: | ---: |
| 0 | 190.333 | 323.000 |
| 1 | 96.667 | 186.000 |
| 2 | 184.667 | 294.000 |
| 3 | 221.000 | 223.333 |
| 4 | 345.000 | 291.333 |
| 5 | 363.667 | 295.333 |
| 6 | 99.000 | 286.667 |
| 7 | 300.333 | 259.000 |

Collection50,832 actions/203,290 frames takes29.754s;full exact replay36.850s.
All24 selected episodes end naturally withzero truncations. The first episode
per lane reproduces every screening return. All31 declared inputs,315 completed
learner inputs,full pixel/action/debug/reward/reset/task replay and video pass.
The68.302s controller completes **13:51:07 UTC November1**,sourcebd3ce22.
No learner,native/GPU work or repeated gameplay stage occurs. All original R8
services remain terminal success and the GPU stays idle.

CPU preflight passes13:49:33 in3.044s/3.167s wrapper,reusing the eight passed
fixtures without repetition. The prior invocation has no success receipt;its
tool output was truncated,so its exact error is unavailable. Original sources
and that limitation are retained in`standard24/preflight-attempt1`. After normal
R8 closure,the passing preflight replaces the obsolete active-process condition
with completed-service/receipt checks. The assessment itself has no failure.

Evidence `/x/Temp/kindle-breakout-unseen-ball-20261101` and
`/x/Temp/kindle-breakout-unseen-center-20261101`,expanded under`standard24`.
Eight reviewSHA256`a88829e90943c0e96f701ca87052bf0007504702b40aff3b04722c5829a84f7d`;
expanded review`9b654c77f987138bc47dba510f6a558da2c37fc7b247667a2cd35e5d93eb8186`,
completion`b2eebb6e7c74b5187d546f315c454086e3fa9796f6956a625d22737f5a032b74`.
Complete;do not repeat the screen,fixtures,preflight or expanded gameplay stages.
