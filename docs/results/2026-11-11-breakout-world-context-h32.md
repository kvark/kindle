# 4M context-frame world with a 32-step training rollout

[Protocol](../experiments/2026-11-11-breakout-world-context-h32.md). Output
`/x/Temp/kindle-world-scale-20261110/train-96-context-4m-h32`, unit
`kindle-world-context-h32-20261111.service` (09:44 to 17:49 UTC, run directly
under a 24G memory cap without the host guard). 32,768 updates in 28,498 s
of native calls (870 ms per update, 8h02m wall); 48-step windows (sixteen
observed, thirty-two imagined), so the run saw 1.57M frames against the
16-step run's 2.10M over 65,536 updates while taking 25% more native time.
Dense development ball tracking (within 1.5 px at 96, side-agnostic) and
brick balanced error per point; the 16-step run's points at the same frame
count and at its end are shown for comparison:

| Run | Updates | Frames | 1 | 4 | 8 | 16 | 32 | Brick error 8 / 16 / 32 |
|---|---|---|---|---|---|---|---|---|
| 32-step rollout | 8,192 | .39M | 98/112 | 60/112 | 34/112 | 14/115 | 9/102 | .010 / .024 / .057 |
| 32-step rollout | 16,384 | .79M | 110/112 | 83/112 | 64/112 | 33/115 | 19/102 | .008 / .022 / .053 |
| 32-step rollout | 24,576 | 1.18M | 110/112 | 93/112 | 65/112 | 46/115 | 25/102 | .007 / .022 / .053 |
| **32-step rollout** | **32,768** | **1.57M** | 110/112 | 96/112 | 64/112 | 41/115 | **31/102** | .008 / .020 / .051 |
| 16-step rollout | 24,576 | .79M | 110/112 | 84/112 | 56/112 | 25/115 | 6/102 | .010 / .023 / .059 |
| 16-step rollout | 49,152 | 1.57M | 111/112 | 95/112 | 71/112 | 52/115 | 21/102 | .007 / .022 / .054 |
| 16-step rollout | 65,536 | 2.10M | 111/112 | 97/112 | 72/112 | 53/115 | 22/102 | .007 / .022 / .053 |

## Held-out corpus (seed 777001, 128 origins, probe mode, `heldout/context-4m-h32-32k`)

| Checkpoint | Observed | 1 | 4 | 8 | 16 | 32 | Brick error 8 / 32 |
|---|---|---|---|---|---|---|---|
| 32-step rollout, 32,768 | 114/115 | 109/115 | 89/117 | 63/113 | 42/111 | **28/116** | .007 / .050 |
| 16-step rollout, 65,536 | 115/115 | 112/115 | 93/117 | 72/113 | 41/111 | 19/116 | .009 / .051 |

Fitting-corpus scores of the new run: 113/123, 92/125, 71/118, 45/109,
26/116, so the training corpus is not memorized. Linear state probe on the
held-out origins (prior RMSE, native pixels, ball x / y): 11.5 / 5.2 at one
step, 12.7 / 6.7 at four, 16.9 / 7.7 at eight, 17.7 / 9.3 at sixteen, 29.2 /
14.1 at thirty-two, against the 16-step run's 16.4 / 5.2, 17.2 / 5.5, 23.2
/ 7.0, 18.7 / 9.0 and 33.8 / 15.1: the state carries the ball's horizontal
position better at every horizon.

## Reading

Supervising the thirty-two-step images raises thirty-two-step ball tracking
from 22/102 to 31/102 on the development origins and from 19/116 to 28/116
held-out, the first time any Breakout world has passed a quarter of the
origins at that horizon, with the bricks still sharp. At equal frames
(.79M and 1.57M) the long horizons are ahead of the 16-step run by 8 to 15
counts at eight and sixteen steps and by 10 to 13 at thirty-two steps. The
cost is at the short horizons and in time: held-out one-, four- and
eight-step counts are 3, 4 and 9 lower than the 16-step run's final values
(which had 34% more frames), and each update costs 2.5 times as much, so
reaching 65k updates with this window would take sixteen hours. The
sixteen-step count is unchanged held-out (42 against 41) and lower on the
development origins (41 against 53), the 57,344-style dips notwithstanding;
this is one run and one held-out corpus. The long-horizon limit of the
Breakout world was therefore partly an objective-horizon limit, and
thirty-two-step tracking at 24% to 30% is the new standing number. The
`world_model_proven` field stays a constant `False`; this remains a
development result with one held-out point.
