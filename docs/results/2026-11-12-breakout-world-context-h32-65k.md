# 4M context-frame world with a 32-step rollout for 65,536 updates

[Protocol](../experiments/2026-11-12-breakout-world-context-h32-65k.md). Output
`/x/Temp/kindle-world-scale-20261110/train-96-context-4m-h32-65k`, unit
`kindle-world-context-h32-65k-20261112.service` (06:43 to 22:45 UTC on
November 12, run directly under a 24G memory cap without the host guard).
65,536 updates in 56,842 s of native calls (867 ms per update, 16h02m wall),
48-step windows (sixteen observed, thirty-two imagined), 3.15M frames. The
first four points reproduce the [32k run](../results/2026-11-11-breakout-world-context-h32.md)
bit for bit. Dense development ball tracking (within 1.5 px at 96,
side-agnostic) and brick balanced error per point:

| Updates | Observed | 1 | 4 | 8 | 16 | 32 | Brick error 8 / 16 / 32 |
|---|---|---|---|---|---|---|---|
| 8,192 | 112/113 | 98/112 | 60/112 | 34/112 | 14/115 | 9/102 | .010 / .024 / .057 |
| 16,384 | 112/113 | 110/112 | 83/112 | 64/112 | 33/115 | 19/102 | .008 / .022 / .053 |
| 24,576 | 113/113 | 110/112 | 93/112 | 65/112 | 46/115 | 25/102 | .007 / .022 / .053 |
| 32,768 | 113/113 | 110/112 | 96/112 | 64/112 | 41/115 | 31/102 | .008 / .020 / .051 |
| 40,960 | 113/113 | 112/112 | 96/112 | 74/112 | 57/115 | 22/102 | .008 / .020 / .051 |
| 49,152 | 113/113 | 111/112 | 96/112 | 72/112 | 53/115 | 24/102 | .008 / .019 / .048 |
| 57,344 | 113/113 | 112/112 | 96/112 | 71/112 | 50/115 | 29/102 | .008 / .019 / .047 |
| **65,536** | **113/113** | **112/112** | **99/112** | **75/112** | **61/115** | **32/102** | **.008 / .018 / .046** |
| 16-step rollout, 65,536 (reference) | 113/113 | 111/112 | 97/112 | 72/112 | 53/115 | 22/102 | .007 / .022 / .053 |
| 64x64 control (reference) | 106/112 | 101/109 | 66/109 | 53/107 | 25/114 | 11/95 | |

## Held-out corpus (seed 777001, 128 origins, probe mode, `heldout/context-4m-h32-65k`)

| Checkpoint | Native time | 1 | 4 | 8 | 16 | 32 | Brick error 8 / 32 |
|---|---|---|---|---|---|---|---|
| **4M, 32-step rollout, 65,536** | 56,842 s | 110/115 | 90/117 | 72/113 | **51/111** | **37/116** | .007 / .045 |
| 4M, 32-step rollout, 32,768 | 28,498 s | 109/115 | 89/117 | 63/113 | 42/111 | 28/116 | .007 / .050 |
| 4M, 16-step rollout, 65,536 | 22,855 s | 112/115 | 93/117 | 72/113 | 41/111 | 19/116 | .009 / .051 |
| 1M, 32-step rollout, 65,536 | 44,669 s | 110/115 | 88/117 | 69/113 | 45/111 | 18/116 | .010 / .050 |

Fitting-corpus scores 111/123, 96/125, 75/118, 50/109, 37/116 with brick
error .004 / .033: the training corpus is not memorized. Linear state probe
on the held-out origins (prior RMSE, native pixels, ball x / y): 11.4 / 5.8
at one step, 13.3 / 6.5 at four, 17.3 / 7.6 at eight, 16.6 / 8.8 at sixteen,
25.1 / 12.4 at thirty-two (the 16-step 65k world: 33.8 / 15.1 at thirty-two).

## Reading

This is the best Breakout world model on record at every horizon, on both
the development origins and the held-out corpus. Thirty-two-step ball
tracking reaches 32/102 on the development origins and 37/116 held-out, up
from 22 and 19 for the 16-step objective at the same update count and from
31 and 28 for this objective at half the budget; sixteen-step held-out
tracking rises from 41 to 51 of 111; the eight-step held-out count matches
the 16-step world's 72; the bricks are the sharpest yet (.007 at eight steps,
.045 at thirty-two); and the state carries the ball's horizontal position at
thirty-two steps with a 25 px linear readout where the 16-step world managed
34. The held-out and fitting scores agree, so none of this is memorization of
the training corpus, and the second half of the run added as much at
sixteen and thirty-two steps held-out (+9 and +9) as the first half did.

The cost is sixteen hours of GPU for 65k updates, 2.5 times the 16-step
world's per-update price, and the short horizons gain nothing (one- and
four-step held-out counts are 2 and 3 below the 16-step world). Thirty-two-step
tracking at 32% held-out remains far from the observed frame's 99%; the curve
was still rising at the end, so a larger budget or model would likely add
more, at proportionate cost. One run, one held-out corpus of 16 episodes;
the driver's `world_model_proven` field stays a constant `False`.
