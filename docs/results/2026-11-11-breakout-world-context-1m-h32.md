# 1M context-frame world with a 32-step training rollout for 65,536 updates

[Protocol](../experiments/2026-11-11-breakout-world-context-1m-h32.md). Output
`/x/Temp/kindle-world-scale-20261110/train-96-context-1m-h32`, unit
`kindle-world-context-1m-h32-20261111.service` (17:53 UTC November 11 to
06:34 UTC November 12, run directly under a 24G memory cap without the host
guard). 65,536 updates in 44,669 s of native calls (682 ms per update,
12h41m wall), 48-step windows (sixteen observed, thirty-two imagined), 3.15M
frames. Dense development ball tracking (within 1.5 px at 96, side-agnostic)
and brick balanced error per point:

| Updates | Observed | 1 | 4 | 8 | 16 | 32 | Brick error 8 / 16 / 32 |
|---|---|---|---|---|---|---|---|
| 8,192 | 106/113 | 82/112 | 34/112 | 8/112 | 7/115 | 1/102 | .012 / .025 / .060 |
| 16,384 | 111/113 | 102/112 | 71/112 | 43/112 | 9/115 | 5/102 | .011 / .023 / .058 |
| 24,576 | 113/113 | 107/112 | 75/112 | 41/112 | 25/115 | 8/102 | .010 / .024 / .057 |
| 32,768 | 113/113 | 107/112 | 82/112 | 51/112 | 29/115 | 22/102 | .009 / .023 / .058 |
| 40,960 | 112/113 | 108/112 | 88/112 | 56/112 | 41/115 | 18/102 | .008 / .024 / .056 |
| 49,152 | 112/113 | 110/112 | 86/112 | 62/112 | 39/115 | 20/102 | .008 / .023 / .056 |
| 57,344 | 113/113 | 110/112 | 89/112 | 65/112 | 40/115 | 23/102 | .009 / .024 / .057 |
| **65,536** | 113/113 | 112/112 | 95/112 | 63/112 | 42/115 | 24/102 | .008 / .022 / .055 |

## Held-out corpus (seed 777001, 128 origins, probe mode, `heldout/context-1m-h32-65k`)

| Checkpoint | Native time | 1 | 4 | 8 | 16 | 32 | Brick error 8 / 32 |
|---|---|---|---|---|---|---|---|
| 1M, 32-step rollout, 65,536 | 44,669 s | 110/115 | 88/117 | 69/113 | 45/111 | 18/116 | .010 / .050 |
| 4M, 32-step rollout, 32,768 | 28,498 s | 109/115 | 89/117 | 63/113 | 42/111 | 28/116 | .007 / .050 |
| 4M, 16-step rollout, 65,536 | 22,855 s | 112/115 | 93/117 | 72/113 | 41/111 | 19/116 | .009 / .051 |

Fitting-corpus scores 111/123, 93/125, 67/118, 37/109, 23/116 (no
memorization). Linear state probe on the held-out origins (prior RMSE,
native pixels, ball x / y): 14.8 / 3.9 at one step, 14.9 / 5.2 at four, 20.6
/ 7.0 at eight, 17.1 / 9.3 at sixteen, 31.8 / 13.1 at thirty-two.

## Reading

The 1M model does carry the 32-step objective, but not to the 4M model's
long-horizon level: at 65k updates its held-out thirty-two-step count (18/116)
matches the 4M 16-step world's and falls short of the 4M 32-step world's
28/116, while its sixteen-step count (45/111) is the best held-out value so
far by a small margin and its one- to eight-step counts sit between the two
4M runs. On the development origins it ends at 24/102 at thirty-two steps
against the 4M 32-step run's 31. It also cost the most native time of the
three (12.4 hours against 7.9 and 6.3), because 682 ms per update over
65k updates exceeds 870 ms over 32k. Capacity therefore limits the
thirty-two-step horizon under the longer objective, and the 4M model with
the 32-step rollout is the better use of compute. The development curve
rose to the end (thirty-two steps 22, 18, 20, 23, 24 over the last five
points) with the usual point-to-point swings; one run, one held-out corpus;
the driver's `world_model_proven` field stays a constant `False`.
