# 4M context-frame world with a 32-step rollout for 131,072 updates

[Protocol](../experiments/2026-11-13-breakout-world-context-h32-131k.md). Output
`/x/Temp/kindle-world-scale-20261110/train-96-context-4m-h32-131k`, unit
`kindle-world-context-h32-131k-20261113.service` (11:08 UTC November 13 to
20:10 UTC November 14, run directly under a 24G memory cap without the host
guard; memory peak 15.9G). 131,072 updates in 117,140 s of native calls
(894 ms per update, 33h02m wall), 48-step windows (sixteen observed,
thirty-two imagined) on a batch of eight, so each frame of the 524,288-action
fitting corpus was visited about 96 times (48 at 65k). The first eight points
reproduce the [65k run](2026-11-12-breakout-world-context-h32-65k.md) bit for
bit. Dense development ball tracking (within 1.5 px at 96, side-agnostic) and
brick balanced error per point:

| Updates | Observed | 1 | 4 | 8 | 16 | 32 | Brick error 8 / 16 / 32 |
|---|---|---|---|---|---|---|---|
| 8,192 to 57,344 | as in the 65k run | | | | | | |
| 65,536 | 113/113 | 112/112 | 99/112 | 75/112 | 61/115 | 32/102 | .008 / .018 / .046 |
| 73,728 | 113/113 | 112/112 | 97/112 | 78/112 | 58/115 | 35/102 | .007 / .015 / .046 |
| 81,920 | 113/113 | 111/112 | 95/112 | 75/112 | 48/115 | 30/102 | .006 / .016 / .048 |
| 90,112 | 113/113 | 111/112 | 99/112 | 77/112 | 61/115 | 37/102 | .006 / .016 / .044 |
| 98,304 | 113/113 | 112/112 | 99/112 | 77/112 | 56/115 | 39/102 | .006 / .015 / .044 |
| 106,496 | 113/113 | 112/112 | 96/112 | 81/112 | 61/115 | 41/102 | .003 / .014 / .043 |
| 114,688 | 113/113 | 112/112 | 97/112 | 78/112 | 57/115 | 37/102 | .004 / .013 / .042 |
| 122,880 | 113/113 | 111/112 | 99/112 | 77/112 | 60/115 | 43/102 | .004 / .013 / .037 |
| **131,072** | **113/113** | **112/112** | **96/112** | **75/112** | **56/115** | **35/102** | **.004 / .012 / .040** |

## Held-out corpus (seed 777001, 128 origins, probe mode, `heldout/context-4m-h32-131k`)

| Checkpoint | Native time | 1 | 4 | 8 | 16 | 32 | Brick error 8 / 16 / 32 |
|---|---|---|---|---|---|---|---|
| **4M, 32-step rollout, 131,072** | 117,140 s | 110/115 | 91/117 | 71/113 | **56/111** | **38/116** | **.006 / .015 / .040** |
| 4M, 32-step rollout, 65,536 | 56,842 s | 110/115 | 90/117 | 72/113 | 51/111 | 37/116 | .007 / .021 / .045 |
| 4M, 32-step rollout, 32,768 | 28,498 s | 109/115 | 89/117 | 63/113 | 42/111 | 28/116 | .007 / .022 / .050 |
| 4M, 16-step rollout, 65,536 | 22,855 s | 112/115 | 93/117 | 72/113 | 41/111 | 19/116 | .009 / .027 / .051 |

The same probe on 128 origins of the fitting corpus:

| Checkpoint | 1 | 4 | 8 | 16 | 32 | Brick error 8 / 16 / 32 |
|---|---|---|---|---|---|---|
| 131,072 | 111/123 | 94/125 | 79/118 | 59/109 | **47/116** | .002 / .011 / .028 |
| 65,536 | 111/123 | 96/125 | 75/118 | 50/109 | 37/116 | .004 / .015 / .033 |
| 32,768 | 113/123 | 92/125 | 71/118 | 45/109 | 26/116 | .005 / .016 / .036 |

Linear state probe on the held-out origins (prior RMSE, native pixels, ball
x / y; 65k in parentheses): 13.1 / 6.8 (11.4 / 5.8) at one step, 15.6 / 7.3
(13.3 / 6.5) at four, 18.9 / 7.8 (17.3 / 7.6) at eight, 16.7 / 8.7 (16.6 /
8.8) at sixteen, 23.4 / 10.3 (25.1 / 12.4) at thirty-two.

## Held-out curve (post hoc)

Not declared in advance: the same probe on every saved checkpoint
(`heldout/context-4m-h32-131k-curve-v2`, one minute each; a first attempt
opened the integrated GPU because Blade's default adapter had switched, and
was rerun with `MEGANEURA_DEVICE_ID=0x744c`). The 32,768 and 65,536 points
reproduce the earlier runs' held-out scores exactly.

| Updates | Held-out 1 / 4 / 8 / 16 / 32 | Brick error 8 / 32 | Fitting 1 / 4 / 8 / 16 / 32 | Prior ball x RMSE at 32 |
|---|---|---|---|---|
| 8,192 | 102 / 51 / 30 / 9 / 2 | .011 / .054 | 96 / 67 / 25 / 13 / 5 | 33.9 |
| 16,384 | 108 / 81 / 56 / 24 / 16 | .009 / .051 | 106 / 87 / 61 / 28 / 15 | 32.6 |
| 24,576 | 107 / 84 / 67 / 34 / 23 | .007 / .052 | 109 / 91 / 61 / 42 / 20 | 30.1 |
| 32,768 | 109 / 89 / 63 / 42 / 28 | .007 / .050 | 113 / 92 / 71 / 45 / 26 | 29.2 |
| 40,960 | 110 / 92 / 75 / 53 / 35 | .008 / .049 | 110 / 95 / 76 / 49 / 34 | 28.0 |
| 49,152 | 110 / 88 / 65 / 40 / 29 | .008 / .049 | 110 / 91 / 77 / 49 / 27 | 25.4 |
| 57,344 | 109 / 92 / 69 / 44 / 25 | .008 / .046 | 110 / 96 / 71 / 52 / 39 | 24.8 |
| 65,536 | 110 / 90 / 72 / 51 / 37 | .007 / .045 | 111 / 96 / 75 / 50 / 37 | 25.1 |
| 73,728 | 110 / 90 / 74 / 51 / 36 | .007 / .047 | 112 / 93 / 81 / 53 / 42 | 25.7 |
| 81,920 | 109 / 88 / 70 / 47 / 23 | .007 / .045 | 109 / 91 / 76 / 46 / 43 | 25.6 |
| 90,112 | 111 / 93 / 78 / 52 / 41 | .007 / .045 | 112 / 98 / 82 / 52 / 42 | 25.1 |
| 98,304 | 110 / 91 / 70 / 50 / 39 | .006 / .043 | 112 / 96 / 83 / 55 / 41 | 25.4 |
| 106,496 | 109 / 92 / 73 / 51 / 37 | .006 / .043 | 113 / 97 / 84 / 54 / 45 | 23.1 |
| 114,688 | 109 / 96 / 78 / 48 / 31 | .006 / .043 | 111 / 94 / 86 / 55 / 45 | 23.5 |
| 122,880 | 109 / 89 / 70 / 55 / 40 | .005 / .041 | 114 / 98 / 84 / 61 / 54 | 23.4 |
| 131,072 | 110 / 91 / 71 / 56 / 38 | .006 / .040 | 111 / 94 / 79 / 59 / 47 | 23.4 |

Neighbouring checkpoints differ by up to 18 counts at thirty-two steps (23
to 41 after 73,728) and 13 at sixteen, so single checkpoints resolve
differences of a few counts poorly; this applies to the earlier single-point
comparisons as well (the 12M model's 3 to 4 count gain over the 4M model at
32,768 updates is inside it). Averaged over windows, held-out tracking at
eight / sixteen / thirty-two steps moves from 70.2 / 47.0 / 31.5 (four
points, 40,960 to 65,536) to 73.0 / 51.2 / 35.6 (eight points, 73,728 to
131,072), while fitting-corpus tracking moves from 74.8 / 50.0 / 34.2 to
81.9 / 54.4 / 44.9: the second half of the budget mostly fits the training
corpus, and the thirty-two-step gap between fitting and held-out origins
grows from about 3 to about 9 counts.

## Reading

Doubling the budget again buys little on held-out play. The final
checkpoint's held-out scores (110/91/71/56/38) sit within checkpoint noise
of the 65k point (110/90/72/51/37); averaged over the second half, sixteen-
and thirty-two-step tracking gain about 4 counts each, and the bricks keep
sharpening (eight-step balanced error .007 to .005 or .006, thirty-two-step
.045 to .040). The fitting corpus gains about twice as much, so by the
declared reading the long horizon is no longer budget-limited for this
model and corpus: beyond about 65,536 updates (about 48 visits per corpus
frame) extra updates go mostly into fitting the 524,288-action corpus. The
earlier finding that budget buys more than capacity holds up to 65k; past
that, more distinct play or a different objective is the lever, not more
updates.

The [sticky-action oracle](2026-11-14-breakout-sticky-oracle.md) puts these
numbers in proportion: a perfect simulator scored with one draw tracks the
recorded continuation 90% / 70% / 41% of the time at eight / sixteen /
thirty-two steps, so this world stands at 70% / 72% / 81% of what the
metric allows, and its remaining misses are concentrated in ball contacts
with bricks and the paddle. The 32-hour run is far from TASK.md's fast
iteration; at 894 ms per update, a second budget doubling is not justified.
One run, one held-out corpus of 16 episodes; the driver's
`world_model_proven` field stays a constant `False`.
