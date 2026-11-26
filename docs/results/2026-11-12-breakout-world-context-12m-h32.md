# 12M context-frame world with a 32-step rollout

[Protocol](../experiments/2026-11-12-breakout-world-context-12m-h32.md). Output
`/x/Temp/kindle-world-scale-20261110/train-96-context-12m-h32`, unit
`kindle-world-context-12m-h32-20261112.service` (23:37 UTC November 12 to
11:04 UTC November 13, run directly under a 24G memory cap without the host
guard). 32,768 updates in 40,677 s of native calls (1,241 ms per update, 1.43
times the 4M model's; 11h27m wall), VRAM 11.6 GiB, 48-step windows (sixteen
observed, thirty-two imagined), 1.57M frames. Dense development ball tracking
(within 1.5 px at 96, side-agnostic) and brick balanced error per point, with
the 4M 32-step run's points for comparison:

| Run | Updates | 1 | 4 | 8 | 16 | 32 | Brick error 8 / 16 / 32 |
|---|---|---|---|---|---|---|---|
| 12M | 8,192 | 106/112 | 72/112 | 47/112 | 21/115 | 13/102 | .008 / .024 / .055 |
| 12M | 16,384 | 110/112 | 87/112 | 65/112 | 42/115 | 21/102 | .008 / .020 / .053 |
| 12M | 24,576 | 111/112 | 96/112 | 73/112 | 45/115 | 25/102 | .007 / .021 / .051 |
| **12M** | **32,768** | 112/112 | 94/112 | 71/112 | 40/115 | 23/102 | .006 / .015 / .047 |
| 4M | 8,192 | 98/112 | 60/112 | 34/112 | 14/115 | 9/102 | .010 / .024 / .057 |
| 4M | 16,384 | 110/112 | 83/112 | 64/112 | 33/115 | 19/102 | .008 / .022 / .053 |
| 4M | 24,576 | 110/112 | 93/112 | 65/112 | 46/115 | 25/102 | .007 / .022 / .053 |
| 4M | 32,768 | 110/112 | 96/112 | 64/112 | 41/115 | 31/102 | .008 / .020 / .051 |
| 4M | 65,536 | 112/112 | 99/112 | 75/112 | 61/115 | 32/102 | .008 / .018 / .046 |

## Held-out corpus (seed 777001, 128 origins, probe mode, `heldout/context-12m-h32-32k`)

| Checkpoint | Native time | 1 | 4 | 8 | 16 | 32 | Brick error 8 / 32 |
|---|---|---|---|---|---|---|---|
| 12M, 32-step rollout, 32,768 | 40,677 s | 110/115 | 89/117 | 66/113 | 46/111 | 32/116 | .008 / .042 |
| 4M, 32-step rollout, 32,768 | 28,498 s | 109/115 | 89/117 | 63/113 | 42/111 | 28/116 | .007 / .050 |
| 4M, 32-step rollout, 65,536 | 56,842 s | 110/115 | 90/117 | 72/113 | 51/111 | 37/116 | .007 / .045 |

Fitting-corpus scores 108/123, 97/125, 74/118, 37/109, 21/116 (brick error
.005 / .033); the training corpus is not memorized. Linear state probe on the
held-out origins (prior RMSE, native pixels, ball x / y): 12.1 / 4.6 at one
step, 13.6 / 6.4 at four, 19.0 / 7.4 at eight, 17.0 / 8.4 at sixteen, 28.5 /
11.9 at thirty-two (4M at 65k: 25.1 / 12.4 at thirty-two).

## Reading

Three times the parameters buy a small, consistent gain at equal updates and
nothing at equal time. Against the 4M model at 32,768 updates the 12M model is
ahead held-out at eight, sixteen and thirty-two steps by 3, 4 and 4 counts
with sharper bricks (.042 against .050 at thirty-two), after leading more
clearly early in training (47 against 34 at eight steps at 8,192 updates).
Against the 4M model's 65k point, which cost 40% more native time, it is
behind by 6, 5 and 5 counts at those horizons. The development curve's last
point dipped at sixteen and thirty-two steps (45 to 40, 25 to 23), the usual
point-to-point swing. Under the 32-step objective, then, budget buys more
long-horizon tracking than capacity does in this range, and the 4M model
with the 32-step rollout at 65k remains the best Breakout world on record;
a 12M run at 65k would cost about 22 hours. One run, one held-out corpus;
`world_model_proven` stays a constant `False`.
