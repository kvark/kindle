# Committing fed-back frames to a palette does not rescue long forecasts

[Protocol](../experiments/2026-11-10-breakout-world-palette-feedback.md). Root
`/x/Temp/kindle-world-scale-20261110`; output `train-96-palette`, guard
directory `guards-96-palette`, unit `kindle-world-scale-20261110.service`
(invocation `82b1eb2a03464705987b6d9c8ca492dc`, PID 2346800). Source `f279543b`
on Meganeura main `a0eff600`. Same 96x96 corpus, re-rendered development
origins, recipe, budget and seed as the [96x96 run](2026-11-10-breakout-world-rgb96.md);
the only change is `rgb_history_palette` with the 32 most frequent corpus
colours (97.1% exact pixel coverage), snapping every fed-back generated frame
with a straight-through gradient.

| Learner, cohort, updates | Observed | 1 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| 96x96 palette feedback, dense development, 2048 | 70/113 | 75/112 | 20/112 | 5/112 | 0/115 | 0/102 |
| 96x96 palette feedback, dense development, 4096 | 85/113 | 86/112 | 8/112 | 1/112 | 0/115 | 0/102 |
| 96x96 palette feedback, dense development, 6144 | 101/113 | 101/112 | 27/112 | 7/112 | 2/115 | 0/102 |
| 96x96 palette feedback, dense development, 8192 | 95/113 | 97/112 | 31/112 | 9/112 | 5/115 | 1/102 |
| 96x96 palette feedback, dense development, 10240 | 102/113 | 105/112 | 38/112 | 15/112 | 5/115 | 0/102 |
| 96x96 palette feedback, dense development, 12288 | 103/113 | 106/112 | 40/112 | 17/112 | 6/115 | 1/102 |
| 96x96 palette feedback, dense development, 14336 | 104/113 | 107/112 | 46/112 | 25/112 | 6/115 | 4/102 |
| 96x96 palette feedback, dense development, 16384 | 107/113 | 106/112 | 40/112 | 23/112 | 8/115 | 3/102 |
| **96x96 soft feedback, dense development, 16,384** | 103/113 | 105/112 | 54/112 | 36/112 | 19/115 | 4/102 |
| 64x64 corpus-scale, dense development, 16,384 | 101/112 | 104/109 | 59/109 | 38/107 | 18/114 | 0/95 |
| 64x64 control, dense development, 16,384 | 106/112 | 101/109 | 66/109 | 53/107 | 25/114 | 11/95 |
| 96x96 soft feedback, dense fitting, 16,384 | 99/114 | 107/123 | 47/125 | 24/118 | 13/109 | 2/116 |
| 96x96 palette feedback, dense fitting, 16,384 | 104/114 | 101/123 | 32/125 | 16/118 | 5/109 | 0/116 |
| 64x64 corpus-scale, dense fitting, 16,384 | 98/111 | 95/118 | 47/117 | 28/116 | 6/105 | 2/116 |
| 96x96 soft feedback, natural development, 16,384 | 27/28 | 25/27 | 10/26 | 7/27 | 8/27 | 1/25 |
| 96x96 palette feedback, natural development, 16,384 | 26/28 | 25/27 | 7/26 | 3/27 | 1/27 | 0/25 |
| 64x64 corpus-scale, natural development, 16,384 | 25/26 | 26/27 | 15/26 | 8/27 | 3/27 | 1/25 |
| 64x64 control, natural development, 16,384 | 25/26 | 23/27 | 17/26 | 11/27 | 5/27 | 0/25 |

## Reading

Snapping the fed-back frame to a palette colour makes the long horizons
worse, not better. One-step prediction is unchanged (106/112 against 105/112)
and observed reconstruction slightly better (107/113), but four-, eight- and
sixteen-step development tracking fall to 40/112, 23/112 and 8/115 from the
soft run's 54, 36 and 19, and the fitting cohort falls the same way (eight-step
16/118 against 24/118). A hedged ball whose candidates each carry less than
half the red intensity snaps to black and is gone for the rest of the rollout;
the soft frame at least keeps a faint ball the motion path can follow. The
straight-through gradient did not teach the model to commit within 16,384
updates: the curve is noisy and never reaches the soft run at eight steps
(peak 25/112 at 14,336).

Together with the corpus-scale and 96x96 screens, this closes the feedback
side of the question: more data, a multi-pixel ball and hard fed-back frames
all leave eight-step tracking at or below the 64x64 control. The ball is being
lost in the one-step prediction's hedging itself, which the recurrent state
does not resolve because the history path, not the state, carries the ball.
The next runs therefore remove the image history altogether at 96 (the state
must carry the ball), then revisit budget and capacity on the large corpus;
after those, the latent-prediction formulation from TASK.md is the candidate.

Native calls took 4,158.3 s for 16,384 updates (253.8 ms each, within 0.3% of
the soft run), loop 4,502.9 s. Development screen on reused seed 41001; the
palette option stays available but is not retained in any recipe.
