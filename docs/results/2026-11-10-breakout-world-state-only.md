# Without the image shortcut, the Breakout world state learns the ball

[Protocol](../experiments/2026-11-10-breakout-world-state-only.md). Root
`/x/Temp/kindle-world-scale-20261110`; output `train-96-plain-v2`, guard
directory `guards-next-v2`, unit `kindle-world-scale-20261110.service`
(invocation `0af282d8ab9c4cba961404dfbe467259`, PID 2356516). Source
`f4dd1adf` on Meganeura main `a0eff600`. Same 96x96 corpus, re-rendered
development origins, 1M preset, budget and seed as the spatial 96x96 runs; the
decoder renders from the RSSM state alone (`rgb_spatial_decoder`, `fusion` and
`action` off, `motion_cross_entropy`, prediction length 16).

## Pixel forecasts

Ball within one 64-scale pixel over visible targets, first latent draw.

| Learner, dense development, updates | Observed | 1 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| State-only 1M 96x96, 2048 | 4/113 | 7/112 | 1/112 | 0/112 | 0/115 | 0/102 |
| State-only 1M 96x96, 4096 | 38/113 | 24/112 | 9/112 | 3/112 | 1/115 | 0/102 |
| State-only 1M 96x96, 6144 | 82/113 | 60/112 | 26/112 | 8/112 | 1/115 | 0/102 |
| State-only 1M 96x96, 8192 | 90/113 | 68/112 | 23/112 | 7/112 | 7/115 | 1/102 |
| State-only 1M 96x96, 10240 | 91/113 | 79/112 | 40/112 | 15/112 | 2/115 | 0/102 |
| State-only 1M 96x96, 12288 | 104/113 | 84/112 | 38/112 | 13/112 | 6/115 | 3/102 |
| State-only 1M 96x96, 14336 | 104/113 | 92/112 | 38/112 | 10/112 | 8/115 | 2/102 |
| State-only 1M 96x96, 16384 | 110/113 | 97/112 | 59/112 | 24/112 | 7/115 | 2/102 |
| Spatial 1M 96x96 (soft feedback), 16,384 | 103/113 | 105/112 | 54/112 | 36/112 | 19/115 | 4/102 |
| Spatial 1M 64x64 corpus-scale, 16,384 | 101/112 | 104/109 | 59/109 | 38/107 | 18/114 | 0/95 |
| Spatial 1M 64x64 control, 16,384 | 106/112 | 101/109 | 66/109 | 53/107 | 25/114 | 11/95 |

## State probe

Development RMSE in native pixels of ridge readouts (fitting → development,
128 origins each, ridge 300) of the RAM ball and paddle positions from the
saved features: current-frame encoder features, posterior state, and the
prior state after 1 / 4 / 8 / 16 / 32 transitions. The spatial recipes never
beat the predict-the-mean baseline for the ball at any horizon.

| Updates | Encoder ball x / y | Posterior ball x / y | Prior 1 | Prior 4 | Prior 8 | Prior 16 | Prior 32 | Paddle 1 / 32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 18 / 16 | 18 / 12 | 15 / 11 | 19 / 12 | 23 / 18 | 28 / 21 | 34 / 26 | 7 / 15 |
| 4096 | 16 / 14 | 15 / 9 | 14 / 8 | 16 / 8 | 20 / 12 | 25 / 15 | 34 / 27 | 7 / 15 |
| 6144 | 16 / 14 | 15 / 9 | 15 / 8 | 14 / 7 | 18 / 11 | 22 / 12 | 33 / 22 | 7 / 15 |
| 8192 | 16 / 13 | 15 / 9 | 14 / 8 | 15 / 7 | 18 / 10 | 21 / 12 | 31 / 20 | 7 / 14 |
| 10240 | 16 / 13 | 15 / 8 | 14 / 7 | 15 / 7 | 18 / 9 | 22 / 12 | 31 / 21 | 7 / 13 |
| 12288 | 16 / 13 | 14 / 9 | 13 / 7 | 15 / 7 | 18 / 9 | 24 / 12 | 30 / 22 | 6 / 14 |
| 14336 | 15 / 13 | 14 / 8 | 13 / 7 | 15 / 7 | 18 / 9 | 22 / 12 | 29 / 20 | 7 / 14 |
| 16384 | 15 / 13 | 14 / 8 | 13 / 7 | 15 / 7 | 18 / 9 | 22 / 11 | 30 / 21 | 7 / 14 |
| predict-the-mean baseline | 40 / 28 | same | same | same | same | same | same | 47 |

## Capacity: the 4M preset

The same recipe with `size4_m` (1,024 deterministic units, 32 eight-way
categoricals, hidden 128), output `train-96-plain-4m`, 1,602.3 s of native
calls for 16,384 updates (97.8 ms each, 2.6x cheaper than the spatial 96 run).

| Learner, dense development, updates | Observed | 1 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| State-only 4M 96x96, 2048 | 43/113 | 19/112 | 2/112 | 1/112 | 0/115 | 0/102 |
| State-only 4M 96x96, 4096 | 96/113 | 66/112 | 15/112 | 12/112 | 4/115 | 0/102 |
| State-only 4M 96x96, 6144 | 97/113 | 78/112 | 38/112 | 16/112 | 8/115 | 1/102 |
| State-only 4M 96x96, 8192 | 109/113 | 82/112 | 41/112 | 18/112 | 8/115 | 2/102 |
| State-only 4M 96x96, 10240 | 111/113 | 97/112 | 46/112 | 27/112 | 7/115 | 0/102 |
| State-only 4M 96x96, 12288 | 113/113 | 108/112 | 67/112 | 35/112 | 8/115 | 3/102 |
| State-only 4M 96x96, 14336 | 110/113 | 98/112 | 44/112 | 21/112 | 11/115 | 3/102 |
| State-only 4M 96x96, 16384 | 113/113 | 105/112 | 78/112 | 45/112 | 15/115 | 4/102 |
| State-only 1M 96x96, 16,384 | 110/113 | 97/112 | 59/112 | 24/112 | 7/115 | 2/102 |
| Spatial 1M 96x96 (soft feedback), 16,384 | 103/113 | 105/112 | 54/112 | 36/112 | 19/115 | 4/102 |
| Spatial 1M 64x64 control, 16,384 | 106/112 | 101/109 | 66/109 | 53/107 | 25/114 | 11/95 |

State probe at 16,384 (same columns as above):

| Updates | Encoder ball x / y | Posterior ball x / y | Prior 1 | Prior 4 | Prior 8 | Prior 16 | Prior 32 | Paddle 1 / 32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4M, 16384 | 13 / 12 | 18 / 6 | 18 / 5 | 16 / 5 | 19 / 8 | 23 / 11 | 31 / 19 | 7 / 12 |

Four times the state lifts the pixel forecasts at every horizon up to sixteen
steps: observed reconstruction is perfect on the cohort, four-step tracking
(78/112) exceeds every spatial recipe including the 64x64 control's 66/109,
and eight-step (45/112) closes most of the gap to the control's 53/107 while
the ball is carried by the state rather than by fed-back frames. The linear
state readout is unchanged from 1M (prior-state ball 16/5 at four steps,
19/8 at eight, 23/11 at sixteen), so the extra capacity went into rendering
and the scene, not into a sharper linear ball code; the evaluation-to-evaluation
oscillation (eight-step 35 → 21 → 45 over the last three points) remains.

## Reward events

Brier error on the positive-reward targets of the dense development cohort
(2 / 4 / 4 / 1 / 2 positives at 1 / 4 / 8 / 16 / 32 steps; 1.0 means the
model never expects the brick hit):

| Learner | 1 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|
| State-only 4M 96x96 | .036 | .335 | .206 | 1.000 | 1.000 |
| Context frame 1M 96x96 | .320 | .422 | .253 | .557 | .922 |
| Spatial 1M 96x96 (soft feedback) | .929 | .909 | .810 | .937 | .936 |
| Spatial 1M 64x64 control | .936 | .934 | .832 | .947 | .905 |

The spatial recipes never expect a brick hit at any horizon: their reward head
reads a state that does not know where the ball is. The state-only and
context-frame models predict the near-term hits, which is the signal an
imagination-trained policy needs. The counts are tiny and this is not a
calibration result; the game-over head is still poor for every learner on the
ten development endings (positive Brier .88–.97 at one step).

## Reading

Removing the shortcut changes what the state learns. After 2,048 updates the
current frame's encoder features already read the ball to 18 / 16 native
pixels (baselines 40 / 28), the posterior to 18 / 12, and the prior carries
it through the rollout with a drift that grows from 15 / 11 at one step to
34 / 26 at thirty-two; the spatial recipes never beat baseline at any horizon
with the same probe. From about 6,000 updates the probe plateaus (13 / 7 at
one step, 18 / 9 at eight, 22 / 11 at sixteen, 30 / 21 at thirty-two) while the
pixel metrics keep climbing, so the linear readout understates what the state
holds, and the paddle stays at 7 / 14.

Pixel forecasts start far behind the history-based recipes and overtake them
at four steps by the end (59/112 against 54/112), with observed reconstruction
110/113, one-step 97/112, eight-step 24/112 against the spatial 96 run's
36/112, and sixteen- and thirty-two-step still weak (7/115, 2/102). The
curve was still rising steeply over the last 2,048 updates (four-step 38→59,
eight-step 10→24). The forecast tiles show a different failure from the
spatial recipes: the ball is rendered as a soft blob that widens with the
horizon rather than fading from fed-back frames, and the brick rows are
blurred bands because a 1M state (512 deterministic units, 32 four-way
categoricals) cannot hold the 6x18 brick configuration exactly. Native
calls took 1,108.8 s for 16,384 updates (67.7 ms each, 3.8x cheaper than the
spatial 96 recipe), loop 1,356.1 s.

Two consequences. The recurrent state can learn Breakout's ball when the
training signal reaches it, so the world model's dynamics are not the obstacle
the earlier series suggested; and the state alone is too small to also store
the static scene. The running 4M and 65,536-update state-only jobs measure
capacity and budget. A decoder that reads the static scene from the last
observed frame, kept fixed through the rollout, while the ball and paddle come
from the state, would remove the storage burden without reintroducing
autoregressive feedback; that is the next model-side candidate. Development
screen on reused seed 41001; nothing here qualifies the world model.
