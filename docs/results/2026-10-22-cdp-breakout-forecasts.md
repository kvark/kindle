# Breakout world forecasts are useful, but reward timing is imperfect

The2M model improves15-step cumulative reward MAE1.856→1.338 versus its500k
prefix on the same recorded final-policy trajectories. It beats unrelated
actions, zero, calibration mean and persistence on cumulative MAE/RMSE. This
supports further learning with the existing model, not a CDP replacement or
a claim that the world model is complete.

|15-step cumulative reward |500k |2M |
| --- | ---: | ---: |
| Actual-action MAE |1.856 |1.338 |
| Actual-action RMSE |5.143 |3.529 |
| Unrelated-action MAE |1.799 |1.446 |
| Zero MAE |2.257 |2.257 |
| Calibration-mean MAE |2.835 |2.835 |
| Persistence MAE |3.657 |3.657 |

There are685 test origins,356 with positive cumulative rewards. Positive-sum
MAE improves3.490→1.780, but false-positive sums worsen: zero-sum MAE.087→.860.
Positive-sum AUC falls.946→.901. Do not describe every metric as improved.

Exact endpoint timing is weaker. H15 endpoint MAE worsens.241→.288, both above
zero's.215;2M unrelated-action MAE.267 is also lower. Positive-endpoint AUC
improves.858→.909 on only36 positive endpoints. Posterior endpoint MAE.162 is
an easier post-observation estimate, not an open-loop forecast. H1 reward MAE
.155 also exceeds zero's.146, though RMSE improves.849→.735 and positive-event
AUC.948→.969 across471 positive events. Errors and ranking measure different
properties; one sampled latent path is not a predictive-mean estimate.

H1 terminal Brier improves.001843→.000765 versus constant.001426;2M AUC.965.
There are only16 game-over events, below the declared20-event minimum; h15 has
none. Survival evidence remains limited, and game-over is not every lost life.
Continuation is corrected using its actual F32 discount1−1/333.

## Scope and audit

Both checkpoints replay the full27,152-action final-policy log, including excess
episodes and tails:54,304 diagnostic actions, zero actor/readout updates, no new
policy decisions. Eight first natural episodes calibrate constant controls;
the next16 test. H1 has11,201 test transitions; h15 excludes34 reset-crossing/
short-tail origins. Repeated samples are not independent learner seeds.
Scalar re-observation differs from the vector actor's categorical draws;
recorded future actions are supplied, never future images/rewards/boundaries.

Both full guards, complete source-action/reward/frame/reset traces, repeated
forecast/prefix/live-belief checks and all346 unchanged tensors/model pass.
Capture times88.25/89.35s; service ends02:19:42UTC, no worker remains.
The repaired128-action smoke adds11.90s and no updates. The original failed
driver-string preflight remains: initialization only, zero actions/updates.
78 CPU tests qualify the explicit driver/device expectation repair; native code,
model arithmetic, checkpoints and2GiB headroom gate are unchanged.

[Full metrics/audit](../../runs/cdp-breakout-forecasts-v2-20261022.bcDtlMcb/full-review.json).
[Learning results and videos](2026-10-22-cdp-breakout-budget.md).

## Decision

Keep CDP and the current recipe. More experience improved both play and useful
reward forecasts; there is no evidence here for a wholesale architecture rewrite.
Use the existing checkpoint restore path for a separately declared warm-start
learning life, retaining learned parameters/optimizer/normalizer while starting
fresh replay and games. Check scheduling with a short live/frozen smoke first.
This is not an equivalent uninterrupted lifetime or an independent training seed.
No backend optimization, broad queue or five-game success claim follows.
