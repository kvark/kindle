# The Breakout world state carries the paddle, not the ball

A CPU-only ridge probe on the saved dense forecasts of the
[64x64 corpus-scale learner](2026-11-10-breakout-world-scale.md) at 16,384
updates (`train/evaluations/16384/dense-{fitting,development}.npz`: 128
origins per split, one latent draw, RAM positions as labels). Readouts fit on
the 128 fitting origins and are scored on the 128 development origins;
features are standardized, ridge 100–1000, targets in native pixels.

| Target | Source | Horizon | Development RMSE | Predict-the-mean baseline |
|---|---|---:|---:|---:|
| Paddle x | prior state (640-d) | 1 / 4 / 8 / 16 / 32 | 3.9 / 7.7 / 9.3 / 12.5 / 13.7 | 47 |
| Paddle x | posterior state, current frame | 0 | 3.1 | 48 |
| Paddle x | encoder features (256-d), current frame | 0 | 5.6 | 48 |
| Ball x | prior state | 1 / 4 / 8 / 16 / 32 | 31–36 at every horizon | 40 |
| Ball y | prior state | 1 / 4 / 8 / 16 / 32 | 26–31 at every horizon | 28 |
| Ball x | posterior state, current frame | 0 | 34.7 | 41 |
| Ball x | encoder features, current frame | 0 | 31.8 | 41 |

The paddle calibrates the method: it is read from the state to within four
native pixels at one step and still within fourteen after thirty-two prior
transitions. The ball is not read at any horizon, and, decisively, not from the
current frame's encoder features or posterior state either. The learner
reconstructs the observed ball 90% of the time and predicts it one step ahead
93% of the time, so that information travels through the decoder's two-frame
image history, not through the encoder and recurrent state.

The [96x96 soft-feedback checkpoint](2026-11-10-breakout-world-rgb96.md) gives
the same picture when probed with `breakout_world_scale.py probe` (ridge 300,
fresh forecasts of the same dense origins, `probe-96-soft/`): paddle 3.8 from
the posterior and 4.3 / 7.1 / 8.9 / 13.7 / 13.4 from the prior state at 1 / 4 /
8 / 16 / 32 steps, ball x 31–34 against 40 and ball y at its 28 baseline
everywhere, encoder features included. A multi-pixel ball did not make the
encoder encode it.

This explains the whole series: with no gradient asking the encoder to encode
the ball (the history path serves the decoder more cheaply), the state never
learns ball dynamics, the only ball pathway is autoregressive pixel feedback,
and that pathway fades under hedging whatever the corpus size, frame side or
feedback discretization. A linear probe with 128 fitting samples can miss a
nonlinear code, but a code the paddle probe finds trivially and the ball probe
cannot find at all is at least not a usable one.

Consequences: the state-only decoder run at 96x96 (`train-96-plain`, in the
active queue) is the direct test of whether removing the shortcut forces the
ball into the state; `breakout_world_scale.py probe` now runs this readout on
any checkpoint, and the driver saves encoder, posterior and prior features in
every evaluation. A latent-prediction objective on these encoder features
would be satisfied without the ball; it needs an encoder that is made to see
the ball first.
