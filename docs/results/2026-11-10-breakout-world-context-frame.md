# A fixed observed context frame keeps the scene sharp while the state carries the ball

[Protocol](../experiments/2026-11-10-breakout-world-context-frame.md). Root
`/x/Temp/kindle-world-scale-20261110`; output `train-96-context`, guard
directory `guards-after-4m`, unit `kindle-world-scale-20261110.service`
(invocation `be48affde4f24089a94e70552781eaec`, PID 2365191). Source
`df7b39e7` on Meganeura main `a0eff600`. Same corpus, development origins,
1M preset, budget, seed and spatial decoder/fusion/action recipe as the
[96x96 soft-feedback run](2026-11-10-breakout-world-rgb96.md); the decoder's
history slots hold one real frame from eight arrivals back in the observed
phase and the last observed frame, fixed, through forecasts
(`rgb_context_lag: 8`). Nothing generated is fed back.

## Pixel forecasts

PIXEL_TABLE

## State probe

PROBE_TABLE

## Reading

The decoder reads the scene from a real frame and the ball from the state,
and both halves show up. The state probe matches the state-only runs from the
first evaluation (encoder ball 19 / 15 at 2,048 updates, prior-state ball
17 / 6 at one step, 21 / 8 at eight and 22 / 10 at sixteen at the end, against
baselines 40 / 28), which the spatial soft-feedback recipe never achieved
with the same decoder weights. At the same time the brick balanced error at
eight steps is 0.011 (0.058 at thirty-two), against 0.35 for the state-only
decoders and 0.006 for the spatial recipe that copies bricks from its fed-back
frames.

Against the 1M state-only run the pixel forecasts improve at every horizon up
to sixteen steps: four-step 72/112 against 59/112, eight-step 32/112 against
24/112, sixteen-step 10/115 against 7/115, with observed 111/113 and one-step
100/112. Against the spatial soft-feedback run it is better at four steps
(72 against 54), slightly behind at eight and sixteen (32 and 10 against 36
and 19), and against the 64x64 control (66 / 53 / 25) ahead at four, behind at
eight and sixteen. The 4M state-only run remains the best at eight steps
(45/112); its 4M context-frame counterpart is queued next. The curve was still
rising at the end (four-step 51 → 72, eight-step 18 → 32 over the last 2,048
updates). Native calls took 4,022.4 s for 16,384 updates (245.5 ms each, the
full-resolution motion path dominates), loop 4,296.6 s.

Nothing generated is fed back, so the long-horizon weakness here is the prior
drifting (prior-state ball RMSE 34 / 20 at thirty-two steps) and the renderer's
precision from a coarse latent, not fading. Development screen on reused seed
41001; nothing here qualifies the world model.
