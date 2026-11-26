# Breakout world model: commit the fed-back frame to a palette colour

The spatial-history recipe is an autoregressive pixel model: the decoder reads
the two previous frames through a local motion path and, in a forecast, those
frames are its own soft sigmoid outputs. Any hedged prediction of the ball
(mass split over candidate pixels) re-enters as a dimmer, wider blob, so the
ball fades geometrically over the rollout. Replacing the generated history with
the true frames was already known to lift eight-step tracking from 46/108 to
96/108; sharpening the soft frames at forecast time collapses the forecasts
because the motion convolutions never saw sharpened statistics.

`rgb_history_palette` makes the commitment part of training. Every generated
frame that re-enters the image history, in the training rollout and in
forecasts, is snapped per pixel to the nearest of a fixed set of centered RGB
colours (argmin over squared distance, first colour on ties); the forward value
is discrete, the gradient passes straight through to the soft frame. Pixel
losses, the event heads, the returned forecasts and the observed-phase history
(true frames) are unchanged. The whole-agent paths reject the option.

At 96x96 the 32 most frequent corpus colours cover 97.1% of pixels exactly and
the ball is a bright multi-pixel object, so the snap preserves every sprite;
at 64x64 the sub-pixel ball would snap to black, which is why the experiment
runs at 96. The palette is extracted from `corpus/fit96` by
`breakout_world_scale.py palette` (stride 50, 32 colours) into
`palette96/config.json`.

## Protocol

Same corpus, development re-render, recipe, budget, evaluation and seed as the
[96x96 run](2026-11-10-breakout-world-rgb96.md); the only change is
`rgb_history_palette` with those 32 colours. Output `train-96-palette`, queue
`queue-96-palette.json`, guard directory `guards-96-palette`. The straight-through
snap adds one argmax and two small matmuls per rollout step.

## Reading

Compare development ball tracking at every horizon against the soft-feedback
96 run on the same origins. A gain at eight steps and beyond, without a loss
at one step, supports committed feedback; if the long horizons do not move, the
ball is lost before the feedback compounds, in the state or the one-step
prediction, and the next step is the model that carries it. The unit tests
check that the graph snap equals the CPU snap, that gradients pass straight
through, and that only the fed-back history is snapped.
