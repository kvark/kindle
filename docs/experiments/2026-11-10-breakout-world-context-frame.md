# Breakout world model: static scene from a fixed observed frame

The [state-only run](../results/2026-11-10-breakout-world-state-only.md) showed
that the recurrent state learns the ball once the decoder cannot read it from
fed-back frames, and that a 1M state then blurs the brick rows because it also
has to store the static scene. The spatial-history recipes had the opposite
split: sharp scenery from the two previous frames, the ball lost in
autoregressive feedback.

`rgb_context_lag` keeps the spatial decoder's image path but changes what it
sees. In the observed phase the history slots hold one real frame from `lag`
arrivals back (clamped to the window start, zero at the first arrival or across
a reset) repeated twice; in forecasts they hold the last observed frame and
stay fixed through the whole rollout. A single frame carries no velocity, so
the motion path can copy walls, bricks and score but cannot propagate the ball;
ball and paddle must come from the state. Nothing is fed back, so nothing
fades. Weights and parameter names are identical to the spatial recipe.

## Protocol

Same 96x96 corpus, re-rendered development origins, 1M preset, B8/T32/H16,
seed 41001, `motion_event_rollout`, RGB100, KL1/.1, rates 4e-4, spatial
decoder/fusion/action and the LEFT/RIGHT/FIRE table as the
[96x96 soft-feedback run](2026-11-10-breakout-world-rgb96.md); only
`rgb_context_lag: 8` is added. 16,384 updates, evaluation every 2,048, output
`train-96-context`, queue `queue-96-context.json`. Unit tests check the context
frame selection on the graph and on the host, reset masking, unchanged
parameters and the absence of the rollout shift.

After the 1M result, the 4M preset with the same context frame runs for
65,536 updates with evaluation every 8,192 (`context-config-96-4m.json`,
output `train-96-context-4m-65k`, launched automatically after the 65k
state-only run). Training is deterministic for a fixed seed and plan, so its
16,384-update point equals a separate 16k run.

## Reading

Against the state-only run: brick balanced error and observed reconstruction
should return to the spatial recipe's level while the state probe keeps
reading the ball. Against the spatial soft-feedback run: eight- and
sixteen-step ball tracking should no longer fade. If the ball probe collapses
back to baseline, the lag-1 observed-phase contexts still leak enough to
rebuild the shortcut and the lag must be bounded below. Development screen on
reused seed 41001.
