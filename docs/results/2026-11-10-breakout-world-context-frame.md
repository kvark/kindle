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

## 4M preset with the context frame

`train-96-context-4m-65k` (4M, context lag 8, 346 ms per update) was stopped
after its 16,384-update point to hand the GPU to the Breakout agent run; its
8,192 and 16,384 points are retained and the run is reproducible from the
queue file. At 16,384: observed 113/113, one-step 107/112, four-step 74/112,
eight-step 30/112, sixteen-step 14/115, thirty-two-step 2/102, brick balanced
error .010 at eight steps; state probe prior-ball 18 / 5 at one step, 20 / 8 at
eight, 23 / 11 at sixteen. Against the 4M state-only run at the same point
(105 / 78 / 45 / 15 / 4, bricks .35) the context frame keeps the scene exact
but trails at eight steps, so at this budget the two designs trade scene
fidelity against ball tracking; whether the gap closes with budget, as it did
for the state-only 1M run, is the open question for the 65k continuation.

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

## 4M context frame at 65,536 updates (November 11)

`train-96-context-4m-65k-v4` (same config and seed as the stopped run; the
run is deterministic and its 8,192 and 16,384 points reproduce the earlier
ones bit for bit) completed 65,536 updates in 22,855 s of native calls (349
ms per update; 6h31m wall, 00:26 to 06:57 UTC, unit
`kindle-world-context-4m-65k-v4-20261110.service`, MemoryMax 24G). Two
earlier relaunches failed before producing new data: `-v2` was OOM-killed at
a 14G memory cap while loading the corpus, and `-v3` was killed at update
14,848 by the host guard's kernel-probe timeout (a known false positive of
the 3 s `journalctl` probe, child exit -15, no GPU fault); the completed run
therefore ran without the host guard. Dense development ball tracking
(within 1.5 px at 96, side-agnostic) and brick balanced error per point:

| Updates | Observed | 1 | 4 | 8 | 16 | 32 | Brick error 8 / 16 / 32 |
|---|---|---|---|---|---|---|---|
| 8,192 | 112/113 | 103/112 | 44/112 | 14/112 | 6/115 | 2/102 | .011 / .024 / .058 |
| 16,384 | 113/113 | 107/112 | 74/112 | 30/112 | 14/115 | 2/102 | .010 / .024 / .058 |
| 24,576 | 113/113 | 110/112 | 84/112 | 56/112 | 25/115 | 6/102 | .010 / .023 / .059 |
| 32,768 | 113/113 | 111/112 | 92/112 | 66/112 | 46/115 | 15/102 | .007 / .023 / .058 |
| 40,960 | 113/113 | 111/112 | 95/112 | 70/112 | 48/115 | 19/102 | .008 / .024 / .053 |
| 49,152 | 113/113 | 111/112 | 95/112 | 71/112 | 52/115 | 21/102 | .007 / .022 / .054 |
| 57,344 | 113/113 | 111/112 | 93/112 | 67/112 | 37/115 | 17/102 | .007 / .023 / .054 |
| **65,536** | **113/113** | **111/112** | **97/112** | **72/112** | **53/115** | **22/102** | **.007 / .022 / .053** |
| 64x64 control (final) | 106/112 | 101/109 | 66/109 | 53/107 | 25/114 | 11/95 | |
| State-only 1M, 65,536 | 113/113 | 109/112 | 93/112 | 70/112 | 46/115 | 15/102 | .32 at 8 |

This is the best Breakout world model so far at every horizon on the shared
development origins, and the first that keeps the bricks sharp while doing
it (balanced error .007 at eight steps against the state-only world's .32).
Natural development episodes at 65,536: observed 28/28, one-step 26/27,
four-step 23/26, eight-step 16/27, sixteen-step 9/27, thirty-two-step 1/25;
contact origins 8/8, 8/8, 7/7, 1/6, 5/8, 1/8. The 57,344 point dips at
sixteen and thirty-two steps and recovers at 65,536; single-checkpoint
points move by ten or more counts between evaluations, so the curve, not one
point, is the evidence. The linear state probe (development RMSE in native
pixels, baselines ball 40 / 28, paddle 47) plateaus from 16,384 updates
while pixel tracking keeps improving: at 65,536, encoder ball 10.4 / 10.4,
posterior 20.1 / 6.6, prior at one step 18.8 / 5.4, eight 20.3 / 7.0,
sixteen 21.2 / 10.4, thirty-two 30.8 / 17.9; paddle 8 to 14 at every
horizon. The driver's `world_model_proven` field in `result.json` is a
constant `False`: the driver never claims proof, and this remains a
development screen on reused seed 41001 and reused origins rather than a
held-out qualification.
