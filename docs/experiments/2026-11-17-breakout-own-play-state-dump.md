# Declaration: does the model state know where the ball lands? Own-play state dumps and four probes

November 17, 14:30 UTC. Follows [the held-out and longer-budget result](../results/2026-11-17-breakout-heldout-and-budget.md).

## Why

The clean life-terminal recipe stops improving at 1.2M to 1.6M actions.
Its balls are lost on the 12-px paddle by near misses of about 10 px;
the misses are not policy noise and not reach, and a planner that reads
the emulator returns 90% of the fastest balls there against the
learners' 76%. Two things could hold the learners back: the model state
does not carry the landing point precisely enough, or it does and the
actor does not use it. The standing probe cannot tell: it reads ball and
paddle positions on a scripted teacher's play, which rarely reaches the
narrow paddle. Separately, the 3.6M run's state reads the paddle worse
after its collapse (9 px against 4.5 to 5.8 px), and four of its
checkpoints have not been probed.

## What runs

Evaluation only, about 10 minutes of GPU. Root
`/x/Temp/kindle-breakout-statedump-20261117`, `queue-14.json`, guards
`guards-14`, unit `kindle-breakout-statedump-20261117`, launched by hand
with `run_gpu_queue_reviewed.py` on the reviewed boot. Nothing is
chained after it.

1. **Own-play state dumps** with the new
   `python/examples/breakout_agent_state_dump.py`: the frozen agent
   plays eight lanes itself (sampled actions, sticky .25, 18 actions,
   belief restarted at each lost life, seeds 6660001 to 6660008, 8,000
   actions per lane), and the actor's input, logits and value are saved
   with RAM labels (ball, paddle, lives, on-screen paddle width).
   Checkpoints: `L22` at 1.6M (best on record), `B22` at 1.2M, `H50` at
   1.2M, `L22` at 2.4M (collapsed). A 64-action smoke of the script
   runs first.
2. **Reset probes** (pinned `breakout_agent_probe_life.py
   --life-loss-reset`) of `L22` at 1.6M, 2.0M, 2.8M and 3.2M.

## Reading

**Landing readout (CPU, from the dumps).** For every paddle descent, the
landing point is the ball's x at its last tracked position before the
bounce or the loss. A ridge readout from the model state of *landing x
minus paddle x*, k actions before that moment, is fitted on lanes 0 to 3,
its penalty chosen on lanes 4 and 5 and scored on lanes 6 and 7, for
k = 1, 2, 4 and 8, overall and on the narrow paddle's fastest balls. A
returned ball ends a median 4 px from the paddle centre and the 12-px
paddle tolerates about 7 px.

- **The state holds it** if the mean absolute error two actions before
  contact is 4 px or less on the narrow paddle's fastest balls: the next
  step is the actor (how it is trained to use the state).
- **The state does not hold it** if that error is 8 px or more: the next
  step is the representation (resolution, latent size, the objective).
- Between the two, both are open and the result says so.

Predicted: 6 to 9 px on the narrow paddle's fastest balls at k = 2 and
5 px or less on wide-paddle balls slower than 9 px per action; the
collapsed 2.4M checkpoint worse than the others by 2 px or more. The
readout is linear, so a large error bounds what a linear reader gets,
not what the actor could; a second readout on random nonlinear features
of the state is reported beside it. Lanes share one learner, and 24 to
40 games per dump give about 150 to 250 narrow fast descents, so
differences under 1.5 px are not read.

**When the paddle reading broke.** Predicted paddle error 6 px or less
at 1.6M and 2.0M and 8 px or more at 2.8M and 3.2M. If 2.0M already
reads 8 px or more, the loss of the paddle preceded the fall in play
(online returns fall from 2.25M).

Neither reading changes the project gate, which stays unfinished.
