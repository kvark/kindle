# Declaration: the clean life-terminal recipe with a two-frame encoder input

November 17, 14:55 UTC. Follows [the own-play state-dump result](../results/2026-11-17-breakout-own-play-state-dump.md).

## Why

The recipe stops improving because its state does not hold where the
ball lands: a linear readout is 8.5 to 10 px off one action before
contact, no better than a constant guess, while it reads the paddle to
3 px and the ball's present position to 4 to 5 px. What is missing is
the ball's travel over the next action, that is its velocity; the
critic cannot tell a ball it will lose from one it returns. The encoder
sees one frame per action and the recurrent state has to carry all
motion.

`--rgb-frame-pair` gives the encoder the previous and the current
resized frame as six channels (the first frame repeated at a reset). It
is the generic frame stacking of DQN-style agents, with no game
knowledge, labels or guidance. It was [tried once](../results/2026-11-03-cdp-rgb-frame-pair.md)
on the guided 64x64 recipe at 200k actions and not retained (114 / 39
against 144 / 71 greedy / sampled); that learner scored about 100 and
its limit was not located. Here the limit is measured and the flag is
aimed at it.

## What runs

One learner, `P22`: the recipe of `K` / `B22` / `L22` unchanged (12M,
96x96, B16/T16, train ratio 16, replay 100k, centered CDP,
`--actor-critic-gradient`, `--life-loss-terminal`, sticky .25, 18
actions) plus `--rgb-frame-pair`, seed 22003, 1.6M actions (the budget
at which `L22` peaked), checkpoints every 400k. About 3.8 hours of
training (the pair cost 13% in the earlier test), then 20 matched cells
(`--life-loss-reset`, 24 games each: 1.6M and 1.2M on the two reused
cohorts and the fresh one, 800k and 400k on the reused ones), reset
probes and own-play state dumps at 1.2M and 1.6M. Root
`/x/Temp/kindle-breakout-framepair-20261117`, `queue-15.json` (25 jobs,
built by `make_framepair_queue.py` there), guards `guards-15`, unit
`kindle-breakout-framepair-20261117`, launched by hand with
`run_gpu_queue_reviewed.py` on the reviewed boot. Nothing is chained
after it. CPU replays afterwards.

## Reading

Baselines are the same seed without the pair: `L22` at 1.6M (median
380.5, 8 of 96 games past 432, narrow fastest balls returned .77 on the
cells, landing readout 8.5 px one action out) and `B22` at 1.2M (334,
3, .76, 8.6 px). The pair widens the first layer, so `P22` is not
`B22`'s trajectory with one change; the three seeds at 1.2M span
medians of 260 to 350, and a single run's score inside that span says
little. The mechanism is read first.

- **Mechanism.** The state now holds the landing point if, at 1.6M, the
  four-fold linear readout of landing minus paddle one action before
  contact is 6 px or less over all descents (baseline 8.5).
- **Play.** The fastest balls on the narrow paddle are returned .85 or
  more at 1.6M on the reused cohorts (CPU replay, both modes pooled).
- **Score.** At 1.6M the pooled median on the reused cohorts is at
  least 400, or at least 20 of the 96 games pass 432.
- **The pair is the lever** if the mechanism passes and play or score
  passes. If the readout passes and neither play nor score does, the
  state has the information and the actor is next. If the readout stays
  at 8 px or more, velocity at the encoder is not what is missing and
  the next step is the precision of the image itself (resolution, crop,
  latent size).

Predicted: readout 5.5 to 7 px; narrow fastest balls .80 to .86; median
360 to 410; 8 to 20 games past 432; slower than `B22` at 400k (its 73).
Named risks: a slow start that 1.6M does not recover; the collapse seen
in `L22` after 2.2M arriving earlier; memory (six input channels) over
the headroom guard, which stops the queue. One seed, a development
screen. The number of games at 864 is reported; the project gate stays
unfinished unless met.
