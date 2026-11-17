# Declaration: the life-terminal recipe with the continuation head supervised on one-step priors

November 17, 23:35 UTC. Follows [the frame-pair result](../results/2026-11-17-breakout-frame-pair.md).

## Why

Post hoc readings of the own-play dumps: one and two actions before
contact a held-out linear readout of the actor's input separates the
balls the agent will lose at AUC .70 to .85, and the agent's critic, on
the same input, at .46 to .63. Under `--life-loss-terminal` a lost ball
ends the episode, so the value of the state should fall to nothing; it
does not until the ball is at the paddle row. The learners stop at a
return rate of .77 to .83 on the narrow paddle's fastest balls, where a
planner with the emulator's state reaches .90.

The actor and critic learn from imagined rollouts. Those end where the
continuation head, asked about prior states, says they end; the head is
trained only on posterior states, which have already seen the life end.
Imagination does end lives (1.5 to 2 times as often as play does, by the
logged means), but nothing trains those endings to fall on the right
states. `--prior-continuation-scale` adds the same continuation target on
the sampled one-step prior, with gradient into the head and the
dynamics. It is not a hand-made prior: the target is the episode
boundary the recipe already uses.

It was [tried once](../results/2026-11-04-cdp-breakout-prior-continuation.md)
on the guided whole-game learner (200k actions: 147.6 / 49.2 against
143.7 / 70.8 greedy / sampled, not retained). There the only endings
were game-overs, 24 in the diagnostic, and the supervised head fired
falsely on 29 of 96 intermediate lost lives, which looked the same and
were not endings. With the life as the episode every lost life is an
ending, about 140 per 50k actions.

## What runs

One learner, `T22`: the recipe of `K` / `B22` / `L22` unchanged (12M,
96x96, B16/T16, train ratio 16, replay 100k, centered CDP,
`--actor-critic-gradient`, `--life-loss-terminal`, sticky .25, 18
actions, seed 22003) plus `--prior-continuation-scale 1`, 1.6M actions,
checkpoints every 400k; about 3.4 hours of training. Then 18 matched
cells (`--life-loss-reset`, 24 games each: 1.6M on the two reused
cohorts and the fresh one, 1.2M, 800k and 400k on the reused ones),
reset probes and own-play state dumps at 1.2M and 1.6M. Root
`/x/Temp/kindle-breakout-priorcont-20261117`, `queue-16.json` (23 jobs,
built by `make_priorcont_queue.py` there), guards `guards-16`, unit
`kindle-breakout-priorcont-20261117`, launched by hand with
`run_gpu_queue_reviewed.py` on the reviewed boot; about 4.2 hours.
Nothing is chained after it. CPU replays and `loss_probe.py` afterwards.

## Reading

Baselines are the same seed and replay without the loss: `L22` at 1.6M
(median 380.5, 8 of 96 games past 432, narrow fastest .77; critic AUC
.63 / .58 one and two actions out, state readout .83 / .74) and `B22`
at 1.2M (334, 3, .76; critic .51 / .46). Until the loss changes the
weights `T22` is not `B22`'s trajectory, and three seeds at 1.2M span
medians of 260 to 350, so a score inside that span says little. The
mechanism is read first.

- **Priors predict the ending** if the logged prior-continuation loss
  averages .028 or less over the last 200k actions (floor .0204 for the
  discounted target, about .036 for a constant).
- **The critic prices the loss** if, on the 1.6M dump, the critic's AUC
  (low value predicting the loss) is at least .72 one action before
  contact and at least .65 two actions before.
- **Play.** The fastest balls on the narrow paddle are returned .85 or
  more at 1.6M on the reused cohorts (CPU replay, both modes pooled).
- **Score.** At 1.6M the pooled median on the reused cohorts is at
  least 400, or at least 20 of the 96 games pass 432.
- **The prior ending is the lever** if the critic test passes and play
  or score passes. If the critic test passes and neither does, the
  value knows and the actor is next. If the prior loss passes and the
  critic test does not, the imagined ending is not what the critic was
  missing (its targets or its replay term are next). If the prior loss
  stays above .032, one-step priors cannot place the ending and the
  dynamics are next.

Predicted: prior loss .024 to .028; critic AUC .66 to .78 and .58 to
.70; narrow fastest balls .78 to .86; median 340 to 420; 5 to 20 games
past 432. Named risks: false endings that make imagination pessimistic
and hurt sampled play, as on November 4; the collapse seen in `L22`
after 2.2M arriving earlier; more GPU memory for the extra prior states
(the headroom guard stops the queue). One seed, a development screen.
The number of games at 864 is reported; the project gate stays
unfinished unless met.
