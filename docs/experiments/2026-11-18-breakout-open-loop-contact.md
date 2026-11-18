# Open-loop supervision across paddle contact in the life-terminal recipe

Prepared 04:10 UTC November 18, 2026, while a GPU kernel fault of 04:02
UTC was unreviewed. The user reviewed it ("Reviewed, use this boot"); the
review file `gpu-fault-20261118/reviewed-kernel-faults.json` in root
`/x/Temp/kindle-breakout-imagined-20261118` lists both faults' records.
**Launched by hand at 05:47 UTC November 18**, unchanged from the prepared
text below. [Result](../results/2026-11-18-breakout-open-loop-contact.md).

## Why

[The imagined-contact probe](../results/2026-11-18-breakout-imagined-contact.md)
shows the best checkpoint's imagination is right when it calls a ball lost
(96%) and wrong when it calls it safe: a quarter of those sequences lose
the ball in the emulator, a third on the narrow paddle. Of the pairs the
emulator loses, imagination rates 20% (`L22`) and 22% (`T22`) below .1.
The model is forgiving at the paddle's edge, and the critic and actor learn
from it.

Nothing in the recipe trains a rollout to end where the game ends. The
continuation head is fitted on posterior states, which have seen the
outcome; the dynamics are fitted one step at a time. The one-step prior
term tried yesterday supervises the step after the ball is already gone.
The existing `--open-loop-length` option rolls the prior from one
posterior start per replay sequence through the recorded actions and
supervises continuation, reward and the latent prediction on every rollout
state, with gradients into the dynamics. At eight steps that covers every
lead time from which a contact is decided. It was tried once, at three
steps in the guided 200k recipe ([negative](../results/2026-11-03-cdp-breakout-open-loop.md));
not in the clean life-terminal recipe and not at a horizon that spans
contact.

## Run

`O22`: the life-terminal recipe (`K`: published protocol, sticky .25,
eight environments, 12M, 96x96, B16/T16, train ratio 16, replay 100k,
centered CDP, encoder rate 4e-5, behaviour rate 8e-5, replay value 1,
actor-critic gradient, life-loss terminal) plus
`--open-loop-length 8 --open-loop-scale 1`, seed 22003, 1.6M actions,
checkpoints every 400k. No native change.

Root `/x/Temp/kindle-breakout-openloop-20261118`, `queue-17.json` (27 jobs,
`make_openloop_queue.py`; pinned copies in `sources.sha256`), guards
`guards-17`, unit `kindle-breakout-openloop-20261118`, about 4.5 hours:

- training, about 200 minutes plus the rollout's cost;
- 18 matched cells with the belief reset (1.6M on the two reused cohorts
  and the fresh one; 1.2M, 800k and 400k on the reused cohorts);
- reset probes, own-play dumps and imagined-contact probes at 1.2M and
  1.6M;
- the two imagined-contact replications the fault left unrun (`B22` and
  `H50` at 1.2M).

## Declared tests

At 1.6M. Baselines are `L22` at 1.6M, the same recipe and seed without the
term, and `T22` at 1.6M.

1. **Imagination stops missing losses.** On imagined-contact probes more
   than one and at most three actions from the paddle row, of the pairs
   the emulator loses in both clones the share imagination rates below .1
   is .10 or less (`L22` .203, `T22` .216), while of the pairs it keeps
   the share rated .5 or more stays at .10 or less (.031, .030).
2. **The critic prices the loss.** One action before contact, the AUC of
   the value's ratio to its level at the start of the descent is .78 or
   more (`L22` .71, `T22` .66).
3. **Play.** Fastest balls on the narrow paddle are returned .85 or more
   on the reused cohorts (`L22` .77).
4. **Score.** Pooled median of 96 games 400 or more, or 20 or more games
   past 432 (`L22` 380.5 and 8).

Readings:

- 1 passes and 3 or 4 passes: the forgiving model was the limit and
  multi-step supervision is the lever; confirm on a second seed.
- 1 passes, 3 and 4 fail: imagination is calibrated and play does not
  follow; the actor is next.
- 1 fails: supervision at this horizon and scale does not sharpen
  contact. The state's 5 to 6 px landing error against a 7 px tolerance
  is then the limit, and input resolution is next.

## Predictions

Missed-loss share .10 to .16, so test 1 is uncertain; false alarms rise
to .04 to .08. Critic AUC .70 to .78. Narrow fastest .76 to .84. Median
300 to 400, 3 to 15 games past 432. I expect a partial move in the
mechanism and no clear move in play.

## Risks

- The one earlier trial of this option lowered play.
- Gradients through eight sampled prior steps may destabilise the
  dynamics; the latent-prediction loss and the KL are logged.
- More imagined endings make the critic pessimistic and may hurt sampled
  play, as on November 4.
- The recipe collapsed between 2.2M and 2.6M without the term; an earlier
  collapse would show in the 1.2M and 1.6M cells.
- Another GPU fault. The queue stops on any fault the review does not
  list.
