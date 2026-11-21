# Does imagination lose the balls the emulator loses?

Declared 03:40 UTC November 18, 2026, after a 300-action smoke of the tool
(its readings are disclosed below). Evaluation only; no learner is trained.

## Why

The [prior-continuation learner](2026-11-17-breakout-prior-continuation-life.md)
finished and changed nothing that matters: the one-step prior predicts the
ending as well as the posterior does (loss .0216 against .0215), yet the
critic is no better and play is no better. The chain we have measured is

- the actor's input separates the balls the agent will lose at AUC .76 to
  .85 one action before contact;
- the critic's value has fallen to 2% to 14% of normal in the last two
  actions of a life, but one action before contact it is still at 77% to
  88% of its value at the start of the descent (returned balls: 94% to 99%).

The link between them is imagination: the critic and the actor learn from
rollouts of the world model, and nothing so far measures whether those
rollouts lose the right balls for the right actions.

## Tool

`python/examples/breakout_imagined_contact.py` (new; SHA-256 in the run
root's `sources.sha256`). One lane plays itself under the published protocol
with its belief restarted at each lost life. Whenever the ball is visible,
descending and at or below y = 100, it

- rolls the world model open loop from the live posterior for seven fixed
  eight-action sequences (stay, right, left, one or two actions right or
  left then stay) through the existing `Agent.prior_behavior_rollout`,
  recording continuation probability, reward and critic value per step;
- clones the emulator and steps it through the same sequence twice with
  fresh sticky draws, recording the action on which a life is lost.

Imagined loss probability is one minus the product of continuation over
the discount. The imagined return is the rollout's rewards and final value
folded through its continuations, which is what the actor's advantage is
built from. Every action also gets a one-step rollout of the action
actually taken, compared with the real outcome.

No native change; the single-lane `Agent` restores the vector checkpoint.

## Run

Root `/x/Temp/kindle-breakout-imagined-20261118`, `queue-probe.json`, guards
`guards-probe`, unit `kindle-breakout-imagined-20261118`, reviewed runner on
the reviewed boot. Four checkpoints, 24,000 actions each, seed 6660101,
about ten minutes each:

- L22 at 1.6M (seed 22003, the best checkpoint on record), the declared one;
- T22 at 1.6M (the prior-continuation learner);
- B22 at 1.2M (seed 22003) and H50 at 1.2M (held-out seed 5011).

## Calibration of the instrument

- The second emulator clone, scored against the first, is the ceiling for
  any predictor of these labels; the reader prints it beside every AUC.
- The one-step rollout of the action taken must separate the real life
  losses at AUC .95 or better. If it does not, state and label are
  misaligned and the readings below are void.

## Declared tests

On L22 at 1.6M, on probes more than one and at most three actions from the
paddle row (the eight-action horizon covers the loss there, and the critic
has not moved yet):

1. **Imagination discriminates.** AUC of imagined loss probability against
   the emulator's outcome is .85 or more.
2. **Imagination ranks actions.** In states where one sequence loses the
   ball in both clones and another keeps it in both, the imagined loss
   probability orders the pair correctly in .85 or more of pairs, and in
   .80 or more on the narrow paddle.
3. **Imagination prices the loss.** The mean imagined loss probability is
   .60 or more on pairs the emulator loses and .05 or less on pairs it
   keeps.

Readings:

- 1 and 2 fail: the dynamics drop what the state holds. The next lever is
  the dynamics model's contact prediction.
- 1 and 2 pass, 3 fails: imagination knows which actions lose the ball but
  hedges, so the loss is under-priced in every target built from it. The
  next lever is the prior's sharpness at contact.
- All three pass: imagination is right and the critic and actor fail to
  learn from it. The next lever is behaviour learning.

The other three checkpoints are replications, read the same way.

## Predictions

The smoke (L22 at 1.6M, 300 actions, 138 probes, wide paddle only, no real
life lost) read on the declared rows: AUC .78 against a ceiling of .96,
pair ranking .86 over 10 states, imagined loss .43 on lost pairs and .04 on
kept ones. With that seen, I expect

- AUC .78 to .88, pair ranking .82 to .92, lower on the narrow paddle (.70
  to .85);
- imagined loss .35 to .55 on lost pairs, so test 3 fails;
- the same picture on all four checkpoints.

## Limits

Fixed sequences are off the agent's own policy, which mixes directions; the
model may be less accurate there than on its own trajectories. One draw of
the prior per sequence, with the same noise stream at every state. One lane
and one seed of play per checkpoint.
