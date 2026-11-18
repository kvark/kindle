# Imagination knows which actions lose the ball, and is wrong on the safe side

Result of [the imagined-contact declaration](../experiments/2026-11-18-breakout-imagined-contact.md).
Root `/x/Temp/kindle-breakout-imagined-20261118`, `queue-probe.json`, guards
`guards-probe`. **Two of four jobs finished; a GPU kernel fault stopped the
queue during the third** (see the last section). `L22` at 1.6M (the declared
checkpoint) ran 03:40 to 03:49 UTC November 18 and `T22` at 1.6M 03:49 to
03:59, both cleanly: 24,000 actions each, 9,466 and 10,538 probed states,
seven action sequences per state, two emulator clones per sequence. `B22`
and `H50` at 1.2M were not measured. CPU readings in `analysis/`, readers
in `scratch-scripts/`.

## Instrument

- One-step imagined continuation of the action actually taken separates
  the real life losses at AUC 1.000 on `L22` (47 lives lost; continuation
  .168 on those steps, .9969 elsewhere) and .999 on `T22` (70). State and
  label are aligned.
- The second emulator clone predicts the first at AUC .947 (`L22`) and .954
  (`T22`) on the declared rows; that is the ceiling under sticky actions.

## Declared tests

On probes more than one and at most three actions from the paddle row.

| Test | Bar | `L22` 1.6M | `T22` 1.6M | |
|---|---|---:|---:|---|
| 1. AUC of imagined loss against the emulator | .85 or more | .885 | .868 | passes |
| 2. Losing-against-saving pairs ordered correctly | .85 or more | .915 (1,324 states) | .907 | passes |
| 2. Same, narrow paddle | .80 or more | .921 (346 states) | .922 (103) | passes |
| 3. Imagined loss on pairs the emulator loses | .60 or more | .750 | .737 | passes |
| 3. Imagined loss on pairs the emulator keeps | .05 or less | .071 | .061 | fails |

Tests 1 and 2 pass and test 3 fails on its second half, so by the
declaration imagination "knows which actions lose the ball but hedges".
That reading is wrong in its second half, as the next section shows: the
model does not hedge. Predictions: AUC .78 to .88 (just above), pair
ranking .82 to .92 (met), lower on the narrow paddle (missed: it is not
lower), imagined loss .35 to .55 on lost pairs (missed: .75), same picture
on both checkpoints (met on the two measured).

Where imagination sees a loss it prices it: across states where the
sequences differ in outcome, the imagined return of a saving sequence
exceeds that of a losing one by 60 on average against a posterior value
of 80 (`T22`: 54 against 75).

## Post hoc: the declared tests were too easy, and the errors are one-sided

The pooled AUC and the pair ranking are dominated by gross cases, such as
holding the paddle away from the ball. What the actor needs is the
marginal case. Emulator loss rate by imagined loss probability, declared
rows (pairs in brackets):

| Imagined loss | below .02 | .02 to .1 | .1 to .5 | .5 to .9 | .9 or more |
|---|---:|---:|---:|---:|---:|
| `L22`, all | .25 (6,320) | .42 (523) | .63 (431) | .88 (292) | .96 (6,210) |
| `L22`, narrow paddle | .32 (1,049) | .63 (91) | .83 (69) | .84 (29) | .95 (2,269) |
| `T22`, all | .20 (6,923) | .25 (696) | .35 (413) | .75 (334) | .96 (4,794) |
| `T22`, narrow paddle | .28 (535) | .61 (18) | .22 (16) | 1.00 (12) | .98 (525) |

Nine in ten pairs sit in the two outer columns: the model is confident
either way. When it says the ball is lost it is right 96% of the time.
**When it says the ball is safe, the emulator loses it a quarter of the
time, and a third of the time on the narrow paddle.** The model returns
balls that the game does not.

The same thing seen as a choice. From each probed state, what the emulator
does with the sequence the model rates safest, against the best of the
seven and against what the agent's own policy went on to do (`L22`):

| States | Agent's own play loses | Sequence with the lowest imagined loss | Best of the seven | Stay |
|---|---:|---:|---:|---:|
| All, 1 to 2 actions from the paddle row (1,023) | .045 | .205 | .055 | .296 |
| All, 2 to 3 actions (945) | .051 | .272 | .140 | .486 |
| Narrow paddle, fastest balls, 1 to 2 actions (211) | .137 | .329 | .152 | .550 |
| Narrow paddle, fastest balls, 2 to 3 actions (240) | .138 | .392 | .269 | .631 |

The fixed eight-action sequences are a poor action set (the closed-loop
policy beats the best of them), so this is no planner. But the model's
pick loses one and a half to four times as often as the best available
sequence.
In the 46 and 48 states the agent went on to lose, the model rates its
safest sequence at .067 and .048 loss; the emulator loses that sequence
53% of the time, and 32% to 33% for the best of the seven.

Closer in, with at most one action left, the picture is worse: AUC .716
against a ceiling of .944, and a call of "lost" is right only 64% of the
time.

## Reading

The chain is now measured end to end on the best checkpoint:

- the state holds the landing point relative to the paddle to 5 or 6 px
  one action out, against a tolerance of about 7 px on the narrow paddle;
- imagination resolves that uncertainty toward the common outcome, a
  return, and does so confidently;
- the critic and the actor learn from that imagination. One action before
  a real loss the critic's value has fallen by 12% to 23%, which is what a
  model that sees a safe way out in those states should teach it.

So the agent is not failing to learn from its model. Its model is
forgiving at the paddle's edge, and an actor trained in it has no reason
to place the paddle more precisely than the model demands. The
[prior-continuation term](2026-11-18-breakout-prior-continuation-life.md)
could not change this: it supervised the step after the ball is already
gone.

What this points at, not yet declared or run: the existing
`--open-loop-length` option supervises continuation, reward and the latent
prediction on multi-step prior rollouts with gradients into the dynamics,
which is the training signal for imagined contact outcomes. It was
negative once, at three steps in the guided 200k recipe; it has not been
tried in the clean life-terminal recipe or at a horizon that spans
contact. This probe's reliability table is the direct test for it.

## Limits

One lane and one play seed per checkpoint; two checkpoints of one training
seed. One draw of the prior per sequence, with the same noise stream at
every state. The miss distance of the falsely safe sequences was not
recorded, so "forgiving at the edge" is inferred from the agent's known
near misses (median 10 px off), not measured here. The fixed sequences
are off the agent's policy. In this lane's own play `L22` scored a mean
of 418.8 over 9 games and `T22` 210.1 over 14.

## GPU kernel fault, 04:02:35 UTC November 18

Three minutes into the third job (`B22` at 1.2M) the kernel reported a
`gfx_0.0.0` ring timeout from the probe's process with "Illegal opcode in
command stream", failed MES resets, a mode-1 reset with VRAM lost and
"GPU reset(2) succeeded". The host guard stopped the queue; nothing was
retried, reset, reloaded or rebooted. It is the second such fault on boot
`a8648671`, after the one of November 16 that the user reviewed, and the
existing review does not cover it. Evidence is in
`gpu-fault-20261118/` in the root: the kernel journal, the nine new fault
records the guard matches (`unreviewed-fault-records.json`), the unit log
and a copy of the guard records. **No GPU work runs until the user has
reviewed it.**
