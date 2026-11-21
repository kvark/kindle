# Does the latent's sampling noise cost the precision the paddle needs? Evaluation-only check

Prepared 13:08 UTC November 19. Follows [the pixel-auxiliary
result](../results/2026-11-19-breakout-pixel-auxiliary-precision.md).
Evaluation only, no training; existing checkpoints.

## Why

Post hoc readings after that result (CPU, own-play dumps and emulator
clones; written up with this check's result):

- A planner with the emulator's exact state and a controlled error in the
  ball-to-paddle offset loses play fast: with a persistent error of 2 px
  (sd) it returns .965 of wide and .838 of narrow balls and scores a
  median of 387 with no game of six past the first wall, which is our
  best learner's level; with no error and the same search it scores a
  median of 714. Position precision of about a pixel is what the gate
  asks for.
- The state's sampled categorical latent changes class in .82 of its 32
  variables at every action, and its marginal entropy is 2.34 of 2.77
  nats per variable. A linear readout of the recurrent part alone reads
  the paddle and ball as well as the whole state does.

The belief is updated with one sample of that latent per action. If that
noise is part of the 2 px, taking the posterior's mode instead of a
sample, or averaging the actor over several samples, should return more
balls with no retraining. Both are existing frozen-evaluation options.

## Run

Root `/x/Temp/kindle-breakout-posterior-20261119`, `queue-21.json` (16
evaluation cells, built by `make_posterior_queue.py`), guards `guards-21`,
unit `kindle-breakout-posterior-20261119`, reviewed runner with the
review file of this boot. Pinned `atari_vector.py` (cafd0ed7).

- Learners: `L22` at 1.6M (seed 22003, the best checkpoint) and `H50` at
  1.2M (held-out seed 5011).
- Variants: `modal` (`--greedy-posterior`, the belief uses the
  posterior's mode) and `avg8` (`--actor-posterior-samples 8`, the actor's
  probabilities averaged over eight further posterior draws; the belief
  itself still uses one sample).
- Cells: cohorts 4000777001 and 4000022003, sampled and greedy actions,
  24 games each, sticky .25, 18 actions, `--life-loss-reset`; the same
  cells these checkpoints already have with sampled posteriors.

## Tests

Per learner and variant, 96 games against the same learner's existing 96.
Baselines: `L22` wide .950, narrow .830, median 380.5, 8 games past 432;
`H50` wide .926, narrow .775, median 260.

1. **Ball return** (`descent_rows.py`): wide up by .015 or more and
   narrow up by .03 or more.
2. **Score:** pooled median up by 30 or more.

Readings:

- **`modal` passes both on both learners:** the sampling noise in the
  belief costs precision. Modal evaluation becomes the matched protocol
  for these agents, and the next training run reduces the noise at its
  source.
- **`avg8` passes and `modal` does not:** the noise matters at the
  actor's input, not in the belief; same direction, smaller.
- **Neither passes:** sampling noise at decision time is not what limits
  the paddle. The precision is what the recurrent state holds, and the
  next step is the input or the objective.

Truncated games are counted and reported; a variant whose greedy cells
stall is not a pass.

Predictions: `modal` changes the wide rate by -.02 to +.01 and passes
with one chance in five (a policy trained on samples sees a different
state distribution); `avg8` changes nothing measurable, one chance in
ten.

## Limits

Two learners, reused cohorts, frozen inference. The policy and the
recurrent model were trained on sampled latents, so a failure of `modal`
does not show that the noise is harmless during training.
