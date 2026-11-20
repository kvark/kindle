# The collapse after 2.2M: the encoder drifts to slow content; a run with the encoder frozen at 1.2M

Prepared 16:20 UTC November 20; **launched by hand at 16:20 UTC**. The user chose on November 20 to find
the cause of [the collapse of the 3.6M-action run](../results/2026-11-17-breakout-heldout-and-budget.md)
(`L22`: median 380.5 at 1.6M, 331.5 at 2.0M, 45 at 2.4M) before any
other run. This file records the CPU diagnosis and declares one causal
test.

## Diagnosis so far (CPU only)

Root `/x/Temp/kindle-breakout-collapse-20261120` (`analysis/`,
`scratch-scripts/`, `encoder/`).

**Timeline** (`collapse_timeline.py` on `L22`'s training log, 10k- and
50k-action bins). The onset is gradual, not an event: from 2.11M the
latent-prediction loss climbs from .045 to .078 at 2.18M, recedes to
.059 and climbs again to .088 at 2.29M, with the dynamics KL going from
10.1 to 12.7; training returns dip in the same bins (152 to 248 per
game from 2.11M to 2.19M against 202 to 334 before) and fall for good
from 2.3M. Policy entropy (2.1 to 2.4 nats), the entropy of executed
actions, the continuation loss and the return scale do not move first.

**The encoder on identical frames** (`encoder_precision.py`, a NumPy
port of the encoder; the same 64,000 frames, replays of `L22`'s 1.6M
sampled cells, through the encoder of each saved checkpoint;
`embedding_shares.py`, `embedding_spectrum.py`). Shares are of the
centered output's variance explained by group means.

| Checkpoint | Follows the paddle (4 px bins) | Follows the ball (8x8 px cells) | Constant within a lane's 50-tick block | Change over one action, relative | Paddle readout | Pooled median |
|---|---:|---:|---:|---:|---:|---:|
| 800k | .105 | .221 | .734 | .141 | 2.0 px | |
| 1.2M | .100 | .201 | .728 | .140 | 2.0 px | 334 |
| 1.6M | .076 | .172 | .768 | .112 | 2.1 px | 380.5 |
| 2.0M | .045 | .124 | .881 | .056 | 2.0 px | 331.5 |
| 2.4M | .018 | .065 | .926 | .024 | 2.1 px | 45 |
| 2.8M | .021 | .082 | .925 | .019 | 2.2 px | 44 |
| 3.2M | .015 | .065 | .926 | .017 | 2.1 px | 62.5 |
| 3.6M | .016 | .064 | .915 | .024 | 2.0 px | 141 |

The encoder never loses the paddle: a linear readout finds it to 2.0 to
2.2 px at every checkpoint (R² .995 to .997), and the ball as before.
What changes is how much of the output moves with it. From 1.2M the
output is taken over by slow scene content; the part that follows the
paddle falls from 10% to under 2%, and one action changes the output a
sixth as much as before. The drift starts between 1.2M and 1.6M, where
online returns stop rising, is half done at 2.0M, before play falls, and
does not reverse afterwards. The norm of the mean output grows from 12
to 29 while the centered norm stays between 12 and 18.

The latent objective is a cosine distance between a prediction from the
recurrent state and the stop-gradient encoder output, both centered on
the batch. When one action changes the target by 2% of its spread, a
prediction that ignores the paddle loses almost nothing. That fits the
rest of the record: after the collapse the loss is lower than ever
(.027 by 3.5M), the model state reads the paddle at 8.9 px on its own
play at 2.4M (3.5 px at 1.6M) and the ball better than before (x 3.8 px
against 5.5), and the policy that depends on the paddle fails.

**It is a drift of scale, not of content** (`embedding_shares_std.py`,
added 16:35 UTC, after the launch). With every output dimension scaled
to unit variance over the same frames, the paddle's share is the same
at every checkpoint (.124, .115, .123, .112 and .127 at 1.2M, 1.6M,
2.0M, 2.4M and 3.6M), the ball's is .25 to .32 and one action changes
the output by .19 to .25. Most dimensions carry the paddle and the ball
as before; a minority that carries slow content has grown and dominates
the cosine. Dimensions that are nearly constant rise from 238 to 737 of
2,304. A target standardized per dimension would not see this drift;
that is the candidate for the objective named in the readings below.

Other learners at 1.6M, on their own frames: `V22` paddle share .164 and
one-action change .172; `S25` .123 and .217. Neither had reached `L22`'s
level of play; whether they drift later is not known.

Not established: why the encoder drifts (the target is its own
stop-gradient output, so features that are easy to predict can feed
themselves), and whether the drift causes the collapse or accompanies
it. The run below tests the second.

## Change

Native: `encoder_freeze_update` in the Dreamer configuration; from that
learner step the encoder's learning-rate multiplier is zero in both
training sessions (its optimizer moments keep tracking). Runner:
`--encoder-freeze-update N` (fresh CDP only; `SUPPORTS_ENCODER_FREEZE`).
Config tests pass (8 passed); the eight runner tests pass.

## Run

`F22`: `L22`'s own training command (life-terminal recipe, clean
centered CDP, 12M, B16/T16, train ratio 16, replay 100k, sticky .25, 18
actions, seed 22003) with `--encoder-freeze-update 75000`, which is 1.2M
actions, for 2.8M actions. One variable against `L22`, and none before
1.2M. Root `/x/Temp/kindle-breakout-collapse-20261120`, `queue-25.json`
(27 jobs: training; cells on the two reused cohorts at 1.2M, 1.6M, 2.0M,
2.4M and 2.8M; own-play dumps and reset probes at 2.0M, 2.4M and 2.8M),
guards `guards-25`, unit `kindle-breakout-freeze-20261120`, reviewed
runner with this boot's review file. Pinned `atari_vector.py` (11a0578a).

Smoke (16:13 to 16:16 UTC, 16,384 actions, freeze at update 400, no
fault): all 12 encoder tensors are identical between the checkpoints at
8,192 and 16,384 actions and differ from the initial ones; all 172 other
world tensors keep changing; 138.8 actions per second; the evaluation
cell, the dump and the probe ran on the checkpoint. Expected: 5.6 hours
of training, about 6.3 hours in all; peak memory as `L22` (15.6 GiB).

## Tests

Baseline `L22` at the same points.

1. **Play does not collapse:** pooled median of 96 games 250 or more at
   2.4M and at 2.8M (`L22` 45; 44 over its 48 sampled games, its greedy
   games stalled), and training returns of 200 or more per game averaged
   over 2.4M to 2.8M (`L22` 35 to 70 per 50k bin).
2. **The state keeps the paddle:** `state_precision.py` on the own-play
   dumps, paddle x mean error 5 px or less at 2.4M and at 2.8M (`L22`
   8.9 at 2.4M; 3.5 at 1.6M).

Checked, not tests: the encoder tensors are identical in every
checkpoint from 1.2M; whether the first 1.2M repeats `L22` game for game
(a seed's first 400k did, on an earlier build); the latent-prediction
loss and the dynamics KL from 2.1M to 2.4M (`L22` up to .088 and 12.7).
Observations for the gate: the medians at 1.6M and 2.0M (`L22` 380.5 and
331.5), and whether training returns after 1.2M leave the band they
held in `L22` (233 to 316 per 50k bin).

Readings:

- **1 and 2 pass:** the encoder's drift is necessary for the collapse.
  Longer budgets are then usable with the schedule as a stopgap; the
  next steps, each declared separately, are a second seed and an
  objective that keeps the fast content without freezing.
- **1 fails:** freezing the encoder does not prevent the collapse, so
  the drift accompanies it or is not sufficient; the recurrent path is
  next (the dynamics KL rose from 10 to 12.7 at the onset).
- **1 passes, 2 fails:** play survives without a sharp paddle in the
  state; the readout is then the wrong instrument here.
- **A median under 250 at 1.6M or 2.0M:** freezing at 1.2M costs play
  by itself and the later points cannot be read as a test of the drift.

Predictions: no collapse, three chances in five; paddle 5 px or less,
three in five; median at 2.4M 250 to 380; training returns stay in
`L22`'s band and do not rise above it, two in three.

## Limits known in advance

- One seed, and one collapse to compare with. If this build does not
  repeat `L22` to 1.2M, the two runs differ by more than the freeze, and
  a collapse that came at 2.2M once may come elsewhere or not at all.
- A hard freeze is a blunt test. The encoder at 1.2M already shows the
  first of the drift at 1.6M (.100 to .076), so 1.2M was chosen as the
  last checkpoint before it, not tuned.
- The shares are group-mean shares on one set of frames from good play;
  the paddle's and ball's groups overlap with the slow blocks.
- It says nothing about why the encoder drifts.
