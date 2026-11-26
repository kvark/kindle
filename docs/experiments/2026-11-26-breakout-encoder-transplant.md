# Seed 5011 on seed 6007's frozen encoder: does the encoder at the freeze set a seed's level?

Declared 10:48 UTC November 26, before launch. The assistant's design;
the user was not asked. The gate, the play mode it reads and the
direction after the failed precision route remain the user's.

## Why

The fixed recipe ends at a different level on each seed: 364.2 per
training game over 2.8M to 3.6M on seed 5011 (`D50`), 382.3 on seed
22003 (`D22`) and 411.2 on seed 6007 (`D60` and its repeat
[`B60`](../results/2026-11-26-breakout-byte-replay-100k.md), 1.9
apart). That spread is larger than the effect of any change tried since
the learning-rate cut, and it is not run-to-run noise on the one seed
where that was measured.

Read on CPU after `B60` (root `/x/Temp/kindle-breakout-replay-20261125`,
`scratch-scripts/encoder_seeds.sh`, `analysis/encoder-readout-*`): the
frozen encoders of five runs applied to the same 64,000 frames (the
frames of November 20's encoder diagnosis, from `L22`'s play), with a
linear readout of positions from each single frame's encoder output.

| Run | Seed | Training return, 2.8M to 3.6M | Paddle (px, mean abs) | Ball x, mean (median) | Ball height, mean (median) | Ball x / height within 3 px |
|---|---:|---:|---:|---:|---:|---:|
| `D60` | 6007 | 412.2 | 1.7 | 8.0 (4.1) | 6.8 (4.6) | .39 / .35 |
| `B60` | 6007 | 410.3 | 1.8 | 8.4 (4.1) | 7.6 (5.1) | .39 / .31 |
| `R60` (300,000-frame replay) | 6007 | 391.7 | 1.9 | 9.9 (5.1) | 7.9 (5.1) | .33 / .32 |
| `D22` (frozen from 0.8M) | 22003 | 382.3 | 2.0 | 10.6 (5.9) | 8.2 (5.8) | .29 / .28 |
| `D50` | 5011 | 364.2 | 1.8 | 10.2 (5.2) | 8.5 (6.0) | .32 / .27 |

- The two runs at 410 and more have the encoders that read the ball
  best; the three lower runs read it 1.5 to 2.6 px worse in x. The
  frames are seed 22003's own play, and that seed's encoder reads them
  worst.
- The unfrozen encoders of seeds 22003 and 5011 read the ball's x at
  9.7 to 11.0 px on these frames at every checkpoint from 0.8M to 3.6M
  (November 20), so the difference is not when the encoder was frozen.
- `R60`'s encoder is worse than its two siblings': the long replay's
  deficit may have entered through the encoder it froze.
- Five runs; a correlation. Across architectures it does not hold (the
  128x128 encoder read every object better and its learner was far
  worse).

The causal test is to give a low seed the high seed's encoder.

## What was built

`--encoder-transplant CHECKPOINT` in the runner (commit 3a5e20a4; no native change, the extension is the one of
November 24): a fresh agent is saved, the twelve encoder tensors of its
world file (220,416 values) are replaced by the donor checkpoint's, and
the result is restored before the first update. The optimizer moments
stay the fresh agent's zeros. Off unless set.

Smokes (`queue-43-smoke.json` in the root, 16,384 actions each, seed
22003):

- the unchanged command reproduces the November 24 reference in every
  learner report and episode (the only differences are file identities
  and timing);
- a self-transplant (the run's own initial encoder) changes no tensor
  and yields an identical initial checkpoint, **but not an identical
  run**: a restored agent draws its random streams from a different
  seed than a fresh one (`DreamerRngs::resumed`), so a transplanted run
  has the seed's initial weights and games and its own sampling;
- with `D60`'s encoder and `--encoder-freeze-update 1` the encoder's
  twelve tensors are `D60`'s at the start and after 1,001 updates, and
  the latent loss falls faster (.081 against .172 and .228 at 1,001
  updates).

## Run

`X50`: `D50`'s training command (the fixed recipe on seed 5011, with
`D60`'s checkpoint cadence) with three changes: `--encoder-transplant`
from `D60`'s 3.6M checkpoint (its encoder as frozen at 1.2M),
`--encoder-freeze-update 1` instead of 75000, and `--replay-compact`
(cleared by `B60`; 6.8 GiB instead of 15.4). Then `D60`'s 18 matched
cells at 2.8M, 3.2M and 3.6M on the three cohorts, the 3.6M own-play
dump and the reset probe: 21 jobs. Root
`/x/Temp/kindle-breakout-transplant-20261126` (pinned runner, probe,
dump and replay scripts, `sources-44.sha256`),
`make_transplant_queue.py`, `queue-44.json`, guards `guards-44`, unit
`kindle-breakout-transplant5011-20261126`, run directory
`X50-cdp-life-ratio16-96-encoderD60-freeze1-lrdrop150k-3600k-s5011`.
The review file is the one of 20:44 UTC November 24 (same boot, no
fault since). About 7 hours of training.

## Checks

21 of 21 jobs; the kernel log; the run's header records the transplant
with 12 tensors changed; the encoder's tensors equal `D60`'s in every
checkpoint; no training game cut; peak device memory.

## Tests

1. **Level:** the training return per game over 2.8M to 3.6M is 388 or
   more: half of the gap from `D50`'s 364.2 (standard error 5.6) to
   seed 6007's 411.2 (4.4) is closed.
2. **Safety:** that return is 340 or more and no training game is cut:
   a foreign encoder, frozen from the first update, does not break the
   learner.

Read in three bands: 400 or more, the level follows the encoder; 376 to
400, part of the gap; below 376 (two standard errors above `D50`), the
encoder is not what separates the seeds.

The gate-level reading is as before (a pooled median of 430 or more
with 72 or more of 144 games past 432 at any checkpoint, no game cut).

Observations: the return by 400k stretch against `D50` and `D60` (with
a mature encoder from the first update the take-off may come earlier;
`D50` 110.9 per game over 0.4M to 1.2M, `D60` 170.8, `B60` 225.9); the
latent loss and KL from the start; the state readouts on the 3.6M dump
(`D50` paddle 3.1 px, ball 4.6 / 4.1, landing minus paddle over all
descending rows 8.9; `D60` 2.3, 3.3 / 3.3, 7.6); sampled and greedy
medians and games past 432 (`D50` 388.5 / 404.5 / 398 pooled with 11 /
17 / 22 of 144); return rates on both paddles; serve delay.

## Readings

- **Test 1 passes:** the encoder carries at least half of the seed gap
  in this run. It is not yet attributed: the run also freezes the
  encoder from the first update and draws its own random streams. The
  control is the same command with seed 5011's own frozen encoder as
  the donor (`D50`'s 3.6M checkpoint), under its own declaration; the
  difference between the two runs is the encoder alone. If that holds,
  what makes an encoder read the ball well, and how to get one on every
  seed without labels, is the next question.
- **Test 1 fails, test 2 passes, the return is 376 or more:** part of
  the gap; the control is still needed to say whose.
- **The return is below 376:** a better-reading encoder does not lift
  seed 5011. The seed's level sits elsewhere (its initial recurrent
  model and heads, or its games), and the correlation above is not
  causal. The control is not run.
- **Test 2 fails:** an encoder frozen from the first update, or a
  foreign one, breaks learning; the control (own encoder, frozen from
  the first update) would say which, and is run only if the cause
  matters for something else.

## Predictions

Test 1: two in five (400 or more: one in four). Test 2: four in five.
The gate-level reading: one in thirty.

## Limits

- One run on one seed with one donor.
- Three things differ from `D50`: the encoder, the freeze from the
  first update, and the random streams (plus the replay's storage,
  cleared by `B60`). Seed 5011's own run-to-run spread is not measured;
  on seed 6007 two runs were 1.9 apart (standard error 8.7).
- The donor's encoder was shaped by seed 6007's own play; it has seen
  the same game and none of seed 5011's frames.
- The recipe's cost is then 1.2M actions of a donor run on top of the
  3.6M; if the encoder is the lever, the donor is paid once.
- The same three cohorts as every run since November 17; 24 games per
  cell.
