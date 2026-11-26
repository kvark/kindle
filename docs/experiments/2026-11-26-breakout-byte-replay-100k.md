# The fixed recipe with the byte replay at its own 100,000 frames: the separating arm, and the recipe's run-to-run spread on one seed

Declared 02:18 UTC November 26, before launch. The assistant's design;
the user was not asked. The gate, the play mode it reads and the
direction after the failed precision route remain the user's.

## Why

[`R60`](../results/2026-11-26-breakout-replay-300k.md) changed two
things at once, the replay's length (300,000 frames) and its storage
(bytes), and trained 20.5 per game below `D60` (standard error 7.7).
Its declared reading for a failed safety test names this run: the byte
replay at 100,000 frames.

Read after that report, on the five full runs of seed 6007 (training
return per game; `analysis/train-returns-seed6007-five-runs.txt` under
the replay root):

| Stretch | `D60` | `N60` | `C60` (crop) | `A60` (1:1, byte replay 80k) | `R60` (byte replay 300k) |
|---|---:|---:|---:|---:|---:|
| 0.4M to 1.2M | 170.8 | 170.8 | 170.9 | 174.0 | 134.7 |
| 1.2M to 2.4M | 330.1 | 330.1 | 320.1 | 298.0 | 323.8 |
| 2.4M to 3.6M | 410.0 | 418.3 | 389.2 | 358.0 | 385.1 |

(standard errors 3.8 to 5.8; `N60` is `D60` itself to 2.4M.)

- The late take-off is `R60`'s alone. `A60` stored bytes too, at 80,000
  frames, and took off with the others, so the take-off points at the
  replay's length, not its storage.
- The level after the rate cut does not separate that way. The only
  runs at 410 and more are `D60` and `N60`, which are one trajectory
  until 2.4M; `C60` (the recipe with a cropped input) and `R60` both
  sit 21 to 25 lower after the cut, having been within 10 of `D60`
  over 1.2M to 2.4M.
- The byte replay holds the same values as the float replay: the
  preprocessing emits Pillow-rounded bytes, the pack shader stores
  them, and the host decodes `byte / 255 - 0.5`. What it changes is the
  last place of a float, which is enough to send the run down a
  different trajectory from the first update.

So this run is two things. It is the declared separating arm. And it is
the first repeat of the recipe on a seed with nothing changed but a
last-place perturbation: **how far the same configuration on the same
seed moves between runs has not been measured**, and four verdicts
(`N60`, `C60`, `A60`, `R60`) were read against a single `D60`.

## Run

`B60`: `D60`'s training command with `--replay-compact` appended and
nothing else changed (`--replay-capacity 100000` as in `D60`), then
`D60`'s 18 matched cells at 2.8M, 3.2M and 3.6M on the three cohorts,
the 3.6M own-play dump and the reset probe: 21 jobs. Root
`/x/Temp/kindle-breakout-replay-20261125` (the `R60` root, same pinned
runner, probe, dump and replay scripts, `sources-42.sha256`),
`make_bytereplay6007_queue.py`, `queue-42.json`, guards `guards-42`,
unit `kindle-breakout-bytereplay6007-20261126`, run directory
`B60-cdp-life-ratio16-96-bytereplay100k-freeze75k-lrdrop150k-3600k-s6007`.
The review file is the one of 20:44 UTC November 24 (same boot, no
fault since). Projected device memory about 6.8 GiB (117 pages of 32
MiB on a base of 3.15 GiB); about 7 hours of training.

## Checks

21 of 21 jobs; the kernel log; `replay_compact_rgb` true and
`replay_capacity` 100000 in every checkpoint's metadata; no training
game cut; peak device memory against the projection.

## Tests

1. **Take-off:** the training return per game over 0.4M to 1.2M is 153
   or more (the midpoint of `R60`'s 134.7 and the other runs' 171 to
   174; more than three standard errors from either).
2. **Level:** the training return per game over 2.8M to 3.6M is 400 or
   more (`D60` 412.2, `N60` 422.9; `C60` 392.1, `R60` 391.7).
3. **Greedy play:** no greedy cell at 2.8M, 3.2M or 3.6M has a game cut
   at the cap or ending at zero.

The gate-level reading is as before (a pooled median of 430 or more
with 72 or more of 144 games past 432 at any checkpoint, no game cut).

Observations: the return by 400k stretch and by 100k bin against `D60`
and `R60`; the latent loss, KL and return scale through the freeze and
the cut (`R60`'s latent loss was half of `D60`'s); weight distances
after the cut; sampled and greedy medians and games past 432; return
rates on both paddles; serve delay; the share of descents with a move
among the last two actions; the 3.6M dump readouts; actions per second
and peak memory.

## Readings

- **Tests 1 and 2 pass:** the byte storage is cleared and the recipe on
  this seed holds its level under a last-place perturbation. `R60`'s
  two deficits are the replay's length, and `C60`'s is the crop's.
  `--replay-compact` can be used wherever memory is short.
- **Test 1 passes, test 2 fails:** the late take-off is the long
  replay's. The level after the cut moves by 20 between runs of one
  configuration on one seed, so `D60`'s 412 overstated the seed and
  `R60`'s and `C60`'s late deficits are not distinguishable from the
  recipe's own spread. Single-run differences of that size on the late
  level are then not read as effects, and the recipe's level on seed
  6007 is quoted as the mean of its unchanged-configuration runs.
- **Test 1 fails:** the byte path delays learning, against `A60`'s
  evidence; look for a defect in it before it is used again, and treat
  `A60` and `R60` as confounded by it.
- **Test 3** is read on its own: a failure with tests 1 and 2 passing
  says the fragile greedy serve is part of the recipe's spread, not of
  `R60`'s replay.

## Predictions

Test 1: five in six. Test 2: four in nine. Test 3: three in five. The
gate-level reading: one in thirty.

## Limits

- One run. It measures one draw of the run-to-run spread, not the
  spread; a level of 400 to 412 leaves both readings partly open.
- The perturbation is the byte path's, not a designed one: it enters at
  the replay's decode. A storage effect that is not a last-place effect
  would show as a failed test 1 or 2 and could not be told from spread
  by this run alone.
- The same three cohorts as every run since November 17; 24 games per
  cell.
