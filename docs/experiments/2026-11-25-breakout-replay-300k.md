# The fixed recipe with three times the replay

Declared 18:36 UTC November 25, before launch. The assistant's design;
the user was not asked. The user's last direction (an input aligned 1:1
and a sharp loss) [ended on November 25 with all four tests
failing](../results/2026-11-25-breakout-aligned-input-sharp-loss.md);
what follows it is the user's to choose, and this run is a low-cost
test that does not depend on that choice.

## Why

- Every run of the fixed recipe replays from the last 100,000 frames,
  2.8% of a 3.6M-action life and about forty games at the level the
  learner ends on. DreamerV3's Atari runs replay from a million or more. The
  limit here has been device memory: the replay keeps frames as 32-bit
  floats and was most of the 15.4 GiB a run used.
- The one comparison of replay sizes on record points the same way.
  On seed 22003, the 50,000-frame control of November 17 (`C22r50`)
  trained at 215.2 per game over 0.8M to 1.2M (235 games, standard
  error 8.1) against 249.1 for the same recipe with 100,000 frames
  (`B22`, 226 games, 8.7): a difference of -33.9 (standard error 11.9),
  and behind in each 200k stretch from 400k on (89 / 153 / 207 / 223
  against 105 / 158 / 228 / 271). It was reported then as "costs
  nothing", read from one frozen checkpoint; training returns have been
  the primary measure since November 21 and say otherwise. One pair of
  runs on one seed.
- The three changes that raised this recipe's level (the encoder
  freeze, the learning-rate drop, averaged checkpoints) all reduce how
  far the learner is carried by its most recent data. A longer replay
  is of that family and costs no time.
- The byte replay built for the 1:1 run (`--replay-compact`) stores a
  96x96 frame in 37,888 bytes with its state instead of 120,832, so
  300,000 frames take 11.0 GiB where 100,000 float frames took 12.2.

## Run

`R60`: [`D60`](../results/2026-11-24-breakout-recipe-seed6007.md)'s
training command on seed 6007 with `--replay-capacity 300000
--replay-compact` and nothing else changed (life-terminal training,
encoder frozen from update 75,000, learning rates cut to a quarter from
update 150,000, 3.6M actions, checkpoints every 400k), then `D60`'s 18
matched cells at 2.8M, 3.2M and 3.6M on the three cohorts and the
own-play dump and reset probe at 3.6M: 21 jobs. Root
`/x/Temp/kindle-breakout-replay-20261125`, `queue-41.json` built by
`make_replay6007_queue.py 300000`, guards `guards-41`, unit
`kindle-breakout-replay6007-20261125`, sources in `sources-41.sha256`
(runner as committed at ecba7d99; extension unchanged since 17:49 UTC
November 24). The GPU review file is the one of 20:44 UTC November 24
(62 records, this boot); no fault has occurred since.

Projected device memory: 3.15 GiB without the replay plus 351 pages of
32 MiB, 14.1 GiB with the replay full at 300,000 actions (`D60` 15.4;
400,000 frames would project to 17.7, inside the margin other projects'
use of the card can take). About 7 hours of training at the byte
replay's 145 actions per second, then about 25 minutes of cells.

## Checks

21 of 21 jobs; the kernel log; the byte replay and its capacity in
every checkpoint's metadata; no training game cut; device memory
against the projection, read when the replay fills. If memory exceeds
15.4 GiB the run is stopped by hand and the capacity redeclared.

## Tests

1. **Level:** the training return per game over 2.8M to 3.6M is 432 or
   more (`D60` 412.2, `N60` 422.9, pooled 417.4 over 595 games with
   standard error 4.3).
2. **Safety:** that return is 364 or more (the lowest of the recipe's
   three seeds), no training game is cut, and no greedy cell has a game
   cut at the cap or ending at zero.
3. **Play:** pooled play at 3.6M has a median of 430 or more over 144
   games with 55 or more past 432, no game cut (`D60` 424.5 with 39,
   `N60` 424 with 41).

The gate-level reading is as for the last five runs: a pooled median of
430 or more with 72 or more of 144 games past 432 at any of the three
checkpoints, no game cut. Whether the gate is met or changed, and which
play mode it reads, remain the user's.

Observations: training return by 400k stretch against `D60`, with the
stretch 0.8M to 1.2M read against the 50,000-frame comparison; the
latent loss and KL through the freeze and the drop; weight distances
between checkpoints after the drop; sampled and greedy medians and
games past 432 per checkpoint; wide and narrow return rates in both
modes against the planner's .98 and .94; serve delay; the share of
descents with a move among the last two actions; the state readouts,
the critic's timing and policy entropy on the 3.6M dump; actions per
second and peak memory.

## Readings

- **Test 1 passes:** replay length is a lever on this recipe. Repeat on
  held-out seed 5011 before it enters the recipe, and look at storing
  the state in half precision to reach 400,000 frames and more.
- **Test 1 fails, the return is 400 or more:** no gain shown at three
  times the replay; the 50,000-frame comparison was then either that
  seed's or specific to going below 100,000. Replay length is closed at
  this budget.
- **The return is below 400 with test 2 passing:** read as no gain; a
  level between 364 and 400 on this seed is where both of its runs
  with a changed input have landed (392 and 364).
- **Test 2 fails:** the longer replay harms, or the byte replay does;
  the byte replay at 100,000 frames would separate them, under its own
  declaration.

## Predictions

Test 1: one in six. Test 2: four in five. Test 3: one in seven. The
gate-level reading: one in twenty-five.

## Limits

- One seed, one run; the recipe's seeds spread from 364 to 412 per
  training game and this seed's two full-frame runs are at 412 and 423.
- Two things change together: the replay's length and its storage. The
  byte replay is not a bit-exact twin of the float replay (its decode
  can differ from the card's division in the last place), so the run
  shares `D60`'s initial weights and not its trajectory.
- The replay is still short by published standards: 300,000 frames is
  8% of the life.
- The same three cohorts as every run since November 17; 24 games per
  cell.
