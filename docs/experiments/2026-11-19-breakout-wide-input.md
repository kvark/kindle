# A wider input: the life-terminal recipe at 128x128

Prepared 13:47 UTC November 19. Follows [the posterior-noise check and
the planner readings](../results/2026-11-19-breakout-posterior-noise-check.md):
a planner with exact control plays at the gate with 1 to 3 px of error in
the ball-to-paddle offset, the best learner plays just below that band,
its paddle ends 4 to 6 px from the landing, and at 96x96 the ball is 1.2
input pixels wide.

## Change

Native: observation sizes 128 and 160 (`ObservationKind::Rgb128`,
`Rgb160`; `--image-size 128|160`; `SUPPORTS_RGB160`). The encoder and the
resize are unchanged code: four convolution and pooling stages end on an
8x8 grid at 128 (6x6 at 96), and the same antialiased resize maps the
160x210 frame to 128x128, so one input pixel is 1.25 source pixels wide
instead of 1.67. The world model grows from 17.9M to 33.5M parameters,
because the layers that read and predict the flattened encoder output
grow with the grid; the actor and critic are unchanged (1.6M). Compile
check passes; the GPU geometry test added beside the 96x96 one is ignored
by default and has not been run. The smokes below exercised both sizes
end to end.

## Run

`W22`: the life-terminal recipe exactly as the 50k-replay control
`C22r50` ran it (clean centered CDP, 12M, B16/T16, train ratio 16, replay
50k, sticky .25, 18 actions, seed 22003) with `--image-size 128`, for
1.6M actions. Root `/x/Temp/kindle-breakout-wide-20261119`, `queue-22.json`
(23 jobs: training, cells at 400k / 800k / 1.2M / 1.6M on the two reused
cohorts, the fresh cohort at 1.6M, own-play dumps and reset probes at
1.2M and 1.6M), guards `guards-22`, unit `kindle-breakout-wide-20261119`,
reviewed runner with this boot's review file. Pinned `atari_vector.py`
(a8c8016a).

Replay is 50k because the GPU replay stores frames as 32-bit floats: at
128x128 a frame is 207 kB, so 50k frames are 9.6 GiB. 160x160 does not
fit with a useful replay (measured below) and waits for a compact replay.

Smokes (13:35 to 13:44 UTC November 19, 16,384 actions each, no fault):

| Input | Time per update | Peak memory at 16k frames | Without the replay | Projected at replay 50k |
|---|---:|---:|---:|---:|
| 96x96 (`C22r50`, same stage) | .121 s | | 4.3 GiB (from `O22`'s full run) | 9.9 GiB |
| 128x128 | .174 s (1.44x) | 8.05 GiB | 4.9 GiB | 14.5 GiB of about 19.4 |
| 160x160 | .249 s (2.06x) | 12.28 GiB | 7.4 GiB | 22.2 GiB; about 30k frames at most |

World microbatches are refused with centered CDP, so memory cannot be
traded for time that way. The frozen tools (evaluation cell, own-play
dump, reset probe) ran on the 128x128 smoke checkpoint. Expected: 4.8
hours of training, 5.4 hours in all, 2 GiB headroom check unchanged with
about 2.9 GiB to spare beyond it.

## Tests

At 1.6M. Baselines: `L22` at 1.6M (same recipe, 96x96, replay 100k;
median 380.5, 8 of 96 games past 432, wide .950, narrow .830) and
`C22r50` at 1.2M (96x96, replay 50k; median 322).

1. **Precision** (`state_precision.py`, own-play dump, 64,000 rows,
   linear): ball x mean error 4.1 px or less (`L22` 5.5) and landing
   minus paddle 5.5 px or less in each of the three bands up to four
   actions out (`L22` 6.7 / 6.8 / 7.3). The bars are `L22`'s errors
   scaled by the pixel size.
2. **Ball return** (96 games, `descent_rows.py`): wide .96 or more and
   narrow .86 or more.
3. **Score:** pooled median 410 or more, or 20 or more games past 432.

Readings:

- **2 or 3 passes:** more pixels per ball buy play. Confirm on held-out
  seed 5011 before any claim, and build the compact replay that 160x160
  needs.
- **1 passes, 2 and 3 fail:** the state reads positions better and the
  paddle does not use it; the readouts are then measuring something play
  does not depend on, and the actor's training signal is next.
- **All fail:** the input's pixel size is not what limits the state at
  this budget. Not retained; the next candidate is the objective.

At 1.2M the run is also compared with `C22r50` (median 322, same replay),
as an observation.

Predictions: ball x 3.6 to 5.0 px; landing minus paddle 5.4 to 6.8 px;
wide .94 to .965; narrow .80 to .87; median 330 to 420; slower than `L22`
through 800k. Test 1 passes with one chance in three, test 2 one in
five, test 3 one in four.

## Limits known in advance

- One seed. Replay 50k against `L22`'s 100k (no difference at 1.2M on
  one seed) and 33.5M world parameters against 17.9M change with the
  input.
- Linear readouts do not order the existing learners by play: `O22`
  reads the ball's x to 3.8 px and plays a median of 317, `L22` 5.5 px
  and 380.5, `X22b` reads the paddle to 2.3 px and plays 274. Test 1
  says whether the input changes what the state holds; tests 2 and 3
  decide.
- 128x128 is a third finer, not the native width. A null here does not
  rule out 160x160.
