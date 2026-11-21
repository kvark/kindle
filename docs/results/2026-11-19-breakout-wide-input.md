# A 128x128 input learns far slower per action and does not read the ball's horizontal position better

Result of [the wide-input declaration](../experiments/2026-11-19-breakout-wide-input.md).
Root `/x/Temp/kindle-breakout-wide-20261119`, `queue-22.json` (23 jobs,
all complete, no GPU fault, no amdgpu kernel message during the queue),
guards `guards-22`, unit `kindle-breakout-wide-20261119`, 13:48 to 18:53
UTC November 19. `W22` is the life-terminal recipe exactly as the
50k-replay control `C22r50` ran it, with `--image-size 128`, seed 22003,
1.6M actions: 293.8 minutes, 90.8 actions per second (`L22` 137.4, so
1.51 times the time per action; the smoke measured 1.44), 99,977 updates,
peak 14.44 GiB of VRAM (projected 14.5), no truncated game in any cell.
CPU readings in `analysis/` (`*-W22-*`), readers in `scratch-scripts/`,
batch `replay22.sh`.

**Result.** All three declared tests fail, and none is close. At 1.6M
actions the 128x128 learner plays a pooled median of 122.5, below the
96x96 control's 800k checkpoint (174.5), and it was behind at every
checkpoint. Its state reads the ball's height better (3.7 px against
4.6), the ball's horizontal position the same (5.5 px against 5.5) and
the paddle worse (5.5 px against 3.5), so where the ball lands relative
to the paddle is read worse (9.3 px against 6.7). By the declared
reading the input's pixel size is not what limits the state at this
budget; 128x128 is not retained and the compact replay that 160x160
would need is not built.

## Declared tests

At 1.6M. Baseline `L22` at 1.6M (96x96, replay 100k, same seed).

| Test | Bar | `W22` | `L22` | |
|---|---|---:|---:|---|
| 1. State readout, ball x, mean error | 4.1 px or less | 5.5 | 5.5 | fails |
| 1. Landing minus paddle, 0.5 to 1.5 / 1.5 to 2.5 / 2.5 to 4 actions out | 5.5 px or less in each | 9.3 / 9.4 / 9.3 | 6.7 / 6.8 / 7.3 | fails |
| 2. Balls returned, wide / narrow paddle | .96 / .86 or more | .904 (4,330) / .748 (262) | .950 / .830 | fails |
| 3. Pooled median of 96 games; games past 432 | 410, or 20 games | 122.5; 0 | 380.5; 8 | fails |

Predictions: only "slower than `L22` through 800k" held, and it
understated the result. Ball x 3.6 to 5.0 px (5.5), landing minus paddle
5.4 to 6.8 px (9.3 to 9.4), wide .94 to .965 (.904), narrow .80 to .87
(.748) and median 330 to 420 (122.5) all missed. I gave the three tests
one chance in three, five and four; I expected a slow start and a
learner that caught up, and it did not catch up.

## Play

24 games per cell, mean (median; best), belief reset at each lost life.

| Actions | 777001 sampled | 777001 greedy | 022003 sampled | 022003 greedy | Pooled median | 400 or more / past 432, of 96 | `C22r50` | `L22` |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| 400k | 48.5 (47; 93) | 50.5 (50; 92) | 53.2 (44; 212) | 52.2 (51; 112) | 48.5 | 0 / 0 | 60.5 | 73 |
| 800k | 78.3 (63.5; 237) | 127.2 (75; 385) | 82.3 (55; 245) | 85.7 (68.5; 252) | 69 | 0 / 0 | 174.5 | 246 |
| 1.2M | 120.8 (87; 371) | 167.6 (127; 358) | 135.7 (117; 354) | 132.5 (97; 364) | 103.5 | 0 / 0 | 322 | 334 |
| 1.6M | 155.5 (114; 401) | 211.2 (175.5; 402) | 159.5 (117.5; 356) | 186.5 (112; 414) | 122.5 | 3 / 0 | | 380.5 |

Fresh cohort 4000313131 at 1.6M: 150.5 (97; 395) sampled, 220.0 (217;
397) greedy, median 131 of 48 games. Online game returns per 200k
actions: 13, 39, 68, 86, 112, 116, 126, 150 (`C22r50` 10, 46, 89, 153,
207, 223; `L22` 9, 45, 105, 159, 228, 271, 276, 282). The three learners
are level through 400k and part after it. `W22` was still rising at the
end.

At 1.6M the wide paddle returns .904 of 4,330 balls (fastest .813) and
the narrow one .748 of 262 (fastest .664); 41 of 96 games reach the
narrow paddle, and 414 of 480 lives are lost on the wide one. At 1.2M:
wide .898, narrow .493 of 73 (`C22r50` .932 and .823); at 800k: .878 and
.452 of 31.

## What the wider input did to the state

Linear readout of the actor's input on own-play dumps (64,000 rows,
48,000 for fitting, held out by lane), mean absolute error in source
pixels. `C22r50` at 1.2M is the matched comparison (same replay, same
budget); its reading was computed for this report.

| | `W22` 1.6M | `W22` 1.2M | `L22` 1.6M | `C22r50` 1.2M |
|---|---:|---:|---:|---:|
| Paddle x | 5.5 | 6.0 | 3.5 | 3.3 |
| Ball x | 5.5 | 4.8 | 5.5 | 5.2 |
| Ball y | 3.7 | 3.2 | 4.6 | 4.7 |
| Landing minus paddle, 0.5 to 1.5 actions out | 9.3 | 9.6 | 6.7 | 8.0 |
| 1.5 to 2.5 | 9.4 | 9.8 | 6.8 | 8.3 |
| 2.5 to 4 | 9.3 | 9.8 | 7.3 | 8.2 |

More pixels per ball improve the read of the ball's height by 1 to
1.5 px, and leave its horizontal position where it was (4.8 to
5.5 px against 5.2 to 5.5). The paddle is read 2 to 3 px worse. I do not
know why; a learner that plays worse also moves its paddle differently,
so this is not a clean statement about the encoder.

The other readings look like a less trained agent, not a different one:

- The policy is less decided. One action before contact, with the paddle
  within 5 px of the landing, it puts .35 of its mass on holding still
  (`O22` .48), and a move is among the last two actions in .82 of wide
  descents (.73 to .80 on the 96x96 learners).
- Holding still still pays: a paddle within 5 px of the landing three
  actions early that holds still returns .995 of 193 balls, and .917 of
  739 when it moves.
- The critic prices a loss as late as before (AUC of the value drop one
  action before contact .65; .66 to .71 on the 96x96 learners), and a
  linear readout of the state separates the balls the agent will lose at
  .89 there.
- The latent-prediction loss over the last 200k actions is .057 (`L22`
  .036); at 1.0M to 1.2M it is .053 against `C22r50`'s .050.

## Limits

- One seed, one budget. The comparison is at equal actions; `W22` cost
  1.51 times the time per action and was still rising, so this does not
  say what a 128x128 learner reaches with more budget.
- Nothing was retuned for the wider input. The world model has 33.5M
  parameters against 17.9M (the posterior's input and the CDP head grow
  with the 8x8 grid), at the same learning rates, batch and train ratio.
  A recipe tuned for 128x128 has not been tried.
- The encoder itself is not the suspect: the ignored GPU reference test
  for the new sizes (`tiny_rgb128_and_rgb160_encoders_match_spatial_reference`)
  was run after the queue and passes, with the nine other ignored
  GPU tests the same filter selects (log `test-wide-encoder-2.log` in the
  root; no kernel message). The filter's eleventh test is a capture
  utility that stops without an explicit output path; it is not a check.
- Readouts are fitted on each learner's own play, and a weaker learner
  visits different states (41 of 96 games reach the narrow paddle
  against 63 and 67 on `X22b` and `O22`).

## Reading

Five changes to what the world model sees or predicts have now been run
on this seed at 1.6M actions (frame pair 353.5, prior continuation 269,
open-loop 317, pixel auxiliary 274, 128x128 input 122.5), and none beats
the unchanged recipe (380.5). The two aimed at position precision did
not deliver precision where the decision needs it: the pixel auxiliary
sharpened the paddle only, the wider input the ball's height only.

## Next

Not launched. The native 128 and 160 sizes stay in the tree as options;
96x96 stays the default. The compact replay for 160x160 is not built.
The declared next candidate is the objective; the run the user chose on
November 18 and that has still not been made is the actor's training
signal. Either needs its own declaration.
