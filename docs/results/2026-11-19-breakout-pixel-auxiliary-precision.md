# Pixel reconstruction sharpens the paddle read, not the offset the decision needs; play does not improve

Result of [the pixel-auxiliary precision declaration](../experiments/2026-11-18-breakout-pixel-auxiliary-precision.md).
Root `/x/Temp/kindle-breakout-pixelaux-20261118`, `queue-20b.json` (23 jobs,
all complete, no GPU fault, no amdgpu kernel message during the queue),
guards `guards-20b`, unit `kindle-breakout-pixelaux-b-20261119`, 06:26 to
12:08 UTC November 19. `X22b` is the open-loop recipe `O22` plus
`--world-pixel-scale 1` (an RGB decoder on posterior states, plain squared
error), seed 22003, 1.6M actions: 324.6 minutes, 82.2 actions per second
(`O22` 123.8, so 1.51 times the time per action), 99,977 updates, peak
16.68 GiB of VRAM (`O22` 15.57). The card's budget never fell below
19.41 GiB this time, so the run kept 2.73 GiB free throughout. CPU readings
in `analysis/` (`*-X22-*`), readers in `scratch-scripts/`, batch
`replay20.sh`.

**Result.** All four declared tests fail. The decoder does what a decoder
does to the paddle: the linear readout of paddle x improves from 3.5 to
2.3 px. The quantity the decision needs, where the ball lands relative to
the paddle, does not move (6.5 / 6.8 / 6.3 px against 6.6 / 6.4 / 6.2).
The policy near contact is less decided than the baseline's, not more,
the ball is returned as often on the wide paddle and less often on the
narrow one, and the pooled median is 274 with no game past the first
wall. By the declared reading, squared-error reconstruction does not
sharpen the state where it matters and the auxiliary is not retained.

## Declared tests

At 1.6M. Baseline `O22` at 1.6M (same recipe and seed without the
auxiliary); `L22` is the recipe without the open-loop term.

| Test | Bar | `X22b` | `O22` | |
|---|---|---:|---:|---|
| 1. State readout, paddle x, mean error | 2.5 px or less | 2.3 | 3.5 | passes |
| 1. Landing minus paddle, 0.5 to 1.5 / 1.5 to 2.5 / 2.5 to 4 actions out | 5.0 px or less in each | 6.5 / 6.8 / 6.3 | 6.6 / 6.4 / 6.2 | fails |
| 2. One action before contact, wide paddle: still mass within 5 px minus still mass at 5 to 12 px | +.15 or more | -.21 (.41 against .62) | +.05 (.48 against .43) | fails, wrong sign |
| 3. Balls returned, wide / narrow paddle | .955 / .85 or more | .931 (4,818) / .754 (581) | .929 / .803 | fails |
| 4. Pooled median of 96 games; games past 432 | 400, or 20 games | 274; 0 | 317; 0 | fails |

Test 1 needs both parts and fails on the second. Predictions: paddle 2.2
to 3.2 px (2.3), wide paddle .93 to .96 (.931) and time per action 1.3 to
1.6 times `O22` (1.51) held; landing minus paddle 5.0 to 6.5 px (6.5 /
6.8 / 6.3, one band outside), narrow paddle .78 to .86 (.754) and median
280 to 400 (274) did not. I gave test 1 one chance in two and test 3 one
in four; both failed.

## Play

24 games per cell, mean (median; best), belief reset at each lost life.

| Actions | 777001 sampled | 777001 greedy | 022003 sampled | 022003 greedy | Pooled median | 400 or more / past 432, of 96 | `O22` | `L22` |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| 400k | 39.2 (38.5; 73) | 39.2 (40.5; 57) | 37.3 (34; 76) | 37.7 (38.5; 61) | 38.5 | 0 / 0 | 50 | 73 |
| 800k | 103.9 (92; 299) | 109.2 (84; 360) | 105.8 (82.5; 381) | 114.4 (93; 340) | 88 | 0 / 0 | 208 | 246 |
| 1.2M | 259.0 (306.5; 424) | 281.5 (312.5; 428) | 215.0 (178.5; 426) | 293.8 (328.5; 424) | 311.5 | 12 / 0 | 319 | 334 |
| 1.6M | 235.8 (243.5; 393) | 296.0 (333.5; 425) | 228.6 (258.5; 419) | 262.3 (309.5; 408) | 274 | 5 / 0 | 317 | 380.5 |

Fresh cohort 4000313131 at 1.6M: 240.1 (282.5; 426) sampled, 307.1 (332;
404) greedy, median 311.5 of 48 games. The learner is slower than `O22`
through 800k and level with it at 1.2M; its 1.6M point is lower than its
1.2M point. Online game returns per 200k actions: 5, 21, 53, 114, 156,
205, 220, 226 (`O22` 10, 31, 84, 147, 218, 232, 248, 229).

At 1.6M the wide paddle returns .931 of 4,818 balls (fastest .866) and
the narrow one .754 of 581 (fastest .680); 63 of 96 games reach the
narrow paddle (`O22` 67). Lives lost: 333 wide, 143 narrow. At 1.2M: wide
.929, narrow .794.

## What the auxiliary did

Linear readout of the actor's input on own-play dumps (64,000 rows,
48,000 for fitting, held out by lane), mean absolute error in source
pixels:

| | `X22b` 1.6M | `X22b` 1.2M | `O22` 1.6M |
|---|---:|---:|---:|
| Paddle x | 2.3 | 1.9 | 3.5 |
| Ball x | 4.4 | 5.9 | 3.8 |
| Ball y | 3.0 | 3.6 | 3.7 |
| Landing minus paddle, 0.5 to 1.5 actions out | 6.5 | 6.2 | 6.6 |
| 1.5 to 2.5 | 6.8 | 6.8 | 6.4 |
| 2.5 to 4 | 6.3 | 6.6 | 6.2 |
| 4 to 8 | 8.6 | 9.4 | 8.3 |

The random-feature readout agrees (landing minus paddle 7.9 / 8.1 / 8.1
against `O22`'s 8.2 / 8.2 / 8.3). The pinned reset probe (3,072 held-out
rows) reads the paddle at 3.7 px against `O22`'s 4.9 and the ball within
4 px in .32 of rows against .25 (within 8 px .65 against .63), with a
larger mean horizontal ball error (9.8 px against 6.0; 5.4 at 1.2M). The
largest, slowest object is drawn more exactly; the small fast one is not.
A likely reason: squared error pays for the paddle and the wall and
almost nothing for a ball of a few pixels. The November 10 world runs
weighted motion for that reason; that weighting is a hand prior and stays
out.

The latent losses end lower than `O22`'s (one-step prediction .032
against .059 over the last 200k actions, eight-step open-loop prediction
.065 against .103; `O22`'s were rising at the end). The critic is
unchanged: AUC of the value's drop one action before contact .78 (`O22`
.77), state readout .88.

## The policy is less decided

Observations on the 1.6M own-play dump, not declared tests.

| Wide paddle, one action before contact | `X22b` | `O22` |
|---|---:|---:|
| Still mass, paddle within 5 px of the landing | .41 | .48 |
| Still mass, 5 to 12 px off | .62 | .43 |
| Mass toward the landing, more than 12 px off | .68 | .76 |
| Policy entropy, nats | 1.43 | .80 |

Entropy over all rows is 1.95 nats against 1.26. From a paddle within
5 px two actions before the end the agent moves again in .73 of wide
descents (`O22` .54, `L22` .49), a move among the last two actions
appears in .835 of wide descents (`O22` .77), and holding still from in
place no longer returns more balls than moving (.940 of 215 against .979
of 583; `O22` .980 against .961). Sampled cells trail greedy cells by 34
to 67 points at 1.6M. The pixel objective has made actors more diffuse
before (the November 10 pixel-world agents lost in sampled play for the
same reason); here it does so as an auxiliary on the CDP recipe.

Eight greedy games at 1.6M (three on cohort 022003, five on the fresh
cohort) ran to the 100,000-frame cap with 0 to 6 points: after a lost
ball the greedy policy never serves. `O22`'s 1.6M greedy cells and this
learner's earlier ones have none. Two of the eight fall among a stream's first three
games; the table above counts natural games only, and counting the two
gives a pooled median of 269.5 instead of 274. `L22` showed the same
stall at 2.8M and 3.2M, after its collapse.

## Reading

Declared for a failed test 1: squared-error reconstruction does not
sharpen the state; precision then needs the input or the architecture, or
is not available to a readout at all; the auxiliary is not retained. That
stands, with one qualification the test itself supplies: the paddle read
did sharpen and nothing followed from it, so the paddle's position was
not what limited the offset.

This is the fourth change to the world model's side of the recipe on this
seed (two-frame input, continuation on priors, open-loop supervision,
pixel reconstruction). Each moved its own measure and none moved play.
Medians at 1.6M: 353.5, 269, 317, 274 against the unchanged recipe's
380.5.

## Limits

One seed, development cohorts plus one fresh cohort, one checkpoint
schedule. The readouts are linear and random-feature fits; a state that
holds the offset in a form neither finds would read as imprecise. The
dump samples the agent's own play, so `X22b` and `O22` are read on
different state distributions. The first attempt (`X22`) stopped at 333k
actions on the memory-headroom check and is not part of any figure.

## Next

Not launched. The remaining untested side is the actor's training signal
on the open-loop recipe: late in training the return scale is 200 to 300
and the mean absolute normalized advantage about .015, so the difference
between returning and losing one ball is a small part of what the actor
is trained on. A wider native input is the other candidate; it needs
native work (the runner stops at 96x96) and memory the 96x96 recipe
already nearly exhausts (15.6 of about 19.4 GiB).
