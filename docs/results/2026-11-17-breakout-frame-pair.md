# The two-frame input fails its three tests; the landing readout it was aimed at was miscalibrated

Result of [the frame-pair declaration](../experiments/2026-11-17-breakout-frame-pair.md)
as amended (50k replay, single-frame control). Root
`/x/Temp/kindle-breakout-framepair-20261117`, `queue-15b.json`, guards
`guards-15b`: 40 of 40 jobs in 6 h 53 min (15:05 to 21:58 UTC November
17), no kernel GPU message. `P22r50` (pair, replay 50k, seed 22003, 1.6M
actions) trained in 241.6 minutes at 110 actions per second (the
single-frame recipe: 133) with GPU memory at 14.2 GB; `C22r50` (single
frame, replay 50k, 1.2M) in 151.3 minutes. CPU readings afterwards
(`analysis/` and `scratch-scripts/` in the root).

## Declared tests

| Test at 1.6M, `P22r50` | Bar | Result | Baseline `L22` 1.6M | |
|---|---|---:|---:|---|
| Mechanism: pooled linear landing-minus-paddle readout, one action out | 6 px or less | 8.5 px | 8.5 px | fails |
| Play: fastest balls returned on the narrow paddle, reused cohorts | .85 or more | .83 (706 descents) | .77 (728) | fails |
| Score: pooled median of 96 games, or games past 432 | 400, or 20 | 353.5; 1 | 380.5; 8 | fails |

**The pair is not the lever.** Predictions: readout 5.5 to 7 px (missed),
narrow fastest .80 to .86 (met), median 360 to 410 (just under), 8 to 20
games past 432 (missed), slower than `B22` at 400k (met: 44 against 73).
Of the named risks, the slow start happened and the memory limit forced
the amendment. 0 games at 864.

The mechanism reading said that a readout at 8 px or more sends the work
to image precision. That reading is withdrawn: the readout does not
measure what it was declared to measure (see the calibration below).

## Play

Matched cells (`--life-loss-reset`, 24 games each); pooled over the two
reused cohorts and both modes, 96 games.

| Learner | Point | Cell means | Median | 400 or more | Past 432 | Narrow paddle: all / fastest (descents) | Wide: all / fastest |
|---|---|---|---:|---:|---:|---|---|
| `P22r50` | 400k | 43.3 to 48.9 | 44 | 0 | 0 | | |
| | 800k | 141.2 to 155.5 | 108.5 | 0 | 0 | .70 / .63 (102) | .90 / .77 |
| | 1.2M | 178.7 to 206.2 | 136.5 | 6 | 0 | .77 / .70 (199) | .91 / .83 |
| | 1.6M | 235.0 to 373.8 | 353.5 | 25 | 1 | **.86 / .83** (706) | .94 / .88 |
| `C22r50` | 400k | 56.2 to 78.0 | 60.5 | 0 | 0 | | |
| | 800k | 188.4 to 245.8 | 174.5 | 6 | 0 | .76 / .73 (244) | .92 / .80 |
| | 1.2M | 249.8 to 326.9 | 322 | 17 | 0 | .82 / .76 (441) | .93 / .86 |
| `B22` / `L22` (replay 100k) | 400k | 80.7 to 101.7 | 73 | | | | |
| | 800k | 193.5 to 309.5 | 246 | | | | |
| | 1.2M | 276.3 to 322.3 | 334 | 18 | 3 | .81 / .76 (564) | .94 / .88 |
| | 1.6M | 321.9 to 364.0 | 380.5 | 33 | 8 | .83 / .77 (728) | .95 / .90 |

On the fresh cohort (48 games) `P22r50` plays a median of 155 at 1.2M and
353.5 at 1.6M, 0 games past 432.

- **At equal replay the pair is behind at 1.2M**: median 136.5 against
  `C22r50`'s 322, narrow-paddle return .77 against .82 (declared: at
  least .05 higher), pooled readout 9.2 against 9.9 px (declared: at
  least 1.5 px lower). It does not help there. It learns more slowly at
  every checkpoint (44 / 108.5 / 136.5 against 60.5 / 174.5 / 322) and
  was still rising at the end: online returns of 263 to 292 in the last
  100k actions against 179 to 254 from 1.0M to 1.45M.
- **Its 1.6M point returns the most narrow-paddle balls on record** (.86
  overall, .83 on the fastest, against .83 and .77 for `L22` at the same
  budget), yet scores lower:
  sampled cells average 235 and 261 against greedy 374 and 333, sampled
  games reach the narrow paddle in 28 of 48 against 43 of 48 greedy, and
  the sampled wide-paddle return on the fastest balls is .84 against .91.
  One seed and one checkpoint; there is no single-frame 50k-replay point
  at 1.6M to compare with.
- **The 50k replay does not hurt at 1.2M**: `C22r50` 322 against `B22`
  334, inside the predicted span, with the same narrow-paddle return
  (.82 / .76 against .81 / .76). It is slower early (60.5 against 73 at
  400k, 174.5 against 246 at 800k).

The two-frame input is not retained: 21% more time per action, half the
replay, and a lower score at every budget measured.

Reset state probes: `P22r50` reads the ball within 8 px .57 at 1.2M and
.49 at 1.6M (ball 6.9 / 6.8 and 6.9 / 8.2 px, paddle 5.1 and 5.2 px);
`C22r50` .60 (7.0 / 6.0 px) with the paddle at 8.6 px, against `B22`'s
.62 and 5.6 px. An 8.6 px paddle reading is the level `L22` showed after
its collapse; `C22r50` plays normally at that point and its own-play
dump reads the paddle to 3.6 px. Both new learners show the rise in
latent-prediction loss and KL that preceded `L22`'s collapse (`P22r50`:
loss .038 to .073 and KL 7.7 to 10.9 from 1.0M to 1.45M, back to .055 by 1.5M;
`C22r50`: KL 10.1 at 1.1M) without a fall in play inside their budgets.

## Post hoc: the landing readout was miscalibrated

The declared readout fits one linear map on descending rows at every
distance from contact (1 to 10 actions) and scores it at one distance.
Fitted the same way on the emulator's exact coordinates (ball x and y
now and one action earlier, paddle x) it scores 15.5 to 16.7 px one
action out: the landing point is not a linear function of position and
velocity when the time to contact varies, so 8.5 px never showed missing
information. `landing_calibration2.py` refits per distance, four lane
folds, same dumps (mean absolute error in native pixels, one action
before contact; about 1,900 to 2,050 descents per dump).

| Dump | Declared pooled fit, state | Same pooled fit, exact coordinates | Per-distance fit, state: landing minus paddle / landing x | Per-distance fit, exact coordinates | State: ball dx per action (constant guess) | State, balls returned / lost | Exact coordinates, returned / lost |
|---|---:|---:|---|---:|---|---|---|
| `L22` 1.6M | 8.5 | 16.2 | 4.8 / 3.5 | 1.9 | 1.4 (6.6) | 4.7 / 6.6 | 3.0 / 2.5 |
| `P22r50` 1.6M | 8.5 | 16.2 | 5.7 / 3.6 | 1.7 | 1.3 (6.6) | 5.2 / 9.8 | 3.0 / 2.5 |
| `B22` 1.2M | 8.6 | 15.5 | 5.1 / 3.7 | 1.7 | 1.4 (6.4) | 4.9 / 7.4 | 2.9 / 2.2 |
| `C22r50` 1.2M | 9.9 | 16.2 | 5.8 / 4.0 | 2.1 | 1.7 (6.3) | 5.6 / 7.6 | 3.0 / 2.5 |
| `P22r50` 1.2M | 9.2 | 16.7 | 5.9 / 3.9 | 1.9 | 1.5 (6.1) | 5.7 / 9.0 | 2.9 / 2.5 |
| `H50` 1.2M | 10.0 | 16.2 | 5.9 / 4.2 | 1.7 | 1.7 (6.1) | 5.6 / 9.5 | 2.8 / 2.5 |

(The last two column pairs fit on rows one and two actions out.)

- The single-frame state already holds the ball's velocity (1.3 to 1.7
  px per action against 6.1 to 6.6 for a constant) and where the ball
  lands to 3.5 to 4.2 px, and the landing relative to the paddle to 4.8
  to 5.9 px. The exact coordinates give 1.7 to 2.1 px. These fits use
  about 2,000 rows on 2,560 features, so they bound the state's error
  from above.
- The pair does not improve any of these (5.7 and 5.9 against 4.8, 5.1
  and 5.8 px). Velocity at the encoder was not missing.
- The state reads the balls the agent goes on to lose worse than the
  ones it returns (6.6 to 9.8 against 4.7 to 5.7 px); the exact
  coordinates do not (2.2 to 2.5 against 2.8 to 3.0).

This **corrects [the own-play state-dump result](2026-11-17-breakout-own-play-state-dump.md)**:
its headline, that the state does not hold where the ball lands and is
no better than a constant guess, rested on the pooled fit and does not
stand. Its paddle, present-ball and collapse readings are single-target
fits at one time and are unaffected, as is its critic table.

## Post hoc: the state can tell which balls will be lost; the critic does not

`loss_probe.py`: a per-distance ridge fit to the 0/1 outcome of each
descent (the agent's own play, sampled), four lane folds, scored as AUC
on held-out lanes, beside the AUC of the agent's critic value (low value
predicting the loss) and of the same fit on exact coordinates (ball
position now and one action earlier, paddle x, paddle width).

| Dump (descents, lost) | Critic, k = 1 / 2 / 4 | State, linear | Exact coordinates, linear | Exact coordinates, random-ReLU |
|---|---|---|---|---|
| `L22` 1.6M (1,984, 152) | .63 / .58 / .57 | .83 / .74 / .68 | .79 / .79 / .79 | .85 / .81 / .79 |
| `P22r50` 1.6M (1,958, 152) | .60 / .52 / .48 | .80 / .74 / .66 | .81 / .80 / .79 | .87 / .85 / .75 |
| `B22` 1.2M (2,055, 171) | .51 / .46 / .45 | .85 / .75 / .70 | .81 / .81 / .81 | .90 / .86 / .83 |
| `C22r50` 1.2M (1,870, 174) | .59 / .53 / .49 | .78 / .70 / .64 | .85 / .84 / .82 | .90 / .86 / .80 |
| `H50` 1.2M (1,936, 187) | .50 / .47 / .45 | .76 / .74 / .66 | .82 / .81 / .81 | .89 / .86 / .79 |

One and two actions before contact a linear readout of the actor's own
input separates the balls the agent will lose at AUC .70 to .85; the
critic, reading the same input, is at .46 to .63. The losses are
foreseeable from the geometry four actions out (.79 to .82 from exact
coordinates) and the state keeps part of that (.64 to .70). So the
information that a ball is about to be lost is in the state, and the
value function does not price it. A lost life ends the training episode,
so that price is the whole remaining return.

Why the critic misses it is not established. The continuation head is
trained only on posterior states, which have seen the frame in which the
life ended (loss .021 to .022 against a floor of .0204 for its
discounted target and .036 for a constant), while the actor and critic
learn from imagined rollouts that ask the same head about prior states.
Imagined rollouts do end: on `L22` from 0.5M to 2.0M the mean imagined
continuation is short of the discount by .0042 to .0061 per step against
.0027 to .0034 in replay, so imagination ends lives 1.5 to 2 times as
often as play does. Whether it ends them in the right states was not
measured. The one earlier measurement of that head on priors, on [the
guided whole-game learner](2026-11-04-cdp-breakout-prior-continuation-arrivals.md),
found it below .5 on 0 of 24 true endings. [The next
declaration](../experiments/2026-11-17-breakout-prior-continuation-life.md)
supervises the head on one-step priors.

## Limits

One seed per arm on reused cohorts plus one fresh cohort for the pair.
The pair and the control share the replay size but the pair's first
layer is wider, and there is no single-frame 50k-replay point at 1.6M.
The calibration and loss probes are post hoc, were not declared, and
score linear fits on about 2,000 descents per dump; AUCs on 150 to 190
lost balls carry a standard error near .02. Critic AUC uses the raw
value, which also varies with the score and the bricks left. The 864 /
90% gate is not met.
