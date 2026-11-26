# The frozen-encoder schedule on held-out seed 5011: no collapse, 382 at 2.4M, and greedy play that stalls at the serve

Result of [the second-seed declaration](../experiments/2026-11-20-breakout-encoder-freeze-seed5011.md).
Root `/x/Temp/kindle-breakout-collapse-20261120`, `queue-26.json` (29
jobs), unit `kindle-breakout-freeze5011-20261120`, 23:02 UTC November 20
to 05:39 UTC November 21, all jobs complete, no GPU fault. `F50` is the
held-out learner `H50`'s own training command (life-terminal recipe,
12m, replay 100k, seed 5011) with `--encoder-freeze-update 75000`, to
2.8M actions: 348.3 minutes, 134.0 actions per second, 174,977 updates,
peak 15.41 GiB of VRAM. CPU readings in `analysis/` (`*-F50*`), batch
`scratch-scripts/replay26.sh`.

**Result.** Test 1 passes, tests 2 and 3 fail. The schedule is safe on
the held-out seed: no collapse, pooled medians 382 at 2.4M and 315.5 at
2.8M against a bar of 250, training returns 320.3 per game over 2.4M to
2.8M and rising through every 400k stretch. The 2.4M checkpoint is the
best held-out point on record (median 382 over 144 games, 57 at 400 or
more, 6 past the first wall; `H50` 276 at 1.2M with none past it). Test
2 asked for 350 at 2.8M and gets 315.5, because **greedy play stalls at
the serve**: at 2.8M 22 of the 72 greedy games run to the action cap
with the policy sitting on NOOP, and at 1.6M all 72 do, scoring 0.
Sampled play serves within about one action at every checkpoint and
scores 271 / 333 / 370 / 351 at 1.6M / 2.0M / 2.4M / 2.8M (`H50` 245 at
1.2M). The gate is not met.

## Declared tests

| Test | Bar | `F50` | |
|---|---|---:|---|
| 1a. Pooled median of 144 games at 2.4M / 2.8M | 250 or more at both | 382 / 315.5 | passes |
| 1b. Training return per game, 2.4M to 2.8M | 200 or more | 320.3 (198 games) | passes |
| 2. Pooled median at 2.8M | 350 or more | 315.5 | fails |
| 3. Gate marker at 2.8M | median 410 or more, or 30 of 144 past 432 | 315.5; 1 of 144 | fails |

The pooled median counts the first three games of each stream. At 2.8M
22 of those 144 games ended at the action cap (25,000 actions) and are
counted with the score they had; counting natural endings only (137
games, later and shorter games filling in) gives 307. Both conventions
give the same verdicts. Sampled play alone is 351 at 2.8M (72 games, 1
past 432); the 2.4M checkpoint would pass test 2 (382) and not test 3
(6 of 144).

## Checks

- **Twin.** `F50`'s 400k, 800k and 1.2M checkpoints are byte-identical
  to `H50`'s (all 184 world tensors) and its first 1,385 training games,
  all that `H50` played, match in step and return.
- **Frozen.** The 12 encoder tensors are identical in the 1.6M, 2.0M,
  2.4M and 2.8M checkpoints while all 172 other world tensors change.
  As with `F22`, the 1.2M checkpoint is 23 updates before the freeze.
- **Losses** (50k-action bins). On this seed the latent loss was already
  rising before the freeze: .031 to .032 over 800k to 1.0M, then .039,
  .041, .043 and .051 over 1.0M to 1.2M (KL 7.0, then 8.2, 8.6, 8.7,
  9.1), while training returns went 193, 172, 161, 157. After the freeze
  the latent loss is .041, .036, .035, .036 and .033 by 1.6M (KL 7.4,
  6.8, 6.6, 6.7, 6.1) and .030 and 5.7 at the end; training returns are
  217, 204, 212, 215 in the first four bins and 260 in the fifth. The
  fall is smaller than `F22`'s (.048 to .024), and it returns the loss
  to its level before the rise.

## Play

First three games per stream, three cohorts (4000313131, 4000777001,
4000022003), 72 sampled and 72 greedy games per checkpoint:

| Checkpoint | Pooled median | Sampled | Greedy | At 400 or more | Past 432 | Best | Games cut at the cap |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1.2M (`H50`) | 276 | 245 | 303 | 15 | 0 | 431 | 0 |
| 1.6M | 8 | 271 | 0 | 8 | 0 | 423 | 72 (all greedy) |
| 2.0M | 340.5 | 333 | 342 | 18 | 0 | 427 | 0 |
| 2.4M | 382 | 370 | 390 | 57 | 6 | 818 | 0 |
| 2.8M | 315.5 | 351 | 254 | 22 | 1 | 475 | 22 (all greedy) |

Balls returned, by paddle width (descents in brackets):

| Checkpoint | Games | Wide | Narrow | Narrow, fastest balls |
|---|---|---:|---:|---:|
| 1.2M (`H50`) | 144 | .928 (7,176) | .771 (874) | .71 (532) |
| 1.6M | 72 sampled | .919 (3,330) | .816 (484) | .77 (288) |
| 2.0M | 144 | .931 (7,108) | .791 (1,116) | .74 (690) |
| 2.4M | 144 | .949 (7,788) | .853 (2,197) | .80 (1,267) |
| 2.8M | 72 sampled | .941 (3,624) | .832 (863) | .78 (491) |
| 2.8M | 72 greedy | .918 (3,073) | .811 (312) | .75 (190) |

At 2.4M sampled and greedy agree (wide .941 and .957, narrow .849 and
.856). `F22` at 2.8M returned .956 and .884.

Training returns per game by 400k stretch:

| Actions | `F50` |
|---|---:|
| 800k to 1.2M (identical to `H50`) | 159.8 (255 games) |
| 1.2M to 1.6M | 232.3 (231) |
| 1.6M to 2.0M | 265.6 (216) |
| 2.0M to 2.4M | 301.6 (200) |
| 2.4M to 2.8M | 320.3 (198) |

Six of the 845 training games after 1.2M pass 432 (best 689); none of
the 1,385 before did. `F22` over 2.4M to 2.8M: 342.5.

## The serve in greedy play

Read from the cell logs (`serve_delay.py`, `cut_games.py`; one emulator
replay, `cut_replay.py`).

- **Sampled play serves at once.** Over all three cohorts the first
  FIRE-type action comes a median of 0 to 1 actions into a life at every
  checkpoint (mean 0.9 to 1.2, longest 10).
- **Greedy at 1.6M never serves.** The most probable action at the
  serve is DOWN, in every game of all three cohorts. An emulator replay
  of one cell shows no FIRE-type action and no ball for the first 16,470
  actions of each game with the paddle standing at one place; after
  that the policy does press FIRE-type actions and no ball appears, and
  the game is cut at 25,000 actions with 0 points.
- **Greedy at 2.0M and 2.4M serves at once** (median 0; longest wait 87
  actions at 2.0M and 0 at 2.4M).
- **Greedy at 2.8M waits on NOOP:** a median of 43 actions per life, 148
  of 889 lives more than 100 actions. The three greedy cells log 39
  games cut at the cap (22 of them among the 72 counted): eight never
  score, the rest stop after their second to fourth lost life with 41 to
  419 points. When this policy does play it also stands still under a
  ball more often: 86 wide-paddle descents end with the paddle more than
  24 px from the landing (sampled 32), and .19 of those are returned.

The policy's entropy in training is 2.2 to 2.5 nats of a possible 2.9 over 18
actions, of which nine serve. About half the probability at the serve
is on serving actions (the sampled policy serves in one action), spread
over nine, and the single most probable action can be one that does not
serve. Greedy play then sits in a state training never showed it for
longer than a few actions. This is a property of reading an
18-action policy by its mode, not of the freeze: earlier learners
showed it too (`L22` at 2.8M and 3.2M, eight games of `X22b`). `F22`
had no cut game, and its greedy policy came close: its longest waits at
the serve are 88, 366, 53 and 0 actions at 1.6M, 2.0M, 2.4M and 2.8M.
Every pooled median in the earlier reports mixes the two modes.

## Other readings

- **State readouts** (own play, linear). Paddle 3.9 / 3.4 px at 2.4M /
  2.8M (`F22` 3.8 / 3.7), ball x 5.3 / 5.0, ball height 4.8 / 5.0,
  landing minus paddle one action out 8.6 / 7.8 (`F22` 7.9 / 7.5).
- **The critic.** Mean value over the life's real return is 1.5 at 2.4M
  and 1.4 at 2.8M (48.4 / 31.4 and 46.9 / 34.3; `F22` 1.9 and 1.7). It
  separates lost balls one action before contact at AUC .69 at both
  points.
- **The habit is unchanged.** In sampled play a move among the last two
  actions in .76 of wide-paddle descents and .75 to .78 of narrow ones;
  from a narrow paddle within 4 px two actions out, holding still
  returns .97 and 1.00 and moving .79.

## Reading

The declared reading for "1 passes, 2 fails" is that the schedule is
safe and the gain does not repeat. The first half holds. The second is
what test 2 says as written, and it is not what the rest of the run
shows: in sampled play the held-out learner improves from 245 at 1.2M
to 351 to 370, its training returns double (159.8 to 320.3), its
narrow-paddle returns go from .771 to .83 to .85, and its 2.4M
checkpoint plays 370 sampled and 390 greedy. What fails at 2.8M is the greedy
mode at the serve. The test stays failed; the schedule is not declared
the default on this result alone.

1. **Two seeds, no collapse with the freeze.** Seed 22003 without it
   collapsed between 2.2M and 2.6M; seed 5011 has no unfrozen run past
   1.2M, so for this seed the freeze is shown safe, not shown necessary.
2. **The cost of the moving target appeared earlier on this seed.** Its
   latent loss rose by two thirds and its training returns fell over the
   200k actions before the freeze, and both turned within 50k actions
   of it. The encoder-share reading taken before launch (paddle share
   .178 at 1.2M, not drifted) did not show this. An earlier freeze is
   the direct test.
3. **Held-out play is still behind the development seed.** Its last
   stretch of training returns (320.3) is where `F22` was over 1.6M to
   2.4M (315.5, 314.3), and its narrow paddle is at .83 to .85 against
   .884; `H50` was behind in the same way at 1.2M. Both frozen runs
   were still rising at 2.8M.
4. **Single checkpoints move.** Sampled medians go 370 then 351, the
   narrow paddle .849 then .832, while training returns rise; this is
   the checkpoint noise seen before (`S25`).

Predictions: test 1 passes (four in five) held; the 2.8M median (320 to
400) came out at 315.5 pooled and 351 sampled; test 2 (one in two)
failed; test 3 (one in eight) failed; narrow paddle at 2.8M (.82 to
.88) came out at .832 sampled. Nothing predicted the greedy stalls.

## Limits

- One run of one held-out seed; no unfrozen control past 1.2M on it.
- The freeze point was fitted to seed 22003's curve.
- The serve readings are from the cell logs and one emulator replay;
  the policy's probabilities at the serve were not dumped.
- Greedy and sampled play are each 72 games per point; the pooled
  figure at 1.6M and 2.8M is a mix of a playing mode and a stalled one
  and should not be compared with other runs' pooled medians.
- Why the encoder drifts is still not established.

## Next

Not declared here. The open questions this run leaves, in the order
they bear on the gate: how far the frozen-encoder learner climbs with
more actions (both seeds were rising at 2.8M); whether an earlier
freeze gets there sooner (seed 5011's loss turned up at 1.0M); and
which play mode the declared tests should read (sampled play is the
policy that was trained; greedy play fails at the serve on some
checkpoints for a reason unrelated to the change under test).
