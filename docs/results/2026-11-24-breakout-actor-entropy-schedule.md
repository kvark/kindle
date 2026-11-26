# The actor's entropy coefficient cut to a tenth from 2.4M: the policy is far sharper and returns no more balls

Result of [the entropy-schedule declaration](../experiments/2026-11-24-breakout-actor-entropy-schedule.md).
`N60` is [`D60`](2026-11-24-breakout-recipe-seed6007.md)'s training
command (the fixed recipe on seed 6007) plus
`--actor-entropy-drop-update 150000 --actor-entropy-drop-factor 0.1`:
from 2.4M actions, where the learning rates are cut to a quarter, the
actor's entropy coefficient is 3e-5 instead of 3e-4. Root
`/x/Temp/kindle-breakout-entropy-20261124`, `queue-37.json` (21 jobs,
00:18 to 08:10 UTC November 24; training 454 minutes, 132 actions per
second, peak 15.4 GiB), under the reviewed guard, no kernel fault.
Readings in `analysis/` under the root (`*N60*`,
`scratch-scripts/replay37.sh`). The assistant's design; the user was not
asked.

**Result.** Tests 1 and 2 pass, tests 3 and 4 fail, and the declared
reading for that pattern applies: a policy trained sharper returns no
more balls. The cut does what it was meant to do to the policy (entropy
over the three move classes .29 nats against `D60`'s .72 on the 3.6M
dump, total 1.31 against 2.46) and nothing measurable to the game:
frozen play returns the same share of balls in both modes (sampled .968
wide and .903 narrow against .966 and .899; greedy .975 and .916
against .975 and .920), pools to 418 / 424.5 / 424 at 2.8M / 3.2M / 3.6M
with 32 / 44 / 41 of 144 games past the first wall (`D60` 424 / 423 /
424.5 with 45 / 40 / 39), and trains at 422.9 per game over 2.8M to 3.6M
against 412.2 (difference 10.7, standard error 8.6; bar 428). The cut is
not added to the recipe. With [tempered play](2026-11-22-breakout-tempered-play.md)
the actor's entropy is closed as a lever.

## Checks

- 21 of 21 jobs complete; no `amdgpu` or out-of-memory line in the
  kernel log over the queue; the boot is the reviewed one.
- `N60` is a true twin of `D60` to the cut: all 184 tensors of the
  400k, 1.2M and 2.4M checkpoints are identical, and the first 2,346
  training games are identical (the last ends at action 2,399,688). At
  2.8M 12 of 184 tensors are identical (the frozen encoder's).
- No training game was cut (2,783 natural games) and none ends at zero
  after the cut.

## Tests

| Test | Bar | `N60` | `D60` | |
|---|---|---:|---:|---|
| 1. Mechanism: between-class policy entropy on the 3.6M own-play dump (nats) | .60 or less | .29 | .72 | passes |
| 2. Safety: training return per game in each 400k stretch from 2.4M; sampled cells with a cut or zero game | 380 or more; none | 409.2 / 434.3 / 411.9; none in 9 cells | 405.6 / 405.7 / 418.9; none | passes |
| 3. Level: training return per game, 2.8M to 3.6M | 428 or more | 422.9 (288 games, se 6.5) | 412.2 (307 games, se 5.7) | fails |
| 4. Play: sampled play at 3.6M, median over 72 games and games past 432 | 425 or more and 20 or more | 422 and 15 | 418 and 14 | fails |

The gate-level reading (pooled median 430 or more and 72 or more of 144
games past 432 at any checkpoint, no game cut) is not met: the closest
checkpoint is 3.2M with 424.5 and 44.

## Training

Returns per training game by 400k stretch (games; standard error):

| Actions | `N60` | `D60` |
|---|---:|---:|
| 2.0M to 2.4M (identical, before the cut) | 347.8 (176; 8.5) | 347.8 (176; 8.5) |
| 2.4M to 2.8M | 409.2 (149; 8.6) | 405.6 (152; 8.1) |
| 2.8M to 3.2M | 434.3 (142; 11.5) | 405.7 (157; 8.6) |
| 3.2M to 3.6M | 411.9 (146; 6.1) | 418.9 (150; 7.3) |
| 2.8M to 3.6M | 422.9 (288; 6.5) | 412.2 (307; 5.7) |
| 2.4M to 3.6M | 418.3 (437; 5.2) | 410.0 (459; 4.6) |

- The two runs do not differ: 10.7 over 2.8M to 3.6M (standard error
  8.6) and 8.3 over all 1.2M actions after the cut (standard error 6.9).
  The middle stretch is the highest on record and the next one is below
  `D60`'s; the median training game is 417 / 418 / 420 against 413.5 /
  412 / 416.
- Training games past 432 after the cut: 83 of 437 (19%) against 70 of
  459 (15%). Lives lost per 1,000 actions: 1.34 to 1.57 by 100k bin
  (mean 1.46) against 1.47 to 1.60 (mean 1.53). Both lean the same way
  as the return and neither is outside the noise of one run.
- **The policy entropy reported in training falls as intended**: 2.44
  to 2.48 nats in the bins before the cut, then 2.16, 1.83, 1.65, 1.52,
  1.41 over the first 500k actions and 1.28 to 1.52 from 2.9M on (`D60`
  2.33 to 2.48 throughout). It was still drifting down at the end
  (1.28 in the last bin).
- The world model is untouched: the latent loss is .035 to .040 after
  the cut and .033 in the last bin (`D60` .036 to .039, then .034), the
  dynamics KL 5.5 to 6.4 (5.8 to 6.2), the mean absolute advantage .015
  to .017 in both.

Relative distance between checkpoints 400k apart:

| | 2.4M to 2.8M | 2.8M to 3.2M | 3.2M to 3.6M | `D60`, same pairs |
|---|---:|---:|---:|---|
| Actor | .077 | .063 | .065 | .077 / .065 / .064 |
| Actor, output layer | .116 | .099 | .084 | .085 / .071 / .068 |
| Dynamics core | .177 | .165 | .156 | .180 / .166 / .159 |
| Posterior | .122 | .112 | .105 | .122 / .112 / .107 |
| Future predictor, first layer | .224 | .210 | .201 | .226 / .211 / .201 |

The actor as a whole moves as far as `D60`'s (.124 from its 2.4M
weights at 3.6M against .127); only its output layer moves further. The
two runs' actors are .092 / .123 / .147 apart at 2.8M / 3.2M / 3.6M, as
far from each other as each is from its own starting point.

## Play

Three cohorts (4000313131, 4000777001, 4000022003), 72 sampled and 72
greedy games per checkpoint, `--life-loss-reset`. No game is cut at the
cap and no game ends at zero in any of the 18 cells.

| Checkpoint | Pooled median | At 400 or more | Past 432 | Sampled median | Greedy median | Sampled past 432 | Greedy past 432 | Best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `D60` 2.8M | 424 | 117 | 45 | 420 | 428 | 14 | 31 | 864 |
| `D60` 3.2M | 423 | 108 | 40 | 421.5 | 426 | 14 | 26 | 864 |
| `D60` 3.6M | 424.5 | 106 | 39 | 418 | 427.5 | 14 | 25 | 859 |
| `N60` 2.8M | 418 | 99 | 32 | 418 | 416.5 | 15 | 17 | 850 |
| `N60` 3.2M | 424.5 | 111 | 44 | 423.5 | 428 | 17 | 27 | 856 |
| `N60` 3.6M | 424 | 113 | 41 | 422 | 427.5 | 15 | 26 | 864 |

- Sampled play is 418 / 423.5 / 422 with 15 / 17 / 15 of 72 past the
  wall against `D60`'s 420 / 421.5 / 418 with 14 at each checkpoint: 47
  of 216 sampled games past the wall against 42.
- Greedy play is 416.5 / 428 / 427.5 with 17 / 27 / 26 against 428 /
  426 / 427.5 with 31 / 26 / 25: 70 of 216 against 82. It is still the
  stronger mode at 3.2M and 3.6M.
- Per cohort at 3.6M the pooled medians are 424, 424 and 423 with 14,
  14 and 13 games past 432.
- Both modes serve: the delay from a lost life to the serve is 0
  actions in every greedy life and has a median of 0 in sampled play
  (maximum 1, 5 and 14 actions at the three checkpoints).

Balls returned, by paddle width (descents in brackets):

| Checkpoint | Sampled wide | Sampled narrow | Greedy wide | Greedy narrow | Narrow fastest, sampled / greedy |
|---|---:|---:|---:|---:|---:|
| `N60` 2.8M | .968 (3,984) | .893 (2,172) | .974 (4,061) | .897 (2,466) | .86 / .86 |
| `N60` 3.2M | .968 (3,924) | .903 (2,433) | .977 (4,199) | .920 (3,331) | .86 / .87 |
| `N60` 3.6M | .968 (3,583) | .910 (2,719) | .973 (3,853) | .926 (3,469) | .87 / .88 |
| `N60` pooled | .968 (11,491) | .903 (7,324) | .975 (12,113) | .916 (9,266) | .86 / .87 |
| `D60` pooled | .966 (11,529) | .899 (6,862) | .975 (12,588) | .920 (9,611) | .85 / .89 |

- No rate moves by more than its sampling error (about .002 on the wide
  paddle and .005 on the narrow one for a difference of two pooled
  rates). Sampled play of the sharper policy does not reach the greedy
  rates of either run.
- The move into contact is unchanged: a move among the last two actions
  in .53 to .68 of wide descents and .54 to .66 of narrow ones (`D60`
  .58 to .68 and .56 to .64); from a narrow paddle within 4 px of the
  landing two actions before the end the agent holds still in .54 to
  .66 of cases (`D60` .56 to .68). Holding returns .98 to 1.00 and
  moving again .89 to .92, as before.

## The 3.6M dump

Own play, 64,000 rows, linear readouts held out by lane:

| | `N60` 3.6M | `D60` 3.6M |
|---|---:|---:|
| Policy entropy, total / between move classes / within a class (nats) | 1.31 / .29 / 1.03 | 2.46 / .72 / 1.74 |
| Mean mass on the top move class | .89 | .64 |
| Between-class entropy, ball rising / descending below row 170 | .33 / .15 | .86 / .33 |
| Moves, share of actions (while the ball rises) | .86 (.94) | .59 (.63) |
| Paddle in place two actions / one action before contact: mass on holding | .94 / .78 | .79 / .69 |
| Policy entropy over the last four actions of a wide descent | .76 to .94 | 1.86 to 2.25 |
| Descents lost | 101 of 1,886 | 107 of 1,879 |
| Paddle (px, mean abs) | 2.2 | 2.3 |
| Ball x / ball height | 3.3 / 3.3 | 3.3 / 3.3 |
| Landing minus paddle, one action out | 6.4 | 6.3 |
| Critic AUC for a lost ball, at contact / one action before | .90 / .71 | .88 / .73 |
| Mean value over the life's real return | 1.91 (50.3 / 26.4) | 1.49 (51.7 / 34.7) |

- The policy commits everywhere, and most where it matters least: with
  the ball rising it now moves in .94 of actions (`D60` .63), to the
  left in .70 of them. Where the decision is made it is sharper too: from
  a paddle in place two actions before contact the mass on holding
  still is .94 against .79.
- The same descents are lost (5.4% against 5.7% on the dumps). The
  state reads the positions as before and the critic prices a lost ball
  as late as before.
- The value is further above the real return per life (1.91 times
  against 1.49) on a dump of 101 lives; one dump, not read further.

## Reading

The declaration's second reading applies. The entropy coefficient does
set the spread of a trained policy at this stage (test 1), the cut is
safe (test 2: no stretch below 380, both modes serve, nothing cut), and
it changes nothing that scores: return rates, the move into contact,
the games past the wall and the training return are `D60`'s within the
noise of one run. What the policy was spreading its mass over was not
the choice that loses the ball. With the mass concentrated, the agent
still arrives at the landing late and moving in more than half of its
descents, and loses the ball in 3% of wide-paddle and 10% of
narrow-paddle descents in sampled play, as before.

Together with tempered play (a trained policy sharpened at decision
time returns no more balls) the actor's entropy is closed, in play and
in training. The recipe stays `D60`'s: life-terminal training, encoder
frozen from 1.2M, learning rates cut to a quarter from 2.4M, 3.6M
actions.

What the result leaves is what the November 19 planner readings
already pointed to: the ball-to-paddle offset is read from the state to
6 px, and most games past the wall needs it to under 1 px. Nothing on
the actor's side tried so far (tempered play, completed returns, the
entropy cut) and nothing on the world model's side that kept the 96x96
input moved that reading. Which direction follows (an input aligned 1:1
with the game's pixels, a loss sharp on small objects, or accepting the
level) is the user's choice and is open.

## Predictions

Test 1 (three in five) passes; test 2 (three in four) passes; test 3
(one in five) fails; test 4 (one in five) fails; the gate-level reading
(one in thirty) fails. This is the pattern the declared odds made most
likely.

## Limits

- One seed, the best of the three, and one run per arm: a difference of
  less than about 17 points of training return (two standard errors)
  could not have been seen, and the 10.7 measured is inside that.
- One factor (a tenth) at one point (2.4M), with the learning rates at
  a quarter; the training policy's entropy was still falling slowly at
  3.6M. A cut from the start of training, or to zero, is not tested.
- Three cohorts that every run since November 17 has used; 24 games per
  cell.
- The dump comparison is one dump per run (about 100 lost lives each).
