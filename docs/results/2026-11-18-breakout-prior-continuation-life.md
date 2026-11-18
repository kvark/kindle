# Supervising the continuation head on one-step priors changes nothing that matters

Result of [the prior-continuation declaration](../experiments/2026-11-17-breakout-prior-continuation-life.md).
Root `/x/Temp/kindle-breakout-priorcont-20261117`, `queue-16.json`, guards
`guards-16`: 23 of 23 jobs in 3 h 33 min (23:32 UTC November 17 to 03:05
UTC November 18), no kernel GPU message. `T22` (the life-terminal recipe
plus `--prior-continuation-scale 1`, seed 22003, replay 100k, 1.6M actions)
trained in 196.2 minutes at 136 actions per second, the same speed as the
recipe without the term. CPU readings afterwards (`analysis/` and
`scratch-scripts/` in the root).

## Declared tests

| Test, `T22` at 1.6M | Bar | Result | Baseline `L22` 1.6M | |
|---|---|---:|---:|---|
| Priors predict the ending: prior-continuation loss, last 200k actions | .028 or less | .0216 | not logged | passes |
| The critic prices the loss: AUC of low value, one / two actions before contact | .72 / .65 | .54 / .51 | .63 / .58 | fails |
| Play: fastest balls returned on the narrow paddle, reused cohorts | .85 or more | .73 (324 descents) | .77 (728) | fails |
| Score: pooled median of 96 games, or games past 432 | 400, or 20 | 269; 0 | 380.5; 8 | fails |

**The one-step prior ending is not the lever.** The declared reading for
this outcome is that the critic's targets are next. Predictions: prior
loss .024 to .028 (better than predicted), critic AUC .66 to .78 and .58
to .70 (missed), narrow fastest .78 to .86 (missed), median 340 to 420
(missed), 5 to 20 games past 432 (missed). Of the named risks, none of the
three happened as described: there is no collapse and no memory problem,
and sampled play is not systematically below greedy. 0 games at 864.

## Play

Pooled over the two reused cohorts, sampled and greedy, 96 games per
point; cell means in brackets.

| Point | `T22` median (cell means) | 400 or more / past 432 | `B22`/`L22` median | 400 or more / past 432 |
|---|---:|---:|---:|---:|
| 400k | 62 (61.2 to 68.4) | 0 / 0 | 73 | 0 / 0 |
| 800k | 268.5 (151.2 to 301.0) | 9 / 1 | 246 | 6 / 1 |
| 1.2M | 221 (200.4 to 246.0) | 9 / 0 | 334 | 18 / 3 |
| 1.6M | 269 (215.8 to 301.0) | 7 / 0 | 380.5 | 33 / 8 |

On the fresh cohort 4000313131 at 1.6M the median is 297 over 48 games
(one game at 530). The learner stops improving at 800k; online returns
sit at 201 to 286 per game from 900k on (`L22`: 235 to 302). A median of
269 is inside the range the same recipe shows between seeds at 1.2M
(276 on held-out seed 5011, 334 and 350 on the development seeds), so
this run does not show that the term hurts. It does show that it does
not help.

Balls returned by paddle width at 1.6M (reused cohorts): wide .930 over
all descents and .85 on the fastest, narrow .789 and .73; `L22`: .950 and
.90, .830 and .77. Only 59 of 96 games reach the narrow paddle (`L22`:
83), so more of `T22`'s lives are lost on the wide paddle. At 1.2M: wide
.922 / .83, narrow .801 / .75 (`B22`: .941 / .88, .806 / .76).

State probe with the belief reset: ball within 8 px .70 at 1.2M and .68 at
1.6M (`B22` at 1.2M: .62), paddle 4.9 and 5.2 px.

## What the term did

| Mean over 200k actions | `T22` 1.0M to 1.6M | `L22` 1.0M to 1.6M |
|---|---:|---:|
| Posterior continuation loss | .0215 to .0217 | .0218 |
| One-step prior continuation loss | .0216 to .0218 | not logged |
| Imagined steps that end, per step | .0076 to .0086 | .0071 to .0082 |
| Replay steps that end, per step | .0057 to .0059 | .0057 to .0058 |

The perfect-prediction floor is .0204 and a constant predictor scores
.036. The one-step prior reaches the posterior's loss, so one action
before the life ends the state already says so, with or without the term.
Imagination still ends lives about 1.3 to 1.5 times as often as replay
does, as before.

## Post hoc: the declared critic test understates the critic, and the experiment aimed at the wrong step

Two things about my own instruments, found while reading this result.

**The life ends two to three actions after the ball passes the paddle.**
Aligning the dumps on the action that ends the life: the critic's value is
at 0% to 2% of its earlier level on the last state of the life and 2% to
14% one action earlier, when the ball is already below the paddle; two
actions earlier (ball at the paddle row) it is at 45% to 74%, and three
actions earlier (one action above the paddle) at 79% to 89%. A one-step
prior is therefore asked a trivial question: the ball is already gone when
it has to predict the ending. The declaration treated the one-step prior
as the link between contact and loss; it is not.

**Raw value varies with the score, so its AUC understates what the critic
knows.** The fair reading is the value's drop within the descent. As a
ratio to the value at the start of the same descent:

| Dump | One action before contact: returned / lost | AUC, ratio | AUC, raw (declared) | Two actions before: AUC ratio / raw |
|---|---:|---:|---:|---:|
| `L22` 1.6M | .94 / .77 | .71 | .63 | .64 / .58 |
| `T22` 1.6M | .98 / .86 | .66 | .54 | .59 / .51 |
| `B22` 1.2M | .99 / .88 | .68 | .51 | .59 / .46 |

By the ratio `T22` still misses the declared bars (.72 and .65) and is not
better than `L22`. The finding of the [frame-pair report](2026-11-17-breakout-frame-pair.md),
that the state separates the balls the agent will lose (AUC .76 to .85 one
action before contact and .70 to .75 two before; here .80 and .75 on `T22`) better than the critic does,
stands with a smaller gap than reported there: the critic is at .66 to .71
and .59 to .64, not .46 to .63. One action before a loss the critic's
value has fallen by 12% to 23%.

Per-distance landing fits on `T22` at 1.6M: landing relative to the paddle
6.6 px one action out (`L22` 4.8, exact coordinates 2.0).

## Limits

One seed. The baseline `L22` is the best of the three learners on this
recipe and seed 22003 has no second run of the unchanged recipe past
1.2M, so the gap to it mixes the term with run-to-run variation. The
critic readings use the agent's own play, so they measure the value of
the states the policy reaches.
