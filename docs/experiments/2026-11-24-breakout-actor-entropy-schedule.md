# Declaration: the fixed recipe with the actor's entropy coefficient cut to a tenth from 2.4M

Prepared 00:18 UTC November 24 and launched by hand at 00:18 UTC
(`queue-37.json`, unit `kindle-breakout-entropy6007-20261124`). The
assistant's design; the user has not been asked. On November 23 the user was told twice that this run would
follow `D60` unless they chose another direction (an aligned 1:1 input
or a loss sharp on small objects); no answer has come. The run does not
settle that choice and can be stopped at any time. One training run and
its evaluation; nothing is chained.

## Why

[`D60`](../results/2026-11-24-breakout-recipe-seed6007.md) fixed the
recipe and left one cheap item on the actor's side untested in
training. On every seed greedy play returns more balls than sampled
play (`D60` .975 against .966 on the wide paddle, .920 against .899 on
the narrow one), the policy's distribution over the three move classes
(still, right, left) is far from decided (between-class entropy .72 to
.78 of a possible 1.10 nats on the 3.6M dumps), and the lost ball is a
paddle that moves into contact. [Tempered play](../results/2026-11-22-breakout-tempered-play.md)
showed that sharpening a trained policy at decision time returns no
more balls; it did not test training with a smaller entropy bonus,
which lets the actor commit where its advantages point while the 1%
unimix keeps every action possible.

The option is new (`--actor-entropy-drop-update` /
`--actor-entropy-drop-factor`, commit da1519b7; the entropy coefficient
3e-4 becomes a graph input multiplied by the factor from a learner
step). The extension was rebuilt at 00:10 UTC November 24 and a
three-run smoke (`queue-37-smoke.json`, 16,384 actions each,
`analysis/smoke-identity.txt` under the root) shows:

- the rebuilt extension without the option reproduces the November 22
  smoke bit for bit (184 of 184 tensors);
- with the schedule set beyond the run, every tensor and every learner
  report equals the run without it, so the option changes nothing
  before its update;
- with the coefficient cut to a tenth from update 300, tensors are
  identical at 4,096 actions (update 256) and all differ at 8,192; the
  first differing learner report is step 301, and by step 1,000 the
  policy entropy is .79 nats against 2.21. On an early agent the
  coefficient does hold the policy's spread.

## Run

`N60`: `D60`'s training command plus
`--actor-entropy-drop-update 150000 --actor-entropy-drop-factor 0.1`
(seed 6007, encoder frozen from update 75,000, every learning rate cut
to a quarter from update 150,000, 3.6M actions, a checkpoint every
400,000). Root `/x/Temp/kindle-breakout-entropy-20261124`,
`queue-37.json` built by `make_entropy6007_queue.py`, which asserts that
the two flags are the only difference from `D60`'s command. 21 jobs:
training, then sampled and greedy cells on cohorts 4000313131,
4000777001 and 4000022003 at 2.8M, 3.2M and 3.6M (24 games per cell,
`--life-loss-reset`), the own-play dump and the reset probe at 3.6M.
About eight hours on the card under the reviewed guard (review file of
November 22, this boot only).

The cut starts at the same update as the learning-rate drop, so `N60`
is a twin of `D60` to 2.4M and differs from it by the entropy
coefficient alone afterwards.

Checks: all 184 world tensors equal `D60`'s at 400k, 1.2M and 2.4M; the
two runs' training games are identical up to the cut; 21 of 21 jobs;
the kernel log.

## Tests

1. **Mechanism:** on the 3.6M own-play dump the policy's between-class
   entropy is .60 nats or less (`D60` .72; the three seeds .72 to .78).
2. **Safety:** training return per game in each 400k stretch from 2.4M
   is 380 or more (`D60` 405.6 / 405.7 / 418.9; 347.8 before the cut),
   and no sampled cell has a game cut at the cap or ending at zero.
3. **Level:** training return per game over 2.8M to 3.6M is 428 or more
   (`D60` 412.2 with a standard error of 5.7; the bar is two standard
   errors of the difference above it).
4. **Play:** sampled play at 3.6M has a median of 425 or more over 72
   games with 20 or more past 432 (`D60` sampled 418 with 14, greedy
   427.5 with 25).

Observations: the policy entropy reported in training by 100k bin; the
total and within-class entropy on the dump; sampled and greedy medians
and games past 432 at every checkpoint; wide and narrow return rates in
both modes; the serve delay in both modes; the share of descents with a
move among the last two actions; the state readout and the critic on
the dump; the actor's distance between 3.2M and 3.6M.

The gate-level reading is the one fixed for `D50` and `D60`: pooled
median 430 or more and 72 or more of 144 games past 432 at any of the
three checkpoints, no game cut. Whether the gate is met is the user's
to say.

## Readings

- **1 fails:** the coefficient does not set the trained policy's spread
  at this stage (it did on the smoke's early agent); with tempered play
  the entropy side is closed. Tests 3 and 4 are then read as the
  spread between two runs that differ by nothing effective.
- **1 passes, 3 and 4 fail:** a policy trained sharper returns no more
  balls. Closed; the recipe stays `D60`'s.
- **1 and 3 pass** (4 either way): the cut helps on this seed. It is
  repeated on seed 5011 as a twin of `D50` before it is added to the
  recipe.
- **2 fails:** the cut harms (lost exploration, or a policy that no
  longer serves); not retained.

Predictions: test 1 three in five; test 2 three in four; test 3 one in
five; test 4 one in five; the gate-level reading one in thirty.

## Limits

- One seed, and the best of the three; the three cohorts every run
  since November 17 has used; 24 games per cell.
- One factor (a tenth) at one point (2.4M); an earlier or deeper cut is
  not tested.
- The learning rates are a quarter of their starting values while the
  cut acts, so the policy moves slowly; a null result at 3.6M does not
  show what a longer run would do.
- The run does not touch the limit the planner readings point to (the
  ball-to-paddle offset, read to 6 px against the 1 px that most games
  past the wall needs). The expected ceiling is greedy play's level,
  not the gate.
