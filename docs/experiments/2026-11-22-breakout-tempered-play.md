# Tempered play on the frozen recipe's checkpoints

Prepared 05:46 UTC November 22. Evaluation only, on existing
checkpoints; no training and no GPU. Designed by the assistant after
[the open-loop freeze result](../results/2026-11-22-breakout-open-loop-freeze.md);
the user has not been asked. Not launched yet: it runs after test 2's
cells of that result finish on the CPU renderer.

## Why

The open-loop result's declared reading leaves the actor's side: how the
policy is shaped from imagination. Three readings of existing own-play
data point at how diffuse the frozen learner's policy is:

- `E22`'s policy entropy in training rose from 2.12 nats at 600k to 2.30
  to 2.45 from 1.5M on, while the mean normalized advantage fell from
  .022 to .016. On its own play at 3.6M the entropy is 2.34 nats, of
  which .69 lies between the three move classes (still, right, left;
  the maximum is 1.10) and the rest among actions with the same effect.
- While the ball rises the policy is close to uniform between the
  classes (.85 to .92 nats), the paddle moves on .60 to .70 of all
  actions, and 44% to 49% of consecutive moves reverse direction.
  Below y 170 in a descent the top class still holds only .84 to .85 of
  the mass.
- Greedy play, the same ranking with no noise, returns more balls
  wherever it serves (wide paddle .957 to .968 against .947 to .954 at
  2.0M, 2.4M, 3.2M and 3.6M) but stalls at the serve at other
  checkpoints, on both seeds.

A lower action temperature keeps the policy stochastic, so it should
serve, while moving it toward its own ranking. If that plays better,
the entropy term of the actor's loss (`--actor-entropy`, default 3e-4)
is the training-side lever to declare next; if it does not, it is not.

## Cells

The declared sampled-cell commands of `queue-26.json` and
`queue-27.json` with `--action-temperature 0.5` added (first three
games per stream, sticky .25, 18 actions, `--life-loss-reset`):

| Checkpoint | Cohorts | Games | Sampled at temperature 1 (on record) |
|---|---|---:|---|
| `E22` 3.6M (seed 22003) | 4000777001, 4000022003 | 48 | median 394.5; wide .954, narrow .871 |
| `E22` 2.8M | same | 48 | median 376; wide .951, narrow .858; greedy cut 13 of 48 |
| `F50` 2.8M (held-out seed 5011) | 4000313131, 4000777001, 4000022003 | 72 | median 351; wide .941, narrow .832; greedy cut 22 of 72 |

Root `/x/Temp/kindle-breakout-collapse-20261120`, outputs in
`temperature/`, script `run_llvmpipe_temperature.sh` (builder
`make_llvmpipe_temperature.py`), one cell at a time on Mesa's llvmpipe
as one memory-capped unit, about 90 minutes per cell. If the user
reviews the November 22 GPU fault first, the same seven commands run on
the card under the reviewed queue runner instead. The baselines were
measured on the card; four paired cells on November 16 did not
distinguish the two devices.

## Tests

1. **Serves:** no game is cut at the action cap in any of the seven
   cells.
2. **Returns more:** over all three checkpoints the wide-paddle return
   rate is .958 or more (sampled .948 over 8,916 descents; the bar is
   three standard errors of the difference above it).
3. **Scores more:** the median over `E22`'s 96 games is 405 or more
   (sampled 387.5, greedy 393 with its cut games) and over `F50`'s 72
   games 370 or more (sampled 351).

Observations, not tests: narrow-paddle return rate (sampled .865 and
.832), the share of descents with a move among the last two actions,
games past 432, serve delay.

## Readings

- **1 and 2 pass:** the policy's own ranking, sharpened, plays better
  without the stall. A training run of the frozen recipe with a lower
  `--actor-entropy` is declared next (separately, and it needs the
  card), and temperature .5 is put to the user as a candidate for the
  play mode the gate reads.
- **1 passes, 2 fails:** tempering is at most a stall-free equal of
  sampled play; a lower entropy term in training is not supported by
  this and is not pursued on this evidence.
- **1 fails:** the stall is not a matter of the mode alone; reported.

Predictions: test 1 nine chances in ten; test 2 one in two (wide .953
to .962); test 3 one in three.

## Limits known in advance

- One temperature, 48 to 72 games per checkpoint; a median over 48
  games moves by 10 to 12 points between neighbouring checkpoints.
- Tempering a fixed policy is not training with a sharper one: the
  policy never saw the states its tempered play reaches, and a learner
  trained with less entropy also explores less.
- Run on llvmpipe against baselines from the card.
- The play mode the gate reads is the user's decision.
