# Clean centered-CDP Breakout agent at 96x96

Declared in [the clean CDP arm](../experiments/2026-11-15-breakout-clean-cdp-96.md):
the [clean 96x96 ratio-64 pixel learner](2026-11-15-breakout-clean-ratio64-96.md)
with its world objective swapped to centered CDP (`--cdp --cdp-centered
--encoder-learning-rate 4e-5`; CDP scale 500, dynamics rate 4e-4,
reconstruction 0) and nothing else changed: published protocol (18
actions, sticky .25, no no-op starts), 12M, 96x96 RGB, B16/T16, replay
100k, train ratio 64, `--actor-critic-gradient`, 400k actions, seed 1009,
no guidance, corrective labels, imitation, demonstrations, motion
weighting, context frame or exploration bonus (run config:
`disagreement_bonus` and `visitation_bonus` false, `reconstruction` 0,
`future_prediction` 500). 99,907 updates in 187.4 minutes (113 ms per
update by wall clock, against the pixel recipe's 293 minutes), VRAM peak
15.4 GiB. Unit `kindle-breakout-clean-cdp96-20261115`, root
`/x/Temp/kindle-breakout-clean-20261115/I-cdp-acgrads-ratio64-96`, queue
`queue-7.json`, guards `guards-7` (status complete); pinned runner and probe
unchanged.

**The latent objective learns the ball without labels.** The linear probe
reads the ball at 7.4 / 7.7 source pixels after 100k actions and 6.0 / 6.3
at 300k (6.5 / 8.4 at 400k), within 8 px .53 to .67, against the pixel
recipe's 18.3 to 18.6 / 11.1 to 13.9 and .12 at 400k and the guided CDP
learner's 8.4 / 8.8 and .39. Every kept checkpoint of this clean learner
reads the ball more precisely than the guided learner does, so in the
retained recipe the latent objective, not the teacher's labels, carried the
ball. Frozen play at 400k is **71.1 to 91.4 mean** in all four cells, three
to four times the clean pixel recipe's 20 to 25, with 4 to 6 of 24
episodes at or above 100 per cell and a best episode of 410; no episode was
cut and greedy play never stalls.

## Frozen play

Three natural episodes per lane, 24 per cell: mean (best), then median.

| Checkpoint | 777001 sampled | 777001 greedy | 022003 sampled | 022003 greedy |
|---|---:|---:|---:|---:|
| Clean CDP, 100k actions (24,907 updates) | 38.4 (68) | 36.4 (77) | 32.1 (62) | 39.5 (99) |
| Clean CDP, 400k actions | **78.1** (385) | **91.4** (410) | **71.1** (356) | **85.5** (394) |
| Clean CDP 400k, median | 28.0 | 34.0 | 30.0 | 37.5 |
| Clean pixel 1009, 100k | 7.0 (11) | 7.8 (12) | 7.2 (12) | 7.4 (13) |
| Clean pixel 1009, 400k | 20.8 (38) | 21.1 (33) | 25.3 (38) | 20.3 (34) |
| Clean pixel 22003, 400k | 19.6 (29) | 20.5 (35) | 19.1 (28) | 22.1 (38) |
| Clean pixel 400k, medians | 20.0 / 20.5 | 21.5 / 20.5 | 26.0 / 18.0 | 19.5 / 20.5 |
| Guided CDP 1009, 400k (64x64, ratio 16) | 90.6 | 192.0 | 120.6 | 150.8 |

The guided row is the retained learner's earlier evaluation on the same
protocol and cohorts ([cohort 4000777001](2026-11-12-breakout-agent-heldout.md),
medians 77 / 143; [cohort 4000022003](2026-11-10-breakout-agent-cdp-pixel-budget400k.md), medians 90 / 114).
The clean CDP learner's play is bimodal: of its 96 episodes at 400k, 68
score 50 or less, 21 score 100 or more and 8 pass 300 (the best Kindle
episode is 414; none reached the 432 of a cleared wall). Its median is
about 1.5 times the pixel recipe's and its mean three to four times. At
100k actions, with the same updates as the pixel run's 100k checkpoint, it
already plays five times better.

## State probe

Same script and trajectories as every clean run (seed 5550001, 3,050 ball
test rows, policy-input features).

| Checkpoint | Ball x / y MAE (px) | Within 8 px | Paddle x MAE | Ball dx / dy R² |
|---|---:|---:|---:|---:|
| Clean CDP initial | 26.5 / 25.8 | .027 | 23.0 | .15 / .00 |
| Clean CDP 100k | 7.4 / 7.7 | .533 | 4.6 | .71 / .69 |
| Clean CDP 200k | 6.5 / 6.6 | .582 | 5.1 | .80 / .74 |
| Clean CDP 300k | **6.0 / 6.3** | **.665** | 5.8 | .81 / .78 |
| Clean CDP 400k | 6.5 / 8.4 | .506 | 5.2 | .80 / .80 |
| Clean pixel 1009, 400k | 18.3 / 11.1 | .124 | 5.4 | .32 / .47 |
| Clean pixel 22003, 400k | 18.6 / 13.9 | .115 | 5.2 | .31 / .41 |
| Guided CDP 1009, 400k | 8.4 / 8.8 | .390 | 5.9 | .70 / .77 |

Ball R² at 400k: x .92, y .76 (300k: .93, .85; guided .90, .77). The 400k
point is worse than 300k in vertical position while play improved; the
131k world run showed neighbouring checkpoints differing this much, so
the dip is not read as forgetting.

## Training curve

Online episodes and learner means per 50k actions.

| Actions | Episodes | Mean return | Policy entropy | Return scale | Advantage | CDP loss |
|---|---:|---:|---:|---:|---:|---:|
| 0-50k | 208 | 2.7 | 1.19 | 3.0 | .032 | .144 |
| 50-100k | 67 | 19.0 | .51 | 4.3 | .042 | .045 |
| 100-150k | 48 | 45.2 | .46 | 5.6 | .051 | .031 |
| 150-200k | 35 | **123.8** | .61 | 19.7 | .037 | .034 |
| 200-250k | 61 | 77.8 | 1.20 | 93 | .024 | .045 |
| 250-300k | 58 | 58.8 | 1.21 | 113 | .026 | .049 |
| 300-350k | 51 | 73.8 | 1.35 | 118 | .025 | .042 |
| 350-400k | 47 | 95.6 | 1.39 | 127 | .026 | .040 |

Online returns peak at 124 per episode between 150k and 200k actions. Then
the return scale (the 5th-to-95th percentile range of imagined returns that
divides the advantages) jumps from 20 to 93, the mean normalized advantage
halves, policy entropy rises from .5 to 1.2 to 1.4 nats and online returns
fall to 59 to 96. The jump is DreamerV3's normalizer working as designed
once a few long episodes appear. Here it coincides with a softer policy and
with lower and more variable play. That is an observation from one run, not
an isolated cause. The world KL rises from 4.5 to 8.0 over the same span.

## Reading

With everything else held at the clean pixel recipe, replacing
reconstruction with the centered latent prediction (and its encoder rate)
is the largest clean gain so far. The representation holds the ball's
position and velocity within 25k updates, better than the guided learner's
at 400k. Frozen play is three to four times the pixel recipe's in mean and
about 1.5 times in median, in 64% of the training time. This answers the
pixels-versus-latents question raised with the clean direction for
Breakout: the encoder, world state, value and policy co-trained through a
latent objective with value and actor gradients into the encoder learn the
ball and paddle without any hand-made relevance prior or teacher signal.
Reconstruction at the same resolution and update count does not.

The clean learner still trails the guided learner in play, most clearly in
greedy play and medians (28 / 34 against 77 / 143 on cohort 777001, 30 /
37.5 against 90 / 114 on 022003). That
comparison changes guidance, imitation, corrective labels, the actor-critic
gradient, resolution and train ratio at once, so it does not isolate the
teacher's contribution. The training curve points at the policy side, not
the representation: the probe is at its best when online play falls, and
the entropy rise follows the return-scale jump. One seed on development
cohorts; the claim needs a second seed before it guides the recipe.

## Next

A [second seed](../experiments/2026-11-15-breakout-clean-cdp-96-seed22003.md)
(22003, nothing else changed) repeats the run, as the pixel recipe was
repeated. Changes that need a decision first stay open: the game's four
needed actions, demonstrations kept out of the return normalizer or used as
separate pretraining, and a return-normalization or entropy change on the
policy side.

**Update.** The [second seed](2026-11-15-breakout-clean-cdp-96-seed22003.md)
does not repeat the ball result (probe within 8 px .08 to .12, frozen play
34 to 59): the statements above about the latent objective learning the
ball hold for seed 1009 only.
