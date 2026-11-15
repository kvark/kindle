# Clean Breakout agent with a latent objective (centered CDP) at 96x96

The clean pixel recipe (DreamerV3 reconstruction, 96x96, train ratio 64,
`--actor-critic-gradient`) is now repeated on two seeds: frozen about 21 on
both cohorts, ball probe 18.3 to 18.6 px horizontally and 11.1 to 13.9 px
vertically ([seed 1009](../results/2026-11-15-breakout-clean-ratio64-96.md),
[seed 22003](../results/2026-11-15-breakout-clean-ratio64-96-seed22003.md)).
Reconstruction spends the representation on every pixel equally; the
question raised with the clean direction is whether a latent objective,
trained jointly with value gradients into the encoder, keeps more of what
the policy needs. The retained Breakout learner already uses one, centered
CDP (the deterministic state predicts the next CNN embedding by cosine
distance to a stop-gradient target, centered over the replay batch, with no
reconstruction), but only with teacher guidance.

**Recipe.** The clean 96x96 ratio-64 run exactly (published protocol, 12M,
96x96 RGB, B16/T16, replay 100k, train ratio 64, actor unimix .01, behavior
learning rate 8e-5, replay value 1.0, `--actor-critic-gradient`, 400k
actions, seed 1009, checkpoints kept at 100k to 400k), with the retained
learner's world objective: `--cdp --cdp-centered --encoder-learning-rate
4e-5` (CDP scale 500, dynamics rate 4e-4, no reconstruction). Its guidance,
corrective labels, imitation and disagreement bonus stay off, so the
encoder is trained only by the latent prediction, reward, continuation, KL
and the value and actor gradients. Pinned runner and probe unchanged. Root
`/x/Temp/kindle-breakout-clean-20261115/I-cdp-acgrads-ratio64-96`.

**Measures.** As for the pixel runs: the linear state probe on the initial
and all four kept checkpoints, and frozen sampled and greedy play on cohorts
4000777001 and 4000022003 at 100k and 400k actions.

**Matched guided reference.** Probed just before launch with the same
script and trajectories, the retained guided CDP learner (400k actions,
64x64, ratio 16, teacher guidance and corrective labels, seed 1009;
`/x/Temp/kindle-breakout-budget400k-20261103/.../final`) reads the ball at
8.4 / 8.8 px (R² .90 / .77), within 8 px .390, velocity R² .70 / .77, paddle
5.9 px; output `reference/probe-guided-cdp-400k-seed1009.json`. The guided
learner holds the ball's horizontal position that no clean learner holds.

**Reading.** Against the pixel recipe's two seeds (play about 21; ball 18.5 /
12.5 px): a lower ball error or higher play says the latent objective with
value gradients keeps more of what matters; a collapse (ball near the
untrained 26 px, play near the clean screen's 4) says CDP needs the
teacher's labels to learn the ball. A clean CDP ball near the guided
reference would say the latent objective, not the labels, carried the
ball in the retained learner. One seed; development cohorts.
