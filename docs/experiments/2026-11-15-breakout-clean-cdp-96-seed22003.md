# Second seed of the clean centered-CDP Breakout agent at 96x96

The [clean centered-CDP learner](../results/2026-11-15-breakout-clean-cdp-96.md)
(seed 1009) is the first clean learner whose state holds the ball (probe
6.0 / 6.3 px at 300k, within 8 px .67) and plays 71 to 91 frozen against the
clean pixel recipe's 20 to 25. That is one seed. The pixel recipe was
repeated before it became the clean reference, and this claim, which would
make the latent objective the clean default, gets the same check.

**Recipe.** `queue-7.json` exactly, with `--seed 22003` and the arm renamed
`J-cdp-acgrads-ratio64-96-s22003`: published protocol, 12M, 96x96 RGB,
B16/T16, replay 100k, train ratio 64, actor unimix .01, behavior rate 8e-5,
replay value 1.0, `--actor-critic-gradient`, `--cdp --cdp-centered
--encoder-learning-rate 4e-5`, 400k actions, checkpoints kept at 100k to
400k, no guidance, labels, imitation, demonstrations or bonus. Pinned
runner and probe unchanged (`sources.sha256`). Root
`/x/Temp/kindle-breakout-clean-20261115/J-cdp-acgrads-ratio64-96-s22003`,
queue `queue-8.json`, guards `guards-8`, unit
`kindle-breakout-clean-cdp96-s22003-20261115`, 15 GiB memory cap, about 3.2
hours.

**Measures.** As for seed 1009: the linear state probe on the initial and
all four kept checkpoints, and frozen sampled and greedy play on cohorts
4000777001 and 4000022003 at 100k and 400k actions.

**Reading.** The result repeats if the ball probe at 300k to 400k stays
near seed 1009's (within 8 px above .4, clearly beyond the pixel seeds' .12
and the guided learner's .39) and frozen play stays well above the pixel
recipe's 21 (cell means above 40). A seed whose ball reads like the pixel
recipe's, or whose play falls near 21, would make the seed-1009 result a
lucky draw, not a recipe property. Two seeds remain a replication check,
not a reliability claim; development cohorts.
