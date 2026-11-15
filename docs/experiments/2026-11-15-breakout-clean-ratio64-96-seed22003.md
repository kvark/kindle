# Second seed of the clean 96x96 ratio-64 Breakout agent

The [clean 96x96 ratio-64 learner](../results/2026-11-15-breakout-clean-ratio64-96.md)
(seed 1009) is the best clean Breakout agent so far: frozen 20.3 to 25.3 on
cohorts 4000777001 and 4000022003, vertical ball probe error 11.1 source
pixels. Earlier Breakout recipes varied widely between seeds (the pixel-world
recipe's greedy score spanned 44 to 161 over four seeds), so the recipe is
repeated once before further clean changes are built on it.

**Recipe.** Identical to the seed-1009 run (published protocol, 12M, 96x96
RGB, B16/T16, replay 100k, train ratio 64, actor unimix .01, behavior
learning rate 8e-5, replay value 1.0, `--actor-critic-gradient`, 400k
actions, checkpoints kept at 100k to 400k) with `--seed 22003`. About 4.9
hours, 16.3 GiB VRAM. Pinned runner and probe unchanged. Root
`/x/Temp/kindle-breakout-clean-20261115/H-acgrads-ratio64-96-s22003`.

**Measures.** As for seed 1009: the linear state probe on the initial and
all four kept checkpoints, and frozen sampled and greedy play on cohorts
4000777001 and 4000022003 at 100k and 400k actions.

**Reading.** Frozen play within a few points of seed 1009's 20 to 25 and a
vertical ball error near 11 px confirm the recipe; a result near the 64x64
run (about 10, 16 px) says seed 1009 was a favourable draw. Two seeds are a
replication check, not a reliability claim.
