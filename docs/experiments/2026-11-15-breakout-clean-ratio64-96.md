# Clean Breakout agent at 96x96 and four times the replay ratio

The [ratio-64 run](../results/2026-11-15-breakout-clean-ratio64.md) raised
clean frozen play to about 10 and removed the greedy FIRE stalls, but the
ball enters the model state only slowly: probe error 19.3 / 16.1 source
pixels at 400k actions, within 8 px .081, against 9 to 13 px for the guided
learners. At 64x64 the ball covers about 0.8 by 1.2 pixels after the
antialiased resize; at 96x96 it covers 1.2 by 1.8, and DreamerV3's own
full-length Atari setting uses 96x96. This run changes only the image size.

**Recipe.** The ratio-64 run exactly (arm B of the clean screen with
`--train-ratio 64`: published protocol, 12M, B16/T16, replay 100k, actor
unimix .01, behavior learning rate 8e-5, replay value 1.0,
`--actor-critic-gradient`, 400k actions, seed 1009, checkpoints kept at
100k to 400k) with `--image-size 96`. Expected 99,907 updates at roughly 160
ms each, about 4.5 hours, and about 16 GiB of VRAM (the guided 96x96 learner
took 1.7 times the 64x64 update time and 1.85 times its memory). Pinned
runner and probe unchanged. Root
`/x/Temp/kindle-breakout-clean-20261115/G-acgrads-ratio64-96`.

**Measures.** As in the ratio-64 run: the linear state probe on the initial
checkpoint (its own floor, since the encoder geometry differs) and all four
kept checkpoints (the probe feeds native frames and the agent
resizes them, so the trajectories are identical), and frozen sampled and
greedy play on cohorts 4000777001 and 4000022003 at 100k and 400k actions.

**Reading.** Against the ratio-64 run at 400k (ball 19.3 / 16.1 px, within
8 px .081, play 9.8 to 11.1): a ball error several pixels lower on both
axes says resolution limits the clean representation and 96x96 becomes the
clean default; no ball gain says the 64x64 frame is not the limit and the
budget (more actions at this ratio) is the remaining clean lever. Play is
read alongside but resolution can cost play at a fixed budget. One seed;
development cohorts; read probe differences of a few pixels as noise.
