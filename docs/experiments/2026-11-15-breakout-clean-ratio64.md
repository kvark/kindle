# Clean Breakout agent at four times the replay ratio

The [clean screen](../results/2026-11-15-breakout-clean-representation.md)
left the ball out of the model state in every arm (linear probe error 19.5 to
24.8 source pixels against 27 untrained) and its best arm, B
(`--actor-critic-gradient`), plays about 7 points. Every clean arm replays
each action 16 times. DreamerV3's own configuration replays each action 32
times for full-length Atari on this protocol (sticky actions, all 18 actions,
at 96x96 grey) and 256 times for Atari100k, so the screen cannot tell a
representation that cannot find the ball from one that was not trained
enough to. This run gives arm B four times the training per action and
changes nothing else.

**Recipe.** Arm B exactly (published protocol, 12M, 64x64 RGB, B16/T16,
replay 100k, actor unimix .01, behavior learning rate 8e-5, replay value 1.0,
`--actor-critic-gradient`, 400k actions, seed 1009) with `--train-ratio 64`:
about 99,900 updates, 2.7 hours at the screen's 97 ms per update. Checkpoints
are kept at 100k, 200k, 300k and 400k actions (`--checkpoint-history`).
Runner and probe are the screen's pinned copies (`atari_vector.py` 143b9a2b,
`breakout_agent_probe.py` 89511a67). Root
`/x/Temp/kindle-breakout-clean-20261115/F-acgrads-ratio64`.

**Measures.** The screen's linear state probe on the initial checkpoint and
all four kept checkpoints; frozen sampled and greedy play on cohorts
4000777001 and 4000022003 (three natural episodes per lane) for the 400k
checkpoint and for the 100k checkpoint, which has taken about as many updates
as arm B's final (about 25,000) from a quarter of the experience.

**Reading.** Against arm B's final (ball 23.0 px, within 8 px .043; play 7.2 /
7.4 and 6.2 / 5.8): a ball error well below the screen's best 19.5 px with
play clearly above B says the clean recipe was compute-starved and the
replay ratio is the lever to scale; no ball gain at four times the updates
says the 64x64 squared-error objective does not see the ball at any
affordable budget, and the next clean change is resolution (96x96). The 100k
point against B separates updates from experience. One seed; development
cohorts; read probe differences of a few pixels as noise.
