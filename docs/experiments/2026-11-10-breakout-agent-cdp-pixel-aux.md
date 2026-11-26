# Breakout agent: retained CDP recipe plus an auxiliary pixel rollout

The [matched pixel-world agent](2026-11-10-breakout-agent-world96-matched.md)
replaced centered CDP with the state-only pixel objective and scored frozen
greedy 114.625 / sampled 38.231 on cohort 4000022003 against the retained CDP
learner's 94.625 / 70.500. Each objective wins one mode. This protocol keeps
the retained CDP recipe unchanged and adds the pixel objective as an
auxiliary: a second decoder renders RGB from the recurrent state, trained
with motion cross-entropy on posterior states and on an 8-step prior image
rollout, while the CDP embedding prediction keeps its scale and rates.

## Recipe

The retained Breakout learner's exact command (12M, 64x64, B16/T16, centered
CDP scale 500, action-effects bonus 1, corrective labels, actor unimix .01,
encoder rate 4e-5, dynamics rate 4e-4, behavior rate 8e-5, replay-value scale
1, 200k actions, seed 1009) plus
`--world-pixel-scale S --world-pixel-loss motion_cross_entropy --world-prediction-length 8`.
The 768-action smoke (`smoke/`, scale 1, 139 ms per update, VRAM 3.6 GiB)
read the centered CDP loss at .998 (weighted 500) against a pixel
reconstruction of 8,600 plus a prior-rollout loss of 8,500 (weighted 17,000
at scale 1; the matched 64x64 pixel world ended near 2,800 plus 2,800). Two
scales run in one queue, 200k actions each: **S = 0.1** (`s01/`: weighted
pixel term about 1,700 early and 560 late, the same order as the CDP term)
and **S = 1** (`s1/`: pixel-dominated as in the matched pixel world, with CDP
retained underneath). Resolution stays at the retained 64x64, which the
matched twin also showed to be the better and cheaper agent setting. Root
`/x/Temp/kindle-breakout-cdp-pixel-aux-20261110`, `queue.json`, unit
`kindle-breakout-cdp-pixel-aux-20261110.service` (launched 16:51 UTC).

Implementation: `world_pixel_scale` (commit 4201056e) builds `world.decoder`
in a CDP world, targets raw replay pixels (the CDP heads still target
embeddings), weights the posterior reconstruction and the prior rollout by
the scale, and leaves the staged graph, inference and checkpoints otherwise
unchanged. CPU graph/config tests and a GPU tiny-agent test cover it.

## Evaluation

Frozen evaluations (three natural episodes per environment, zero updates,
25k-action truncation) on cohort 4000022003, sampled and greedy, for the
actual-initial and 200k checkpoints. References: retained CDP 1009 94.625 /
70.500; matched pixel world 96x96 114.625 / 38.231; fresh CDP 22003 75.250 /
44.708.

## Reading

One development seed. Greedy at or above 114 with sampled near or above 70
would say the two objectives compose; greedy near 95 with sampled unchanged
would say the auxiliary did not alter what the actor uses; a loss in both
would say the pixel gradient disturbs the CDP latent at this scale. Report
updates, wall time, VRAM, two-wall counts and the auxiliary's loss curve
alongside the CDP loss.
