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
`S` is chosen once from a 768-action smoke so that the weighted pixel term
(S times the summed per-pixel cross-entropy, about 6,000 to 17,000 at 64x64)
is of the same order as the weighted CDP term (500 times the embedding loss),
not larger; the value and the smoke's loss readings are recorded in the
queue and the result. Resolution stays at the retained 64x64 so the only
change from the retained learner is the auxiliary; the 96x96 variant is a
separate decision after the matched 64x64 twin reports.

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
