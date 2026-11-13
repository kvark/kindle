# Pixel-world Breakout agent with the context-frame decoder

[Protocol](../experiments/2026-11-11-breakout-agent-context-frame.md). Root
`/x/Temp/kindle-breakout-pixel-context-20261111`, output `candidate/`, guards
`guards/`, unit `kindle-breakout-pixel-context-20261111.service` (07:41 to
08:23 UTC). Source `11b8c04c` (runner copy hash in `launch.txt`), Meganeura
main `a0eff600`. The command is the matched pixel world 64x64 200k command
(seed 1009) plus `--world-context-lag 8`.

## Run

200,000 actions (799,517 frames), 12,477 updates, 2,397.4 s wall (40 minutes
against 33 without the context frame), 83.4 actions/s, 177 ms per update
(123 ms world), VRAM 10.2 GiB. 297 online episodes with guided play, mean
48.5 (25k buckets: 17.2, 43.3, 46.7, 53.6, 76.2, 67.0, 78.9, 63.1). World
losses fell from 10,798 reconstruction / 11,021 prior rollout at update 60
to 2,760 / 2,822 at the end: the same final level as the state-only decoder
(2,738 / 2,857), so at 64x64 with the 12M preset the context frame did not
lower the pixel loss the way it did for the 4M world at 96x96. Exploration
loss .02, reward loss .04, KL 14.5; imagined return mean 10.3, return scale
12.3, policy entropy .91.

## Frozen evaluations (200k, three natural episodes per environment, zero updates)

| Learner (seed 1009) | Cohort | Mode | Episodes | Mean | Natural-24 | Median | Max | At or above 100 | Two-wall |
|---|---|---|---|---|---|---|---|---|---|
| Context frame | 4000001009 | sampled | 27 | 27.444 | 28.083 | 24 | 79 | 0 | 0/27 |
| Context frame | 4000001009 | greedy | 25 | 77.240 | 78.792 | 65 | 202 | 5 | 0/25 |
| Context frame | 4000022003 | sampled | 25 | 35.280 | 36.167 | 25 | 199 | 1 | 0/25 |
| Context frame | 4000022003 | greedy | 25 | 72.600 | 74.750 | 59 | 181 | 6 | 0/25 |
| State-only (reference) | 4000001009 | sampled / greedy | | 37.680 / 156.840 | | | | | 0 |
| State-only (reference) | 4000022003 | sampled / greedy | | 40.286 / 151.080 | | | | | 0 |

## Reading

Negative: the context frame does not help the agent at this geometry. Both
modes fall below the state-only pixel world, the pixel losses end at the
same level, and the run costs 20% more time. The standalone benefit of the
context frame (sharp bricks with the ball in state) was measured on a 4M
world at 96x96 over 65k updates; at 64x64 with 12M for 12.5k updates the
decoder evidently does not need the frame to render the scene, and the stale
frame held through the 8-step rollout gives imagination nothing the state
did not have.

A second reading matters more. Five single-seed runs of the pixel-world
recipe now exist on seed 1009 (baseline, entropy 1e-4, entropy 3e-5, unimix
0, context frame) and their greedy scores on cohort 1009 are 156.8, 93.9,
129.1, 87.0 and 77.2. Each variant changed one setting, but the spread is
also what five draws of a high-variance learner look like, so the baseline's
156.8 (and 151.1 on the other cohort) may be the top of the distribution
rather than its centre. The seed 22003 pixel-world point (99.3 against fresh
CDP's 75.3) is the one independent draw that supports the greedy gain. The
[seed screen](2026-11-11-breakout-agent-pixel-seeds.md) adds two fresh
seeds of the unchanged baseline recipe to settle this before any recipe is
retained.
