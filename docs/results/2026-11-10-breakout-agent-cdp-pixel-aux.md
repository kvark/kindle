# Breakout agent: retained CDP recipe plus an auxiliary pixel rollout

[Protocol](../experiments/2026-11-10-breakout-agent-cdp-pixel-aux.md). Root
`/x/Temp/kindle-breakout-cdp-pixel-aux-20261110`; outputs `s01/` (scale .1)
and `s1/` (scale 1), guards `guards/`, unit
`kindle-breakout-cdp-pixel-aux-20261110.service` (launched 16:51 UTC,
invocation `9efd06d31d5441faa1c68ab4b7f508a2`). Source `401ed0bb` (runner
copy hash in `launch.txt`), Meganeura main `a0eff600`. The command is the
retained learner's with `--world-pixel-scale S --world-pixel-loss
motion_cross_entropy --world-prediction-length 8` appended; 64x64, 12M,
B16/T16, centered CDP 500, bonus 1, corrective labels, unimix .01, encoder
4e-5, behavior 8e-5, replay-value 1, seed 1009, 200k actions.

## Scale .1

Training 16:51 to 17:23 UTC: 200,000 actions (799,570 frames), 12,477
updates, 1,968.9 s wall, 101.6 actions/s, 143 ms per update (87 ms world),
VRAM 9.6 GiB. 257 online episodes with guided play, mean 49.7 (25k buckets:
12.5, 61.0, 51.6, 69.1, 54.2, 80.0, 51.7, 82.0). The centered CDP loss fell
from .993 at update 60 to .073 at the end (weighted 500 to 37); pixel
reconstruction 7,104 to about 2,790 and prior rollout 7,218 to 2,917
(weighted .1: 1,430 to 570), KL 14.1, exploration .03; imagined return mean
5.2, return scale 27.9, policy entropy .69.

## Scale 1

RUN_S1

## Frozen evaluations (cohort 4000022003)

Three natural episodes per environment, zero updates, 25k-action truncation;
no episode stalled. "Natural-24" is the first three natural episodes per
stream.

| Scale | Checkpoint | Mode | Episodes | Mean | Natural-24 | Median | Max | Two-wall |
|---|---|---|---|---|---|---|---|---|
| .1 | actual initial | sampled | 29 | 1.241 | 1.208 | | 4 | 0/29 |
| .1 | actual initial | greedy | 28 | 2.214 | 2.375 | | 7 | 0/28 |
| .1 | 200k | sampled | 30 | 47.667 | 47.750 | 44.5 | 107 | 0/30 |
| .1 | 200k | greedy | 24 | 68.292 | 68.292 | 53.5 | 173 | 0/24 |
EVAL_S1

Scale .1 greedy sorted: 12, 27, 34, 35, 40, 43, 43, 44, 45, 46, 49, 49, 58,
59, 65, 67, 70, 87, 97, 113, 116, 125, 142, 173 (five of 24 at or above 100).
The actual-initial controls are identical to the matched pixel-world run's
(same seed, same initial weights, same cohort).

References on this cohort: retained CDP 1009 94.625 greedy / 70.500 sampled;
matched pixel world 64x64 151.080 / 40.286 and 96x96 114.625 / 38.231; first
unmatched pixel agent 4.785 / 4.760.

## Reading

READING
