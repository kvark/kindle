# Second seed for the pixel-world Breakout agents

[Protocol](../experiments/2026-11-10-breakout-agent-pixel-seed22003.md). Root
`/x/Temp/kindle-breakout-pixel-seed22003-20261110`; outputs `pixel64/` and
`aux1/`, guards `guards/`, unit `kindle-breakout-pixel-seed22003-20261110.service`
(launched 18:02 UTC, invocation `5931965e01cf48c0af46a23f213ff2d4`). Source
`14af7e17` (runner copy hash in `launch.txt`), Meganeura main `a0eff600`.
Both commands are the seed 1009 commands with `--seed 22003`.

## pixel64 (matched pixel world, 64x64, no CDP)

Training 18:02 to 18:34 UTC: 200,000 actions (799,496 frames), 12,477
updates, 1,953.5 s wall, 102.4 actions/s, 142 ms per update, VRAM 9.5 GiB.
304 online episodes with guided play, mean 48.8 (25k buckets: 17.4, 58.1,
58.7, 34.3, 47.5, 56.2, 94.0, 90.7). Final world losses: reconstruction
2,738, prior rollout 2,857, exploration .03, reward .10, KL 16.3; imagined
return mean 16.7, return scale 29.8, policy entropy .83.

## aux1 (retained CDP plus pixel auxiliary at scale 1)

RUN_AUX1

## Frozen evaluations (cohort 4000022003)

Three natural episodes per environment, zero updates, 25k-action truncation;
no episode stalled. "Natural-24" is the first three natural episodes per
stream.

| Learner | Checkpoint | Mode | Episodes | Mean | Natural-24 | Median | Max | Two-wall |
|---|---|---|---|---|---|---|---|---|
| pixel64 | actual initial | sampled | 30 | 1.633 | 1.875 | | 5 | 0/30 |
| pixel64 | actual initial | greedy | 26 | 2.192 | 2.292 | | 11 | 0/26 |
| pixel64 | 200k | sampled | 32 | 28.375 | 27.208 | 29 | 59 | 0/32 |
| pixel64 | 200k | greedy | 27 | 99.259 | 98.292 | 80 | 371 | 0/27 |
EVAL_AUX1

pixel64 greedy sorted: 24, 31, 39, 40, 40, 43, 43, 44, 44, 52, 64, 71, 78, 80,
82, 82, 97, 98, 100, 109, 116, 120, 137, 141, 181, 353, 371 (nine of 27 at or
above 100, mean length 1,328); sampled 7 to 59, mean length 842.

Seed pairs on this cohort, greedy / sampled:

| Learner | Seed 1009 | Seed 22003 |
|---|---|---|
| Retained CDP | 94.625 / 70.500 | 75.250 / 44.708 |
| Pixel world 64x64 | 151.080 / 40.286 | 99.259 / 28.375 |
| CDP + pixel scale 1 | 104.036 / 61.692 | AUX1_PAIR |

## Reading

READING
