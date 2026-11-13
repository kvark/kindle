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

Training 18:36 to 19:08 UTC: 200,000 actions (799,603 frames), 12,477
updates, 1,951.7 s wall, 102.5 actions/s, 142 ms per update, VRAM 9.6 GiB.
251 online episodes with guided play, mean 51.6 (25k buckets: 13.0, 61.0,
61.0, 63.2, 39.2, 42.6, 117.1, 107.8). Final world losses: centered CDP .061
(weighted 31), reconstruction 2,753, prior rollout 2,822, exploration .04,
reward .09, KL 16.6; imagined return mean 22.7, return scale 42.8, policy
entropy .73.

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
| aux1 | actual initial | sampled | 30 | 1.633 | 1.875 | | 5 | 0/30 |
| aux1 | actual initial | greedy | 26 | 2.192 | 2.292 | | 11 | 0/26 |
| aux1 | 200k | sampled | 25 | 68.280 | 69.042 | 53 | 316 | 0/25 |
| aux1 | 200k | greedy | 25 | 101.040 | 104.083 | 74 | 310 | 0/25 |

aux1 greedy sorted: 28, 48, 49, 49, 57, 59, 65, 66, 67, 69, 71, 72, 74, 86, 87,
87, 90, 104, 130, 137, 140, 146, 162, 273, 310 (eight of 25 at or above 100,
mean length 1,381); sampled 22 to 316, median 53, mean length 1,218.

pixel64 greedy sorted: 24, 31, 39, 40, 40, 43, 43, 44, 44, 52, 64, 71, 78, 80,
82, 82, 97, 98, 100, 109, 116, 120, 137, 141, 181, 353, 371 (nine of 27 at or
above 100, mean length 1,328); sampled 7 to 59, mean length 842.

Seed pairs on this cohort, greedy / sampled:

| Learner | Seed 1009 | Seed 22003 |
|---|---|---|
| Retained CDP | 94.625 / 70.500 | 75.250 / 44.708 |
| Pixel world 64x64 | 151.080 / 40.286 | 99.259 / 28.375 |
| CDP + pixel scale 1 | 104.036 / 61.692 | 101.040 / 68.280 |

## Reading

Both pixel-world learners beat the fresh CDP seed 22003 in greedy play on
this cohort (99.3 and 101.0 against 75.3), as they beat the retained seed
1009 (151.1 and 104.0 against 94.6). In sampled play the pure pixel world
loses on both seeds (28.4 and 40.3 against 44.7 and 70.5) while CDP plus the
pixel auxiliary at scale 1 wins on seed 22003 (68.3 against 44.7) and loses
narrowly on seed 1009 (61.7 against 70.5). Two-seed means, greedy / sampled:
retained CDP 84.9 / 57.6; pixel world 125.2 / 34.3; CDP plus pixel 102.5 /
65.0. The combined learner is the only one above CDP in both modes on
average, and its two seeds sit within 3 greedy and 7 sampled points of each
other where CDP's differ by 19 and 26.

Two seeds on one shared, reused development cohort are not a reliability
claim and no run reached a two-wall episode (0 of 211 evaluation episodes
across these ten frozen evaluations); the original 864/90% gate stays
unfinished. Everything here is the retained 200k budget at 64x64 with the
same 12,477 updates; the pixel runs cost 33 minutes against CDP's 21 because
the RGB decoder and the 8-step image rollout add 65 ms per update. The
retained CDP recipe has a 400k budget point; the combined learner does not
yet.
