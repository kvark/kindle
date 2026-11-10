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

Training 17:25 to 17:57 UTC: 200,000 actions (799,580 frames), 12,477
updates, 1,957.8 s wall, 102.2 actions/s, 141 ms per update (87 ms world),
VRAM 9.6 GiB. 258 online episodes with guided play, mean 53.2 (25k buckets:
12.8, 47.3, 74.4, 37.3, 65.7, 104.5, 69.0, 111.5). The centered CDP loss fell
from .994 at update 60 to .056 (weighted 500 to 28); pixel reconstruction
7,063 to about 2,750 and prior rollout 7,129 to 2,820 (weighted 1: the pixel
term dominates the total 5,683), KL 17.3, exploration .04; imagined return
mean 22.8, return scale 46.5, policy entropy .71.

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
| 1 | actual initial | sampled | 29 | 1.241 | 1.208 | | 4 | 0/29 |
| 1 | actual initial | greedy | 28 | 2.214 | 2.375 | | 7 | 0/28 |
| 1 | 200k | sampled | 26 | 61.692 | 63.333 | 55 | 151 | 0/26 |
| 1 | 200k | greedy | 28 | 104.036 | 102.833 | 88.5 | 286 | 0/28 |

Scale 1 greedy sorted: 20, 30, 33, 37, 38, 38, 42, 45, 47, 49, 63, 68, 83, 88,
89, 97, 98, 101, 107, 114, 124, 129, 152, 195, 202, 261, 277, 286 (11 of 28 at
or above 100, mean length 1,363); sampled 12 to 151, median 55, mean length
1,204.

Scale .1 greedy sorted: 12, 27, 34, 35, 40, 43, 43, 44, 45, 46, 49, 49, 58,
59, 65, 67, 70, 87, 97, 113, 116, 125, 142, 173 (five of 24 at or above 100).
The actual-initial controls are identical to the matched pixel-world run's
(same seed, same initial weights, same cohort).

References on this cohort: retained CDP 1009 94.625 greedy / 70.500 sampled;
matched pixel world 64x64 151.080 / 40.286 and 96x96 114.625 / 38.231; first
unmatched pixel agent 4.785 / 4.760.

## Reading

On this cohort and seed the five 200k learners rank, greedy / sampled:
pixel world 64x64 151.080 / 40.286; pixel world 96x96 114.625 / 38.231;
CDP plus pixel at scale 1 104.036 / 61.692; retained CDP 94.625 / 70.500;
CDP plus pixel at scale .1 68.292 / 47.667. The auxiliary at scale 1 is the
most balanced learner: it keeps most of CDP's sampled score and beats its
greedy score, with the CDP loss still falling to .056 under a pixel term that
dominates the total. At scale .1 the two objectives interfere rather than
compose, losing to both parents in greedy. The pixel objective therefore
helps the actor most when it dominates the world's gradient, whether or not
CDP is present; CDP's contribution shows in the sampled mode (61.7 against
40.3 for pixels alone).

Not established: anything beyond one development seed on a shared cohort;
the original two-wall gate (0 two-wall episodes in every run); why the
greedy and sampled modes diverge so strongly under the pixel objective
(tight sampled medians of 33 to 55 against greedy maxima of 286 to 383);
whether a larger scale, a longer rollout or the 4e-4 dynamics rate changes
the balance. The next check is a second seed of the two best learners
against the fresh CDP seed 22003's 75.250 / 44.708 on the same cohort.
