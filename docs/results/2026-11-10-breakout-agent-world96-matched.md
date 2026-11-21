# Matched Breakout agent: retained recipe with the pixel world objective

[Protocol](../experiments/2026-11-10-breakout-agent-world96-matched.md). Root
`/x/Temp/kindle-breakout-world96-matched-20261110`; outputs `w96/` and
`w64/`, guard directory `guards/`, unit
`kindle-breakout-world96-matched-20261110.service` (launched 15:18 UTC,
invocation `89bcb1c5dcaa4f94a124139c3c4b7f30`). Source `5123d715` (runner
copy hash in `launch.txt`) on Meganeura main `a0eff600`. The retained
learner's command differs only in `--cdp --cdp-centered
--encoder-learning-rate 4e-5` (removed) and `--image-size 96|64
--world-pixel-loss motion_cross_entropy --world-prediction-length 8` (added).

## 96x96 run

Training ran 15:18 to 16:11 UTC: 200,000 aggregate actions (799,579 emulator
frames), 12,477 learner updates (the retained learner's exact count), 3,215.5 s
wall, 62.2 actions/s; the learn stage took 3,072 s at 242 ms per update (166 ms
world, 36 ms imagination), against the retained CDP learner's 21m28s. VRAM
peaked at 17.6 GiB (the 768-action smoke showed 5.5 GiB; replay and the
graph's steady state are not visible in a smoke). 273 online episodes
completed with guided play included, mean return 51.8 (25k-action buckets:
15.8, 73.5, 48.5, 43.6, 97.7, 54.8, 56.4, 87.7); five online episodes scored
345 to 388. World losses fell from 17,469 reconstruction / 17,289
prior-rollout at update 60 to about 6,040 / 6,360 over the last 500 updates,
exploration loss .02, reward loss .11, KL 21.5; the imagined return mean
ended at 16.6 with return scale 19.9 and policy entropy .88.

## 64x64 run

Training ran 16:13 to 16:46 UTC: 200,000 aggregate actions (799,494 emulator
frames), 12,477 learner updates, 1,971.1 s wall, 101.5 actions/s; the learn
stage took 1,825 s at 143 ms per update (87 ms world, 34 ms imagination),
against the retained CDP learner's 21m28s at the same geometry. VRAM peaked
at 9.5 GiB. 309 online episodes completed with guided play included, mean
return 55.7 (25k-action buckets: 19.6, 57.7, 62.4, 42.0, 89.2, 98.8, 72.9,
57.8). World losses fell from 7,885 reconstruction / 7,876 prior-rollout at
update 60 to about 2,775 / 2,830 over the last 500 updates (64x64 has 4/9 of
the pixels), exploration loss .03, reward loss .07, KL 15.6; the imagined
return mean ended at 16.8 with return scale 37.5 and policy entropy .87.

## Frozen evaluations (cohort 4000022003)

Three natural episodes per environment, zero updates, 25k-action truncation.
No episode stalled in either 96x96 evaluation; "natural-24" is the first
three natural episodes of each stream.

| Run | Checkpoint | Mode | Episodes | Mean | Natural-24 | Median | Max | Two-wall |
|---|---|---|---|---|---|---|---|---|
| 96x96 | actual initial | sampled | 28 | 1.179 | 1.083 | | 4 | 0/28 |
| 96x96 | actual initial | greedy | 24 | 1.875 | 1.875 | | 5 | 0/24 |
| 96x96 | 200k | sampled | 26 | 38.231 | 39.208 | 32 | 94 | 0/26 |
| 96x96 | 200k | greedy | 24 | **114.625** | 114.625 | 86.5 | 342 | 0/24 |
| 64x64 | actual initial | sampled | 29 | 1.241 | 1.208 | | 4 | 0/29 |
| 64x64 | actual initial | greedy | 28 | 2.214 | 2.375 | | 7 | 0/28 |
| 64x64 | 200k | sampled | 28 | 40.286 | 42.833 | 33 | 84 | 0/28 |
| 64x64 | 200k | greedy | 25 | **151.080** | 154.167 | 111 | 383 | 0/25 |

96x96 greedy 200k scores sorted: 36, 38, 51, 59, 60, 63, 65, 70, 76, 80, 84,
85, 88, 90, 98, 105, 119, 128, 142, 145, 163, 246, 318, 342 (nine of 24 at or
above 100, mean episode length 1,425 actions); sampled 14 to 94, all positive,
mean length 969. 64x64 greedy: 37, 40, 47, 49, 69, 77, 81, 86, 91, 96, 105,
106, 111, 121, 131, 145, 150, 170, 211, 227, 292, 301, 321, 330, 383 (15 of 25
at or above 100, mean length 1,440); sampled 12 to 84, all positive, mean
length 975. No evaluation episode stalled.

References on cohort 4000022003 (same protocol): retained CDP 1009 94.625
greedy / 70.500 sampled; fresh CDP 22003 75.250 / 44.708; the first,
unmatched pixel-world agent 4.785 / 4.760.

## Reading

Swapping only the world objective turns the retained recipe's frozen greedy
score on this cohort from 94.625 (centered CDP) into 114.625 at 96x96 and
151.080 at 64x64, with no stalled episodes and a 383 single-episode maximum,
while the sampled score falls from 70.500 to 38 to 43. The 64x64 run is the
cleaner comparison: same geometry, same update count (12,477), 33 minutes
against the CDP learner's 21, and it beats even the retained learner's
original-cohort greedy 143.667. The 96x96 run costs 54 minutes and 17.6 GiB
and scores lower, so resolution is not what the agent needed; the objective
is. Both pixel-world learners have broad greedy distributions (medians 86
and 111) and tight sampled ones (medians 32 and 33 against a greedy max of
342 and 383), so the stochastic policy still spreads probability over
actions the greedy choice avoids; actor unimix .01 and the entropy term are
unchanged from the retained recipe.

Unmatched by design: the dynamics group trains at the base 4e-5 (CDP's
recipe sets it to 4e-4) and the rollout is 8 steps in a 16-step window. This
is one development seed on a shared, reused cohort against references on the
same cohort; it does not establish reliability across seeds, the original
864/90% two-wall gate (0/25 and 0/24 two-wall here), or that the sampled
mode can be recovered. The [auxiliary protocol](../experiments/2026-11-10-breakout-agent-cdp-pixel-aux.md)
keeps CDP and adds this pixel objective to ask whether the two compose.
