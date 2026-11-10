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

RUN64

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
| 64x64 | | | | EVAL64 | | | | |

Greedy 200k scores sorted: 36, 38, 51, 59, 60, 63, 65, 70, 76, 80, 84, 85, 88,
90, 98, 105, 119, 128, 142, 145, 163, 246, 318, 342 (nine of 24 at or above
100, mean episode length 1,425 actions). Sampled 200k: 14 to 94, all positive,
mean length 969.

References on cohort 4000022003 (same protocol): retained CDP 1009 94.625
greedy / 70.500 sampled; fresh CDP 22003 75.250 / 44.708; the first,
unmatched pixel-world agent 4.785 / 4.760.

## Reading

READING
