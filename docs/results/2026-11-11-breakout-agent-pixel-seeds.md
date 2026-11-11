# Two fresh seeds of the unchanged pixel-world Breakout recipe

[Protocol](../experiments/2026-11-11-breakout-agent-pixel-seeds.md). Root
`/x/Temp/kindle-breakout-pixel-seeds-20261111`, outputs `s3005/` and
`s4007/`, guards `guards/`, unit `kindle-breakout-pixel-seeds-20261111.service`
(launched 08:26 UTC). Source `9c2883c7`, Meganeura main `a0eff600`. The
command is the matched pixel world 64x64 200k command with the seed changed.

## Seed 3005

Training 08:26 to 08:59 UTC: 200,000 actions, 12,477 updates, 1,961.7 s
wall; 322 online episodes with guided play, mean 42.7. Final policy entropy
.90, return scale 34.1; world losses reconstruction 2,770, prior rollout
2,771.

## Seed 4007

RUN_4007

## Frozen evaluations (200k, three natural episodes per environment, zero updates)

| Seed | Cohort | Mode | Episodes | Mean | Natural-24 | Median | Max | At or above 100 | Two-wall |
|---|---|---|---|---|---|---|---|---|---|
| 1009 (reference) | 4000001009 | sampled / greedy | 25 / 25 | 37.680 / 156.840 | | | | | 0 |
| 1009 (reference) | 4000022003 | sampled / greedy | 28 / 25 | 40.286 / 151.080 | | | | | 0 |
| 22003 (reference) | 4000022003 | sampled / greedy | 32 / 27 | 28.375 / 99.259 | | | | | 0 |
| 3005 | 4000001009 | sampled | 27 | 46.000 | 49.667 | 36 | 178 | 2 | 0/27 |
| 3005 | 4000001009 | greedy | 26 | 108.500 | 112.458 | 91 | 307 | 11 | 0/26 |
| 3005 | 4000022003 | sampled | 24 | 35.125 | 35.125 | 39 | 67 | 0 | 0/24 |
| 3005 | 4000022003 | greedy | 24 | 160.875 | 160.875 | 119 | 379 | 15 | 0/24 |
EVAL_4007

CDP references, greedy / sampled: seed 1009 143.667 / 70.750 (cohort 1009)
and 94.625 / 70.500 (cohort 22003); seed 22003 75.250 / 44.708 (cohort 22003).

## Reading

READING
