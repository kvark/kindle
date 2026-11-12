# Retained CDP recipe on the pixel-world recipe's fresh seeds

[Protocol](../experiments/2026-11-12-breakout-agent-cdp-seeds.md). Root
`/x/Temp/kindle-breakout-cdp-seeds-20261112`, outputs `s3005/` and `s4007/`,
guards `guards/`, unit `kindle-breakout-cdp-seeds-20261112.service` (launched
22:49 UTC on November 12). Source `45854db3`, Meganeura main `a0eff600`. The
command is the retained Breakout learner's with only the seed changed.

## Seed 3005

Training 22:49 to 23:11 UTC: 200,000 actions, 12,477 updates, 1,279.1 s wall
(the retained recipe's 21-minute pace); 272 online episodes with guided
play, mean 50.2.

## Seed 4007

RUN_4007

## Frozen evaluations (200k, three natural episodes per environment, zero updates)

| Recipe, seed | Cohort | Mode | Episodes | Mean | Natural-24 | Median | Max | At or above 100 | Two-wall |
|---|---|---|---|---|---|---|---|---|---|
| CDP 3005 | 4000001009 | sampled | 27 | 39.370 | 41.000 | 40 | 81 | 0 | 0/27 |
| CDP 3005 | 4000001009 | greedy | 28 | 93.036 | 96.875 | 68.5 | 346 | 5 | 0/28 |
| CDP 3005 | 4000022003 | sampled | 26 | 41.115 | 41.292 | 35.5 | 121 | 1 | 0/26 |
| CDP 3005 | 4000022003 | greedy | 28 | 112.036 | 123.375 | 87 | 373 | 11 | 0/28 |
EVAL_4007

## Paired comparison (greedy / sampled)

| Seed | CDP, cohort 1009 | Pixel, cohort 1009 | CDP, cohort 22003 | Pixel, cohort 22003 |
|---|---|---|---|---|
| 1009 | 143.7 / 70.8 | 156.8 / 37.7 | 94.6 / 70.5 | 151.1 / 40.3 |
| 22003 | | | 75.3 / 44.7 | 99.3 / 28.4 |
| 3005 | 93.0 / 39.4 | 108.5 / 46.0 | 112.0 / 41.1 | 160.9 / 35.1 |
| 4007 | PAIR_4007 | 50.5 / 22.5 | PAIR_4007_B | 43.7 / 23.1 |

## Reading

READING
