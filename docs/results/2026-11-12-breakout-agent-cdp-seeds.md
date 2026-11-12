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

Training 23:13 to 23:34 UTC: 200,000 actions, 12,477 updates, 1,288.1 s
wall; 270 online episodes with guided play, mean 48.2.

## Frozen evaluations (200k, three natural episodes per environment, zero updates)

| Recipe, seed | Cohort | Mode | Episodes | Mean | Natural-24 | Median | Max | At or above 100 | Two-wall |
|---|---|---|---|---|---|---|---|---|---|
| CDP 3005 | 4000001009 | sampled | 27 | 39.370 | 41.000 | 40 | 81 | 0 | 0/27 |
| CDP 3005 | 4000001009 | greedy | 28 | 93.036 | 96.875 | 68.5 | 346 | 5 | 0/28 |
| CDP 3005 | 4000022003 | sampled | 26 | 41.115 | 41.292 | 35.5 | 121 | 1 | 0/26 |
| CDP 3005 | 4000022003 | greedy | 28 | 112.036 | 123.375 | 87 | 373 | 11 | 0/28 |
| CDP 4007 | 4000001009 | sampled | 24 | 49.583 | 49.583 | 45.5 | 92 | 0 | 0/24 |
| CDP 4007 | 4000001009 | greedy | 27 | 82.333 | 76.000 | 71 | 252 | 5 | 0/27 |
| CDP 4007 | 4000022003 | sampled | 26 | 43.692 | 45.375 | 40 | 87 | 0 | 0/26 |
| CDP 4007 | 4000022003 | greedy | 25 | 100.360 | 102.750 | 77 | 376 | 6 | 0/25 |

## Paired comparison (greedy / sampled)

| Seed | CDP, cohort 1009 | Pixel, cohort 1009 | CDP, cohort 22003 | Pixel, cohort 22003 |
|---|---|---|---|---|
| 1009 | 143.7 / 70.8 | 156.8 / 37.7 | 94.6 / 70.5 | 151.1 / 40.3 |
| 22003 | | | 75.3 / 44.7 | 99.3 / 28.4 |
| 3005 | 93.0 / 39.4 | 108.5 / 46.0 | 112.0 / 41.1 | 160.9 / 35.1 |
| 4007 | 82.3 / 49.6 | 50.5 / 22.5 | 100.4 / 43.7 | 43.7 / 23.1 |
| mean | 106.3 / 53.3 (3 seeds) | 105.3 / 35.4 (3 seeds) | 95.6 / 50.0 (4 seeds) | 113.7 / 31.7 (4 seeds) |
| range, greedy | 82 to 144 | 51 to 157 | 75 to 112 | 44 to 161 |

## Reading

With four seeds on each side the two recipes separate cleanly. Greedy
means are close (CDP 106.3 against pixel 105.3 on cohort 1009; 95.6 against
113.7 on cohort 22003) but the pixel recipe's greedy range is two to three
times wider (44 to 161 against 75 to 112 on cohort 22003); it wins greedy on
three of four seeds on cohort 22003 and two of three on cohort 1009, and
loses badly on seed 4007. Sampled play is CDP's on every seed but 3005
(means 53.3 against 35.4 and 50.0 against 31.7), and CDP's own sampled
scores also vary widely across seeds (39 to 71 on cohort 1009), so the
retained learner's 70-point sampled scores were themselves a favourable seed.
The honest summary of the pixel objective after four paired seeds: a higher
greedy ceiling and a similar greedy mean with much higher variance, and a
sampled mode about 18 points lower on average. Neither recipe reached a
two-wall episode in any of the 16 evaluations here (CDP's best single episode
376, the pixel world's 379 on these seeds). Four paired seeds on two reused
development cohorts are enough to rank the recipes' variance, not to qualify
either against the 864/90% gate.
