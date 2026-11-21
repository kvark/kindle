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

Training 09:00 to 09:33 UTC: 200,000 actions, 12,477 updates, 1,969.8 s
wall; 327 online episodes with guided play, mean 46.0. Final policy entropy
.75, return scale 35.6; world losses reconstruction 2,806, prior rollout
2,867 (the world learns the same way on every seed; the actor does not).

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
| 4007 | 4000001009 | sampled | 26 | 22.538 | 22.917 | 24 | 43 | 0 | 0/26 |
| 4007 | 4000001009 | greedy | 28 | 50.536 | 50.708 | 43 | 137 | 3 | 0/28 |
| 4007 | 4000022003 | sampled | 25 | 23.120 | 23.625 | 22 | 53 | 0 | 0/25 |
| 4007 | 4000022003 | greedy | 26 | 43.692 | 45.708 | 43.5 | 88 | 0 | 0/26 |

Seed summary of the unchanged pixel-world recipe at 200k, greedy / sampled:

| Seed | Cohort 4000001009 | Cohort 4000022003 |
|---|---|---|
| 1009 | 156.840 / 37.680 | 151.080 / 40.286 |
| 22003 | | 99.259 / 28.375 |
| 3005 | 108.500 / 46.000 | 160.875 / 35.125 |
| 4007 | 50.536 / 22.538 | 43.692 / 23.120 |
| mean (3 / 4 seeds) | 105.3 / 35.4 | 113.7 / 31.7 |
| Retained CDP, seed 1009 | 143.667 / 70.750 | 94.625 / 70.500 |
| Fresh CDP, seed 22003 | | 75.250 / 44.708 |
| CDP mean (1 / 2 seeds) | 143.7 / 70.8 | 84.9 / 57.6 |

CDP references, greedy / sampled: seed 1009 143.667 / 70.750 (cohort 1009)
and 94.625 / 70.500 (cohort 22003); seed 22003 75.250 / 44.708 (cohort 22003).

## Reading

The pixel-world recipe's greedy score varies by a factor of three across
seeds (44 to 161) where the world losses do not, so the seed-1009 result that
opened this line (157 / 151) was a high draw, and the earlier statement that
the pixel objective beats CDP in greedy play on every pair is withdrawn:
seed 4007 loses to both CDP learners in both modes on both cohorts. What
survives: on the new cohort the pixel recipe's four-seed greedy mean (113.7)
is above CDP's two-seed mean (84.9), three of four pixel seeds beat both CDP
seeds there, and the sampled mode is worse on every pixel seed (23 to 46
against 45 to 71). On the retained learner's own cohort CDP's single seed
(143.7) beats two of the three pixel seeds. Four and two seeds on reused
development cohorts cannot rank the two recipes reliably; they do say the
pixel objective raises the ceiling of greedy play (four runs above 150, CDP's
best 150.8 at 400k) while making the learner less predictable and worse
when sampled.

The training-time cost is unchanged (33 against 21 minutes per 200k). No
episode in any of the 46 frozen evaluations run since November 10 reached
the two-wall threshold; the best single episode is 404. The sampled-mode
levers tried (entropy scale, unimix) do not work; the return normalization
and imitation weight are untested; and whether a multi-seed CDP screen would
show the same variance is unknown, since CDP has two seeds here.
