# Held-out evaluation cohort for the retained and pixel-world Breakout agents

[Protocol](../experiments/2026-11-12-breakout-agent-heldout.md). Root
`/x/Temp/kindle-breakout-agent-heldout-20261112`, outputs `out/`, guards
`guards/`; ran 06:35 to 06:41 UTC on November 12 under
`kindle-breakout-agent-heldout-20261112.service`. Cohort 4000777001 (never
used before), three natural episodes per environment, zero updates, no
stalled episodes; all four checkpoints are seed 1009.

| Checkpoint | Mode | Episodes | Mean | Natural-24 | Median | Max | At or above 100 | Two-wall |
|---|---|---|---|---|---|---|---|---|
| Retained CDP, 200k | sampled | 24 | 75.208 | 75.208 | 57.5 | 316 | 3 | 0/24 |
| Retained CDP, 200k | greedy | 25 | 163.200 | 162.625 | 117 | 367 | 17 | 0/25 |
| Pixel world, 200k | sampled | 26 | 48.000 | 48.417 | 34.5 | 228 | 2 | 0/26 |
| Pixel world, 200k | greedy | 24 | 177.833 | 177.833 | 111 | 414 | 14 | 0/24 |
| Retained CDP, 400k | sampled | 26 | 90.615 | 93.375 | 77 | 344 | 7 | 0/26 |
| Retained CDP, 400k | greedy | 25 | 192.000 | 197.417 | 143 | 368 | 16 | 0/25 |
| Pixel world, 400k | sampled | 28 | 89.750 | 93.042 | 72 | 307 | 9 | 0/28 |
| Pixel world, 400k | greedy | 24 | **223.000** | 223.000 | 237.5 | 371 | 19 | 0/24 |

For reference, the same checkpoints on the two reused cohorts (4000001009 /
4000022003), greedy / sampled: CDP 200k 143.7 / 70.8 and 94.6 / 70.5; pixel
200k 156.8 / 37.7 and 151.1 / 40.3; CDP 400k 173.7 / 92.8 and 150.8 / 120.6;
pixel 400k 185.2 / 83.6 and 158.6 / 100.9.

## Reading

On a cohort none of these learners or their recipes ever saw, the seed-1009
pixel-world agent beats the retained CDP agent in greedy play at both budgets
(177.8 against 163.2 at 200k, 223.0 against 192.0 at 400k; nineteen of
twenty-four episodes at or above 100 and a median of 237.5 at 400k), loses
sampled play at 200k (48.0 against 75.2) and ties it at 400k (89.8 against
90.6). This cohort is easier for every learner than the two reused ones (all
eight scores are above their counterparts there), so it changes the
comparison, not the absolute scale. The two-wall gate is still untouched: no
episode in these eight evaluations, or in the 46 before them, reached 432;
the pixel world's 414 is the closest any Kindle agent has come. This is one
fresh cohort for one training seed each; the pixel recipe's seed spread
(44 to 161 greedy at 200k on cohort 22003) stands, and nothing here speaks
to the seeds that scored low.
