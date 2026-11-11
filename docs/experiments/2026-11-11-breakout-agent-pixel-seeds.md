# Two fresh seeds of the unchanged pixel-world Breakout recipe

Five seed-1009 runs of the pixel-world recipe (the matched baseline and four
single-setting variants) score 77 to 157 greedy on cohort 4000001009. Before
any of today's conclusions rest on the baseline's 156.8 / 151.1, this screen
trains the unchanged baseline recipe with two fresh seeds and evaluates them
on both cohorts, so the recipe's own seed spread is measured rather than
inferred from variants.

## Recipe

The [matched pixel world 64x64](2026-11-10-breakout-agent-world96-matched.md)
200k command with `--seed 3005` (`s3005/`) and `--seed 4007` (`s4007/`);
nothing else changes. Root `/x/Temp/kindle-breakout-pixel-seeds-20261111`,
`queue.json`, unit `kindle-breakout-pixel-seeds-20261111.service`.

## Evaluation

Frozen evaluations of each 200k checkpoint on cohorts 4000001009 and
4000022003, sampled and greedy, three natural episodes per environment.
References, greedy / sampled: pixel world seed 1009 156.840 / 37.680 (cohort
1009) and 151.080 / 40.286 (cohort 22003); seed 22003 99.259 / 28.375 (cohort
22003); retained CDP seed 1009 143.667 / 70.750 and 94.625 / 70.500; fresh
CDP seed 22003 75.250 / 44.708 (cohort 22003).

## Reading

Four pixel-world seeds against two CDP seeds is still a development
comparison on reused cohorts, not qualification. If the fresh seeds land
near the baseline, the greedy gain over CDP is a recipe property; if they
land in the 77 to 130 band of the variants, the earlier greedy claims shrink
to "comparable to CDP with less cohort variation" and the sampled deficit
stands as the recipe's main cost.
