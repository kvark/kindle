# Retained CDP recipe on the pixel-world recipe's fresh seeds

The pixel-world recipe's greedy score spans 44 to 161 across four training
seeds; the retained CDP recipe has two seeds (1009 and 22003). A fair
variance comparison needs CDP on the same fresh seeds, so this screen trains
the retained Breakout command unchanged with `--seed 3005` and `--seed 4007`
(the seeds of the [pixel-world seed screen](2026-11-11-breakout-agent-pixel-seeds.md))
and evaluates each 200k checkpoint on cohorts 4000001009 and 4000022003,
sampled and greedy, three natural episodes per environment.

Root `/x/Temp/kindle-breakout-cdp-seeds-20261112`, outputs `s3005/` and
`s4007/`, `queue.json`, unit `kindle-breakout-cdp-seeds-20261112.service`.
Pixel-world references on the same seeds, greedy / sampled: seed 3005 108.5 /
46.0 (cohort 1009) and 160.9 / 35.1 (cohort 22003); seed 4007 50.5 / 22.5 and
43.7 / 23.1. CDP references: seed 1009 143.7 / 70.8 and 94.6 / 70.5; seed 22003
75.3 / 44.7 (cohort 22003). Four seeds each on two reused cohorts is a
paired development comparison, not qualification; it says whether the pixel
recipe's spread is unusual for this training setup or shared by CDP.
