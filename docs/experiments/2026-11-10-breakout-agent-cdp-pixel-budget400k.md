# CDP plus pixel auxiliary: original-cohort scores and a 400k budget point

The retained CDP recipe has two fixed budget points on its original
evaluation cohort 4000001009: 200k scores 70.750 sampled / 143.667 greedy and
[400k](../results/2026-11-03-cdp-breakout-budget400k.md) scores 92.792 /
173.708 in 43m37s. The [CDP plus pixel auxiliary at scale 1](2026-11-10-breakout-agent-cdp-pixel-aux.md)
has beaten CDP on two seeds on cohort 4000022003 only. This protocol puts it
on the original cohort at both budgets.

## Jobs

1. Frozen evaluations on cohort 4000001009 (sampled and greedy, three natural
   episodes per environment, zero updates) of two existing seed 1009
   checkpoints: the CDP plus pixel scale 1 200k learner
   (`/x/Temp/kindle-breakout-cdp-pixel-aux-20261110/s1/final`) and the matched
   pixel world 64x64 200k learner
   (`/x/Temp/kindle-breakout-world96-matched-20261110/w64/final`). Outputs
   `eval200k/`.
2. A fresh 400,000-action life of the CDP plus pixel scale 1 command (seed
   1009, otherwise the 200k command; recovery checkpoints every 100k), then
   frozen evaluations of its actual-initial and 400k checkpoints on cohort
   4000001009 and of the 400k checkpoint on cohort 4000022003, sampled and
   greedy. Output `candidate/`.

Root `/x/Temp/kindle-breakout-cdp-pixel-400k-20261110`, `queue.json`, unit
`kindle-breakout-cdp-pixel-400k-20261110.service`. Expected training time
about 65 minutes at 142 ms per update for 24,977 updates.

## Reading

The 200k evaluations say whether the two-seed gain carries to the original
cohort against 70.750 / 143.667. The 400k point is read against 92.792 /
173.708 and against the recipe's own 200k point for return per training
minute, as the retained report did. One seed, fixed budgets, no extension of
an existing life; the original 864/90% two-wall gate is reported, not
expected.
