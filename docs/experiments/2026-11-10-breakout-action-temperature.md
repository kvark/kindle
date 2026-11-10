# Diagnostic: action temperature on the 400k pixel-world and CDP checkpoints

Every pixel-world learner beats centered CDP in greedy play and loses in
sampled play. If the gap comes from a flatter action distribution rather than
from different greedy policies, sampling at a lower softmax temperature should
move the pixel world's sampled score toward its greedy score faster than it
moves CDP's. This is a frozen-evaluation diagnostic with the runner's
existing `--action-temperature` option (sampled frozen restore only); it
changes no training and is not the retained protocol's sampled mode.

## Jobs

Frozen evaluations on cohort 4000001009, three natural episodes per
environment, zero updates, of the pixel world 400k checkpoint
(`/x/Temp/kindle-breakout-pixel64-400k-20261110/candidate/final`) and the
retained CDP 400k checkpoint
(`/x/Temp/kindle-breakout-budget400k-20261103/full/breakout-12m-64-b16-r16-replayvalue1-expert100-400k-seed1009/final`)
at action temperatures .5 and .25. Temperature 1 is the recorded sampled
score (83.615 and 92.792) and temperature 0 the recorded greedy score
(185.160 and 173.708). Root `/x/Temp/kindle-breakout-temperature-20261110`,
`queue.json`, outputs `out/`.

## Reading

A diagnostic on one seed and one cohort. If the pixel world at .5 or .25
reaches CDP's score at the same temperature, the sampled deficit is a
distribution-sharpness effect that an entropy or unimix setting could
address in training; if it stays behind at every temperature, the sampled
policy itself ranks actions worse and the greedy gain does not transfer.
