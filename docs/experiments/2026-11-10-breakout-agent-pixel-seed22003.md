# Second seed for the pixel-world Breakout agents

Every pixel-world agent result so far is seed 1009 on cohort 4000022003. The
retained CDP recipe has a fresh-seed point on that cohort: seed 22003 scores
75.250 greedy / 44.708 sampled ([report](../results/2026-11-06-cdp-breakout-seed22003.md))
against seed 1009's 94.625 / 70.500. This protocol trains the two best
pixel-world learners with seed 22003 so their fresh-seed points can be set
beside it.

## Recipe

Two 200k-action learners, each the retained Breakout command with seed 22003
(environment seeds follow the runner's rule from the seed) and the stated
change, otherwise identical to the seed 1009 runs:

- `pixel64`: the [matched pixel world](2026-11-10-breakout-agent-world96-matched.md)
  at 64x64 (no CDP; `--world-pixel-loss motion_cross_entropy
  --world-prediction-length 8`, bonus kept).
- `aux1`: the [CDP plus auxiliary pixel rollout](2026-11-10-breakout-agent-cdp-pixel-aux.md)
  at scale 1 (`--world-pixel-scale 1.0` with the same pixel flags).

Root `/x/Temp/kindle-breakout-pixel-seed22003-20261110`, outputs `pixel64/`
and `aux1/`, `queue.json`, unit `kindle-breakout-pixel-seed22003-20261110.service`.

## Evaluation

Frozen evaluations on cohort 4000022003 (three natural episodes per
environment, zero updates, sampled and greedy, actual-initial and 200k
checkpoints). References on that cohort: fresh CDP 22003 75.250 / 44.708;
retained CDP 1009 94.625 / 70.500; pixel64 1009 151.080 / 40.286; aux1 1009
104.036 / 61.692.

## Reading

Two seeds are not a reliability claim. If both seed 22003 learners stay above
the fresh CDP 22003 greedy point, the greedy gain is not a seed 1009
accident; if either falls to the CDP band or below, the seed 1009 result is
seed variation and the ranking stands unconfirmed. Sampled scores are read
the same way against 44.708. Report updates, wall time, two-wall counts and
the per-episode distributions.
