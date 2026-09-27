# Historical scores at the strategy reset

These are existing frozen evaluations, **not new training or a matched benchmark**.
[Self-contained data](2026-09-27-historical-scores.json) includes every learner
seed mean, episode count, protocol qualifier and source path.

Human-normalized score is `(score - random) / (human - random)`, so 1 means the
reference human score, not mastery. Use the
[upstream Dreamer score anchors](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/baselines.yaml).
Means weight learner seeds equally. Do not aggregate this mixed-budget/protocol
snapshot into a suite ranking.

| Game | Frozen result | Learner seeds | Mean score | Human-normalized |
| --- | --- | ---: | ---: | ---: |
| Boxing | historical frozen non-sticky | 3 | 85.9946 | 7.1579 |
| Freeway | historical frozen non-sticky | 3 | 32.6204 | 1.1020 |
| Pong | sticky, equal first 4 episodes per stream | 1 | -7.1667 | 0.3834 |
| Breakout | four actions | 1 | 10.9167 | 0.3200 |
| Breakout | eighteen actions | 1 | 11.6250 | 0.3446 |
| Qbert | 3.2M actions, R64 final | 1 | 12595.3704 | 0.9353 |

Freeway training used probability .5 / hold 64 random-action assistance; frozen
evaluation did not. Tiny was pretrained on 250,000 random-play RGB64 observations
from Boxing, Pong, Freeway, Breakout and Qbert (45k train + 5k validation each).
Large is a different externally pretrained encoder. All rows use historical
RGB64 input; only the labelled Pong result uses .25 sticky actions.

Boxing/Freeway have three learner roots; the other rows have one. No
learner-level confidence interval or learning curve is reconstructed in this
historical snapshot. These omissions are explicit nulls in JSON, not zero
variance. New development comparisons must retain curves and use at least three
learner seeds.

The old Breakout two-wall/864-point gate is **29.94 human-normalized**. Neither
tested 200k-action recipe approaches it. Stop repeating those gate runs; screen
learning curves and representations first. This evidence does not prove the
gate impossible for every method or set the required budget.

See the [current roadmap](../kindle_single_life_dreamer_plan.md) for full
disclosures, historical gates and videos.
