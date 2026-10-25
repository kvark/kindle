# Breakout: greedy actions do not consistently fix the plateau

| Frozen policy | Sampled mean | Greedy mean |
| --- | ---: | ---: |
| Source2M |98.208 |53.458 |
| Warm-start8M lineage |85.167 |97.792 |
| Actual initial |1.417 |2.208 |

Each score uses24 natural rounds. All greedy cohorts complete, with zero
two-wall successes or cutoffs. Keep sampled evaluation as standard; selecting
the better mode per checkpoint would conceal the inconsistent effect. This
does not identify an entropy setting or justify an entropy sweep. Neither
mode approaches the381.81 reference or864-point two-wall gate.

All three guards, exact346-tensor frozen comparisons and whole trajectory/video
replays pass. Total51,944 actions, zero learner updates. The service finishes
October23 03:30:34UTC,90.85s wall time. Same native c8aeba28, model configuration,
evaluation seed and preprocessing; only action selection changes. ALE stickiness
and stochastic latents remain, so greedy does not imply deterministic play.
One lineage and reused development episodes do not establish population effects.

Next: the bounded12M spatial-state check, then a learning decision. Backend
optimization remains parked; no automatic long Breakout extension.

- [Audit](../../runs/cdp-breakout-greedy-20261023.C20j01e3/review.json)
- [Source greedy rollout](../../runs/cdp-breakout-greedy-20261023.C20j01e3/source.mp4)
- [Final greedy rollout](../../runs/cdp-breakout-greedy-20261023.C20j01e3/final.mp4)
- [Untrained greedy rollout](../../runs/cdp-breakout-greedy-20261023.C20j01e3/initial.mp4)
- [Fixed comparison plan](../experiments/2026-10-23-cdp-breakout-action-mode.md)
