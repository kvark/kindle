# Five-game CDP / RGB / Tiny comparison — in progress

**1/45 train/evaluation pairs audited.** The selected games are Boxing, Pong,
Freeway, Breakout and Qbert, with CDP, faithful Dreamer RGB and pretrained
frozen Tiny, seeds1009/2017/3019. The [fixed declaration](../experiments/2026-10-04-cdp-atari-comparison.md)
keeps all methods unassisted and extrinsic-only. Current activity stays in
[PR31](https://github.com/kvark/kindle/pull/31); this report records completed evidence.

## First result: Freeway CDP seed1009

Training completes200,000 actual actions/49,939 updates in23m43s, with zero
learner debt, finite checkpoint tensors and **online last50 score0**. Eight
batched streams achieve1.171x real time each. Mean full update time is26.587ms,
including8.236ms world training and11.717ms imagination. This different-game
timing is not an unchanged-workload speedup over Seaquest.

Frozen sampled-policy evaluation completes the declared24 episodes at49,152
actions: **mean0**, zero learner updates, all250 saved model/optimizer tensors
unchanged. Guards, action/update ledgers, final-checkpoint identity, exact CPU
replay and whole stream-zero video checks pass. Both native processes record
16 historical baseline warnings and **zero new allocation warnings**. No GPU
recovery or NVML polling was performed; utilization remains unmeasured.

This seed does not demonstrate unassisted Freeway learning at the declared
budget. It does not yet compare CDP with RGB/Tiny: those controls and the other
seeds are still required. Do not extend the budget, add action assistance or
drop this negative result. Historical aided Freeway wins are not matched controls.

[Whole rollout video](../../runs/cdp-atari-learning-20261004.XLgCAlkq/freeway-cdp-1009-frozen.mp4) ·
[Full pair audit, all episodes and tails](../../runs/cdp-atari-learning-20261004.XLgCAlkq/freeway-cdp-1009-pair-audit.json) ·
[Compact data/configuration/learning curve](2026-10-04-cdp-atari-comparison.json)

The earlier initialization-only stop and excluded Tiny smokes remain in the
[preparation report](2026-10-04-cdp-atari-initialization-stop.md). The user's
later permission records standalone allocation warnings without blocking GPU
use; it does not reclassify the failed initialization as successful training.
