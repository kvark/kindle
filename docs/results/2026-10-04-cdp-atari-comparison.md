# Five-game CDP / RGB / Tiny comparison — in progress

**3/45 train/evaluation pairs audited.** The selected games are Boxing, Pong,
Freeway, Breakout and Qbert, with CDP, faithful Dreamer RGB and pretrained
frozen Tiny, seeds1009/2017/3019. The [fixed declaration](../experiments/2026-10-04-cdp-atari-comparison.md)
keeps all methods unassisted and extrinsic-only. Current activity stays in
[PR31](https://github.com/kvark/kindle/pull/31); this report records completed evidence.

## First matched group: Freeway seed 1009

All three methods finish 200,000 actual actions/49,939 updates, with zero
learner debt, finite checkpoint tensors and **online last-50 score 0**. Frozen
sampled-policy evaluation also scores **0 for every method**, with 24 natural
episodes/49,152 actions each. No training episode receives a positive reward.

| Method / whole rollout | Training time | Full update ms | World train ms | Real time per stream |
| --- | ---: | ---: | ---: | ---: |
| [CDP](../../runs/cdp-atari-learning-20261004.XLgCAlkq/freeway-cdp-1009-frozen.mp4) | 23m43s | 26.587 | 8.236 | 1.171x |
| [RGB](../../runs/cdp-atari-learning-20261004.XLgCAlkq/freeway-rgb-1009-frozen.mp4) | 27m05s | 30.795 | 12.875 | 1.026x |
| [Pretrained frozen Tiny](../../runs/cdp-atari-learning-20261004.XLgCAlkq/freeway-pretrained_tiny-1009-frozen.mp4) | 29m30s | 33.301 | 6.198 | 0.941x |

Eight streams share batched acting and one learner. Aggregate real-time ratios
are 9.371x/8.207x/7.531x respectively. CDP takes 12.4% less training wall time than
RGB and 19.6% less than Tiny on this seed; its world-training stage is 36.0%
shorter than RGB's. Tiny has the cheapest world-training stage but the slowest
whole-agent training. This is a descriptive matched-package comparison, not
multi-seed uncertainty or an unchanged-learning optimization measurement.

Frozen runs take 18.3s/18.3s/168.3s respectively for the same action count.
These timings include checkpoint export but exclude construction; they are not
isolated kernel timings. Frozen evaluation makes zero learner updates and
preserves all 250/292/241 saved model/optimizer tensors byte-exactly.
Guards, counters, final-checkpoint identity, exact CPU replay and whole
stream-zero video checks pass. All episodes, excess episodes and unfinished
tails remain in the audits below.

Each of the six native processes records 16 historical baseline warnings and
**zero new allocation warnings**. No GPU recovery or NVML polling was performed;
utilization remains unmeasured. CPU video replay is outside the native timings.

This seed demonstrates no unassisted Freeway learning at the declared budget
with any method. It does not establish a CDP learning advantage or an Atari-wide
conclusion; the other two seeds and four games remain required. Do not extend
the budget, add action assistance or drop this negative result. Historical aided
Freeway wins are not matched controls. Tiny's 250k same-title pretraining
observations remain additional experience, not online-only learning.

[CDP full pair audit](../../runs/cdp-atari-learning-20261004.XLgCAlkq/freeway-cdp-1009-pair-audit.json) ·
[RGB full pair audit](../../runs/cdp-atari-learning-20261004.XLgCAlkq/freeway-rgb-1009-pair-audit.json) ·
[Tiny full pair audit](../../runs/cdp-atari-learning-20261004.XLgCAlkq/freeway-pretrained_tiny-1009-pair-audit.json) ·
[Compact data/configurations/learning curves](2026-10-04-cdp-atari-comparison.json)

The earlier initialization-only stop and excluded Tiny smokes remain in the
[preparation report](2026-10-04-cdp-atari-initialization-stop.md). The user's
later permission records standalone allocation warnings without blocking GPU
use; it does not reclassify the failed initialization as successful training.
