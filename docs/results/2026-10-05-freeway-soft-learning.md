# Freeway: soft targets do not improve exploration

All three fresh soft-target candidates finish32,768 actions/8,131 updates each,
with zero real rewards and16 natural episodes per run. The retained hard-target
and extrinsic-only controls also score zero. This is another floor result, not
evidence that the methods have equal learning capacity. Freeway remains unsolved.

| Seed | Soft/hard final bonus | Soft/hard policy entropy | Soft maximum player height | Soft wall time |
| --- | ---: | ---: | ---: | ---: |
|1009|.00900 / .00937|2.8887 /2.8890|132|216.51s|
|2017|.01126 / .01169|2.8859 /2.8868|135|220.40s|
|3019|.01002 / .01049|2.8830 /2.8838|113|210.86s|

The target change removes the expected sampling-loss floor: final regression
MSE is.0051–.0063 versus.136–.148. Those different-target losses are not directly
comparable as a learning gain. Bonus magnitudes fall only about4%; policy and
coverage barely change. UP remains32.2–33.3% of requested actions; no stream
reaches height160 or sustains UP longer than nine decisions. Lower target
sampling variance alone is not a sufficient fix at this budget.

All98,304 actions,24,393 updates,48 completed episodes and unfinished tails are
retained. Guards/seals, transition counters and finite checkpoints pass. CPU
replay matches every recorded reward/boundary/reset/frame count;1,532 of1,536
sampled frames validate the diagnostic RAM coordinate, four are unchecked.
The predeclared all-zero branch skips frozen evaluation. Qualification's3,072
actions/451 updates are excluded. No new successful rollout or competence claim.
Mean wall time215.93s; paired−2.28% [−4.61%,−.32% seed-bootstrap95% interval]
versus retained hard controls is descriptive, not a matched timing experiment.
GPU utilization remains unmeasured.

## Interpretation and next experiment

The saved-feature position probe finds more readable player-height information
in the existing CNN than in sampled categorical targets. An independent F64
posterior readout improves the categorical probe when using probabilities,
but remains below the CNN in all three seeds. These are192/64 chronological
train/test splits, fixed unit ridge regularization and no holdout tuning; not
proof of policy use, transfer or a new GPU numerical qualification.

The next focused target is the **existing CNN embedding**, detached on arrival,
as supported by Plan2Explore's [embedding-target implementation](https://github.com/danijar/dreamerv2/blob/main/dreamerv2/expl.py).
Do not add a pixel encoder or change rewards/actions/budget. This changes the
predicted information, output width and natural target scale; disclose all three
instead of calling it a pure noise ablation. No coefficient sweep or unchanged
training extension follows this negative result.

[Declaration](../experiments/2026-10-05-freeway-soft-disagreement.md) ·
[Compact results and curves](2026-10-05-freeway-soft-learning.json) ·
[Qualification](2026-10-05-freeway-soft-qualification.md).
Raw artifacts: `runs/freeway-soft-disagreement-20261005.NeU4KJoy`.
