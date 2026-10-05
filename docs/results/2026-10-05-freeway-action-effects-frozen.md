# Freeway action-effects bonus: small retained improvement, not competence

Frozen policies trained for32,768 actions score14 crossings across72 natural
episodes, versus1 for the retained extrinsic-only controls and0 for actual
initial-weight controls. All three candidates improve over their paired controls.
Only13 candidate episodes score anything; none reaches25. **Freeway is not unlocked.**

| Learner seed | Candidate crossings /24 episodes | Extrinsic control | Untrained |
| --- | ---: | ---: | ---: |
|1009|4|0|0|
|2017|8|0|0|
|3019|2|1|0|

Mean candidate score .19444, seed-bootstrap95% interval[.08333,.33333]. Paired
candidate minus extrinsic .18056 [.04167,.33333]; minus untrained .19444
[.08333,.33333]. These are three development learner replicates with coarse
uncertainty, not72 independent models. The earlier seed3019 training regression
remains a stability concern. This is evidence to test a longer fixed budget,
not a mastery claim or a guarantee that learning will continue.

## Protocol and audits

The [declared positive branch](../experiments/2026-10-05-freeway-action-effects.md)
is complete: three candidates, three retained extrinsic-trained controls and
three actual initial-weight controls. First3 natural episodes per stream, eight
streams, sampled policy, sticky.25/full18/repeat4, no action aid or reward shaping.
Held-out base seed1,000,000,000 plus learner seed, per-stream offset1,000,003.
All216 episodes complete with zero cutoffs:442,368 additional evaluation actions,
zero learner updates. All tails and zero-score episodes are retained.

All12 guards pass (three zero-action initializations plus nine evaluations).
Full CPU trajectory replays match recorded actions/rewards/state. Exact saved
tensor equality passes:346 tensors per candidate/initial model,250 per extrinsic
model. Each full stream0 video passes its hash/frame-count/ffprobe checks.
Original online pixel hashes are unavailable; this is deterministic replay,
not a claim of original-image equality. GPU utilization remains unmeasured.

Retained extrinsic controls trained on native `ecc935a7` and were evaluated on
current `f4b6a5c7`, with the qualified zero-ensemble path and exact configuration,
protocol and checkpoint identity checks. Their original identity is preserved.
Candidates use `f4b6a5c7` for both training/evaluation. No generic pair-verifier
tolerance or production precision was changed.

## Whole rollout videos

The predeclared stream0 videos below are unfiltered; all three candidate stream0
rollouts happen to score zero. The additional
[seed2017 stream2 video](../../runs/freeway-action-effects-20261005.z9KUOhCS/evaluation/effects-2017-stream2.mp4)
contains crossings near2:43 and4:16 in one round. It was selected **after** seeing
results for illustration, contains all three stream episodes, and adds no new
evaluation experience or statistical evidence.

| Seed | Candidate | Extrinsic | Untrained |
| --- | --- | --- | --- |
|1009|[video](../../runs/freeway-action-effects-20261005.z9KUOhCS/evaluation/effects-1009-frozen.mp4)|[video](../../runs/freeway-action-effects-20261005.z9KUOhCS/evaluation/extrinsic-1009-frozen.mp4)|[video](../../runs/freeway-action-effects-20261005.z9KUOhCS/evaluation/untrained-1009-frozen.mp4)|
|2017|[video](../../runs/freeway-action-effects-20261005.z9KUOhCS/evaluation/effects-2017-frozen.mp4)|[video](../../runs/freeway-action-effects-20261005.z9KUOhCS/evaluation/extrinsic-2017-frozen.mp4)|[video](../../runs/freeway-action-effects-20261005.z9KUOhCS/evaluation/untrained-2017-frozen.mp4)|
|3019|[video](../../runs/freeway-action-effects-20261005.z9KUOhCS/evaluation/effects-3019-frozen.mp4)|[video](../../runs/freeway-action-effects-20261005.z9KUOhCS/evaluation/extrinsic-3019-frozen.mp4)|[video](../../runs/freeway-action-effects-20261005.z9KUOhCS/evaluation/untrained-3019-frozen.mp4)|

## Decision

Run one [fresh200k-action follow-up](../experiments/2026-10-05-freeway-effects-200k.md)
per seed with the same mechanism, then evaluate on a new held-out seed set.
The only training change is the finite interaction budget. No checkpoint resume,
encoder change, scripted aid, coefficient sweep or stopped-matrix restart.
Original mastery gate remains mean>=25, >=90% rounds reaching25 crossings,
>=20 natural rounds and no cutoffs; current result fails it.

[Training result](2026-10-05-freeway-action-effects-learning.md) ·
[Compact evidence](2026-10-05-freeway-action-effects-frozen.json).
Raw artifacts: `runs/freeway-action-effects-20261005.z9KUOhCS/evaluation`.
