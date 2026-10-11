# Greedy actions help the first pyramid, not Qbert's overall score

**Keep sampled evaluation as the standard.** Greedy actions improve first-pyramid
completion in every seed,62/72→69/72, but their mean score is6,953.47 versus
7,367.01 sampled. The paired change is−413.54 [−2,092.71,580.21]. Two seeds
improve modestly;3019 loses2,092.71 points. No model passes the full historical
completion/15,000-score gate or the193,220.77 long-run reference.

[Declaration](../experiments/2026-10-11-cdp-qbert-action-mode.md) ·
[Compact evidence](2026-10-11-cdp-qbert-action-mode.json) ·
[Independent completion review](../../runs/cdp-qbert-greedy-20261011.5vPsrd3M/completion-review.json) ·
[Retained sampled results](2026-10-11-cdp-qbert-budget.md).

| Seed | Sampled score | Greedy score | Paired change | First pyramids,sampled→greedy | Greedy actual-initial score |
| --- | ---: | ---: | ---: | ---: | ---: |
|1009 |5,172.92 |5,444.79 |+271.88 |20/24→22/24 |239.58 |
|2017 |5,812.50 |6,392.71 |+580.21 |22/24→24/24 |278.13 |
|3019 |11,115.63 |9,022.92 |−2,092.71 |20/24→23/24 |251.04 |

All three greedy candidates clear the90% first-pyramid fraction, but remain
below the accompanying score threshold. Every candidate beats its actual
initial; initial controls complete zero pyramids. Their mean256.25 is not the
sampled initial mean200.69. Intervals resample the three learner seeds, not
episodes or streams. These reused development seeds are not a final held-out
test; different action modes naturally produce different interaction counts.

## What ran and passed

Six fresh frozen evaluations restore the same completed2M and actual-initial
checkpoints, changing only the existing `--greedy` runtime flag. Native7311547d,
N8/eight host workers, full18/sticky.25/repeat4, native input/one GPU RGB64 resize
and evaluation seeds are unchanged. No retraining, action hints, model rewrite
or policy-temperature sweep. Greedy action selection does not make ALE or
categorical beliefs deterministic.

The service finishes October11 at17:33:10 UTC in**3m32.63s**. All six guards,
144 selected natural episodes, exact346 saved tensors/model and six whole
CPU replays/videos pass. **218,936 evaluation actions, zero learner updates**;
frozen loops take79.55s, with initialization/replays/audits additional. Retain
all14 excess episodes, unfinished tails and nine allocation warnings. No
selected cutoff, GPU failure or runtime retry. GPU utilization is unmeasured;
this is not a training-throughput comparison.

The final independent CPU review passes in2.33s (231.5MiB,zero swap), rereading
both modes' logs, selecting the first three natural episodes per stream,
recomputing scores/task successes and seed summaries, and verifying58 source
pins, saved tensors, guard seals and replay/video hashes. The initial CPU
preparation's lazy-module-path error and failed source remain; it occurred
before any game/GPU action. No worker remains.

## Decision

Action sampling contributes to first-pyramid unreliability, but removing it
does not consistently improve reward or solve later play. Do not promote a
best-mode/best-seed score, replace the sampled controls, lower a gate or start
an entropy sweep. Together with the [useful reward forecasts](2026-10-11-cdp-qbert-forecasts.md),
this shifts the next review toward longer-horizon value/credit assignment and
measured learning cost, not another representation replacement. Before using
real returns to judge the critic, account for its combined extrinsic/intrinsic
training target. No new learner allocation follows automatically. The full
five-game DreamerV3-quality goal and matched RGB budget comparison remain open.

## Whole rollouts

These are complete stream-zero videos, including excess episodes and tails;
reported scores cover all eight streams. Links require this workspace.

| Seed | Greedy trained | Greedy actual initial |
| --- | --- | --- |
|1009 |[Whole rollout](../../runs/cdp-qbert-greedy-20261011.5vPsrd3M/qbert-centered-1009-candidate-frozen.mp4) |[Whole control](../../runs/cdp-qbert-greedy-20261011.5vPsrd3M/qbert-centered-1009-untrained-frozen.mp4) |
|2017 |[Whole rollout](../../runs/cdp-qbert-greedy-20261011.5vPsrd3M/qbert-centered-2017-candidate-frozen.mp4) |[Whole control](../../runs/cdp-qbert-greedy-20261011.5vPsrd3M/qbert-centered-2017-untrained-frozen.mp4) |
|3019 |[Whole rollout](../../runs/cdp-qbert-greedy-20261011.5vPsrd3M/qbert-centered-3019-candidate-frozen.mp4) |[Whole control](../../runs/cdp-qbert-greedy-20261011.5vPsrd3M/qbert-centered-3019-untrained-frozen.mp4) |
