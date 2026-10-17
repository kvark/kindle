# Longer Pong training does not resolve CDP's weak seeds

The three fresh500k-action runs complete October5 at17:42 UTC, in3h32m29s
including frozen evaluation and replay audits. Increasing the finite budget by
2.5x improves seed2017 substantially but leaves1009/3019 near a complete loss.
**Pong is still unreliable; no further budget extension is declared.**

| Seed | Final online mean | Frozen score | Wins /24 | Actual initial score | Whole trained / initial video |
| --- | ---: | ---: | ---: | ---: | --- |
|1009|-20.46|-21.0000|0/24|-20.2917|[trained](../../runs/cdp-pong-budget-20261005.LTG3h04X/pong-cdp-1009-candidate-frozen.mp4) / [initial](../../runs/cdp-pong-budget-20261005.LTG3h04X/pong-cdp-1009-untrained-frozen.mp4)|
|2017|-4.50|-2.3750|7/24|-20.5000|[trained](../../runs/cdp-pong-budget-20261005.LTG3h04X/pong-cdp-2017-candidate-frozen.mp4) / [initial](../../runs/cdp-pong-budget-20261005.LTG3h04X/pong-cdp-2017-untrained-frozen.mp4)|
|3019|-20.84|-20.9167|0/24|-20.3333|[trained](../../runs/cdp-pong-budget-20261005.LTG3h04X/pong-cdp-3019-candidate-frozen.mp4) / [initial](../../runs/cdp-pong-budget-20261005.LTG3h04X/pong-cdp-3019-untrained-frozen.mp4)|

Frozen learner-seed mean−14.7639,95% seed-bootstrap interval[−21.0000,−2.3750];
initial−20.3750. Paired improvement+5.6111 [−.7083,18.1250] includes no
improvement. Episodes/streams are not independent learner replicates.
All72 trained and72 initial selected episodes are natural; there are7/72
trained wins. This is not the historical Pong mastery gate or DreamerV3 quality.

## What the extra budget shows

The [declaration](../experiments/2026-10-05-cdp-pong-budget.md) changes only
training length, keeping Size1M/N8/B8/T16/H15/R32 and the qualified CDP/action-
effects package. Fresh learners are not checkpoint resumes; repeated prefixes
are additional compute. New held-out evaluation base3,000,000,000 plus learner
seed, first3 completed episodes per each of8 streams, cap200k actions.

Last50-completed-episode online means, at the actual recorded action counts:

| Actions |1009|2017|3019|
| ---: | ---: | ---: | ---: |
|32,768|-20.16|-20.42|-20.39|
|65,536|-20.40|-20.40|-20.38|
|131,072|-20.42|-18.76|-20.52|
|199,680|-20.54|-16.90|-20.56|
|299,520|-20.74|-14.94|-20.42|
|399,872|-20.44|-9.02|-20.88|
|500,000|-20.46|-4.50|-20.84|

The published Atari57 Pong mean is−7.1622 at2M frames versus our−15.2667 near
that interaction count; the original long-run target remains20.4455.
The [released reference](2026-10-05-dreamerv3-five-game-reference.json) and
[five-game evidence](2026-10-05-cdp-five-game-screen.json) retain all six
reference seeds. Protocol, model and averaging-window differences remain;
no exact reproduction or compute-saving claim. The separate Atari100k target
uses fewer frames and a different protocol.

Last100k-action training diagnostics are also seed-dependent:

| Seed | Raw KL | CDP cosine error | Reward loss | Negative-event prediction / zero-event prediction |
| --- | ---: | ---: | ---: | ---: |
|1009|0.0762|0.000251|0.1303|-0.0230 / -0.0226|
|2017|2.2867|0.005173|0.0092|-0.2849 / -0.0002|
|3019|0.2478|0.000743|0.1333|-0.0253 / -0.0242|

The failed seeds have lower prediction error but poorer reward separation and
less posterior/prior information difference. This is consistent with
task-irrelevant representations, **not proof of feature collapse**. These
are update-weighted minibatch averages, not pooled event calibration or
held-out forecasts. Seed2017 learns reward events and improves control; no
adapter-wide all-zero reward failure or nonfinite learner is observed.

Follow-up: the [Meganeura refresh](2026-10-06-meganeura-refresh.md) and
[frozen Pong probes](2026-10-06-cdp-pong-world.md) now pass. The latter find
readable CNN features but weak recurrent ball/reward state in the failed seeds;
they motivate one centered-CDP loss test, not another unchanged extension.
Labels never enter actor training. Do not automatically repeat the suite or
increase capacity.

## Audit and cost

All nine host guards, counters, finite final checkpoints, exact346 saved
tensors per frozen model, complete trajectory replays and whole-video
hash/frame checks pass. The October6 CPU consolidation rechecks these in12.5s
before replacing the installed backend. No recovery or NVML polling occurred.

1.5M training actions /374,817 updates;5,998,843 actual frames,
3h28m36s training time. Frozen evaluation adds258,824 actions and zero updates.
All episodes, excess completions and unfinished tails remain in the artifacts.
Nativef4b6a5c7 /Meganeura592a2f5a /Bladee349cddf identify **this** result; the
October6 backend refresh cannot relabel it. Earlier200k/Freeway development,
the fresh repeated prefixes and diagnostic work remain additional experience
and cost. GPU utilization is unmeasured.

[Configuration, all seed results, score/action/time curves and audits](2026-10-06-cdp-pong-budget.json).
Artifacts: `runs/cdp-pong-budget-20261005.LTG3h04X`.
