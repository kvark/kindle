# CDP five-game screen: four retained improvements, Pong still unreliable

The fixed screen is complete. Boxing and Freeway learn strongly; Breakout and
Qbert improve modestly over their actual initial-weight controls. Pong improves
in only one of three seeds. **None reaches the predeclared long-run DreamerV3
reference. The five-game / lower-budget goal remains open.**

The twelve new training/evaluation pairs finish at **12:36 UTC on October5**,
in **5h43m45s**, with no worker left running. All three earlier Freeway cohorts
are reused, including weaker seeds. The final independent CPU consolidation
rechecks all fifteen learners and thirty evaluations in26s, without GPU work.

## Scores and the budget comparison

Each learner gets200,000 aggregate decisions, approximately800,000 actual
emulator frames. Online means use the last50 completed episodes. Frozen means
use the first3 natural episodes per stream, eight streams/model. Published
scores are online frame bins, with different averaging windows and protocols.

| Game | Kindle online | Published at800k frames | Kindle frozen | Initial control | Published last10% of200M |
| --- | ---: | ---: | ---: | ---: | ---: |
|Boxing|68.49|64.1|69.74|0.39|99.61|
|Pong|-19.33|-20.5|-18.53|-20.31|20.45|
|Freeway|19.15|0|24.14|0|33.4|
|Breakout|4.17|3.21|4.36|1.61|381.81|
|Qbert|355.83|406.93|403.13|152.43|193,220.77|

These early means do **not** show a broad regression behind published DreamerV3
at approximately the same interaction count: Boxing/Pong/Breakout are similar,
Freeway is ahead and Qbert is somewhat behind, with large seed variation.
This is descriptive context, not equivalence or algorithmic superiority.
The long-run targets remain far away. Atari100k is a separate, nonsticky/
minimal-action400k-frame benchmark: our800k-frame recipe spends **more** online
experience and misses its published final means on four of the five games.

The [reference extraction](2026-10-05-dreamerv3-five-game-reference.json) includes
all available released traces, their varying seed counts and hashes, final-bin
scores and the predeclared last10% statistic. Do not select a convenient late
bin or compare frozen results as if they were the published online curve.
No matched same-hardware RGB run exists for this action-effects package, so
there is **no five-game compute-saving claim**. Smaller parameter count alone
does not answer that question.

## Retention and uncertainty

Seeds below are1009/2017/3019. Intervals bootstrap learner seeds, not the
24 episodes or eight streams within each model; three-seed intervals are coarse.

| Game | Frozen seed means | Paired improvement over initial [95% interval] | Frozen human-normalized mean |
| --- | --- | --- | ---: |
|Boxing|72.71 / 64.92 / 71.58|69.35 [64.71, 71.88]|5.8030|
|Pong|-20.83 / -14.08 / -20.67|1.78 [-0.71, 6.42]|0.0615|
|Freeway|22.21 / 23.21 / 27|24.14 [22.21, 27]|0.8155|
|Breakout|3.42 / 2.63 / 7.04|2.75 [1.13, 5.21]|0.0924|
|Qbert|312.5 / 235.42 / 661.46|250.69 [90.63, 491.67]|0.0180|

Human normalization is (score−random)/(human−random), using the pinned upstream
anchors, not a new mastery gate. All three Boxing, Freeway, Breakout and Qbert
models beat their paired initial controls. Pong1009/3019 are slightly worse;
2017 improves but still loses. Frozen score improvement is not game completion.
The original Freeway mastery gate passes only3019. Historical Tiny results
retain their older nonsticky/aided/pretrained protocol; they are not these CDP
learners.

## Failure diagnosis and next decision

There is real task feedback, not the former all-zero Freeway adapter/exploration
floor. The last50k-action diagnostic averages expose a useful distinction:

- Pong1009/3019 have raw KL about.57 and almost indistinguishable reward-head
  outputs for actual negative-reward versus zero-reward samples (roughly−.02).
  Pong2017 has KL2.64 and better negative-event separation (−.469 versus−.004).
- Breakout1009 has KL.46 and weak reward-event separation; the other two separate
  reward events more clearly but still have poor control.
- Qbert1009/2017 also have KL below1 and weak event separation. Qbert3019 has
  KL2.68, clearer reward-event separation and the highest return.
- Boxing separates positive/negative events well in all three seeds.
  Intrinsic rewards remain active, but their scalar means alone do not show
  useful exploration or prove that they cause these failures.

These are **posterior training diagnostics**, not held-out prior forecasts or
proof of feature collapse. Low latent-prediction loss does not establish that
the ball, hazards or task-relevant dynamics are represented. All updates and
checkpoints are finite; this does not establish representation sufficiency.
Event-prediction numbers are update-weighted averages of existing minibatch
reports, not pooled event-weighted estimates; empty event cohorts can dilute
them. Direct held-out event checks would be needed for a calibrated claim.

The released Atari57 Pong mean is also−20.50 at800k frames, then−7.16 at2M.
Before changing capacity, loss or exploration, the next bounded question is
whether the weak CDP seeds leave this early phase by2M frames. Declare three
fresh500k-action Pong runs with the exact same learner, no other factor changed,
and frozen final/initial-control audits. This is a reviewed budget experiment,
not an automatic extension or checkpoint resume. If prediction/control remains
weak, inspect task-state readability and prior forecasts before spending more.
All five games and the budget question stay in scope; no unchanged full-suite
queue or old representation matrix restarts.

## Recipe, time and evidence

The [declaration](../experiments/2026-10-05-cdp-five-game-budget.md) is unchanged:
Size1M/N8/B8/T16/H15/R32, microbatch8/replay100000, full18/sticky.25/repeat4,
no reset no-ops, native capture with one GPU resize toRGB64,100000-frame cutoff.
Joint CNN+CDP, cosine500, encoder6e-6/dynamics4e-4/base4e-5, AGC.3, warmup1000,
ac_grads=false, action-effects-disagreement coefficient1. No action aid, external
reward shaping or video pretraining. Including the ensemble, trainable counts
are940,384 world/exploration plus116,817 behavior parameters (1,057,201 total).
The JSON retains full configuration.

Nativef4b6a5c7, Meganeura592a2f5a and Bladee349cddf are unchanged. October5's
post-run upstream recheck still finds only docs/artifacts in Meganeura6288f885,
not an unadopted runtime fix. No native rebuild or duplicate qualification.

Fifteen runs, including retained Freeway, cost3M actions /749,085 updates /
11,992,159 actual emulator frames and **6h56m28s training time**. This is not
total project cost: prior failed screens, diagnostics, repetition of fresh
prefixes, reference training and pretraining remain additional disclosed work.
New twelve-run wall time above also includes evaluation, CPU replay/video and
audits; those are not all GPU-training seconds.

Mean update31.62ms: imagination16.19ms (51%), world training8.45ms (27%),
posterior4.33ms (14%), with remaining replay/behavior/synchronization work.
Eight environments use batched acting and one serialized learner; training is
about8x aggregate /1x per-stream realtime. GPU utilization is **unmeasured**.
Further wall-time optimization should target measured whole-agent cost,
especially imagination, not assume RGB reconstruction remains the bottleneck.

All45 guards, complete counter/debt checks and finite checkpoints pass.
Frozen evaluation adds789,232 actions,3,155,933 frames and **zero updates**.
The720 selected episodes are all natural, with zero cutoffs; all755 completed
episodes, excess episodes and unfinished tails remain retained. Exact346
saved tensors per frozen model, full trajectory replay, video hashes and
ffprobe frame/format checks pass. No recovery occurred.

## Whole rollout videos

These are unfiltered stream-zero deterministic replays, not best-episode clips.
Each pair is trained / actual initial control. The other seven streams remain
in the audited cohort. Original online pixel hashes were not recorded.

- Boxing: 1009 [trained](../../runs/cdp-five-game-screen-20261005.4e20N61X/boxing-cdp-1009-candidate-frozen.mp4) / [initial](../../runs/cdp-five-game-screen-20261005.4e20N61X/boxing-cdp-1009-untrained-frozen.mp4); 2017 [trained](../../runs/cdp-five-game-screen-20261005.4e20N61X/boxing-cdp-2017-candidate-frozen.mp4) / [initial](../../runs/cdp-five-game-screen-20261005.4e20N61X/boxing-cdp-2017-untrained-frozen.mp4); 3019 [trained](../../runs/cdp-five-game-screen-20261005.4e20N61X/boxing-cdp-3019-candidate-frozen.mp4) / [initial](../../runs/cdp-five-game-screen-20261005.4e20N61X/boxing-cdp-3019-untrained-frozen.mp4).
- Pong: 1009 [trained](../../runs/cdp-five-game-screen-20261005.4e20N61X/pong-cdp-1009-candidate-frozen.mp4) / [initial](../../runs/cdp-five-game-screen-20261005.4e20N61X/pong-cdp-1009-untrained-frozen.mp4); 2017 [trained](../../runs/cdp-five-game-screen-20261005.4e20N61X/pong-cdp-2017-candidate-frozen.mp4) / [initial](../../runs/cdp-five-game-screen-20261005.4e20N61X/pong-cdp-2017-untrained-frozen.mp4); 3019 [trained](../../runs/cdp-five-game-screen-20261005.4e20N61X/pong-cdp-3019-candidate-frozen.mp4) / [initial](../../runs/cdp-five-game-screen-20261005.4e20N61X/pong-cdp-3019-untrained-frozen.mp4).
- Freeway: 1009 [trained](../../runs/freeway-effects-200k-20261005.cK3FhKKq/evaluation/effects-1009-frozen.mp4) / [initial](../../runs/freeway-effects-200k-20261005.cK3FhKKq/evaluation/untrained-1009-frozen.mp4); 2017 [trained](../../runs/freeway-effects-200k-20261005.cK3FhKKq/evaluation/effects-2017-frozen.mp4) / [initial](../../runs/freeway-effects-200k-20261005.cK3FhKKq/evaluation/untrained-2017-frozen.mp4); 3019 [trained](../../runs/freeway-effects-200k-20261005.cK3FhKKq/evaluation/effects-3019-frozen.mp4) / [initial](../../runs/freeway-effects-200k-20261005.cK3FhKKq/evaluation/untrained-3019-frozen.mp4).
- Breakout: 1009 [trained](../../runs/cdp-five-game-screen-20261005.4e20N61X/breakout-cdp-1009-candidate-frozen.mp4) / [initial](../../runs/cdp-five-game-screen-20261005.4e20N61X/breakout-cdp-1009-untrained-frozen.mp4); 2017 [trained](../../runs/cdp-five-game-screen-20261005.4e20N61X/breakout-cdp-2017-candidate-frozen.mp4) / [initial](../../runs/cdp-five-game-screen-20261005.4e20N61X/breakout-cdp-2017-untrained-frozen.mp4); 3019 [trained](../../runs/cdp-five-game-screen-20261005.4e20N61X/breakout-cdp-3019-candidate-frozen.mp4) / [initial](../../runs/cdp-five-game-screen-20261005.4e20N61X/breakout-cdp-3019-untrained-frozen.mp4).
- Qbert: 1009 [trained](../../runs/cdp-five-game-screen-20261005.4e20N61X/qbert-cdp-1009-candidate-frozen.mp4) / [initial](../../runs/cdp-five-game-screen-20261005.4e20N61X/qbert-cdp-1009-untrained-frozen.mp4); 2017 [trained](../../runs/cdp-five-game-screen-20261005.4e20N61X/qbert-cdp-2017-candidate-frozen.mp4) / [initial](../../runs/cdp-five-game-screen-20261005.4e20N61X/qbert-cdp-2017-untrained-frozen.mp4); 3019 [trained](../../runs/cdp-five-game-screen-20261005.4e20N61X/qbert-cdp-3019-candidate-frozen.mp4) / [initial](../../runs/cdp-five-game-screen-20261005.4e20N61X/qbert-cdp-3019-untrained-frozen.mp4).

[Compact evidence, seed curves, diagnostics and reference values](2026-10-05-cdp-five-game-screen.json).
Full audited summaries and raw traces:
`runs/cdp-five-game-screen-20261005.4e20N61X`;
retained Freeway: `runs/freeway-effects-200k-20261005.cK3FhKKq`.
