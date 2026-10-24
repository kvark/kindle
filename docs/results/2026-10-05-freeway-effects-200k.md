# Freeway CDP: unassisted exploration and retained learning are unlocked

All three fresh200k-action learners now cross repeatedly without action aids,
reward shaping or pretraining. Frozen mean scores are22.21/23.21/27.00 versus
zero for all actual initial-weight controls. Every candidate evaluation episode
scores19–30 crossings:1,738 crossings across72 natural episodes. This resolves
the zero-reward exploration blocker, **not three-seed mastery or all of Phase3**.

| Seed | Training crossings, including tails | Final online mean (last50 episodes) | Frozen mean /24 natural episodes | Episodes with>=25 crossings | Whole trained / untrained rollout |
| --- | ---: | ---: | ---: | ---: | --- |
|1009|742|13.88|22.2083|3/24|[trained](../../runs/freeway-effects-200k-20261005.cK3FhKKq/evaluation/effects-1009-frozen.mp4) / [untrained](../../runs/freeway-effects-200k-20261005.cK3FhKKq/evaluation/untrained-1009-frozen.mp4)|
|2017|1,125|21.08|23.2083|3/24|[trained](../../runs/freeway-effects-200k-20261005.cK3FhKKq/evaluation/effects-2017-frozen.mp4) / [untrained](../../runs/freeway-effects-200k-20261005.cK3FhKKq/evaluation/untrained-2017-frozen.mp4)|
|3019|1,249|22.48|27.0000|24/24|[trained](../../runs/freeway-effects-200k-20261005.cK3FhKKq/evaluation/effects-3019-frozen.mp4) / [untrained](../../runs/freeway-effects-200k-20261005.cK3FhKKq/evaluation/untrained-3019-frozen.mp4)|

Frozen seed mean24.1389, seed-bootstrap95% interval[22.2083,27.0000]; paired
improvement over untrained is the same because all72 control episodes score0.
Using the [retained upstream anchors](2026-09-27-historical-scores.json)
(random0, human29.6), human-normalized mean is .8155 [.7503,.9122], with
seed ratios .7503/.7841/.9122. This is a descriptive scale, not a matched human
comparison or a mastery threshold.
Three learner seeds are the replicates; episodes/streams are not independent
models. Only3019 passes the unchanged historical gate: mean>=25, >=90% rounds
reaching25, >=20 natural rounds and no cutoffs. Neither threshold nor protocol
was relaxed. The aggregate30/72 rounds reaching25 does not establish that gate.

## What changed and what did not

Only the interaction budget increased from32,768 to200,000 actions per seed.
The [declaration](../experiments/2026-10-05-freeway-effects-200k.md) uses fresh
learners, not checkpoint resumes. Native `f4b6a5c7`, small CDP recipe and
action-effects bonus are unchanged: Size1M/N8/B8/T16/H15/R32, microbatch8,
replay100000, full18/sticky.25/repeat4/no reset no-ops, native frames with one
GPU RGB64 resize. CDP cosine500, encoder6e-6/dynamics4e-4/base4e-5, warmup1000,
AGC.3, ac_grads=false, intrinsic coefficient1. Full configuration is in the JSON.

The bonus removes each predictor's action-independent component before measuring
ensemble disagreement. CNN-target training stays detached; no game labels,
direction hints, new encoder or scripted action enter the learner. Earlier
[sampled/soft/visual-target failures](2026-10-05-freeway-embedding-learning.md),
[32k discovery](2026-10-05-freeway-action-effects-learning.md), and
[32k frozen controls](2026-10-05-freeway-action-effects-frozen.md) remain retained.
The shorter extrinsic controls are **not a matched200k ablation**. This result
establishes useful learning for this recipe, not its200k causal advantage over
extrinsic-only CDP or Atari-wide superiority.

## Learning curves and cost

Each point below is the mean of the eight natural rounds ending at that action
count, not the smoother last50 statistic above. Full round score/action/time
curves are committed in the JSON; complete reports remain in raw artifacts.

| Actual aggregate actions |1009|2017|3019|
| ---: | ---: | ---: | ---: |
|32,768|.250|.125|.000|
|65,536|.000|.000|.625|
|98,304|.000|3.750|10.625|
|131,072|1.500|21.750|21.250|
|163,840|23.125|22.750|22.875|
|196,608|22.875|21.875|26.875|

Reward discovery at28,504/32,416/6,240 actions does not immediately yield useful
play. The learning transition arrives around100–150k; the small screen alone
understated the eventual result. The two weaker seeds plateau near22, while3019
improves late. Do not select only3019 or relabel all three as mastered.

Training takes27m59s/27m19s/28m10s,83m28s total. Whole guarded training,
evaluation, replay/video and per-pair audits finish in1h28m49s, at05:39 UTC.
Mean update31.54ms;7.89–8.14x aggregate realtime / .986–1.017x per stream with
eight batched environments. GPU utilization remains unmeasured, not inferred
from these frame-clock ratios. No native rebuild or competing GPU work occurred.

## Evidence and decision

All600,000 training actions/149,817 updates,288 completed training episodes and
unfinished tails are retained. Frozen evaluation adds294,912 actions and zero
updates:144 natural episodes, zero cutoffs, first3 per stream/model. New held-out
base2,000,000,000 plus learner seed, stream offset1,000,003. All nine guards,
training counters/finite checkpoints, full frozen CPU trajectory replays and
exact346-tensor equality per frozen model pass. All six unfiltered stream0
videos pass hash/frame-count/ffprobe checks. Original online pixel hashes are
unavailable; videos are deterministic replays. Earlier diagnostics, failed
screens and repeated fresh prefixes remain additional compute, not hidden
inside this600k budget. Services are stopped; no recovery occurred.

The [strategy reset](../strategy_reset_plan.md#2-working-rules-for-this-plan)
uses learning curves, not old mastery gates, for development. The predeclared
[exploration criterion](../experiments/2026-10-05-freeway-action-effects.md)
requires repeated unassisted crossings and retained improvement over actual
untrained controls. All three seeds now demonstrate both, well beyond isolated
reward discovery. **Freeway's exploration blocker is resolved.** Full three-seed
mastery remains unproven, and Phase3 still needs the predeclared second-task
comparison (planned Venture) with fresh same-game extrinsic controls. Do not
extend Freeway solely to tune the historical gate or automatically start another
campaign. Generalization, video priors and native-game learning remain later work.

[Compact evidence](2026-10-05-freeway-effects-200k.json).
Artifacts: `runs/freeway-effects-200k-20261005.cK3FhKKq`.
