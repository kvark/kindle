# Fixed Tiny latents: predictable, not yet control-sufficient

[Configuration, curves, controls, uncertainty and audits](2026-10-03-fixed-latent-sufficiency.json)
· [Predeclared protocol](../experiments/2026-10-03-fixed-latent-sufficiency.md).

## Decision

**Tiny retains useful coarse state, and its frozen latents support prediction.**
A small GPU head beats persistence on all three trained encoders. However,
lower latent error does **not** reliably improve predicted player state or
reward events, and actual versus unrelated actions remain weakly distinguished.
This is conditional evidence for latent prediction, **not validation of the
online JEPA world model or an efficiency advantage over Dreamer RGB**.

The [direct-policy gradient connection](2026-10-03-policy-tiny-qualification.md)
works; the [online comparison](2026-10-03-policy-tiny-learning.md) finds no clear
benefit. This diagnostic rules out a blanket “the latents contain no useful
information” explanation. It does not rule out missing bullets, inadequate
control-relevant features, a poor RSSM bottleneck, target drift or loss scaling.
Keep learned RGB as the qualified 2D default and `actor_critic_gradient` off.
Do not restart a larger unchanged joint-Tiny campaign.

## What the frozen representation contains

Readouts use only the full7×7×64 production features, with RAM as diagnostic
targets—not actor inputs. Test trajectories are completely held out from fitting
and checkpoint selection. Each encoder sees the same12,288 test actions,
95 positive reward events and21 terminals; these are not independent events
across encoders.

| Trained encoder seed | Player x / y R² | Enemy-lane x R² range | Current reward-event AUC |
| --- | ---: | ---: | ---: |
| 1009 | .779 / .838 | .242–.658 | .705 |
| 2017 | .775 / .806 | −.078–.658 | .721 |
| 3019 | .797 / .854 | .005–.683 | .712 |

Player position is readily decoded; enemy detail is uneven. Reward ranking is
moderately informative, but event MSE is worse than a constant predictor.
These unconstrained regression outputs are **not calibrated probabilities**.
The labels contain no bullet coordinates, so this test cannot establish that
small, policy-critical objects survive the representation.

## Fixed-target forecasts

Predictors see previous/current latents, the proposed action sequence and encoder
phase. They predict a residual in all3,136 latent components, never future
images or labels. Persistence predicts a zero residual. Independent random
actions replace only the action input for the unrelated-action control.

Ratios below are predicted error divided by control error; lower is better,
and1 means no advantage. Intervals resample learner seeds and whole held-out
trajectories in a crossed bootstrap, preserving shared trajectories across
models. Only three seeds/three test trajectories are available.

| Horizon, actions | Raw latent MSE / persistence [95% CI] | Standardized MSE / persistence | Actual-action MSE / unrelated-action MSE [95% CI] |
| --- | ---: | ---: | ---: |
| 1 | .840 [.679,.927] | .771 | .9991 [.9973,1.0006] |
| 15 | .489 [.456,.530] | .562 | .9902 [.9756,1.0053] |

Per-encoder raw ratios are **.682/.921/.916** at h1 and
**.463/.519/.485** at h15. The gain is not solely a chunk-reset artifact:
within-chunk h1 ratios are .847/.926/.928. Chunk-crossing results and all
per-trajectory errors remain in the JSON. No horizon crosses an episode reset.

**The important failure:** decoding predicted features through the frozen
readout does not preserve all the apparent gain. At h1, both player coordinates
are worse than persistence for every encoder. At h15, player-y improves, but
player-x R² is only .468–.517 versus .589–.612 for persistence. Reward-event
forecast AUC is .492–.618 at h1 and .411–.455 at h15. These slices contain only
27/18 reward events and3/7 terminals. There is no established useful reward
forecast or action-discrimination advantage.

The current world-model prior was evaluated on a different held-out trajectory;
these head ratios must not be presented as an apples-to-apples RSSM improvement.
They establish an independently fitted predictor's capability, not a fixed agent.

## Fitting, cost and integrity

All three final direct-policy checkpoints are used without score selection.
Collection is32,768 random actions per encoder: native pixels, sticky.25,
repeat4, all18 actions, no reset noops, no learning or action/reward aid.
Whole-trajectory seeds: train9101/9109/9127, validation10103/10111,
test11113/11117/11131;4,096 actions each. Complete terminal targets and reset
arrivals are retained. Artificial recording cuts remain separate from real
game truncations. All action/reward/boundary/label traces match across encoders.

Nine auxiliary GPU heads complete:128hidden, batch64, Adam1e−3,
2,048 updates/head, fixed head seed20261003, training-only F64 normalization,
validation-only selection every128 updates. Selected future heads use512–1,280
updates; all state heads select2,048. Future-head training normalized MSE is
.298–.404 versus held-out .534–.813: fitting/generalization is not perfect.
One head seed does not measure initialization variability.

Collection takes **13m13s summed worker time**. All nine heads take **59.6s
including their guards**, adding no gameplay. This is not full-agent training
throughput: there is no encoder backward or RSSM/behavior learning. CPU build
preparation overlaps collection, so those times are not a matched speed result.

All696 world/behavior/slow-value tensors per checkpoint remain bitwise unchanged,
including saved optimizer tensors, with zero actor updates. All seven full-job
guards/seals and the corrected smoke guard pass; no new warning/fault, unfinished
child, NVML polling or host recovery. Minimum sampled collection budget headroom
is4.762GiB, **not physical free or peak VRAM**. Device utilization is unmeasured.

The widened3,136-output native head independently passes F64 forward/loss/raw
gradient checks; synthetic fitting reduces loss .008381→.000004121. The native
change only widens a diagnostic dimension limit; it does not change the learner.
Strict workspace/binding Clippy,1,043 Python tests, Rust CPU tests and
[CI268](https://github.com/kvark/kindle/actions/runs/37157862626) pass fordc07cc3.
Meganeura13b19d33 includes the qualified dV fix over latest main6268ea5;
Bladee349cddf, driver580.178.04. Backend main was rechecked22:05 and unchanged.

Retained failures: the first collector smoke stops after128 actions on a missing
actor recording boundary; the second collects1,024 actions then falsely rejects
different safetensors serialization hashes. An independent audit proves all
tensors unchanged. The corrected1,024-action smoke compares keys/shapes/dtypes/
bytes and passes. Thus **2,176 excluded smoke actions**, plus98,304 full-corpus
actions, are disclosed:100,480 diagnostic interactions in this follow-up. A
pre-GPU build PATH failure is also retained. None is silently discarded or
counted as fitting data. Raw evidence: `runs/fixed-tiny-latents-20261003.5uHqSE`.

## What to test next

The representation has a large static offset: coordinate-mean RMS1.84–2.03
versus mean coordinate standard deviation .092–.101. The online predictor fits
unstandardized absolute features; this head fits standardized residuals. That
is a concrete **conditioning hypothesis**, not a demonstrated cause: this head
also changes inputs, architecture, optimizer and the fixed-target regime.

Before more expensive joint gameplay, use this saved corpus for one matched
**frozen-encoder RSSM target-standardization ablation**, leaving recurrence,
other losses, data and update budget unchanged. Evaluate prior forecasts with
persistence, unrelated actions and decoded event/state controls—not aggregate
latent MSE alone. This isolates target conditioning without encoder drift or
more interaction collection. It is the proposed next test, **not queued here**.
The present bounded evaluation is complete; useful online JEPA dynamics remain
unconfirmed. No Phase3 campaign, asynchronous learner or swarm starts here.
