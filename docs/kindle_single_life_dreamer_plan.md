# Kindle: CDP learning roadmap

Updated October25,2026. [Handover](handover.md) contains the reproducible runtime
and machine-transfer instructions. [PR31](https://github.com/kvark/kindle/pull/31)
is the status dashboard; individual result reports preserve the evidence.

**Current result:** [AMD takeover](results/2026-10-25-cdp-amd-takeover.md) completes
Breakout2M B2/T64 in11h40m. Frozen500k/final/initial scores are14.71/10.04/1.42,
all0/24 two-wall successes; accounting, frozen tensor equality and full replays
pass. The service has exited. BF16 is unavailable on RX7900XT/RADV; F16 inputs,
ordinary targets, hardware and sequence/batch changes confound comparison.
Paired early updates differ despite matching initial tensors/experience; see
[precision evidence](results/2026-10-25-cdp-amd-precision.md). Auto's4096-action
smoke and exact frozen restore pass at similar throughput. A fresh
[500k Auto comparison](experiments/2026-10-25-cdp-amd-auto.md) is now running in
`kindle-amd-auto500k-20261025`. Do not duplicate it or extend the weak F16 life.

**Compute priority, updated from the user's October25 direction:** finding useful
game-specific configurations matters more than maintaining one universal set of
hyperparameters. Measure frozen learning against GPU time and experience. A
working development recipe is useful evidence; reliable success across seeds is
still a separate requirement. The current declared run keeps its fixed budget.

## Goal and architecture

Learn to play games through experience, initially with sparse explicit rewards
and human guidance, eventually transferring skills between games. First make one
actor learn reliably. Video priors, native games, transfer and swarms follow.

The main path is Dreamer with **centered CDP**: a jointly learned small RGB CNN,
categorical recurrent state-space model, latent prediction instead of a pixel
reconstruction decoder, and imagined actor/critic learning. The action-effects
exploration ensemble currently has coefficient1. Native Rust/Meganeura/Blade
perform learning and inference; Python adapts games and analyzes results.
Frozen Large/Tiny JEPA and DINO are not the active frontend. Learned RGB with
reconstruction remains the control. CDP's five-game advantage over matched RGB
has not been established; the earlier Seaquest comparison is narrower evidence.

Use eight environments with batched inference and one shared learner, independent
stream histories, native max-pooled frames and one GPU64 resize. The current
Atari reference is12M/B8/T16/H15/BF16/replay-ratio32, full18-action space,
sticky actions.25 and action repeat4.500k actions are roughly2M emulator frames;
budgets count aggregate actions across streams, not per-stream actions.

## Verified position

| Game | Latest useful frozen result | What remains |
| --- | --- | --- |
| Boxing | Historical12M500k mean85.26;72/72 wins | Reference99.61; no current-backend replication |
| Pong | Three500k seeds mean15.71;72/72 wins | Reference20.45;2M on1009 only improves16.75→16.92 |
| Freeway | Fixed aided500k seeds32.67/30.67/32.96;mean32.10;72/72 crossing successes | Reference33.40; selected/reused development seeds |
| Breakout | Retained B8/T16:500k58.83→2M98.21; AMD B2/T64:14.71→10.04;initial1.42 | Reference381.81;zero two-wall successes; reliability open |
| Qbert | Historical2M mean7,367;62/72 first pyramids | Reference193,220.77; intermediate15k/90% also unmet |

These references are published long-run Dreamer scores, not equal-budget
controls. Five-game supremacy is **not achieved**. Report matched-frame and
wall-clock learning alongside any comparison to these endpoints.

Freeway's aid is5% persistent-uniform16-action blocks, disabled in evaluation.
Keep its cohort separate from unassisted31.83/30.58/0 (mean20.81,48/72 successes).
Do not hide seed3019's failed unassisted run or the intervention's selection on it.

Breakout warm-starting from2M to4M-total and8M-total yields80.17 and85.17,
below the98.21 source control. No unchanged extension follows. Greedy action
selection has inconsistent effects, so sampled evaluation remains standard.
Reward forecasts improve; trained state contains ball information, but longer
warm-start training does not improve15-step ball-x prediction. This supports a
temporal-learning experiment, not an immediate encoder rewrite.

Evidence and videos:

- [Pong](results/2026-10-21-cdp-pong-budget.md)
- [Freeway replication](results/2026-10-23-cdp-freeway-replication.md)
- [Breakout budget](results/2026-10-22-cdp-breakout-budget.md),
  [warm start](results/2026-10-23-cdp-breakout-warmstart.md),
  [forecasts](results/2026-10-22-cdp-breakout-forecasts.md),
  [state diagnosis](results/2026-10-23-cdp-breakout-state.md)
- [Qbert](results/2026-10-11-cdp-qbert-budget.md)
- [Boxing](results/2026-10-09-cdp-boxing-capacity.md)

## Current execution order — strategy reset

1. **Transfer the actual runtime and unfinished checkpoint.** Follow the handover;
   do not rebuild the older PR source and assume it matches recent training.
   Backend optimization and upstream porting are parked. Native/JAX parity is
   not a prerequisite for learning.
2. **Qualify initialization and frozen restore on the target machine.** Original
   host PMU faults block GPU work there. A new machine needs its own bounded
   check, not the old host's boot/device pins. Preserve the867-update smoke
   checkpoint; compare all346 tensors before/after frozen evaluation.
3. **Completed: the declared Breakout context experiment.** B2/T64 versus retained
   B8/T16 keeps128 sampled positions/update and ratio32; it changes temporal
   gradient length and independent-sequence count together. One fresh seed1009,
   2M actions, fixed500k/final/actual-initial evaluations of24 natural rounds.
   No checkpoint selection or automatic extension. See the
   [exact declaration](experiments/2026-10-23-cdp-breakout-context64.md).
4. **Complete the matched Auto learning comparison.** Paired early updates show
   a reward projection staying unchanged under F16 while Auto updates it; the
   raw derivative mechanism remains unmeasured. The fresh500k Auto life changes
   only precision relative to the completed AMD F16 control. Review fixed
   final/initial evaluations and curves before attributing the score gap to
   precision or context. No automatic extension, broad seed campaign,
   optimization sweep or JAX-parity project.
5. **Find cheaper recipes for the remaining gaps.** Use T16 as the development
   starting point, eight independent lanes and batched GPU inference. Try lower
   replay ratios and smaller capacity in bounded development runs; changing
   several settings together is allowed when seeking a working cheap recipe.
   Use the game-specific capacity evidence below instead of shrinking every
   game blindly. Confirm promising configurations across seeds before claiming
   reliability. Keep useful gains separate from published endpoints.

The original B2/T64 smoke completed4096 actions/867 updates in45.245s with finite
346 tensors and exact initial tensors. Normal restores faulted before run_start;
one traced restore passed. AMD restore now passes with all346 tensors unchanged
under the explicitly adapted precision. The fresh2M life and fixed evaluations
are complete. Hardware, precision and execution-path differences remain
comparison limitations.

[Incident and diagnostic boundaries](results/2026-10-23-cdp-context64-gpu-fault.md).

## Compute-constrained development

The historical500k-action capacity cohort supports specializing by game:

| Game | Size1M frozen mean | Size12M frozen mean | Cost-conscious direction |
| --- | ---: | ---: | --- |
| Boxing |77.69 |85.26 | Prefer1M initially; both sizes won72/72 selected matches |
| Freeway |30.26 |27.54 | Prefer1M initially; preserve separate reporting for action aids |
| Pong |4.54 |12.40 | Test intermediate capacity rather than assuming1M is sufficient |
| Breakout |5.14 |38.21 | Small capacity was weak; test an intermediate preset before a long run |
| Qbert |993.75 |2657.99 | Small capacity was weak; prioritize learning per GPU-hour |

[Archived capacity data](https://github.com/kvark/kindle/blob/archive/phase2-before-cleanup-20261025/docs/results/2026-10-09-cdp-five-game-capacity.json),
blob`b600def5ce8bb55894eacf64e8a0b1c043251ad3`. These are historical development
cohorts, with some runtime differences; the roughly3.56x aggregate wall-cost
ratio is not an AMD speed prediction. They are not the later qualified-backend
results in the main status table. The current native presets jump from1M to12M;
an intermediate preset is a proposed change, not an implemented option.

The first cheap development configuration should use existing controls: centered
CDP, Auto precision, N8/B8/T16/H15, microbatch8, native frames/one GPU64 resize,
and replay ratio16 as a candidate instead of32. Choose capacity by the game;
1M is a justified starting point for Boxing/Freeway. Halving the ratio halves
scheduled replay-position work per action, not necessarily end-to-end time or
the experience needed to learn. Target100–200k actions and a30-minute training
cap for the first learning screen, with fixed frozen/actual-initial evaluation;
record a wall-cap cutoff explicitly. A short weak result does not prove that
the configuration cannot learn. No broad queue or automatic extension follows.

Image resolution is currently fixed at64 in native preprocessing, replay shapes
and CNN geometry. Test lower resolutions through that native path when it is
implemented; do not simulate them with CPU downscale/upscale.48x48 has56.25%
of the pixels and32x32 has25%, but RSSM and imagined actor/critic work do not
shrink proportionally. Check small-object visibility and actual learning before
using lower resolution, especially in ball games. Smaller capacity, shorter
sequences and fewer updates can be tested without that code change first.

## Runtime and performance

Retained NVIDIA Freeway B8 runs take about38 minutes per500k actions; Breakout2M
takes about2.5 hours. The AMD B2/T64 F16 run takes11h40m at47.59 actions/s, with
97.7% of wall time in learning; sampled device-wide busy is typically93–94%.
The failed B32 Freeway control ran faster (28m14s) but learned
worse. Batch size changes optimizer cadence at fixed replay ratio; synthetic
throughput alone is not the objective. AMD busy samples do not establish kernel
efficiency; the historical learning runs lack corresponding utilization data.
Do not recycle old65% utilization or0.51× realtime as current measurements.

The qualified c8aeba28 stack includes reduced-precision kernels, resident targets,
GPU sampling and convolution reuse. Its last retained RGB synthetic comparison
was1.525× JAX cost, with scope in the performance reports. No further kernel
sweeps or upstream porting are scheduled. Preserve the source; rebuild and check
on the new machine. Identical source does not guarantee identical binary bytes,
GPU behavior or speed across devices and drivers.

Long-term pipeline: GPU game → GPU capture → preprocessing → model → action,
with only actions read by CPU. Atari emulation currently remains on CPU. Playing
and updating can run faster than real time; no real-time throttle is required.
“Online learning” describes the coupled acting/training workflow sufficiently.

## After Atari

- **Video priors:** test whether video pretraining improves dynamics and learning
  at equal total experience/compute. Record data provenance and same-game exposure;
  video-pretraining support is not established by the Atari results.
- **Persistent/native games:** use `/x/Code/mind-games`, beginning with vkQuake2
  or TMNF, then suitable GOG/Wine games. Add sparse reward adapters and report
  end-to-end latency, reliable capture and real gameplay.
- **Transfer and retention:** train across related games, then measure adaptation
  to an unseen game, forgetting and dependence on per-game reward engineering.
- **Swarms:** only after strong single-actor native-game results and transfer.
  Experience sharing/concurrent learners are not prerequisites for this phase.

## Evidence and process

Develop cheaply, confirm promising changes across seeds. Keep actual-initial
controls, frozen evaluations with zero updates, full action accounting and
rollout videos. Disclose aids, truncations, selected seeds, warm starts and new
hardware. Checkpoint restore does not preserve replay/RNG/games/live belief.
Keep compact results in Git, large artifacts on shared storage, and only useful
checkpoint milestones. Do not build another status ledger into AGENTS.md.

[Strategy rationale](strategy_reset_plan.md), [historical reports](experiments/README.md)
and [performance history](backend_training_performance_plan.md) remain available.
Historical “next” paragraphs describe past states and do not authorize queues.
The previous full roadmap is preserved in Git at8d464e0.
