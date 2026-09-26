# Kindle: a single actor that learns while acting

This is the authoritative plan. [Current evidence and archive](experiments/README.md)
retain experiments and failures; [AGENTS.md](../AGENTS.md) gives working rules.
Keep the runtime small, comparisons controlled and results reproducible.
For a quick overview of done/in-progress/next work and why progress is costly,
start with the [PR status dashboard](https://github.com/kvark/kindle/pull/29).

## What exists

Native Rust/Meganeura/Blade implements a categorical Dreamer RSSM, sequence replay,
imagined actor/critic training and causal **LeVJEPA** perception. The frontend is
frozen during gameplay, not end-to-end JEPA training and not DINO. DINO remains
a historical control, not an automatic fallback. The new experiments use a
separately pretrained **5.49M causal ViT-Tiny/16**. Native numerical,
optimizer/restore, streaming, fit and noncollapse checks pass; the first bounded
4,096-update pretraining run completes with verified state and encoder exports.
Frozen probes retain useful position features but mixed motion results. Bounded
actor integration and matched-order throughput pass. Tiny Freeway now passes
all three fresh learner roots, but Breakout regresses and broader Tiny reliability
is unproven. Tiny stays opt-in; the 303M frontend remains the default.

```text
previous belief + executed action -> deterministic prior -> predicted features
                                             |
RGB history through now -> frozen encoder -> posterior
                                             |
                              reward / continuation / imagination
                                             |
                                       actor + critic
```

The predictor sees the prior, never the posterior containing its target. LeVJEPA
uses causal prefixes of 16-arrival chunks, projected to 7×7×64 features. Chunk
boundaries reset perception only; episode boundaries also reset belief. Six
streams share batched inference and one learner while retaining separate visual
caches, recurrent state, RNG and causal replay histories.

Core code: [agent](../kindle/src/dreamer/agent.rs),
[vector collection](../kindle/src/dreamer/agent/vector.rs),
[world model](../kindle/src/dreamer/world.rs),
[networks](../kindle/src/dreamer/networks.rs),
[behavior](../kindle/src/dreamer/behavior.rs),
[replay](../kindle/src/dreamer/replay.rs),
[LeVJEPA](../kindle/src/vision/levjepa.rs).

Use **adaptive execution** as shorthand; the established category is online RL.
Keep observing, acting and scheduled learning explicit. Frozen evaluation must
never update weights. No concurrent learner service is needed for this phase.

## Current game status

The five-game goal requires each of fresh model roots **1009/2017/3019** to pass
its final-policy gate and beat a separately restored untrained control. Videos
are whole stream-zero evaluations, including unfinished tails, not selected wins.
Full multi-stream evaluations determine results.

| Game | Measured result | Unchanged gate / next decision | Rollout |
| --- | --- | --- | --- |
| Boxing | Three roots pass: 123/123, 207/207, 51/51 wins; means +83.87/+90.58/+83.53; controls near zero | ≥20 natural matches, ≥90% wins, mean ≥+50, no cutoffs. Complete. | [1009](../runs/boxing-confirmation-20260910.hTEDcu/seed1009-evaluation.mp4), [2017](../runs/boxing-confirmation-20260910.hTEDcu/seed2017-evaluation.mp4), [3019](../runs/boxing-confirmation-20260910.hTEDcu/seed3019-evaluation.mp4) |
| Pong | Three fresh roots 2017/3019/1009 pass: 24/24, 23/24, 24/24 frozen wins; means +20.5417/+17.4583/+20.0833. Controls 0/76 combined; zero updates/cutoffs. | ≥20 natural matches, ≥90% wins, mean ≥+15, no cutoffs. Complete on the fixed recipe; cross-root state/replay/video audit passes. | [2017](../runs/pong-block-confirmation-20260916.rBwdGF/seed2017-evaluation.mp4), [3019](../runs/pong-block-confirmation-20260916.rBwdGF/seed3019-evaluation.mp4), [1009](../runs/pong-block-confirmation-20260916.rBwdGF/seed1009-evaluation.mp4), [controls](experiments/README.md#current-pong-confirmation) |
| Freeway | Three fresh Tiny roots1009/2017/3019 pass: final36/36 each, means32.9167/31.6944/33.25, versus controls0/108 combined, mean0. Complete pairs, cross-root state, replays and videos pass; zero frozen updates/cutoffs. | ≥20 natural rounds, ≥90% reach 25 crossings, mean ≥25, no cutoffs. Complete on the fixed Tiny recipe, conditional on one pretrained encoder. | [1009](../runs/tiny-freeway-confirmation-20260922.tij9QW/seed1009/final.mp4), [2017](../runs/tiny-freeway-seed2017-replacement-20260922.12z27y72/seed2017/final.mp4), [3019](../runs/tiny-freeway-confirmation-20260922.tij9QW/seed3019/final.mp4), [controls and complete report](../runs/tiny-freeway-confirmation-20260922.tij9QW/results.md) |
| Breakout | Pretrained Tiny: mean10.9167 versus .9310 control. Own initial Tiny:12.4583 versus1.10. Large:30.7917 versus .9655. All trained evaluations have0/24 two-wall completions; zero frozen updates/cutoffs. | ≥20 completed episodes, ≥90% clear both walls / reach864 points. Inspect forecasts and improve the small-encoder recipe; no pretraining benefit demonstrated in one seed. Four-action arm remains held. | [Pretrained Tiny](../runs/levjepa-tiny-breakout-20260921.ghJPWG/evaluate.mp4), [its control](../runs/levjepa-tiny-breakout-20260921.ghJPWG/untrained.mp4), [pretraining ablation and videos](../runs/levjepa-tiny-pretraining-ablation-20260921.lrjxlN/results.md), [Large](../runs/breakout-action-pilot-20260920.kNeotb/results.md) |
| Qbert | Completed Tiny R64 seed0: 3.2M final22/27 first pyramids (81.5%), mean12,595.37; 1.6M midpoint24/24, mean8,673.96; control0/24, mean120.83. Complete state/replay/video checks pass. | ≥20 episodes, ≥90% first pyramids **and** mean ≥15,000. Final fails both thresholds; throughput work precedes another learning trial. | [Final](../runs/qbert-r64-3m2-20260925.FrriIH/seed0/final.mp4), [midpoint](../runs/qbert-r64-3m2-20260925.FrriIH/seed0/midpoint.mp4), [control](../runs/qbert-r64-3m2-20260925.FrriIH/seed0/untrained.mp4), [report](../runs/qbert-r64-3m2-20260925.FrriIH/results.md) |

For Breakout/Qbert, a task completed before a later cutoff counts as achieved,
without relabeling that episode natural. Retain all episodes and partial tails.
Task observers are post-hoc evaluation, not privileged policy inputs or rewards.

## Immediate sequence

Boxing, Pong and Freeway satisfy **three of five** game gates. Qbert's completed
[3.2M R64 pair](../runs/qbert-r64-3m2-20260925.FrriIH/results.md) fails both final
thresholds: 22/27 first pyramids (81.5%) and mean12,595.37, versus0/24 and120.83
untrained. Its 1.6M midpoint reaches24/24 and8,673.96. More exposure raises mean
score but does not reliably complete even the first pyramid. Keep the predeclared
final primary; do not select the better-looking checkpoint or relax the gate.

1. **Throughput before another long learner root.** Qualify current Meganeura
   0dbfcc00's optimizer arenas/batched dispatches, separately from the adopted
   9746c9ac control. The [isolated preparation](../runs/meganeura-optimizer-20260926.TfWlJI/README.md)
   keeps Bladefbb4f28c, causal Tiny7fe9b252 and the full learning recipe fixed.
   Require independent optimizer references, production world/behavior gradients,
   Tiny/Large causal/N6 checks, complete state/moments/restore and short pixel
   integration before matched old/new/new/old timing. CPU builds are preparation,
   not GPU qualification. Do not silently change completed campaigns.
2. **Measure the complete learner, then remove measured waste.** Use calibrated
   Vulkan timestamps with optimizers and transfers included, plus host-stage
   spans. Do not report old optimizer-free profiles or blocked-readback time as
   GPU utilization. Separately test GPU-resident parameter synchronization after
   the upstream-only comparison. Retain full state, gradients, action traces and
   untraced matched-order measurements; no concurrent learner service.
3. **Repair the failing game recipes with bounded comparisons.** Keep trained
   causal Tiny fixed initially. Revisit Breakout's minimal action space with a
   fresh declaration, preserving the old four-action hold. For Qbert, inspect
   failure trajectories and reward/continuation forecasts before simply doubling
   the budget again. Compare one change at a time against its qualified control.
   Lower replay ratio is a learning tradeoff, not a parity speedup.
4. **Confirm only a passing recipe.** Fresh roots1009/2017/3019 each need the
   fixed final-policy gate and a restored untrained control. Do not replicate an
   unchanged failure merely to occupy the device. Native games, transfer and
   swarms remain downstream of reliable single-actor results.

Current production is nativea761ee5c / Meganeura9746c9ac / Bladefbb4f28c.
The [completed correctness refresh](../runs/meganeura-correctness-refresh-20260924.Be6kq9/README.md)
passes independent references, full gradients/state/restore and default Large
compatibility. [Matched timing](../runs/meganeura-correctness-timing-20260924.mYGvjj/results.md)
cuts total time7.5–7.8%; the [paired Qbert control](../runs/correctness-qbert-comparison-20260924.Q4NZyO/results.md)
does not improve learning in its one seed. Preserve both findings.

All completed writers, failed-reader evidence, interrupted Freeway2017 attempt
and historical holds remain immutable. Details and checkpoint chronology belong
in [the experiment index](experiments/README.md), not the active decision list.
Checkpoints preserve weights/moments, not replay/live belief/RNG; a fresh run is
required for an equivalent uninterrupted history. Review every GPU successor.

## Complexity and compute: what to change

Dreamer needs several interacting networks and a recurrent training loop, but
286 commits and 13K lines of experiment reports are not architectural necessities.
Keep investigations in the archive, production code for exercised features, and
compact evidence for decisions. Test coverage is not cruft merely because it
is larger than the implementation it protects.

This implementation also tests a costly departure from vanilla Dreamer: a frozen
303M video encoder beside the nominal 12M learner. Selected ratio 256 consumes about
102 million replay positions during a 400k-action run. Repeated learning and
frontend cost must earn their place through sample-efficiency comparisons.

Qualified small-batch block products deliver **27.3% higher throughput** with
exact full-state/action parity. Root 2017's 400,008-action training takes **10.94h**,
10.155 actions/s, **.677× aggregate real time** (about .113× per stream).
Wall time is 65.6% learning, 33.7% observing and .45% emulator stepping. These
stage wall times are not GPU utilization or calibrated idle intervals.

The new matched Tiny/Large AB/BA comparison reduces total time **26.0–26.9%**
and observation time **73.6–74.8%**, with exact same-arm state/trajectory repeats.
Tiny reaches only **.897–.902× aggregate real time** (~.150× per stream) at R256;
learning now accounts for about 87% of wall time. Mean learner step .257s includes
.092s world training and .086s imagination. More environment workers do not
directly remove this cost. [All four windows and stage clocks](../runs/levjepa-tiny-throughput-20260921.cY1QjK/results.md).

The latest-backend [matched comparison](../runs/meganeura-correctness-timing-20260924.mYGvjj/results.md)
further reduces total time7.5–7.8% at the same recipe. Aggregate realtime reaches
.974–.975x;learning still occupies91.8% of wall time,improving only2.6–2.9%.
Observation time falls42.8–45.6%. These are stage clocks,not utilization.
Keep recipe and pretrained Tiny fixed for the first backend learning comparison.
A matched replay-ratio ablation is separate,not a parity optimization.
Historical R64 Boxing exceeds aggregate real time, but has only one successful
root and less score margin; it is not an adopted replacement. Retain AGC/full
recurrence unless an ablation supports changing them. Reconstruction/future
controls remain .25/0, .25/.25 and 0/.25. Backend fixes need numerical checks,
not weeks of blind training. Recheck upstream before diagnosing old bugs.

Uncapped step-driven playing/learning is supported; current R256 training is not
super-real-time. Free-running native gameplay without time control is required
but not validated by Atari. Measure arrival order, observation gaps, executed
action durations and training debt before introducing concurrency.

The completed3.2M-action R64 trial takes18.799h: **3.151× aggregate / .525×
per-stream realtime**. Wall time is74.19% learning,22.77% observation and2.24%
environment stepping. Six environments already use batched encoder/belief/policy
inference; training batches16×64 replay states and imagines1024 starts for15 steps.
Serial emulator stepping is not the dominant cost.

Across all199,917 updates, mean learner time is251.00ms: world training86.62ms,
imagination85.15ms, posterior34.47ms, behavior22.15ms and parameter sync20.90ms.
Current synchronization reads parameters to CPU and writes inference copies back
to GPU. Posterior/imagination make95 blocking readback batches per update.
Neither those waits nor these wall clocks measure GPU idle time or SM occupancy.
Full-learner Vulkan traces are the next measurement; **NVML stays disabled**.

### Right-size the causal encoder

The [checkpoint accounting](../runs/model-sizing-20260920.kPIOWC/README.md)
finds **303,099,904** frontend parameters, **5,921,280** RSSM dynamics/prior/
posterior parameters and **10,281,233** optimizer-owned world+behavior parameters.
Our deterministic width 2048, hidden width 256 and 32×16 categorical state match
[DreamerV3's 12M preset](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/dreamerv3/configs.yaml).
Its size labels refer to whole-agent presets, not RSSM counts. The disproportionate
component is the frontend, not an undersized accidental RSSM configuration.

Use **ViT-Tiny/16: 12 layers, width 192, three heads, MLP 768**, totaling
**5,486,592** encoder parameters. Preserve 224px inputs, 16-frame block-causal
chunks, independent stream histories and JL64/2×2 pooling to 7×7×64. Logical F32
KV storage falls from 588 to 55.125 MiB per stream (330.75 MiB for N6); these are
tensor sizes, not measured device peaks. Keep RSSM capacity and R256 fixed first.
Do not slice Large weights, substitute DINO/RGB or claim proportional speedup.

Tiny requires its own self-supervised video pretraining. Use the original
LeVJEPA multi-view invariance + SIGReg objective with causal token dropping and
an evaluation EMA, not an undisclosed teacher-distillation substitute. Declare
the corpus, train/held-out split, frame sampling, views, step budget, projector,
normalization and exported weight identity before training. Preserve original
token positions in masks/RoPE; patches cannot read the CLS sink or future frames.
Train on Meganeura/Blade; Python may prepare video batches and reference checks.

The [upstream recipe](https://github.com/MLO-lab/LeVJEPA) demonstrates Tiny
pretraining on unlabeled video; the inspected official released checkpoint is
Large. Native Tiny inference support passes CI. A separate
[native pretraining graph](../runs/levjepa-tiny-cpu-20260920.e4QkQH/README.md)
and native AdamW/GPU-EMA/save/restore/export paths pass 100 CPU tests. The
[independent reference and allocation preflight](../runs/levjepa-tiny-reference-20260920.XhB8T8/README.md)
cover every gradient and identify a 25.3→9.4 GiB B128 plan reduction from native
erf GELU. These are prepared numerical expectations and static allocations,
not GPU correctness, runtime fit, throughput or trained features. The native
Python interface and deterministic video adapter are staged. A separately
declared [fresh random-policy corpus](../runs/levjepa-tiny-atari-corpus-20260920.oTjdon/result.md)
contains 250,000 RGB64 observations / 999,112 emulator frames from the five games,
with whole-recording training/validation splits and no clips crossing resets.
The actual B128/V4 pilot measures **.45s native training + .80s batch preparation**
per step: data preparation is the measured bottleneck in this serial pipeline.
These are stage wall times, not GPU utilization. This is additional offline
experience, not reused evaluation
footage or an untrained gameplay control. Latest upstream RMSNorm fusion fixes
are included in staging. The [seven native GPU checks](../runs/levjepa-tiny-accuracy-20260921.qa3GqK/results.md)
now pass, including all 155 gradients and bitwise exact 763-tensor continuation.
Preserve the original device-selection/strict-trig failures and documented
primitive-bound revision; full-model and gameplay gates are unchanged. A
[32-update B128/V4 fit/timing pilot](../runs/levjepa-tiny-fit-20260921.oDmZVn/results.md)
and [frozen held-out noncollapse screen](../runs/levjepa-tiny-feature-check-20260921.wSQUJj/results.md)
complete. The first [fresh 4,096-update candidate](../runs/levjepa-tiny-pretrain-20260921.JaPZpW/results.md)
completes in 84.18 minutes: seed 743, B128/V4, peak LR 1e-4/warmup 128/cosine,
weight decay .04, EMA .99. All nine complete checkpoints and both encoder exports
verify. The separately declared [frozen quality comparison](../runs/levjepa-tiny-quality-20260921.ojeZgt/results.md)
completes against its own untrained export, with RGB/constant controls. All five
noncollapse screens pass; pooled Pong position mean R² improves .904→.938.
Motion is mixed: explicit-history mean R² falls .508→.333, with severe paddle
readout outliers despite lower median error. Preserve every target and the fixed
test split; do not retune this result away. These probes justify a bounded
downstream test, not adoption or a general JEPA advantage.
Gate adoption on numerical/causal and streaming/reset checks, held-out
feature variance/quality, measured N6 memory/time and frozen downstream learning
against its untrained control. Report all offline experience. If Tiny and Large
use different corpora, compare pretrained packages, not a pure size ablation.
Pretraining-only backend extensions do not automatically change the gameplay
runtime: Tiny exports use the existing frozen encoder operations. Prefer the
qualified gameplay backend for the first size comparison; if a newer runtime
is needed, qualify it separately and use it for both Large and Tiny controls.
The [prepared gameplay package](../runs/levjepa-tiny-gameplay-cpu-20260921.YF4BKS/README.md)
adds explicit Tiny selection and checkpoint identity on that unchanged backend.
All 754 Python and 93 Rust CPU checks pass. Its own
[dense-reference and exact N6/serial inference checks](../runs/levjepa-tiny-gameplay-gpu-20260921.X6TnAI/results.md)
now pass. [Bounded N6 pixel/save/restore](../runs/levjepa-tiny-gameplay-pixels-20260921.NQLh0I/results.md)
also passes: 3,840 actions / 611 updates per arm, exact frozen state and complete
replays; the new Large path matches its retained state and trajectories exactly.
Tiny selection is opt-in; Large remains the default until downstream evidence
supports replacement. Initial short-loop times are 194s Tiny versus 289s Large,
with observation cost 33s versus 125s and similar learner cost. These include
replay warmup and are not matched-order or steady-state throughput qualification.
The [post-warmup AB/BA check](../runs/levjepa-tiny-throughput-20260921.cY1QjK/results.md)
now passes all cost and exact repeatability gates. Its terminal writers stay
unchanged. The full-budget Breakout pair completes, but Tiny's frozen mean
10.9167 regresses against Large's 30.7917. Keep Large as the default. A separate
same-architecture [encoder-initialization control](../runs/levjepa-tiny-pretraining-ablation-20260921.lrjxlN/results.md)
now completes: frozen mean12.4583 versus1.10 control, two-wall successes0/24
versus0/30. It exceeds pretrained Tiny by1.5417 points in this single paired
seed; no downstream pretraining benefit is demonstrated. This does not prove
a capacity limit, a reliable negative effect, or justify replacing the trained
causal-video JEPA target with random features.

## World-model evaluation and pretraining

Predict before each target observation in a recorded frozen match, then compare
features, reward and continuation with reality. Keep posterior estimates separate.
Use persistence, unrelated-action and zero-reward controls; report MAE **and** MSE,
positive/negative/terminal counts, and visual-cache/reset strata. Sparse all-frame
MAE or AUC alone cannot establish calibration. Another policy's logged return
is not an unbiased critic target.

Existing [own-policy forecasts](../runs/world-evaluation-20260908.Xzx3pN/report.html)
and [common-recording report](../runs/common-world-report-20260909.O7nqqe/report.html)
show action sensitivity but limited cross-trajectory generalization. This is not
a proven explanation for failed policies. For new Pong roots, preselect the first
four complete stream-zero matches without score filtering. The staged multi-match
probe needs current-source GPU forecast checks; CPU extraction alone is not that.
The [current-package readiness check](../runs/current-world-probe-cpu-20260921.KlMyMe/README.md)
passes 74 CPU tests and selects the first four Tiny trained/control matches.
Its [run-local wrapper](../runs/current-world-probe-wrapper-cpu-20260921.r8GrmS/README.md)
adds per-stage Vulkan memory checks and complete frozen-state comparison, with
39 CPU tests. The separate [one-step GPU report](../runs/tiny-world-one-step-20260921.U7yHOa/results.md)
now completes strict and forced replay: all1,126 actions/four matches, common
forecasts and input/reset identities match exactly; complete frozen state passes.
Scalar restore explicitly changes only collection-stream metadata6->1 and the
executed-action counter. Feature MSE .0001665 beats persistence .0002863; reward
MAE .00892 beats zero .03819. Continuation MSE .00439 is worse than always-continue
.00353, with only four terminals and22 positive rewards. These are useful
one-step measurements, not a causal explanation of the score gap. The separate
[15-step/all-origin check](../runs/tiny-world-horizon15-20260921.bKmiUF/results.md)
also passes, with 16,470 forecast targets, exact one-step overlap and unchanged
complete state. At horizon15, feature MSE is .0001860 versus .0010262 persistence;
reward MAE/MSE are .008938/.019244 versus .040187/.118692 zero prediction.
Continuation MSE remains worse (.005106 versus .003716). There are only22
positive rewards and four terminal targets at each horizon, not independent
new events. Recorded future actions condition these prior forecasts; this is
not counterfactual or imagined-policy validation. Preserve both terminal checks.
Freeway confirmation now passes. Qbert/Breakout still need targeted diagnostics;
a continuation ablation remains separate from throughput qualification.

Pretrained visual weights are supported; a video-dataset world-pretraining
workflow is not adopted. Start with aligned RGB, executed actions/durations and
boundaries from mind-games. Missing actions/rewards are missing labels, not
NOOP/zero. Compare fresh, encoder-only and encoder+world initialization at equal
target budgets. Keep actor/critic unchanged in world-only updates; declare resets
and offline lineage. Useful pretraining means faster retained gameplay learning,
not just lower feature error. No perception expansion before calibration/coverage.

## Beyond five Atari games

Beating most of a predeclared Atari suite is an ambition, not a consequence of
using Dreamer. Our variant does not inherit published DreamerV3 scores. Extend
to Seaquest/Frostbite/Private Eye, then Atari-26 with explicit gates, budgets and
seed distributions. Independent per-game training tests algorithm breadth, not
one transferable policy. Keep the pinned local upstream control and disclose
representation/precision/protocol differences.

Use `/x/Code/mind-games` for launch, time control, capture and input; recheck its
current Kindle API. Prefer **vkQuake2** next, with vkQuake only an integration
reference; then **TMNF** and a small **GOG/Wine** panel. Implement a small RGB8 /
executed-action / reward / boundary adapter, not another training stack. Measure
kills/objectives, track finishes and game completion, not motion alone. Explicit
sparse rewards and documented guidance remain acceptable. Menus, startup scripts
and overrides must not masquerade as autonomous learning.

Reserve an unseen shooter before tuning a general FPS actor. Compare fresh
dynamics/policy, transferred dynamics with fresh policy, and transferred dynamics
**plus policy** at matched budgets. Declare action mappings and optimizer,
normalizer, replay and belief resets. Measure zero-shot play, fixed-budget
adaptation and source-game forgetting; a held-out map is not a held-out title.

Only after strong multi-seed results on at least three GOG titles across two
genres and held-out cross-title adaptation/retention should two independent
Kindles share immutable experience chunks. Natural deaths/respawns are allowed;
cloning/rewinding a live game for training is not. Swarms and shared optimizers
remain later work. Intrinsic reward stays behind its existing seam, off in
controls, with extrinsic-only comparisons before adoption.

## Execution and evidence

Use the GPU; **NVML is temporarily disabled**. Serialize bounded direct native
jobs under the [host-only guard](gpu_incident_response.md), with actual device
assertions and >=2 GiB sampled Vulkan budget headroom. No recovery operation,
blind retry or quarantined candidate reuse. The four old Xid incidents remain
unexplained. Passing driver 580/no-NVML jobs is not a causal fix or safety proof.

Preserve declarations, failures and artifacts. Do not grow a new framework for
each check or rebuild qualified binaries for unchanged code. CI must install
declared test dependencies in a clean environment. Checkpoints preserve weights/
moments, not replay/RNG/live belief; interrupted resume is not equivalent lifetime
continuation. Atomic complete-state recovery and bounded storage remain later work.
