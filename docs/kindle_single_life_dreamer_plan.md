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
| Breakout | Pretrained Tiny: mean10.9167 versus .9310 control. Own initial Tiny:12.4583 versus1.10. Large:30.7917 versus .9655. All trained evaluations have0/24 two-wall completions; zero frozen updates/cutoffs. | ≥20 completed episodes, ≥90% clear both walls / reach864 points. Fresh matched four/eighteen-action comparison is running on the adopted runtime; historical Large four-action arm stays held. No pretraining benefit demonstrated in one seed. | [Pretrained Tiny](../runs/levjepa-tiny-breakout-20260921.ghJPWG/evaluate.mp4), [its control](../runs/levjepa-tiny-breakout-20260921.ghJPWG/untrained.mp4), [pretraining ablation and videos](../runs/levjepa-tiny-pretraining-ablation-20260921.lrjxlN/results.md), [Large](../runs/breakout-action-pilot-20260920.kNeotb/results.md) |
| Qbert | Completed Tiny R64 seed0: 3.2M final22/27 first pyramids (81.5%), mean12,595.37; 1.6M midpoint24/24, mean8,673.96; control0/24, mean120.83. Complete state/replay/video checks pass. | ≥20 episodes, ≥90% first pyramids **and** mean ≥15,000. Final fails both thresholds; inspect early hazards and later progression. | [Final](../runs/qbert-r64-3m2-20260925.FrriIH/seed0/final.mp4), [midpoint](../runs/qbert-r64-3m2-20260925.FrriIH/seed0/midpoint.mp4), [control](../runs/qbert-r64-3m2-20260925.FrriIH/seed0/untrained.mp4), [report](../runs/qbert-r64-3m2-20260925.FrriIH/results.md) |

For Breakout/Qbert, a task completed before a later cutoff counts as achieved,
without relabeling that episode natural. Retain all episodes and partial tails.
Task observers are post-hoc evaluation, not privileged policy inputs or rewards.

## Immediate sequence

Boxing, Pong and Freeway satisfy **three of five** game gates. Qbert's completed
[3.2M R64 pair](../runs/qbert-r64-3m2-20260925.FrriIH/results.md) fails both final
thresholds: 22/27 first pyramids and mean12,595.37. More exposure raises score but
does not reliably complete even the first pyramid. Keep the predeclared final
primary; do not select the better-looking midpoint or relax the gate.

1. **Throughput qualification completed; adopt the minimal runtime change.**
   The [four-window Atari comparison](../runs/chunks-atari-timing-20260926.e2OEhn/results.md)
   cuts wall time **6.5–6.6% in both orders**, with <0.1% total repeat drift and
   exact same-arm complete state, moments, reports and trajectories. It retains
   N6/full18/12M/B16/T64/full BPTT/M16/R256 and the trained causal Tiny encoder.
   Native `b00ce7be` / Meganeura `ee3aea42` / Blade `fbb4f28c` is now the
   production package: exact source `30bfa1a`, one scheduling call plus dependency
   pins, no profiler or tuning API. Reuse the qualified package without rebuilding.
   [Pixel/state/restore](../runs/world-chunks-gameplay-20260926.q4iNnd/results.md)
   and [latest-runtime compatibility](../runs/chunks-compatibility-20260926.opVUiI/results.md)
   pass. Different backend arithmetic means old/new campaign roots stay separate.
2. **Repair failing game recipes with bounded comparisons.** The fresh
   [Breakout four/eighteen-action comparison](../runs/breakout-minimal-comparison-20260926.xsQCaK/README.md)
   keeps trained causal Tiny7fe9, native b00ce7be, seed0 and 200,004 actions /
   49,652 updates per arm fixed. Latest four-action gradients and
   [exact N6 repeat/restore/replay checks](../runs/breakout-width-latest-pixels-20260926.idNbRS/results.md)
   pass. Four-action training is running; review it before frozen evaluation
   and a restored untrained control, then complete the eighteen-action arm.
   Preserve the old hold and all original competence gates. Qbert's
   [retained episode analysis](../runs/qbert-tail-analysis-20260926.m0P7ZI/results.md)
   finds a16,850 median but five early first-pyramid failures and a later8–9k
   plateau. Inspect hazards and reward/continuation forecasts before another
   budget increase. Retain every episode and the original mean/success gates;
   change one scientific variable at a time.
3. **Confirm only a passing recipe.** Fresh roots1009/2017/3019 each need the
   fixed final-policy gate and a restored untrained control. Do not replicate
   unchanged failures just to occupy the device. Native games, transfer and
   swarms remain downstream of reliable single-actor results.

The runtime gain is real but modest: **1.0405–1.0415× aggregate / ~.174× per-stream
real time** at R256, with learning still dominant. A concurrent learner service
is not the next step. Imagination host work and GPU-resident parameter sync are
separate future optimizations; account for derived weights before aliasing.
The optimizer-only update's [negative speed result](../runs/meganeura-optimizer-timing-20260926.A6u3AI/results.md)
and the earlier refresh's [negative Qbert learning result](../runs/correctness-qbert-comparison-20260924.Q4NZyO/results.md)
remain; correctness and throughput do not imply gameplay competence.

All completed writers, failed-reader evidence, interrupted Freeway2017 attempt
and historical holds remain immutable. Checkpoints preserve weights/moments,
not replay/live belief/RNG. Review each GPU successor individually. Detailed
chronology belongs in [the experiment index](experiments/README.md).

## Complexity and compute

Dreamer's interacting networks and recurrent learner are real complexity;
hundreds of investigation commits and chronological reports are not architectural
requirements. Keep production code for exercised features and evidence in linked
reports. Tests are not cruft simply because they exceed implementation size.

The initial 303M frontend was disproportionate to the nominal 12M learner.
Tiny reduces that cost, but R256 still consumes about 102 million replay
positions per 400k-action run. Repeated learning must earn its cost through
sample-efficiency comparisons. Six environments already batch encoder/belief/
policy inference. Training uses B16×T64 replay states and 1,024 imagination
starts for 15 steps; serial emulator stepping is not the main bottleneck.

Completed fixed-recipe optimizations:

| Comparison | Measured result | Scope |
| --- | --- | --- |
| Small-batch block products | 27.3% higher throughput | Exact full-state/action parity; historical Large recipe |
| [Tiny versus Large](../runs/levjepa-tiny-throughput-20260921.cY1QjK/results.md) | 26.0–26.9% less total time; 73.6–74.8% less observation time | Exact same-arm repeats; pretrained packages, not a pure size ablation |
| [Correctness refresh](../runs/meganeura-correctness-timing-20260924.mYGvjj/results.md) | 7.5–7.8% less total time; .974–.975× aggregate real time | Fixed Tiny/R256; learning still 91.8% of wall time |
| [Optimizer-only update](../runs/meganeura-optimizer-timing-20260926.A6u3AI/results.md) | 0.8–0.9% slower | Negative speed result; compatibility passes |
| [Four world submissions + latest runtime](../runs/chunks-atari-timing-20260926.e2OEhn/results.md) | 6.5–6.6% less Atari wall time; ~1.041× aggregate real time | Exact same-arm state/reports/actions; isolated core scheduling gain 7.4–7.5% |

These are separate comparisons, not percentages to add. Uncapped, step-driven
playing/learning works; sustained per-stream super-real-time learning is not
established. Free-running native games without time control remain a separate
requirement. Measure arrival order, observation gaps, action durations and
training debt before introducing concurrency.

The completed Qbert R64 trial takes 18.799h at **3.151× aggregate / .525×
per-stream real time**: learning 74.19%, observation 22.77%, emulator 2.24%.
Mean update is 251.00ms: world 86.62, imagination 85.15, posterior 34.47,
behavior 22.15 and parameter synchronization 20.90. R64 changes the learning
schedule; it is not a parity speedup over R256.

The [full-learner trace](../runs/learner-timeline-20260926.6FJAqf/results.md)
measures a GPU-pass union of 156.50ms per 250.86ms update (62.4%), with 94.36ms
uncovered. This synthetic early-update probe excludes frontend/ALE/N6 live sync.
It is **not SM utilization**; readback waits include computation and pass gaps
are not automatically hardware idle. World command recording alone takes
26.09ms before GPU execution; optimizer passes take only .44ms. Imagination
host work and GPU→CPU→GPU parameter synchronization are separate follow-ups;
derived weights must remain coherent. **NVML stays disabled.**

Keep AGC/full recurrence unless an ablation supports changing them.
Reconstruction/future controls remain .25/0, .25/.25 and 0/.25. Qualify backend
fixes before long training; do not repeatedly replicate unchanged failed recipes.

### Right-size the causal encoder

[Checkpoint accounting](../runs/model-sizing-20260920.kPIOWC/README.md) finds
303,099,904 Large frontend parameters, 5,921,280 RSSM parameters and 10,281,233
optimizer-owned world+behavior parameters. Deterministic width 2048, hidden
width 256 and 32×16 categorical state match
[DreamerV3's 12M preset](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/dreamerv3/configs.yaml);
its name denotes the whole-agent preset, not the RSSM alone.

Keep **causal ViT-Tiny/16: 12 layers, width 192, three heads, MLP 768,
5,486,592 encoder parameters**. Preserve 224px inputs, 16-arrival block-causal
chunks, independent histories and JL64/2×2 pooling to 7×7×64. Logical F32 KV
storage is 55.125 MiB per stream versus Large's 588 MiB; these are tensor sizes,
not measured device peaks. Do not slice Large weights or substitute DINO/RGB.

The native pretrainer uses multi-view invariance + SIGReg, causal token dropping
and evaluation EMA, not undisclosed distillation. Preserve token positions in
masks/RoPE; patches cannot read the CLS sink or future frames. Full-gradient/
AdamW/EMA/restore and causal/N6 [numerical checks pass](../runs/levjepa-tiny-accuracy-20260921.qa3GqK/results.md).
The [first training run](../runs/levjepa-tiny-pretrain-20260921.JaPZpW/results.md)
completes 4,096 updates in 84.18 minutes on 250,000 observations / 999,112 emulator
frames, with whole-recording splits and verified checkpoints/exports. Declare
all offline experience and lineage in later comparisons.

The [frozen quality comparison](../runs/levjepa-tiny-quality-20260921.ojeZgt/results.md)
passes five noncollapse screens, but is mixed: Pong position R² .904→.938;
explicit-history motion R² .508→.333 with severe paddle outliers. Keep all
targets, RGB/constant controls and whole-seed splits; no test-driven retuning.
[Gameplay integration](../runs/levjepa-tiny-gameplay-pixels-20260921.NQLh0I/results.md)
and matched cost pass. Freeway now passes three learner roots, conditional on
one encoder. Breakout regresses versus Large, and its
[own-initial-encoder comparison](../runs/levjepa-tiny-pretraining-ablation-20260921.lrjxlN/results.md)
shows no pretraining benefit in one seed. Neither establishes a capacity limit
or justifies random features as the product. **Tiny stays opt-in; Large remains
default** pending broader downstream evidence.

Keep numerical/causal/streaming checks, held-out quality, N6 memory/time and
frozen downstream controls as adoption gates. Different pretraining corpora
compare packages, not size alone. Pretraining-only runtime changes do not
automatically update gameplay; use the same qualified runtime across each pair.

## World-model evaluation and pretraining

Predict before observing each target, then compare features, reward and
continuation with reality. Keep posterior estimates separate. Use persistence,
unrelated-action and zero-reward controls; report MAE **and** MSE, event counts,
visual-cache resets and episode boundaries. Sparse all-frame MAE/AUC is not
calibration; another policy's return is not an unbiased critic target.

The [one-step check](../runs/tiny-world-one-step-20260921.U7yHOa/results.md) and
[15-step report](../runs/tiny-world-horizon15-20260921.bKmiUF/results.md) replay
the first four complete Tiny Breakout matches: 1,126 actions and 16,470 correlated
forecast targets. Strict/forced replay, exact horizon-one overlap and complete
frozen state pass. Scalar restore explicitly changes collection metadata 6→1
and the action counter, not learning state.

At horizon15, feature MSE .0001860 beats persistence .0010262; reward MAE/MSE
.008938/.019244 beat zero .040187/.118692. Continuation is worse than
always-continue (.005106 versus .003716). There are only 22 positive rewards
and four terminals at each horizon. Recorded future actions condition forecasts:
this is not counterfactual validation or a cause of policy failure.
The older [own-policy](../runs/world-evaluation-20260908.Xzx3pN/report.html) and
[common-recording](../runs/common-world-report-20260909.O7nqqe/report.html)
reports likewise show limited cross-trajectory generalization.

Next diagnostics should target Qbert's early hazards/later plateau and Breakout's
failures without changing evaluation gates. Retain preselected first-four
complete stream-zero matches for new Pong diagnostics, without score filtering.
A continuation ablation is separate from throughput qualification.

Visual video pretraining works; a video-dataset **world-pretraining** workflow is
not adopted. Start with aligned RGB, executed actions/durations and boundaries
from mind-games. Missing actions/rewards are missing labels, not NOOP/zero.
Compare fresh, encoder-only and encoder+world initialization at equal target
budgets. Keep actor/critic unchanged in world-only updates; declare resets and
offline lineage. Useful pretraining means faster retained gameplay learning,
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
