# Kindle: a single actor that learns while acting

This is the authoritative plan. [Current evidence and archive](experiments/README.md)
retain experiments and failures; [AGENTS.md](../AGENTS.md) gives working rules.
Keep the runtime small, comparisons controlled and results reproducible.
For a quick overview of done/in-progress/next work and why progress is costly,
start with the [PR status dashboard](https://github.com/kvark/kindle/pull/29).
Links into `runs/` are local workspace evidence, not publicly hosted artifacts.
The numerical summaries here are public; publish compact result data and selected
videos before relying on those links for external review.

## What exists

The central hypothesis is that learning to predict useful compact representations
is cheaper than reconstructing pixels, while retaining the information needed to
learn arbitrary games from sparse rewards. Frozen video pretraining is one part
of that design, not the entire bet. Test world-update cost, end-to-end cost and
learning curves against a matched Dreamer12M RGB control. A cheaper but less
useful world model does not establish an efficiency advantage.

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
| Breakout | Complete matched Tiny comparison: four actions mean10.9167 versus .875 control; eighteen mean11.625 versus .93103. Both trained arms0/24 two-wall completions. Historical Large mean30.7917 also fails; no demonstrated pretraining benefit in one Tiny seed. | ≥20 completed episodes, ≥90% clear both walls / reach864 points. Fewer actions did not repair this seed. Keep eighteen as reference; diagnose before another recipe. Historical Large four-action arm stays held. | [Complete comparison](../runs/breakout-minimal-comparison-20260926.xsQCaK/results.md), [four-action video](../runs/breakout-minimal-comparison-20260926.xsQCaK/a4/evaluate.mp4), [eighteen-action video](../runs/breakout-minimal-comparison-20260926.xsQCaK/a18/evaluate.mp4), [pretraining ablation](../runs/levjepa-tiny-pretraining-ablation-20260921.lrjxlN/results.md), [Large](../runs/breakout-action-pilot-20260920.kNeotb/results.md) |
| Qbert | Completed Tiny R64 seed0: 3.2M final22/27 first pyramids (81.5%), mean12,595.37; 1.6M midpoint24/24, mean8,673.96; control0/24, mean120.83. Complete state/replay/video checks pass. | ≥20 episodes, ≥90% first pyramids **and** mean ≥15,000. Final fails both thresholds; the first-episode probe misses its terminal and retains high values through a scoreless ending. | [Final](../runs/qbert-r64-3m2-20260925.FrriIH/seed0/final.mp4), [midpoint](../runs/qbert-r64-3m2-20260925.FrriIH/seed0/midpoint.mp4), [control](../runs/qbert-r64-3m2-20260925.FrriIH/seed0/untrained.mp4), [report](../runs/qbert-r64-3m2-20260925.FrriIH/results.md), [world/policy diagnostic](../runs/qbert-hazard-probe-cpu-v2-20260926.GnWOvb/results.md) |

For Breakout/Qbert, a task completed before a later cutoff counts as achieved,
without relabeling that episode natural. Retain all episodes and partial tails.
Task observers are post-hoc evaluation, not privileged policy inputs or rewards.

**Freeway training is exploration-assisted:** probability .5 selects a random
action held for64 agent actions; frozen evaluation is unassisted. Its pass is
not a demonstration of unaided sparse-reward exploration. Tiny also receives
same-title offline video, detailed below. Keep both qualifications visible.

These evaluations use non-sticky `published` Atari with no reset no-ops. A
September26 check reproduces identical observations/rewards/boundaries for the
same512 actions across environment seeds1009/2017/100000 in all five games.
Learner roots and sampled policies vary, but environment seeds do not establish
varied starts. Add separately declared sticky-action evaluation, equal first-N
per-stream summaries and learner-level uncertainty; preserve original cohorts
and gates. Frozen episode counts are not independent learner replicates.
The [new adapter's CPU checks](../runs/native-pixel-protocol-cpu-20260926.WNenAu/results.md)
pass836 tests: native RGB is the fresh vector default, RGB64 is explicit, restore
requires an input choice, and sticky .25 is opt-in. Pixel-detail retention and
real ALE replay pass. The separately declared [native integration](../runs/native-pixel-integration-20260926.z2mimo/results.md)
also passes training, frozen restore and sticky replay with unchanged complete
state during evaluation. Robust policy results are not established by these
plumbing checks. The table above is historical RGB64/non-sticky.

## Immediate sequence

The user-approved order is evaluation/reporting, synthetic encoder CI, a bounded
matched Dreamer control, GPU round-trip reduction, then individual perception
experiments. Keep the historical three-of-five gate results separate from
robustness under a changed protocol.

1. **Preserve visual input — implemented and integrated.** Fresh vector runs
   use native max-pooled RGB, then one aspect-preserving encoder resize. RGB64
   remains explicit, restore requires choosing its input format, and sticky
   .25 is opt-in. The [836-test adapter suite](../runs/native-pixel-protocol-cpu-20260926.WNenAu/results.md)
   and [N6 training/frozen/sticky integration](../runs/native-pixel-integration-20260926.z2mimo/results.md)
   pass with exact state/replays and unchanged native b00ce7be. This removes the
   64→224 bottleneck before encoding, not compression inside the encoder.
   Tiny was pretrained on RGB64; measure the changed input distribution before
   declaring learning improvement.
2. **Make evidence reproducible and test robustness.** Assistance, same-title
   pretraining and deterministic-start limitations are now disclosed. Add
   separately declared sticky evaluations of the historical final policies,
   retaining their RGB64 input to isolate environmental stochasticity. Report
   equal first-N episodes per stream and learner-root uncertainty, alongside all
   episodes/tails. Publish compact results and selected videos for external
   reviewers. New outcomes may invalidate broad reliability claims; preserve
   original cohorts and their gates.
3. **Synthetic encoder CI — implemented; remote execution pending.** The
   [full-Tiny fixture](../runs/encoder-ci-20260926.GTcKXz/results.md) passes local
   dense-reference, projection/pooling, chunk-wrap and batched reset/gap checks.
   CI generates deterministic untrained weights and an independent dense
   PyTorch reference; no local pretrained checkpoint is needed.
4. **Run a bounded pinned Dreamer12M control.** Match the game, action vocabulary,
   emulator version, actual interactions, replay settings, precision and N6
   collection. Keep RGB64 as the explicit historical comparison; native-input
   JEPA is a separate intervention. Report world/whole-agent cost, memory and
   learning, including offline pretraining cost. The control must run directly
   under the host guard with NVML disabled. Its old environment uses ALE0.9,
   versus Kindle's0.12.1, and upstream driver records include action-free resets:
   resolve those mismatches before claiming equal experience.
5. **Reduce measured learner overhead.** The adopted four-submission package
   cuts Atari wall time6.5–6.6%, but only reaches ~1.041× aggregate / .174×
   per-stream real time at R256. Next target posterior/imagination host
   readbacks and redundant parameter transfers. Preserve sampling semantics,
   derived weights, complete state/moments and actions where claiming parity;
   use untraced matched-order timing. No concurrent learner service.
6. **Then change perception or learning, one variable at a time.** Neither the
   [Breakout action-width pair](../runs/breakout-minimal-comparison-20260926.xsQCaK/results.md)
   nor [Qbert's final R64 policy](../runs/qbert-r64-3m2-20260925.FrriIH/results.md)
   passes its gate. Do not confirm either unchanged failed recipe. The Qbert
   [forecast probe](../runs/qbert-hazard-probe-cpu-v2-20260926.GnWOvb/results.md)
   misses the terminal and retains high values through a scoreless ending.
   Its [life-count readout](../runs/qbert-life-representation-20260926.F1lTAq/results.md)
   is stronger before pooling, but covers phase zero only and uses four times
   the dimensions. A same-size pooling test needs a separate recording across
   all chunk phases; never tune on the exposed test split or equate linear
   decodability with gameplay competence.
7. **Confirm only a passing recipe.** Fresh roots1009/2017/3019 each need the
   original final-policy gate and a restored untrained control. Native games,
   transfer and swarms remain downstream of reliable single-actor results.

The qualified package remains native b00ce7be / Meganeura ee3aea42 / Blade
fbb4f28c: [pixel/state/restore](../runs/world-chunks-gameplay-20260926.q4iNnd/results.md),
[runtime compatibility](../runs/chunks-compatibility-20260926.opVUiI/results.md)
and [matched-order timing](../runs/chunks-atari-timing-20260926.e2OEhn/results.md)
pass. Reuse it without rebuilding for adapter/test/documentation-only changes.
Keep the [negative optimizer timing result](../runs/meganeura-optimizer-timing-20260926.A6u3AI/results.md)
and [negative Qbert correctness-refresh result](../runs/correctness-qbert-comparison-20260924.Q4NZyO/results.md);
correctness, speed and learning are distinct outcomes.

All completed writers, failed-reader evidence, interrupted Freeway2017 attempt
and historical holds remain immutable. Checkpoints preserve weights/moments,
not replay/live belief/RNG. Review each GPU successor individually. Detailed
chronology belongs in [the experiment index](experiments/README.md).
## Complexity and compute

Dreamer's interacting networks and recurrent learner are real complexity;
hundreds of investigation commits and chronological reports are not architectural
requirements. Keep production code for exercised features and evidence in linked
reports. Tests are not cruft simply because they exceed implementation size.

Learning a game is not the same target as mastering it. The
[retained DreamerV3 Atari-100k curves](../runs/breakout-reference-context-20260926.kGfmtw/results.md)
give a five-seed Breakout score-window mean8.89, far below our two-wall threshold.
That historical 200M/100k-action online benchmark is not matched to our 12M+Tiny/
200k-action frozen evaluation. It neither diagnoses JEPA nor predicts the budget
needed for mastery; retain the stronger gate and test causes rather than assuming
every modest score means an implementation or representation failure.

Report conventional learning curves and human-normalized scores beside mastery.
Using the [pinned upstream references](https://github.com/danijar/dreamerv3/blob/e3f02248/baselines.yaml),
Qbert's final score is .935 human-normalized; Breakout's864-point gate is29.94.
Protocol differences make these descriptive, not matched benchmark claims.
The gate stays fixed, but missing mastery must not be called absence of learning.
A bounded matched upstream Dreamer12M control remains necessary to assess the
frozen-JEPA design; the single pretraining ablation settles neither benefit nor
harm. Synthetic full-encoder streaming parity is now wired into CI and passes
locally; the checkpoint-specific production hardware tests remain separate.

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
frames, with whole-recording splits and verified checkpoints/exports. The
[random-policy corpus](../runs/levjepa-tiny-atari-corpus-20260920.oTjdon/result.md)
contains **all five target games**, each45,000 training plus5,000 validation
RGB64 observations. This is additional same-title offline experience, not
held-out-title transfer. Native pretraining and the collector are preserved on
[`exp/levjepa-tiny-pretrain-20260920`](https://github.com/kvark/kindle/tree/exp/levjepa-tiny-pretrain-20260920).
Declare this lineage in every downstream comparison.

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

Qbert's [first complete episode probe](../runs/qbert-hazard-probe-cpu-v2-20260926.GnWOvb/results.md)
exactly replays 1,272 actions and 18,975 targets with unchanged full state. At H15,
feature MSE .000666 beats persistence .011081/unrelated actions .001886; reward
MAE/MSE 4.124/421.67 beat zero 7.194/992.65. Continuation is slightly worse than
always-continue, with only one terminal. At that terminal, H1 continuation is
.99734 versus 0. The last 281 actions yield no reward while mean posterior value
remains 2,119.43. This is one realized trajectory, not unbiased critic calibration
or proof of an encoder defect. Check held-out life-count information before
choosing representation versus downstream learning changes; nonterminal deaths
must not silently become terminal labels.

Further diagnostics should retain early hazards/later plateaus and Breakout's
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
