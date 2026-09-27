# Kindle: a single actor that learns while acting

This is the authoritative plan. [Current evidence and archive](experiments/README.md)
retain experiments and failures; [AGENTS.md](../AGENTS.md) gives working rules.
Keep the runtime small, comparisons controlled and results reproducible.
For a quick overview of done/in-progress/next work and why progress is costly,
start with the [Phase 2 PR status dashboard](https://github.com/kvark/kindle/pull/31).
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
frozen during gameplay, not end-to-end JEPA training. The obsolete DINO path has
been removed; its results remain historical. New agents default to a
separately pretrained **5.49M causal ViT-Tiny/16**. Native numerical,
optimizer/restore, streaming, fit and noncollapse checks pass; the first bounded
4,096-update pretraining run completes with verified state and encoder exports.
Frozen probes retain useful position features but mixed motion results. Bounded
actor integration and matched-order throughput pass. Tiny Freeway now passes
all three fresh learner roots, but Breakout regresses and broader Tiny reliability
is unproven. Tiny is now the default; Large is an explicit comparison only.
Old checkpoint/backend pinning must not delay this implementation. Keep current
architecture/encoder integrity checks, not migration machinery.

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

Single and vector pixel actors now share a resident acting path: GPU pixels ->
preprocessing -> encoder -> pooling -> belief/policy -> selected actions. Replay
collection also stays on GPU. Linux Vulkan capture is integrated through
Dullahan's fenced ownership protocol and exercised in real vkQuake at 640x480:
full 12M frozen, 256 actions in 2.758 s, plus a separate small-world 105-update
plumbing test. This is not Quake competence or 12M training throughput.
Explicit diagnostics/checkpoints and sampled learner batches still read back
data. See the
[implementation and validation report](experiments/2026-09-26-gpu-resident-acting.md).

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

The historical five-game campaign required roots **1009/2017/3019** to pass
their final-policy gates and beat separately restored untrained controls. These
gates are retained to interpret old results, not as the development queue. Videos
are whole stream-zero evaluations, including unfinished tails, not selected wins.
Full multi-stream evaluations determine results.

The [human-normalized snapshot](results/2026-09-27-historical-scores.md)
publishes raw learner-seed means, score anchors and protocol qualifications in
compact JSON. It is descriptive historical evidence, not a matched benchmark.

| Game | Measured result | Unchanged gate / next decision | Rollout |
| --- | --- | --- | --- |
| Boxing | Three roots pass: 123/123, 207/207, 51/51 wins; means +83.87/+90.58/+83.53; controls near zero | ≥20 natural matches, ≥90% wins, mean ≥+50, no cutoffs. Complete. | [1009](../runs/boxing-confirmation-20260910.hTEDcu/seed1009-evaluation.mp4), [2017](../runs/boxing-confirmation-20260910.hTEDcu/seed2017-evaluation.mp4), [3019](../runs/boxing-confirmation-20260910.hTEDcu/seed3019-evaluation.mp4) |
| Pong | Historical non-sticky roots pass71/72 wins versus0/76 controls. But root1009 with25% sticky actions wins only2/24 equal-cohort matches, mean−7.1667; all3/31, mean−8.3871. State/replay/video audit passes. | ≥20 natural matches, ≥90% wins, mean ≥+15, no cutoffs. Fixed-recipe pass; **robustness fails**. One stochastic-evaluation root, no new control pair. | [Sticky video](../runs/pong-sticky-evaluation-20260926.SxeHCw/seed1009.mp4), [new report](experiments/2026-09-26-gpu-pixels-and-pong-robustness.md), [historical videos/controls](experiments/README.md#current-pong-confirmation) |
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

## Current execution order — strategy reset

The September 27 user direction adopts
[strategy_reset_plan.md](strategy_reset_plan.md). This document remains the
current evidence/roadmap; the strategy proposal records the rationale and phase
details. The new order supersedes historical queue, package-adoption and
five-game confirmation work. Do not rerun unchanged failed recipes.

1. **Phase 0: close the reporting gaps.** Assistance and same-title pretraining
   are disclosed below and in the PR. Synthetic causal Tiny parity runs in CI
   without pretrained files; Linux/lavapipe, Metal and Python checks pass.
   Publish compact raw scores, human-normalized values and limitations in
   `docs/results/`. The old 864-point Breakout requirement is a historical
   mastery definition, not an appropriate 200k-action development target.
   There is no evidence that this recipe can reach it at that budget; stop gate
   runs. This is not a proof that no algorithm could reach it.
2. **Phase 1a, complete: shorten learner iterations without changing learning.**
   Acting/capture and replay collection are already resident. First share
   compatible direct parameters and batch remaining derived-weight transfers,
   comparing complete saved state and update timings with the unchanged
   implementation. This [first step now passes](results/2026-09-27-shared-parameters.md):
   227.03 -> 212.78 ms/update (6.28% less time) in one fixed synthetic pair,
   exact complete state/reports and short N6 Pong integration. This is not
   sustained game throughput. The [next measured step](results/2026-09-27-fused-learner.md)
   implements GPU Gumbel sampling, complete T64/H15 recurrence and grouped RSSM
   arithmetic: shared-control synthetic **212.70 -> 184.43 ms/update**, native
   N6 Pong **17.36 -> 20.16 steady actions/s at R256**. Independent sampling,
   output/gradient references and short Pong learning statistics pass. Sampling
   draws changed; state/trajectories are not bitwise equivalent. CPU targets and
   independent slow-critic EMA remain. **The 3x target is not achieved.** F32,
   replay ratio, batch/BPTT, losses and optimizer stay fixed. BF16 is not a
   compute option in the current backend; F16 relaxation needs a separate test.
   One numerical/learning check plus matched timing suffices for development;
   no qualification campaign. The user explicitly accepts 3x as a stretch
   target, not a Phase 1 exit gate. Phase 1 closes with that miss disclosed.
3. **Phase 1b, complete: establish a <=1-hour, three-seed screening recipe.** Use a small
   learner and a fast sparse-reward environment. [MinAtar recipe/curve contract](screening.md)
   now [completes in 8m18s for all three seeds](results/2026-09-27-minatar-screen.md):
   each 32,768 actions / 8,135 updates in 164 seconds. Final online scores
   .44/.40/.68 give mean .507, seed-bootstrap 95% CI [.400,.680]. This is a
   working iteration loop, not reliable improvement or competence. JAX/Craftax
   buffers need an additional CUDA/Vulkan interop bridge;
   MinAtar is the permitted CPU-environment fallback, not a CPU learner. Its
   small public observations are packed losslessly and use the existing jointly
   learned encoder; no frozen frontend, pretraining or action/reward aid.
   Do not combine this new environment/recipe and speed change into one claim.
   Record curves against actual actions and elapsed time; promote only useful
   changes to 12M and longer Atari confirmation.
4. **Phase 2, in progress: test whether the representation earns its cost.**
   The [offline/learning protocol](experiments/2026-09-27-representation-comparison.md)
   now has 6,144 Pong/Breakout/Seaquest clips, split by whole trajectories.
   Native batched token diagnostics compare identical frames; privileged labels
   never enter the agent. All five ridge/corrected-MLP controls complete:
   Large decodes best, Tiny pretraining is mixed, and the 173k reconstruction
   CNN is competitive enough to justify a joint-trained RGB learning arm.
   [Full probe results](results/2026-09-27-representation-probes.md) are not RL.
   Held-out motion/
   small-object probes, then a three-seed learning comparison against random
   Tiny and a learned-encoder Dreamer control. Include a title absent from the
   video corpus. Native input is current; RGB64 is an explicit ablation.
   Matched upstream/native smokes pass: all four complete 6,144 actions and
   1,186 updates. The first full declaration stopped in host-log capture before
   launching a GPU worker; use local live logs for the fresh attempt. The three-seed,
   three-game, 200,004-action-per-run comparison remains unfinished.
   The historical ~15.6 versus ~59 actions/s is not a matched
   efficiency comparison.
   If frozen JEPA offers no probe/learning benefit, change the 2D frontend;
   retain JEPA as a testable 3D candidate, not an architectural obligation.
5. **Phase 3: exploration and reward.** Extrinsic-only versus one mechanism,
   without Freeway's random-action assistance, three learner seeds and curves.
   Prefer a GPU-compatible intrinsic mechanism. The old CPU hash-visitation
   experiment is not supported by current pixel collection; do not silently
   enable it or add host feature readback. Disclose shaped rewards, overrides,
   privileged reward observers and every pretraining source.
6. **Phase 4: video priors for dynamics and behavior.** Compare action-free
   dynamics pretraining, inferred actions and behavior priors at equal online
   experience; disclose offline cost and target-game exposure.
7. **Phase 5, later: asynchronous real-time deployment.** Only after the
   single-actor learner is effective, introduce a separately measured async
   actor/learner, learner debt and latency percentiles. Native GPU capture
   already works, but reward/terminal adapters and mind-games controller
   integration remain. Swarm experience sharing comes after useful real-time
   single-actor operation, not before it.

Development results use one changed factor, at least three seeds for learning
comparisons, curves and learner-seed uncertainty (bootstrap intervals/IQM where
appropriate), not pass/fail mastery gates. Numerical/parity smoke checks are
not three-seed learning experiments. Each new result gets a compact JSON and
Markdown summary in `docs/results/`; include configuration, seeds, aids,
curves when available, and limits. Historical summaries may lack curves; say
so rather than inventing them. Large raw artifacts remain in `runs/`.
Keep GPU containment and stop-on-fault rules unchanged.

The earlier production timing package (b00ce7be / ee3aea42 / fbb4f28c) and its
6.5–6.6% gain are historical controls, not the current source identity or a
barrier to new development. Current resident-actor implementation is described
in the [GPU report](experiments/2026-09-26-gpu-resident-acting.md).
Preserve old results/failures without recursively revalidating their archives.
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
needed for mastery; retain the historical definition and test causes rather than assuming
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
derived weights must remain coherent. Application NVML polling stays disabled;
normal JAX backend initialization is permitted for the bounded upstream control.

The capture-to-action target is GPU-resident: capture -> letterbox/normalize/
patch layout -> causal encoder -> belief/policy -> action readback. Only the
acting hot path has the action-only boundary; sparse rewards, checkpoints and
explicit diagnostics remain legitimate host traffic. This path now exists for
uploaded RGB and resident RGB/RGBA/BGRA, with real Dullahan capture integration.
Producer/consumer fence completion plus EXTERNAL queue-ownership barriers make
ring reuse explicit. The conservative socket handshake serializes game frames;
it is not yet pipelined external-semaphore execution. Paged device replay avoids
collection readback, but the learner still downloads sampled feature/context
batches. Track capture-to-action latency separately from update throughput.
Port the lease contract into mind-games GameSession before vector native games;
the old structured-state KindleActor and multi-frame freezer are not that path.

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

The current reference is **causal ViT-Tiny/16: 12 layers, width 192, three heads,
MLP 768, 5,486,592 encoder parameters**. Preserve 224px inputs, 16-arrival block-causal
chunks, independent histories and JL64/2×2 pooling to 7×7×64. Logical F32 KV
storage is 55.125 MiB per stream versus Large's 588 MiB; these are tensor sizes,
not measured device peaks. Do not slice Large weights or silently substitute
features during a speed comparison. Phase 2 may replace the frozen frontend
after controlled probes and learning experiments.

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
or justifies random features as the product. **Tiny is the current default;
Large is explicit.** Phase 2 must establish a benefit over random/learned features.

Keep numerical/causal/streaming checks, held-out quality, N6 memory/time and
frozen downstream controls for final adoption. Different pretraining corpora
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
not just lower feature error. Representation expansion is a separate experiment.

## Beyond five Atari games

Beating most of a predeclared Atari suite is an ambition, not a consequence of
using Dreamer. Our variant does not inherit published DreamerV3 scores. Extend
to held-out Seaquest/Frostbite/Private Eye for representation/exploration tests,
then a broader suite with curves, budgets and learner-seed distributions.
Independent per-game training tests algorithm breadth, not
one transferable policy. Keep a matched local upstream control and disclose
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

Useful real-time single-actor learning and held-out adaptation/retention precede
swarm work. Phase 5 can then test immutable experience sharing or several actors
feeding one learner, with explicit ownership and causal histories. Do not turn
the old GOG milestone into another unchanged-gate queue. Natural deaths/respawns
are allowed; cloning/rewinding a live game for training is not. Intrinsic reward
stays in its separate channel, off in controls, with extrinsic-only comparisons.

## Execution and evidence

Use the GPU; application NVML polling remains disabled, while normal JAX/CUDA
initialization is permitted by the September26 user direction. Serialize bounded direct native
jobs under the [host-only guard](gpu_incident_response.md), with actual device
assertions and >=2 GiB sampled Vulkan budget headroom. No recovery operation,
blind retry or quarantined candidate reuse. The four old Xid incidents remain
unexplained. Passing driver 580/no-NVML jobs is not a causal fix or safety proof.

Preserve declarations, failures and artifacts. Do not grow a new framework for
each check or rebuild qualified binaries for unchanged code. CI must install
declared test dependencies in a clean environment. Checkpoints preserve weights/
moments, not replay/RNG/live belief; interrupted resume is not equivalent lifetime
continuation. Atomic complete-state recovery and bounded storage remain later work.
