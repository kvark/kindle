# Kindle status

Updated: **2026-09-22, 23:48 UTC**. This is the short operational dashboard,
updated at meaningful phase boundaries, not a live log. The
[research plan](docs/kindle_single_life_dreamer_plan.md) remains authoritative
for architecture, budgets and acceptance gates.

**Goal: reliable learning and wins on five Atari games. Current: 2/5.**
Boxing and Pong pass their three-seed gates. **The small-encoder Freeway pilot
now passes:** final policy **36/36** qualifying rounds, mean **33.03 crossings**,
versus **0/36**, mean **0**, for its untrained-policy control. Its midpoint also
passes, mean **30.50**. All five phases, complete state/moments, six-stream replays
and videos verify. **Fresh-seed confirmation: seed1009 passes the complete pair, final36/36
versus control0/36. The approved fresh seed2017 replacement is running under
a systemd service; its interrupted predecessor is preserved. One of three roots passes.**
Freeway is not reliable yet; Breakout and Qbert remain below their gates.

The selected frontend is a separately trained **5.49M causal-video JEPA**, not
DINO or an RGB replacement. The DreamerV3 **12M RSSM preset stays unchanged**.
Tiny pretraining, native integration and matched-order cost checks are complete:
**26–27% less total time**, **74–75% less observation time**. Learning dominates
the remaining cost. Tiny is opt-in; Large remains the default because Breakout
regresses and three-root Tiny reliability is untested. The target remains a
trained causal-video encoder, not random features.

## Workboard

| State | Action item | Result / remaining work |
| --- | --- | --- |
| Done | Native Dreamer + causal LeVJEPA, vector collection | Rust/Meganeura/Blade; six independent streams share batched inference and one learner. No concurrent learner service. |
| Done | Boxing and Pong reliability | Three fresh seeds pass each game's gate. Pong: **71/72 trained wins vs 0/76 untrained**, zero evaluation updates/cutoffs; all six videos and complete state verify. |
| Done | Qualify block-matmul optimization | **27.3% higher throughput**, exact same-state/moment/action checks. Still below real-time training. |
| Done | Clean PR history and fix CI | Old 286-commit history archived; linear review history, no merges. [Implementation ed16e19 passes CI #156](https://github.com/kvark/kindle/actions/runs/35615868079); subsequent changes are documentation-only. The 13-line checkpoint-history option passes 768 local Python tests and exact native state/trajectory/restore checks; native Rust/Cargo inputs are unchanged. [Live PR status/checks](https://github.com/kvark/kindle/pull/29/checks). |
| Done | Breakout four-action runtime qualification | Gradients, full state, initialization/restore and all eight N6 pixel/replay/refusal checks pass. Optional replay/schema support is in the main tree; full eighteen-action remains the default. No new gameplay result. |
| Done, with caveats | First compact causal-video JEPA pretraining candidate | **5,486,592 parameters**; **250,000 observations** with whole-recording held-out splits. **Fresh seed 743, 4,096 updates complete**, numerical/optimizer/restore/streaming/state/export checks pass. Native step .45s plus .78s data preparation. Frozen noncollapse passes; position decoding improves, motion is not uniformly better. [Pretraining](runs/levjepa-tiny-pretrain-20260921.JaPZpW/results.md), [all frozen probe results and caveats](runs/levjepa-tiny-quality-20260921.ojeZgt/results.md). |
| Done | Finish active Breakout Large reference | **200,004 actions / 49,652 updates**. Frozen mean **30.79 vs 0.97**, two-wall successes **0/24 vs 0/29**; both zero updates/cutoffs. Complete state/replays/videos pass. Learning, not a win. Four-action arm held. [Results and videos](runs/breakout-action-pilot-20260920.kNeotb/results.md). |
| Done | Qualify opt-in Tiny actor integration | Same gameplay backend: **754 Python + 93 Rust CPU checks**, dense-reference and exact N6/serial GPU parity pass. Tiny/Large each complete **3,840 actions / 611 updates**, frozen restore and six-stream replay. All state/moments verify; Large matches its retained state and trajectories exactly. [Inference checks](runs/levjepa-tiny-gameplay-gpu-20260921.X6TnAI/results.md), [integration results and videos](runs/levjepa-tiny-gameplay-pixels-20260921.NQLh0I/results.md). |
| Done | Confirm cost in both orders | All four **Tiny/Large/Large/Tiny** runs pass, **10,008 actions / 2,153 updates each**. Post-warmup windows: **26.0–26.9% less total time**, **73.6–74.8% less observation time**; exact state/moment/report/trajectory repeats. Tiny **.897–.902× aggregate / ~.150× per stream**. [Complete comparison](runs/levjepa-tiny-throughput-20260921.cY1QjK/results.md). |
| Complete; quality regresses | Full-budget Tiny/Large learning comparison | Fresh seed0 Tiny: **200,004 actions / 49,652 updates, 4.108h**. Frozen **10.92 vs .93** control, **0/24 vs 0/29** two-wall successes, zero updates/cutoffs. Complete state, common initial learner state and all-stream replays pass. Large scores **30.79** at the same budget; different corpora mean a package comparison, not a pure size ablation. [Results and both videos](runs/levjepa-tiny-breakout-20260921.ghJPWG/results.md). |
| Complete; no demonstrated benefit | Isolate the effect of Tiny pretraining | Own initial encoder: **200,004 actions /49,652 updates, 4.081h**. Frozen **12.46 vs1.10** control, **0/24 vs0/30** two-wall successes; complete state/common initialization/replays pass. Trained-policy mean is **+1.54** versus pretrained Tiny, with one seed only. No random-encoder product switch or default adoption. [Results and videos](runs/levjepa-tiny-pretraining-ablation-20260921.lrjxlN/results.md). |
| Pilot passes | Freeway with fixed pretrained Tiny | Fresh seed0: **400,008 actions /99,652 updates, 7.969h**. Frozen final **33.03 crossings, 36/36** qualifying rounds; midpoint **30.50, 36/36**; untrained policy **0, 0/36**. Each has 75,000 unassisted actions, zero updates/cutoffs. Complete state, all six streams and decoded videos pass. [Results and videos](runs/tiny-freeway-exposure-20260921.ejKgSH/results.md), [combined audit](runs/tiny-freeway-exposure-20260921.ejKgSH/pair.json). |
| Running: approved seed2017 replacement | Freeway stability on fresh roots 1009/2017/3019 | Seed1009 final **36/36, mean32.92** versus control **0/36, mean0**; complete pair/state/replays/videos pass. Original seed2017 is terminal at45,714 actions after guard loss. User-approved fresh replacement starts23:14 UTC: unchanged400,008-action recipe,200,004 midpoint,18h guard bound, tested systemd ownership/cleanup and no automatic restart. Seed3019 unstarted. **1/3 fresh roots passes.** [Seed1009 results/videos](runs/tiny-freeway-confirmation-20260922.tij9QW/results.md), [current run](runs/tiny-freeway-seed2017-replacement-20260922.12z27y72/results.md), [preserved incident](runs/freeway-guard-interruption-20260922.2xt6tnn_/README.md). |
| Qbert prepared; not launched | Qbert exposure and Breakout action vocabulary | [Qbert protocol and CPU input checks](runs/tiny-qbert-exposure-preparation-20260922.9c53nmwp/README.md): fresh seed0 Tiny, unassisted400,008 actions/200,004 midpoint, four frozen episodes per stream with600,000 cap and matched untrained control. Full pyramid/15,000-score gate unchanged; no GPU declaration or queue. Breakout four-action recipe remains held. |
| Open | Lower learner cost / justify representation | Replay-ratio and objective ablations remain needed. Current wins do **not** establish a JEPA advantage. |
| Done, with caveats | World-model forecasts through the actor's 15-step horizon | **1,126 actions /four matches /16,470 targets**, exact one-step overlap and complete frozen state, zero updates. H15 feature MSE **.0001860 vs .0010262** persistence; reward MAE **.00894 vs .04019** zero; continuation MSE **.00511 vs .00372** always-continue (worse). Only22 positive rewards/four terminals; recorded future actions condition forecasts. [15-step report and raw predictions](runs/tiny-world-horizon15-20260921.bKmiUF/results.md), [strict/forced one-step comparison](runs/tiny-world-one-step-20260921.U7yHOa/results.md). |
| Later | Native games, transfer, intrinsic motivation, swarms | Single-actor reliability first. Not current work. |

**Next decision:** review the approved seed2017 replacement after its full
training budget, then separately declare its frozen midpoint evaluation.
The scientific question remains whether the passing Tiny Freeway recipe survives
fresh training seeds. Both pilot checkpoints pass, so the additional exposure improves this
seed's mean by 2.53 crossings but is not shown necessary. Keep the 400,008-action
primary budget fixed for confirmation. Historical Large Freeway pilots also
passed before fresh roots failed; do not infer stability from seed0. The Tiny
Breakout regression and mixed motion/continuation diagnostics remain unresolved.
No evidence yet justifies another encoder redesign or default switch.
[Exact parameter accounting and selected design](runs/model-sizing-20260920.kPIOWC/README.md).
[Native pretraining readiness and next steps](runs/levjepa-tiny-cpu-20260920.e4QkQH/README.md).
[Independent reference and memory results](runs/levjepa-tiny-reference-20260920.XhB8T8/README.md).
[Fresh corpus and measured adapter cost](runs/levjepa-tiny-atari-corpus-20260920.oTjdon/result.md).
[Isolated package and qualification fixtures](runs/levjepa-tiny-package-20260920.rDhKjm/README.md).
[GPU results and preserved failures](runs/levjepa-tiny-accuracy-20260921.qa3GqK/results.md),
[actual B128 timing](runs/levjepa-tiny-fit-20260921.oDmZVn/results.md),
[held-out noncollapse screen](runs/levjepa-tiny-feature-check-20260921.wSQUJj/results.md).

**Measured pretraining bottleneck:** serial crop/batch preparation (.78s) takes
longer than native training (.45s). These are wall times, not utilization.
No CPU learner fallback or concurrent learning service was added.

**Measured gameplay bottleneck after shrinking perception:** about 87% learning.
A steady learner step averages .257s: .092s world training, .086s imagination,
.035s posterior, .022s behavior training, with the remainder mostly sync.
These clocks do not measure GPU idle time. More environment workers do not remove
this fixed R256 learner cost; replay-ratio ablation remains a separate next test.

**Procedural deviation:** a CPU module-filter mistake briefly created GPU shader
pipelines during Breakout; it exited without model execution or an observed
kernel fault. The test is now ignored by default. Preserve the
[record](runs/levjepa-tiny-cpu-20260920.e4QkQH/test-selection-20260920.md); the overlap
is not uncontended timing evidence. No NVML or recovery operation occurred.

**Results and videos:** [all five games and their exact gates](docs/kindle_single_life_dreamer_plan.md#current-game-status),
[Pong seed-by-seed results and six videos](docs/experiments/README.md#current-pong-confirmation).
For a quick look: [small-encoder Freeway rollout](runs/tiny-freeway-exposure-20260921.ejKgSH/final.mp4),
[its untrained-policy control](runs/tiny-freeway-exposure-20260921.ejKgSH/untrained.mp4),
[successful Pong rollout](runs/pong-block-confirmation-20260916.rBwdGF/seed1009-evaluation.mp4).
Breakout checks: [gradients](runs/breakout-gradients-20260920.dtzN0w/results.md),
[complete state](runs/breakout-state-20260920.iuGWoA/results.md),
[initialization/restore](runs/breakout-initial-completion-20260920.DAvUch/results.md),
[completed pixel integration](runs/breakout-pixels-20260920.YrLIfy/results.md).

## Why has this taken so long?

**Not simply because Dreamer cannot get beyond Pong.** We combined three jobs:
implement Dreamer, develop/debug a native GPU training stack, and investigate a
different visual representation/training objective. Backend optimization, four
driver incidents and overgrown experiment-management machinery consumed a lot
of effort. That overhead is not an inherent architectural requirement. The
recent Pong work was three-seed confirmation on a fixed qualified runtime, not
continued attempts to obtain a first win.

**Our recipe is expensive.** Each 400,008-action Pong run takes **10.94–10.99h**,
so three training seeds alone take about **33 GPU-hours**. Replay ratio 256 means
about 102 million replay positions and 99,652 updates per run. Root 2017 spends
65.6% of wall time learning, 33.7% processing observations, and .45% stepping
the emulator. These are stage timings, not GPU-utilization measurements; NVML
remains disabled. More environment workers do not directly address these costs.

**JEPA adds cost and uncertainty, but is not a proven cause of the failures.**
The earlier successful recipe used a frozen **303.10M LeVJEPA frontend** beside
**10.28M optimized learner parameters** (the nominal Dreamer12M preset), with prior
feature prediction and no reconstruction. RSSM dynamics/prior/posterior are
**5.92M** of that learner. The frontend is **29.5×** larger than the optimized
stack; the tested 5.49M candidate is **55.2×** smaller than Large. This is not
plain DreamerV3, nor end-to-end online JEPA training. Standard Dreamer learns
its visual encoder with the world model; its published reconstruction ablations
also make removing that learning signal a substantive research choice.
[Primary paper](https://www.nature.com/articles/s41586-025-08744-2).
Our new results prove learning, not that this departure is better or faster.
We have not yet demonstrated that the large frontend earns its extra cost.
An [earlier local upstream control](https://github.com/kvark/kindle/blob/9ef9a169d52ed36a912d578a89690f992c5982fc/docs/experiments/2026-09-05-kickoff.md#upstream-control-provenance)
exists, but differs in precision, perception and protocol and is not a matched
current-campaign architecture ablation.

**Assessment:** iteration efficiency and engineering scope are the demonstrated
problems; an absolute shortage of GPUs is not established. Confirm the
smaller causal encoder across seeds, make learning/representation choices earn their place,
and do not add concurrency, model size or swarm infrastructure to
explain away weak single-actor results. Keep the five-game success gates unchanged.
