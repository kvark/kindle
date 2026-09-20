# Kindle status

Updated: **2026-09-21, 01:40 UTC**. This is the short operational dashboard,
updated at meaningful phase boundaries, not a live log. The
[research plan](docs/kindle_single_life_dreamer_plan.md) remains authoritative
for architecture, budgets and acceptance gates.

**Goal: reliable learning and wins on five Atari games. Current: 2/5.**
Boxing and Pong pass their three-seed gates. Freeway, Breakout and Qbert do not.
We have moved on from Pong. The new priority is a **5.49M causal-video JEPA
encoder**, replacing the oversized 303M frontend after pretraining and checks.
The RSSM already matches DreamerV3's 12M preset and stays unchanged. Tiny inference
support passes CI; native pretraining, AdamW, GPU EMA and checkpoint/export
support are staged and pass GPU gradient, optimizer, exact-restore and streaming
checks. The B128/four-view pilot and held-out noncollapse screen pass.
**The first 4,096-update pretraining run is running**, with checkpoints every 512
updates (roughly 85 minutes at the pilot rate).
**There is no validated pretrained Tiny encoder or Tiny gameplay result yet**.
The Breakout Large reference is complete: frozen mean **30.79 versus 0.97**
untrained, but **0/24 two-wall completions**. Its unstarted four-action arm is held.

## Workboard

| State | Action item | Result / remaining work |
| --- | --- | --- |
| Done | Native Dreamer + causal LeVJEPA, vector collection | Rust/Meganeura/Blade; six independent streams share batched inference and one learner. No concurrent learner service. |
| Done | Boxing and Pong reliability | Three fresh seeds pass each game's gate. Pong: **71/72 trained wins vs 0/76 untrained**, zero evaluation updates/cutoffs; all six videos and complete state verify. |
| Done | Qualify block-matmul optimization | **27.3% higher throughput**, exact same-state/moment/action checks. Still below real-time training. |
| Done | Clean PR history and fix CI | Old 286-commit history archived; four linear review commits. [Last verified CI #149 passes](https://github.com/kvark/kindle/actions/runs/35546573229); all 91 main Rust CPU tests pass. [Live PR status/checks](https://github.com/kvark/kindle/pull/29/checks). |
| Done | Breakout four-action runtime qualification | Gradients, full state, initialization/restore and all eight N6 pixel/replay/refusal checks pass. Optional replay/schema support is in the main tree; full eighteen-action remains the default. No new gameplay result. |
| In progress | Train a compact causal-video JEPA frontend | **5,486,592 parameters**; **250,000 observations** with whole-recording held-out splits. **751 Python + 102 Rust CPU checks**, **seven GPU checks**, actual-package fit and held-out noncollapse screen pass. **Fresh seed 743, 4,096 updates running**. Native step .45s plus .80s data preparation; quality validation remains. [Current declaration](runs/levjepa-tiny-pretrain-20260921.JaPZpW/README.md). |
| Done | Finish active Breakout Large reference | **200,004 actions / 49,652 updates**. Frozen mean **30.79 vs 0.97**, two-wall successes **0/24 vs 0/29**; both zero updates/cutoffs. Complete state/replays/videos pass. Learning, not a win. Four-action arm held. [Results and videos](runs/breakout-action-pilot-20260920.kNeotb/results.md). |
| Next | Compare Tiny with Large at fixed learner settings | Retain the 12M RSSM, R256 and action vocabulary initially; measure learning and wall-clock cost. Different pretraining corpora must be disclosed, not called a pure size ablation. |
| Deferred | Freeway/Qbert exposure and Breakout action vocabulary | Resume with separate declarations after the compact frontend decision. Confirm successful recipes on three seeds; existing gates unchanged. |
| Open | Lower learner cost / justify representation | Replay-ratio and objective ablations remain needed. Current wins do **not** establish a JEPA advantage. |
| Open | World-model forecasts on current checkpoints | Recorded matches are selected; current-runtime multi-match forecast validation remains unfinished. |
| Later | Native games, transfer, intrinsic motivation, swarms | Single-actor reliability first. Not current work. |

**Next decision:** establish that a pretrained Tiny encoder preserves useful
causal features and lowers measured cost. Preserve the completed Large reference, and
change neither RSSM capacity nor replay ratio in the first comparison. A smaller
parameter count alone establishes neither speed nor gameplay quality.
[Exact parameter accounting and selected design](runs/model-sizing-20260920.kPIOWC/README.md).
[Native pretraining readiness and next steps](runs/levjepa-tiny-cpu-20260920.e4QkQH/README.md).
[Independent reference and memory results](runs/levjepa-tiny-reference-20260920.XhB8T8/README.md).
[Fresh corpus and measured adapter cost](runs/levjepa-tiny-atari-corpus-20260920.oTjdon/result.md).
[Isolated package and qualification fixtures](runs/levjepa-tiny-package-20260920.rDhKjm/README.md).
[GPU results and preserved failures](runs/levjepa-tiny-accuracy-20260921.qa3GqK/results.md),
[actual B128 timing](runs/levjepa-tiny-fit-20260921.oDmZVn/results.md),
[held-out noncollapse screen](runs/levjepa-tiny-feature-check-20260921.wSQUJj/results.md).

**Current pretraining bottleneck:** serial crop/batch preparation (.80s) takes
longer than native training (.45s). These are wall times, not utilization.
No CPU learner fallback or concurrent learning service was added.

**Procedural deviation:** a CPU module-filter mistake briefly created GPU shader
pipelines during Breakout; it exited without model execution or an observed
kernel fault. The test is now ignored by default. Preserve the
[record](runs/levjepa-tiny-cpu-20260920.e4QkQH/test-selection-20260920.md); the overlap
is not uncontended timing evidence. No NVML or recovery operation occurred.

**Results and videos:** [all five games and their exact gates](docs/kindle_single_life_dreamer_plan.md#current-game-status),
[Pong seed-by-seed results and six videos](docs/experiments/README.md#current-pong-confirmation).
For a quick look: [latest successful Pong rollout](runs/pong-block-confirmation-20260916.rBwdGF/seed1009-evaluation.mp4).
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
We run a frozen **303.10M LeVJEPA frontend** beside **10.28M optimized learner
parameters** (the nominal Dreamer12M preset), with prior feature prediction and
no reconstruction in the selected recipe. RSSM dynamics/prior/posterior are
**5.92M** of that learner. The frontend is **29.5×** larger than the optimized
stack; the selected 5.49M replacement is **55.2×** smaller than Large. This is not
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
problems; an absolute shortage of GPUs is not established. Pretrain and test the
smaller causal encoder, make learning/representation choices earn their place,
and do not add concurrency, model size or swarm infrastructure to
explain away weak single-actor results. Keep the five-game success gates unchanged.
