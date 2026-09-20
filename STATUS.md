# Kindle status

Updated: **2026-09-20, 19:19 UTC**. This is the short operational dashboard,
updated at meaningful phase boundaries, not a live log. The
[research plan](docs/kindle_single_life_dreamer_plan.md) remains authoritative
for architecture, budgets and acceptance gates.

**Goal: reliable learning and wins on five Atari games. Current: 2/5.**
Boxing and Pong pass their three-seed gates. Freeway, Breakout and Qbert do not.
We have moved on from Pong. Breakout's runtime qualification is complete,
including all eight pixel/replay/refusal checks. The paired learning pilot's
eighteen-action baseline is running under the host-only guard, started at 19:18
UTC. Its 200,004-action budget should take roughly 5–6 hours at recent throughput;
the declared limit is nine hours. No gameplay result is available yet.

## Workboard

| State | Action item | Result / remaining work |
| --- | --- | --- |
| Done | Native Dreamer + causal LeVJEPA, vector collection | Rust/Meganeura/Blade; six independent streams share batched inference and one learner. No concurrent learner service. |
| Done | Boxing and Pong reliability | Three fresh seeds pass each game's gate. Pong: **71/72 trained wins vs 0/76 untrained**, zero evaluation updates/cutoffs; all six videos and complete state verify. |
| Done | Qualify block-matmul optimization | **27.3% higher throughput**, exact same-state/moment/action checks. Still below real-time training. |
| Done | Clean PR history and fix CI | Old 286-commit history archived; review history is linear. [Latest checked code CI passes](https://github.com/kvark/kindle/actions/runs/35530950331); main CPU suite passes 742 tests. [Live PR checks](https://github.com/kvark/kindle/pull/29/checks). |
| Done | Breakout four-action runtime qualification | Gradients, full state, initialization/restore and all eight N6 pixel/replay/refusal checks pass. Optional replay/schema support is in the main tree; full eighteen-action remains the default. No new gameplay result. |
| In progress | Breakout four-vs-eighteen-action learning | Nine CPU checks pass; commands/readers and 105 inputs are pinned. Eighteen-action training is running; frozen evaluation, untrained control, then the four-action arm remain. Keep the two-wall/864-point gate. |
| Next | Freeway and Qbert exposure experiments | Test the prepared longer continuous runs with retained midpoints; confirm a successful recipe on three seeds, not repeat the unchanged failed recipe. |
| Open | Lower compute cost / justify representation | Matched replay-ratio, cheaper-representation and plain-Dreamer controls remain needed. Current wins do **not** establish a JEPA advantage. |
| Open | World-model forecasts on current checkpoints | Recorded matches are selected; current-runtime multi-match forecast validation remains unfinished. |
| Later | Native games, transfer, intrinsic motivation, swarms | Single-actor reliability first. Not current work. |

**Next decision:** compare Breakout learning with four versus eighteen actions,
at equal budgets and against each arm's untrained control. Runtime checks only
make that comparison trustworthy; they are not progress toward the gameplay gate.
A successful pilot still needs three fresh seeds before Breakout becomes done.

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
We run a frozen **303M LeVJEPA frontend** beside a **12M learner**, with prior
feature prediction and no reconstruction in the selected recipe. This is not
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
problems; an absolute shortage of GPUs is not established. Keep the prepared
Breakout comparison focused, make cheaper learning/representation controls earn
their place, and do not add concurrency, model size or swarm infrastructure to
explain away weak single-actor results. Keep the five-game success gates unchanged.
