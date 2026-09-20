# Current evidence and research archive

The [project plan](../kindle_single_life_dreamer_plan.md) is the only roadmap.
Logs, declarations, checkpoints and videos live in git-ignored `runs/`. Do not
overwrite completed/failed experiments or change their acceptance gates.

## Current Pong confirmation

[Declaration](../../runs/pong-block-confirmation-20260916.rBwdGF/declaration.md):
fresh roots 2017/3019/1009, qualified native `886bae68`, causal LeVJEPA, 400,008
training actions per root. Each root needs final-policy evaluation, fresh
initialization, restored untrained control and complete replay/video checks.
Every phase is individually invoked/reviewed; no automatic successor.

The [complete cross-seed audit](../../runs/pong-block-confirmation-20260916.rBwdGF/completed.json)
passes all three independent fresh roots: **71/72 trained wins versus 0/76 controls**.
Historical root 1009 is not included. This completes Pong's declared reliability
gate, not the five-game goal.

| Fresh root | Frozen trained wins / mean | Restored untrained wins / mean | Evidence and whole-stream videos |
| --- | --- | --- | --- |
| 2017 | 24/24 / +20.5417 | 0/28 / −20.3214 | [Pair](../../runs/pong-block-confirmation-20260916.rBwdGF/seed2017-result.json), [trained](../../runs/pong-block-confirmation-20260916.rBwdGF/seed2017-evaluation.mp4), [control](../../runs/pong-block-confirmation-20260916.rBwdGF/seed2017-untrained-evaluation.mp4) |
| 3019 | 23/24 / +17.4583 | 0/24 / −20.4583 | [Pair](../../runs/pong-block-confirmation-20260916.rBwdGF/seed3019-result.json), [trained](../../runs/pong-block-confirmation-20260916.rBwdGF/seed3019-evaluation.mp4), [control](../../runs/pong-block-confirmation-20260916.rBwdGF/seed3019-untrained-evaluation.mp4) |
| 1009 | 24/24 / +20.0833 | 0/24 / −20.5417 | [Pair](../../runs/pong-block-confirmation-20260916.rBwdGF/seed1009-result.json), [trained](../../runs/pong-block-confirmation-20260916.rBwdGF/seed1009-evaluation.mp4), [control](../../runs/pong-block-confirmation-20260916.rBwdGF/seed1009-untrained-evaluation.mp4) |

Each training run completes **400,008 actions / 99,652 updates** in 10.94–10.99h,
10.114–10.155 actions/s, .674–.677× aggregate real time. All verify complete state/moments,
checkpoint identity, guard completion and every native memory sample (minimum
estimated headroom 7,695,106,048 bytes). No faults or NVML calls. All six frozen
arms have zero updates/cutoffs and unchanged complete checkpoints. Complete ALE
replays match every action, reward and boundary; whole-stream videos retain
unfinished tails at 160×210/60 fps. The first four trained stream-zero matches
per root are selected for later world diagnostics, not new native forecasts.
Full counters, hashes and stage timings remain in the linked paired evidence.

All twelve native phases, three `pair` writers and `finish` completed by
September 20. Never repeat them. `audit.py summary` is the read-only cross-seed
reader. The legacy training initialization reader is memory-heavy on its 3.3 GB
stderr; avoid unnecessary repeat reads. No automatic game successor exists.

Breakout's [four-test gradient group](../../runs/breakout-gradients-20260920.dtzN0w/results.md)
passes both world/behavior tests at eighteen/four actions. The
[twelve complete-state canaries](../../runs/breakout-state-20260920.iuGWoA/results.md)
also pass: all six same-width pairs, six retained-anchor comparisons and four
fresh-process repeats are exact. [Zero-update initialization and restore](../../runs/breakout-initial-completion-20260920.DAvUch/results.md)
also pass at both widths, with exact common parameters and moments. The separate
reader preserves an original serialized-header hash failure without repeating
native work. The [eight-phase pixel integration](../../runs/breakout-pixels-20260920.YrLIfy/results.md)
also passes, including complete six-stream replays, frozen state, valid incomplete
caps and both vocabulary refusals. Main's five replay/schema/test files match the
qualified adapter, with 742 CPU tests passing and no native rebuild. These are
runtime results, not evidence that four actions learn better. The
[paired learning pilot](../../runs/breakout-action-pilot-20260920.kNeotb/declaration.md)
is declared with nine CPU checks and 105 input pins; the
[status dashboard](../../STATUS.md) tracks its active phase and later results.
The September 20 [model-sizing decision](../../runs/model-sizing-20260920.kPIOWC/README.md)
now prioritizes a separately pretrained 5.49M causal-video JEPA frontend. Finish
the active eighteen-action Large quartet unchanged; the explicit
[four-action hold](../../runs/breakout-action-pilot-20260920.kNeotb/a4-train/HOLD.md)
prevents its unstarted arm. This is not a complete two-width pilot or a Tiny
training result. Original pinned declarations and completed evidence are unchanged.
The [CPU readiness review](../../runs/breakout-current-readiness-20260919.ntvDer/README.md)
locates the existing source-matched fixtures and evidence; it is not a GPU result.
Future experiments should use bounded diagnostics and compact records, without
changing existing declarations.

## Qualified runtime

The current backend passes 23 native tests and same-driver complete-state/pixel
checks. Block products improve fixed-recipe N6 throughput **27.3% in both orders**,
retaining all 241 tensors/146 moments, non-timing reports and action traces exactly.
Episode-budgeted evaluation passes six integration phases on the unchanged binary.

Keep the completed [block pixel/timing group](../../runs/current-block-pixels-20260916.M7whE0/),
[episode integration](../../runs/current-block-episode-runtime-20260916.CXHzRj/)
and [source adoption](../../runs/current-block-adoption-20260916.4Smhmw/).
Source cleanup does not qualify a new build or require another backend campaign.

The September 20 final preflight records upstream Meganeura `4a66a276` and Blade
`eaff5092`. Review since `da0e2842` finds direct scalar-gradient broadcasts,
measured/reused readback staging and cached-block/Gemma fixes. LeVJEPA's current
`CachedQueryAttention` is distinct from the repaired split-block path. Do not
rediscover these upstream changes or silently relabel this fixed campaign; an
upstream refresh needs a separate numerical/timing comparison.

## PR verification

The cleanup fixes the missing `safetensors` test dependency rather than skipping
tests. A fresh dependency environment passes all 702 retained Python tests;
both Rust formatting checks pass. [CI run140](https://github.com/kvark/kindle/actions/runs/35490598316)
passes Python bindings/tests, Linux/lavapipe canaries and macOS checks. The learner,
locks, native bindings, active runner and pinned guards are byte-identical to
the pre-cleanup source. Research-history removal changes neither the runtime
nor the campaign's 5,329 verified inputs.

## Archive, not deleted evidence

The complete old history is retained at
[`9ef9a16`](https://github.com/kvark/kindle/tree/9ef9a169d52ed36a912d578a89690f992c5982fc)
and pushed as `archive/dreamer-jepa-pre-cleanup-20260919`. Its
[64 reports](https://github.com/kvark/kindle/tree/9ef9a169d52ed36a912d578a89690f992c5982fc/docs/experiments)
contain the detailed measurements, failed readers, qualification and incident
chronology removed from this changeset. No raw experiment, historical worktree
or package is deleted or relabeled.

Useful entry points:

- [Kickoff, recovered baseline and ablations](https://github.com/kvark/kindle/blob/9ef9a169d52ed36a912d578a89690f992c5982fc/docs/experiments/2026-09-05-kickoff.md).
- [Boxing three-root confirmation](https://github.com/kvark/kindle/blob/9ef9a169d52ed36a912d578a89690f992c5982fc/docs/experiments/2026-09-10-boxing-confirmation.md).
- [Freeway failures and recovered Pong](https://github.com/kvark/kindle/blob/9ef9a169d52ed36a912d578a89690f992c5982fc/docs/experiments/2026-09-11-recovered-confirmations.md).
- [Block correctness and timing](https://github.com/kvark/kindle/blob/9ef9a169d52ed36a912d578a89690f992c5982fc/docs/experiments/2026-09-16-current-block-matmul.md).
- [Prepared Breakout hypothesis](https://github.com/kvark/kindle/blob/9ef9a169d52ed36a912d578a89690f992c5982fc/docs/experiments/2026-09-16-breakout-current-runtime.md).
- [Prepared checkpoint-history/exposure tools](https://github.com/kvark/kindle/blob/9ef9a169d52ed36a912d578a89690f992c5982fc/docs/experiments/2026-09-16-atari-dose-retention.md).

Unused exposure-study/midpoint-retention helpers and tests are removed from the
production tree. Their prepared branches and CPU evidence remain available;
bring back only what a declared experiment needs. Archiving changes no gate.
