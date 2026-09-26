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
[PR status dashboard](https://github.com/kvark/kindle/pull/29) tracks its active phase and later results.
The September 20 [model-sizing decision](../../runs/model-sizing-20260920.kPIOWC/README.md)
now prioritizes a separately pretrained 5.49M causal-video JEPA frontend. The
[eighteen-action Large quartet](../../runs/breakout-action-pilot-20260920.kNeotb/results.md)
is complete: 200,004 actions / 49,652 updates, frozen mean 30.7917 versus .9655,
but two-wall successes 0/24 versus 0/29. Complete state/replay/video and both
zero-update frozen checks pass; the competence gate fails. The explicit
[four-action hold](../../runs/breakout-action-pilot-20260920.kNeotb/a4-train/HOLD.md)
prevents its unstarted arm. This is not a complete two-width pilot or a Tiny
training result. Original pinned declarations and completed evidence are unchanged.
The [CPU readiness review](../../runs/breakout-current-readiness-20260919.ntvDer/README.md)
locates the existing source-matched fixtures and evidence; it is not a GPU result.
Future experiments should use bounded diagnostics and compact records, without
changing existing declarations.

The fresh [Tiny four/eighteen-action comparison](../../runs/breakout-minimal-comparison-20260926.xsQCaK/README.md)
uses the adopted b00ce7be runtime and trained Tiny7fe9, with 200,004 actions /
49,652 updates per arm. [Latest four-action gradients](../../runs/breakout-width-latest-gpu-20260926.VG1oDc/results.md)
and [exact pixel repeat/restore/replays](../../runs/breakout-width-latest-pixels-20260926.idNbRS/results.md)
pass; 34 CPU campaign checks pass. Four-action training started September26
at09:10 UTC. Frozen final/control phases and the fresh eighteen-action arm follow
only after individual review. One paired seed is not three-root reliability;
the old Large four-action hold and two-wall competence gate remain unchanged.

## Compact causal-video encoder

The first **5.49M Tiny** candidate completes native video pretraining: 4,096
updates in 84 minutes, all nine full checkpoints and encoder exports verified.
[Pretraining result](../../runs/levjepa-tiny-pretrain-20260921.JaPZpW/results.md).
Frozen position decoding improves over its own initialization, but motion
readouts have significant outliers; [retain the complete mixed result](../../runs/levjepa-tiny-quality-20260921.ojeZgt/results.md).
[Gameplay-backend inference](../../runs/levjepa-tiny-gameplay-gpu-20260921.X6TnAI/results.md)
and [bounded N6 learning/restore/replay](../../runs/levjepa-tiny-gameplay-pixels-20260921.NQLh0I/results.md)
pass. Large's state and trajectories remain exact. The
[matched-order cost check](../../runs/levjepa-tiny-throughput-20260921.cY1QjK/results.md)
cuts total time 26–27% and observation time 74–75%, with exact same-arm state and
trajectory repeats. The [Breakout package comparison](../../runs/levjepa-tiny-breakout-20260921.ghJPWG/results.md)
regresses: Tiny mean 10.92 versus Large 30.79; neither passes the two-wall gate.
The [own-initial-encoder ablation](../../runs/levjepa-tiny-pretraining-ablation-20260921.lrjxlN/results.md)
shows no pretraining benefit in one paired seed. Tiny remains opt-in.

The [complete Tiny Freeway pilot](../../runs/tiny-freeway-exposure-20260921.ejKgSH/results.md)
passes: 400,008 actions /99,652 updates, 7.969h. Frozen final mean **33.03** with
**36/36** qualifying rounds, midpoint mean **30.50** with **36/36**, restored
untrained-policy mean **0** with **0/36**. Each frozen arm has 75,000 unassisted
actions, zero updates/cutoffs; complete states, all-stream replays and decoded
videos pass. [Final video](../../runs/tiny-freeway-exposure-20260921.ejKgSH/final.mp4),
[control video](../../runs/tiny-freeway-exposure-20260921.ejKgSH/untrained.mp4),
[combined audit](../../runs/tiny-freeway-exposure-20260921.ejKgSH/pair.json).
The subsequent [fresh-root confirmation](../../runs/tiny-freeway-confirmation-20260922.tij9QW/results.md)
now passes all1009/2017/3019 pairs: **108/108** qualifying final rounds versus
**0/108** controls, means32.9167/31.6944/33.25. All midpoints also pass; added
exposure is not shown necessary. The [cross-root certificate](../../runs/tiny-freeway-confirmation-20260922.tij9QW/completed.json)
verifies complete state, replays, videos, distinct parsed initial weights and
the same trained encoder/runtime. Freeway is the **third reliable game** under
the fixed gates, conditional on one pretrained encoder. The interrupted2017
attempt stays separate; its one user-approved replacement is explicit in the
report. All writers are terminal; only unrecorded readers are reusable.
No automatic successor, default adoption or isolated size advantage follows.

The [15-step world-model report](../../runs/tiny-world-horizon15-20260921.bKmiUF/results.md)
retains 16,470 prior forecast targets from four Tiny Breakout matches. Feature
and reward predictions beat simple baselines throughout; continuation does not.
Exact one-step overlap and complete frozen state pass. Recorded future actions
condition the forecasts; this is not imagined-policy validation or a diagnosed
cause of the gameplay regression.

## Qualified runtime

Current production uses native `b00ce7be`, Meganeura `ee3aea42` and Blade
`fbb4f28c`: the [completed four-submission Atari comparison](../../runs/chunks-atari-timing-20260926.e2OEhn/results.md)
cuts wall time 6.5–6.6% in both orders, with exact same-arm state/moments/reports/
trajectories and <0.1% total repeat drift. The exact five-file source delta from
30bfa1a is adopted without rebuilding the qualified package. The earlier
9746c9ac/a761ee5c [correctness refresh](../../runs/meganeura-correctness-refresh-20260924.Be6kq9/README.md)
passes independent primitive/world/behavior references, Tiny/Large frontend
checks, complete state/restore and [matched timing](../../runs/meganeura-correctness-timing-20260924.mYGvjj/results.md):
7.5–7.8% less wall time. The [Qbert backend-only pair](../../runs/correctness-qbert-comparison-20260924.Q4NZyO/results.md)
does not improve learning in seed0: final16/25 pyramids,mean3,414 versus the old
22/24 and4,811.46. Both controls score120.83 with no pyramids. Full state/replays/
videos pass; preserve the separate schema-reader correction and original failure.
The completed [R64/exposure pair](../../runs/qbert-replay-exposure-20260924.AEyDRM/results.md)
keeps that package fixed: matched400k scores1/24 pyramids,mean1,139.58; primary1.6M
scores24/24,mean8,673.96; untrained0/24,mean120.83. Full state/replays/videos pass,
but the15,000 score gate still fails. Training takes9.368h at3.161x aggregate
realtime,not throughput parity. The completed [3.2M exposure-only pair](../../runs/qbert-r64-3m2-20260925.FrriIH/results.md)
keeps that R64 recipe: midpoint24/24 pyramids/mean8,673.96,primary final22/27/
12,595.37,control0/24/120.83. All state/replay/video checks pass,zero frozen
updates/cutoffs. Training takes18.799h at3.151x aggregate/.525x per-stream realtime.
The final fails both fixed thresholds; more exposure does not establish reliability.
Its [retained episode analysis](../../runs/qbert-tail-analysis-20260926.m0P7ZI/results.md)
finds five early failures and six later 8–9k episodes despite a 16,850 median.
This motivates hazard/forecast diagnostics, not dropping episodes or relaxing gates.
The separate [optimizer comparison](../../runs/meganeura-optimizer-20260926.TfWlJI/results.md)
passes thirteen component checks and full pixel/state/restore/default-Large
integration, but [four matched timing windows](../../runs/meganeura-optimizer-timing-20260926.A6u3AI/results.md)
find0dbfcc00 is0.8–0.9% slower; that update alone was not adopted for speed. The
[complete learner trace](../../runs/learner-timeline-20260926.6FJAqf/results.md)
finds62.4% GPU pass coverage (not SM utilization),26.09ms world command recording
and only.44ms optimizer passes per250.86ms update. Exact state/reports pass;
tracing adds0.9%. The [submission comparison](../../runs/world-submissions-20260926.9nJMTy/results.md)
on latestee3aea now completes one/four/four/one windows:7.4–7.5% less core time,
exact full state/moments/reports. Minimal30bfa1a carries one scheduling call plus
dependency pins; its [N6 pixel/restore check](../../runs/world-chunks-gameplay-20260926.q4iNnd/results.md)
matches3,840 training actions/611 updates, complete state and six frozen episodes
exactly. [Latest-runtime compatibility](../../runs/chunks-compatibility-20260926.opVUiI/results.md)
and all four actual Atari timing windows now pass. Preserve the separate Tiny
memory-label reader correction; no native rerun. Candidate aggregate real time
is 1.0405–1.0415×, about .174× per stream: a modest runtime gain, not game competence.
NVML remains disabled; no new long learner root is declared by qualification.
See the [plan](../kindle_single_life_dreamer_plan.md) and PR dashboard for status.

The historical block backend passes 23 native tests and same-driver complete-state/pixel
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

The history-runner implementation at ed16e19 passes
[CI run156](https://github.com/kvark/kindle/actions/runs/35615868079) and all 768
local Python CPU tests, including the 13-line checkpoint-history port. The native
Rust/Cargo inputs are unchanged by that port. Later result/dashboard updates are
documentation-only; consult the [live PR checks](https://github.com/kvark/kindle/pull/29/checks)
for the current documentation head.

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
