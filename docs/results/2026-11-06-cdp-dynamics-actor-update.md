# Added dynamics loss barely changes the retained actor update on two batches

The [archived-batch diagnostic](../experiments/2026-11-06-cdp-dynamics-actor-update.md)
finds that adding the weighted dynamics.5 term changes the full estimated actor
update by **0.01136% /0.01702%** of its baseline norm. The directions remain
almost identical. This limits a large immediate update-change explanation at
the retained200k checkpoint; it does not explain the failed learner's earlier
trajectory. Keep actor dynamics off and retain143.667 greedy/70.750 sampled.

Both batches reuse the completed November3 inputs and baseline derivatives,
with the same retained weights and optimizer moments. No old capture, component
backward stage, optimizer reader, learner or collector is repeated. The new
native probe reconstructs imagination and computes only the combined gradient
and the weighted dynamics gradient. The subsequent CPU analysis estimates two
independent counterfactual updates; neither is applied or chained.

## Actor results

| Measurement | Batch0 | Batch1 |
|---|---:|---:|
| Baseline raw actor gradient norm |305.301907|251.756398|
| Weighted dynamics raw gradient norm |.065045|.085575|
| Dynamics / score-function raw cosine |−.257018|.123861|
| Baseline full estimated update norm |.0194097785|.0182298873|
| Combined full estimated update norm |.0194096442|.0182300316|
| Combined versus baseline update cosine |.9999999936|.9999999856|
| Update difference norm |2.20466e-6|3.10243e-6|
| Difference / baseline update norm |.0113585%|.0170184%|
| Conditional dynamics contribution norm |2.31141e-6|3.15032e-6|
| Conditional score-function contribution norm |1.47523e-6|9.46266e-7|
| Conditional imitation contribution norm |.011053520|.010340310|
| Conditional imitation / dynamics norm |4782.146|3282.309|
| Saved-history contribution norm |.015339660|.015339660|
| Clipped actor parameters, baseline→combined |11→11 of11|10→10 of11|
| Maximum absolute change in AGC multiplier |5.30754e-5|5.06488e-5|

The new term is larger than the score-function contribution on these batches,
but much smaller than imitation and saved momentum. Component transforms hold
the combined total's clipping and variance denominator fixed. Whole baseline
and combined counterfactual updates each recompute their own clipping and
variance. Component norms are not additive shares or independent updates.
All original per-parameter and aggregate baseline update norms reproduce the
archived optimizer report at rtol1e-9/atol1e-12.

The critic's new auxiliary derivative is exactly zero. Its combined raw
gradient and estimated update are exactly the baseline arrays for both batches.
Actor combined gradients match archived baseline plus new dynamics with relative
L2 errors2.555e-8/2.549e-8. Weighted dynamics losses are−.15879649/−.14776047;
combined losses80.877052/101.275475 agree with the scalar sum within2.12e-8
normalized error. No loss magnitude is treated as evidence of gameplay benefit.

## Native qualification and exact reconstruction

Implementation/source `c041afeab9a7589c10b8f698b66064ddc4dbaa48` passes192 Kindle
and5 adapter CPU tests, strict Clippy and release packaging. The two new CPU
checks construct the gated production graph, verify its unchanged prefix and
parameter ownership, and check arrival rewards/bonuses and time-major targets.
Existing auxiliary F64 qualification is reused without repeating its GPU jobs.

The original ChaCha imagination stream is reconstructed from the checkpoint's
seed and counters, continuing across both batches. All sampled actions,
9,830,400 imagined-feature values per batch, weighted advantages and trajectory
weights match the archived arrays **exactly**. Original GPU categorical means
and percentile updates give return scales35.9654083/35.6867256.

The new job completes under the existing host guard: native measurement.369s,
test2.42s, controller/audit3.986s, complete **07:38:44.292 UTC November6**.
All419 pins,22 unchanged behavior parameters and32 unchanged world snapshot
inputs pass. Minimum native budget headroom is19,413,241,856bytes; this is not
independent GPU telemetry. Zero Atari actions, learner updates and optimizer
steps. The native stage succeeds on its original execution.

Executable SHA256
`51c5332c8b8b5953877867ed757a91225714c0f6c4865ea3572c5063c56c3430`.
Root `/x/Temp/kindle-dynamics-actor-update-20261106` retains the executable,
full new derivatives, shared draws, snapshot inputs, reconstructed targets,
before/after weights and exact declarations.

## Preserved CPU failure and separate repair

The first CPU analysis attempt fails before estimating either batch: its
exclusive declaration filename collides with the wrapper's declaration.
The original script, declaration, log and failed receipt remain unchanged.
It creates no analysis output directory and performs no native or GPU work.

The [separate CPU repair](../experiments/2026-11-06-cdp-dynamics-update-analysis-repair.md)
at source `b11b8953` changes only import lookup and declaration/output locations.
All four calculation functions are AST-identical; both new CPU fixtures pass.
The repaired analysis completes in.698s at **07:41:51.891 UTC**, with454 unchanged
pins. Original baseline optimizer arithmetic is reused; the previous native
optimizer comparison failure and its limits remain retained. This repair does
not relabel the original analysis stage as passed.

Independent closure recounts the primary differences/cosines from saved F64
update vectors, verifies exact critic invariance,454 pins and the preserved
failure, then seals92 artifact files in.731s at **07:43:01.701 UTC**. No Kindle
service or render-device holder remains. The4,688,112,504-logical-byte disposable
build cache is released; the packaged executable remains byte-identical.

## Limits and decision

These are two selected final-checkpoint batches and F64 conditional optimizer
estimates, not native parameter writes, early-training measurements, the failed
candidate's own optimizer trajectory or independent-seed evidence. Small late
differences can still precede different action choices and later experience.
The result does not identify the cause of the full dynamics.5 regression.

Together with the [fixed-plan probe](2026-11-06-cdp-breakout-action-path-derivatives.md),
this weakens two simple explanations at these measured states: generally
reversed return directions and a large immediate auxiliary update. It does not
justify a coefficient sweep, a categorical-representation replacement or a new
learner by itself. The raw/conditional imbalance is not an instruction to raise
the auxiliary weight; prior actor-weight and imitation-decay learners regressed.

The diagnostic is closed with the original analysis failure and separate
successful repair. Keep the extra actor loss off. No successor allocation is
selected; original864/90%-two-wall and the five-game goal remain unfinished.
