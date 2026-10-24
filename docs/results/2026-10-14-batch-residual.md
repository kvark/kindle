# Larger batches expose image-training dataflow costs

October14,2026. [Declaration](../experiments/2026-10-14-batch-residual.md) ·
[Data](2026-10-14-batch-residual.json) · [Backend plan](../backend_training_performance_plan.md).

**Convolution and other-than-matrix work account for89.0% of the increase
from the retained B8 fixed graph to B32.** The empty command stream does not
grow. This directs the investigation toward connected image-training work,
not another matrix-tile, launch-grid or row-only fusion trial.

This is attribution, not a new throughput result. Ordinary complete RGB B32
remains68.344ms native versus retained27.187ms JAX,2.51×. B8 remains
22.780/12.737ms,1.79×. The preceding packing intervention saves only1.49%
and stays unpromoted. Production and Atari remain unchanged.

## Locate the growing cost

The already-qualified cached family diagnostic now uses the retained B32
configuration, including its previously reviewed128MiB logical scratch cap.
No shader, precision, optimizer or dependency changes. Four fixed sessions,
all eight work subsets and three rotated orders preserve bindings/barriers.
Full outputs are restored after every round; optimizers are disabled.

| Allocated component | Retained B8 | B32 | Scaling |
| --- | ---: | ---: | ---: |
| Empty command stream |4.295ms|4.102ms|0.96×|
| Convolution |5.587ms|21.858ms|3.91×|
| Matrix |3.388ms|7.543ms|2.23×|
| Other operations |5.463ms|21.325ms|3.90×|
| Full fixed graph |18.733ms|54.828ms|2.93×|

Bracketing B32 full replays average54.824ms. These are interacting
counterfactual allocations, not pure kernel durations, removable savings,
SM utilization or independently additive costs. B8 is retained evidence,
not a fresh paired timing. Tiny signed allocations for absent operations
remain as measurement noise, not clamped values.

World training supplies15.922ms of the21.325ms other-operation allocation.
A separately declared world-only follow-up keeps heavy work active and
uses the existing explicit, independently checked classifier:

| World component | Dispatches | Allocated time |
| --- | ---: | ---: |
| Heavy work plus command stream |514 heavy|26.902ms|
| Layout/cast materialization |1,329|4.442ms|
| Row/reduction |1,154|5.379ms|
| Pointwise/image other |1,104|5.893ms|
| Full fixed world |4,101|42.616ms|

Bracketing world replays average42.595ms versus42.647ms in the first
allocation. These subclasses include all world shapes, not just CNN work.

## A concrete connected-image difference

CPU analysis follows actual buffer dependencies in the native plan and
root-reachable instructions in the compiled B32 JAX executable. Native's
first convolution really outputs BF16; its diagnostic label incorrectly
says otherwise, so the actual shader descriptor determines the contract.

Before the next convolution, native materializes:

| Result | Logical output extent |
| --- | ---: |
| F32 channel bias addition |256MiB|
| F32 max pooling |64MiB|
| F32 NCHW→NHWC transpose |64MiB|
| F32 RMS normalization |64MiB|
| F32 SiLU |64MiB|
| F32 NHWC→NCHW transpose |64MiB|
| BF16 conversion |32MiB|
| Packed channels-last conversion |32MiB|

The sum is640MiB of logical output extents, **not peak VRAM, measured DRAM
traffic or a forecast of removable traffic**. Saved forward values have real
backward consumers: pool derivatives, normalization/SiLU derivatives, bias
and scale reductions, casts and convolution weight gradients.

JAX's actual first convolution combines bias and produces BF16 channels-last
storage. A reachable multi-output fusion combines pooling and normalization
statistics; activation/scale output feeds the next convolution directly in
channels-last order. JAX still has separate pooling masks, reciprocal-root,
activation and backward work. It is not one giant fused training kernel.
The two paths also differ in rounding/gradient contracts; this inspection
does not establish value/derivative equality or isolated block speed.

This strengthens the compiler/dataflow hypothesis beyond counting dispatches.
The earlier wider-JAX precision control retained BF16 convolution islands and
XLA fusion: its6.19% effect does not establish that native's materialized image
pipeline is cheap. No new global dtype sweep follows.

## Decision and checks

Next compare a complete staged image-training block, including live outputs,
forward tapes, late-arriving gradients, actual layouts and rounding. Separate
compiler/layout cost from heavy-kernel algorithm cost. Prefer a general typed
lowering using existing compiler machinery; do not assume a new NHWC API or
BF16-only engine is necessary. Any proposed block improvement must survive a
complete B32 learner timing before promotion. No automatic fusion/geometry
sweep, game allocation or claim that this alone closes the remaining gap.

Both GPU guards pass: two preparatory synthetic updates,2,975 optimizer-free
replays,1,920 measured cached calls, zero game actions. Every measured call
verifies cache reuse. Both allocations reproduce all five complete plans/
pipeline lists, first outputs/report and initial/before/after292 tensors
exactly against the ordinary B32 control. Original dependencies and restored
world output fingerprints also match. One allocation warning is retained;
zero API/numerical/hard faults, recovery, NVML polling or unfinished workers.
Minimum sampled Vulkan estimated budget headroom is12,684,099,584bytes,
not physical free or peak VRAM.

Formatting, release Clippy, two classifier tests and automated CPU audits
pass. Three CPU preparation failures remain: an unapplied dependency-path
edit, a Python delimiter error, and a source-normalizer mismatch caused by
rustfmt's log wrapping. Each was corrected before its GPU allocation; no
numerical gate, shader, replay loop or acceptance tolerance was relaxed.

Artifacts: `runs/batch-residual-20261014.b7KtbHQM`; family/subclass binaries
`571edbdf`/`52597d90`, immutable cached Meganeura056562b plus qualified patches,
Blade96c1dc7 plus cache timestamps, Naga a97559b. Ordinary B32 remains
`e5d890dd`. A16:18UTC upstream recheck finds Meganeura99e3de2's Xe2
rectangular BF16 capability support; its selection still prefers the existing
16×16×16 shape on this5080. This is not an identified image-dataflow fix or
new GPU qualification. Bladee7c3f7b is unchanged; no repin. CI400 on preceding
3ce1dcb passes. The broader root-cause/performance goal stays active.
