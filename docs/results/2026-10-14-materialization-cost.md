# Copies alone are not the missing compiler improvement

October14,2026. [Declaration](../experiments/2026-10-14-materialization-cost.md)
· [Data](2026-10-14-materialization-cost.json)
· [Comparable-work diagnosis](2026-10-14-executable-work.md).

**Fixed-graph cost allocation:1.432ms layout/cast materialization,2.329ms
row/reduction work,1.631ms other operations.** These are the three portions
above15.778ms for heavy kernels plus the unchanged command stream. The
complete fixed graph takes21.171ms in this diagnostic, not a new ordinary
learner time. Retained complete native/JAX28.213/12.836ms remains2.20×.

This closes the attribution question without implementing another optimization.
Copies alone do not explain the gap. Row/reduction and world backward/scalar
work are the larger non-heavy region, alongside the separately demonstrated
heavy-kernel and dependency-execution problems.

| Actual family | Dispatches | Cost allocation | Three-order range |
| --- | ---: | ---: | ---: |
| Layout/cast/broadcast materialization |1,744|1.432ms|1.431–1.433ms|
| Row/reduction, including fused scalar work |1,557|2.329ms|2.325–2.331ms|
| Other pointwise/SiLU/image work |1,246|1.631ms|1.626–1.635ms|
| Heavy work plus retained command stream |739 heavy|15.778ms|15.774–15.781ms|

Rows include matrix split-partial reductions, not just normalization. The last
row still issues every suppressed command and dependency: **it is not a lower
bound for an optimized program**. Removing real intermediate operations may
also change dependencies, overlap, register pressure and memory use. The
allocation averages interactions across all eight subsets; it is neither pure
kernel duration nor independently removable/additive savings. The three-order
ranges are not confidence intervals.

The full world graph contributes1.612ms row/reduction and1.456ms other work.
Imagination materialization contributes.684ms versus world's.655ms, despite
only366 versus1,333 dispatches. Thus dispatch count is a poor priority rule.
Existing time stacking/splitting already uses balanced trees; do not rediscover
or propose that as a new fix. Summed binding extents remain in the CPU preflight,
explicitly not measured DRAM traffic or effective bandwidth.

## Controls and limits

Same staged RGB12M/B8/T16/BF16 graphs and original first update. All matrix and
convolution work stays enabled. Selected other dispatches get zero workgroups,
retaining bindings, pipelines, group membership and dependencies. Optimizers,
gradient accumulation and clipping are disabled before the diagnostic; invalid
masked outputs never train an actor.

Three rotated orders, eight subsets, two warmup/16 measured calls/subset/session;
64 full calls before and after, plus warmups/restoration. Bracketing unmodified
GPU time21.182ms; mask7 full work21.171ms versus retained21.179ms. No speedup
claim. All four sessions restore their exact retained output fingerprints after
every round. Initial/before/after292 tensors, first report/config, every actual
dispatch/buffer/output, pipeline key and runtime group match the control.
Both Rust classifier tests and release Clippy pass; independent CPU
classification covers all5,286 dispatches and rejects unknown families.

One passing guard, zero allocation warnings/faults/recovery/NVML polling;
one original synthetic update,2,380 optimizer-free session replays, zero game
actions. CPU result review passes; no worker remains. No new JAX capture,
numerical-contract relaxation, changed training defaults or installed runtime.
Inherited compiler qualification failures remain separate.

Artifacts: `runs/materialization-cost-20261014.yLfoQTsc`. Complete sources,
locks, exact binarya7c6e45e, all samples/fingerprints/metadata and audits remain.
Latest Meganeura2214412 is unchanged; newly reviewed Bladee7c3f7b/Nagacbbac959
adds Metal BF16 only. No Vulkan/SPIR-V change was found; this diagnostic keeps
qualified96c1dc7/a97559b, not a combined compiler refresh.

## Engineering consequence

Do not turn the demonstrated wide-head concat into another unweighted local
optimization. The next compiler design should represent a **connected row/
backward region with explicit layouts, rounding and multiple live outputs**,
so it can remove both temporary buffers and dependency boundaries. Cost and
validate a representative region before generalizing; preserve reductions and
saved values required by backward. Merely increasing a scalar fusion cap or
collecting independent GEMMs into one dispatch is not that capability.

This still leaves the heavy-kernel algorithm gap and host command construction.
Efficient shared typed matrix/convolution algorithms and coarser execution are
both needed; no individual remaining knob has evidence for a2× learner gain.
No new optimization or learning allocation follows automatically from this
attribution. The broader performance goal remains open.
