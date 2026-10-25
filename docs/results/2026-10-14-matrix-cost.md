# Small and irregular products dominate native matrix cost

October14,2026. [Declaration](../experiments/2026-10-14-matrix-cost.md) ·
[Data](2026-10-14-matrix-cost.json) ·
[Backend plan](../backend_training_performance_plan.md).

**The checked matrix path accounts for2.903ms,63.54% of matrix cost, while
doing only12.01% of nominal matrix arithmetic.** The195 aligned/tiled products
account for1.666ms despite doing the other87.99%. All716 products already use
BF16 operands and multiplication; this is not missing precision coverage.

The recurrent world model supplies2.568ms of the checked path. Its eight-row
products and weight gradients with an eight-element reduction are now a
concentrated target for matching JAX's actual algorithms. Merely counting MACs
or dispatches hid this distinction. It does not yet identify which checked
subtype causes how much cost, prove a particular replacement faster, or explain
the entire remaining gap.

No new ordinary whole-update timing: current isolated RGB remains24.340ms
versus retained complete JAX12.836ms,1.90×. Production and Atari stay unchanged.

## Fixed-checkpoint attribution on the current kernels

Same RGB12M/B8/T16/H15/BF16, incremental convolution addresses and retained
Blade command-cache implementation. One original update initializes state;
optimizer/accumulation/clipping are then disabled. Four masks toggle checked
and aligned matrix workgroups independently. Every other operation, binding,
command and dependency remains. Masked outputs are never learner results.

Three rotated rounds, two warmups+16 measurements per mask/session/round;
64 full baseline measurements before/after. Full work is restored before each
mask and after every round. Command caching is explicitly off for diagnostic
recording; this measures GPU pass intervals, not cached whole-agent wall time.

| Session | Checked products | Aligned products | Checked allocation | Aligned allocation |
| --- | ---: | ---: | ---: | ---: |
| World model, including posterior prefix |470|21|2.568ms|.113ms|
| Imagination |33|155|.155ms|1.222ms|
| Behavior gradients |17|16|.168ms|.267ms|
| Slow value |1|3|.012ms|.064ms|
| Total |521|195|2.903ms|1.666ms|

The full fixed graph brackets at19.910ms; its full-mask mean is19.911ms.
Non-matrix work plus all unchanged commands accounts for15.342ms. These are
two-family Shapley allocations, including overlap/cache interactions, not
pure kernel durations or independently additive possible gains. This conditional
matrix allocation4.568ms is not identical to the preceding three-family4.475ms
study. The retained JAX matrix kernel sum2.366ms also has a different boundary;
do not subtract it from this table as a verified attainable saving.

The checked inventory contains302 products with eight output rows,160 additional
products with reduction length eight, and59 other irregular cases. Examples
in the world model:

- 32 dense8×2048 by2048×256 products, BF16 operands/F32 results.
- 30 accumulated2048×8 by8×256 weight-gradient products, F32 addend/results.
- 16 eight-group1024×8 by8×256 weight-gradient products.
- 16 eight-group8×256 by256×768 forward products, BF16 results.

These counts describe actual compiled work, not independently measured time per
shape. Shapes alone do not imply equivalent JAX layout, accumulation, addend,
output precision, library algorithm or execution overhead.

A read-only check of the retained full reachable JAX HLO narrows the contract:
96 reduction-eight cuBLAS calls return F32 and use `beta=1` with an F32 addend;
their two product operands are BF16. Another16 return BF16 with `beta=0`.
The F32 cases include2048×8 by8×256 and the eight-group1024×8 by8×256 products.
Thus these gradient counterparts do not avoid our F32 accumulation/result
requirement. Raw HLO lines/configurations are retained in `contracts.json`,
SHA2561f51f0c5c233068bc0215cbc32a10fc070660cb0afb059d17654b0d02236b52c.
This verifies library contracts, not value parity, per-shape timing or identical
native/JAX layouts/invocation counts. A grouped control must include the addend.

## Exactness and scope

All current dispatches/pipeline descriptors, first report/initial-first states,
buffers/groups and restored output fingerprints pass the independent CPU audit.
All292 tensors are exact at initial/before/after boundaries, including comparison
with the retained current candidate; all four sessions reproduce their retained
fixed outputs after every round. Every expected measured sample appears once.

One passing GPU guard, zero allocation warnings, one synthetic update,
1,468 optimizer-free session replays, zero actions. Minimum sampled Vulkan
estimated headroom15,096,152,064bytes. Two explicit classifier CPU tests,
formatting/release Clippy and publication review pass. No new shader or logged
validation qualification is claimed by this unchanged-runtime timing fixture.
No worker, recovery, NVML polling or SM-utilization measurement.

Native1339b0b7d6c73ee4d3f4929c6ce4a780f645675f1639ea5bc5f31a06bda69807.
Exact backend source remains the [iterator candidate](2026-10-14-convolution-iterators.md),
Blade96c1dc7/Naga a97559b. New Kindle code is only the bounded diagnostic.
Source, locks, declarations, inventories, all raw samples and audits:
`runs/matrix-cost-20261014.5lrQ8LIi`.

## Native executable inspection is unavailable through the standard API

The separate [compile-only probe](../experiments/2026-10-14-native-executable.md)
enables the advertised pipeline-executable feature and required capture flag.
The5080 driver returns one compute executable and statistics, but **zero
internal representations** for a simple counter shader. The specification
permits implementation-dependent text/binary representations; availability is
not guaranteed. [API contract](https://docs.vulkan.org/refpages/latest/refpages/source/vkGetPipelineExecutableInternalRepresentationsKHR.html).

Close this route without a heavier follow-up or driver change. The returned
local-memory statistic is68,719,476,736bytes even for the tiny shader; retain it
without interpreting it as actual spilling or memory use. No native assembly
or dynamic instruction count was obtained. CUDA translations remain proxies.

This probe passes logged validation and one host guard, zero warnings or
dispatches/actions/updates, sampled headroom16,568,483,840bytes. Binary7047b47f,
isolated Blade source/metadata retained under
`runs/native-code-20261014.I4Ik2df6`; no GPU or build failure occurred. Combined
studies total two passing guards and zero new allocation warnings.

## Next decision

Match the representative small-row and short-reduction dense/grouped products
to a controlled library fixture, using the full-executable contracts above. Verify input/output
types, transpose, addend, grouping and selected algorithm; previous convolution
F32-output controls accidentally widened operands and are not the template.
Include independent F64 values and execution overhead in the comparison.

Then choose between a general small-product algorithm and compiler-level
aggregation across repeated parameter uses, based on measured costs. Do not
assume that the aligned tile, another split count, a larger training batch or
the rejected scalar RGB-convolution alternative solves this problem. No
optimization or learning allocation follows automatically; the full goal stays
open and the other execution/host/kernel costs remain in scope.
