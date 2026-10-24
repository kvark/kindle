# Convolution needs separate arithmetic and data-feeding contracts

October15,2026. [Data](2026-10-15-convolution-contract.json) ·
[Next bounded transfer test](../experiments/2026-10-15-convolution-transfer.md).

**The remaining gap is not just arithmetic throughput.** The retained library
uses several role-specific reuse/staging schedules; our shared arithmetic
generator does not express the same data-feeding contract. This CPU audit
also reproduces a concrete Naga limitation: its cooperative load infers the
matrix component type from the memory pointer, discarding the requested
floating-point type. That prevents direct matrix loads from packed `vec4<u32>`
storage, although equivalent Vulkan SPIR-V validates for F32/F16/BF16.

This identifies a missing lowering contract, **not its performance benefit**.
Complete isolated RGB12M/B32 remains54.224ms versus JAX26.632ms,2.04×;
no new GPU run, production change or Atari restart occurs here.

## Reconcile every actual convolution

The audit matches all23 native products to actual full-JAX HLO, NVTX thunks
and CUDA kernels across eight captured updates: eight forward, seven input
gradients and eight weight gradients. Native parameter/gradient-buffer paths
confirm layer identity, including identical-shaped encoder/decoder layers.
Posterior/world plans are one byte-identical staged session, counted once.
All B512/K5/stride1/pad2 shapes match. Library products output BF16; native
forward outputs BF16 and derivatives F32. Output rounding remains distinct.

| Contract | Native | Actual library |
| --- | --- | --- |
| Forward/dX output reuse |32×32 checked or64×64 packed|Mostly128/256 spatial positions ×32/64 channels; special three-channel forward|
| dW output reuse |32×32 checked or64×64 packed|32×128,64×64 or64×256|
| Explicit/shared staging |One shared tile;128 threads|Named three-, four- or five-stage schedules; mostly128 threads|
| Packed operand expression |`vec4<bf16>`:8bytes|16-byte aligned accesses in CUTLASS contracts; retained dX/dW SASS has128-bit async copies|
| Layout boundary |NCHW/CoCiKhKw;14 products additionally pack channels|NHWC/KRSC at all23 HLO convolution boundaries|
| dW accumulation |256-position compensated blocks;4,096 positions per partial here|Different split schedules; do not infer accumulator precision from BF16 outputs|

The full data file preserves every library tile, stage count, grid, shared
allocation and register count. All23 main kernels plus18 helpers total
11.255ms/update in the retained trace, of which10.940ms is main kernels.
These are instrumented intervals, not ordinary full-update latency or an
additive savings budget. Native instruction counts remain unavailable.
Library register counts up to255 do not justify targeting high occupancy.

## Why the fixed spatial family was not equivalent

The [rejected replacement](2026-10-15-convolution-family.md) covered forward/dX,
not dW, with one fixed spatial×32-channel schedule and compiler-managed matrix2
staging. It did not reproduce the library's pass-specific scheduling.

For CNN0 forward and image-output dX, three reduction channels became32 at
each of25 taps:800 physical reduction positions versus75 useful. The original
checked loop rounds the flat75 positions to96 instead. The replacement
therefore computes8.33× the original padded tile MACs on these cases; measured
complete-operation regressions are461% and342%. This is a source-level work
estimate, not a causal time partition. Conversely, aligned CNN1/CNN2 forward
retain exactly the same padded MAC count and still regress90%/84%: padding is
not the whole explanation.

Across all15 occurrences, replacement padded tile work is335.544G versus
228.170G,47.06% more. Across all23 original native products, nominal226.492G
becomes369.099G after actual tile/channel padding,62.96% more. Neither number
is a native/JAX issued-instruction ratio. The earlier1.40% nominal-heavy-work
comparison excludes this padding. Retained library counters actually execute
more tensor instructions for the two measured derivatives while finishing
faster; minimizing nominal/tiled MACs alone is not the solution.

The first audit incorrectly treated packed48-channel K as flat1,200 rather
than per-tap-padded1,600. Its source/result remain; the reviewed second audit
uses actual generated grids and packed source dimensions for every operation.
No GPU result or sample is changed. The nonstationary native dX samples in the
family report remain unfiltered and unexplained.

## A specific legal contract missing from Naga

The cooperative-matrix SPIR-V specification permits a pointer component type
different from the matrix component type, explicitly supporting wider packed
memory transfers. Stride is measured in pointed-to elements, not BF16 values.
[Specification, issue8](https://github.khronos.org/SPIRV-Registry/extensions/KHR/SPV_KHR_cooperative_matrix.html).

Six CPU cases against the exact linked Naga`a97559b` library establish:

- Typed `vec4<f32/f16/bf16>` loads parse and validate.
- Requesting any of those matrix types from `vec4<u32>` produces a
  `coop_mat16x16<u32,A>` instead and fails the explicit result-type check.
- Equivalent GLSL for all three types compiles and validates as Vulkan1.3
  SPIR-V. Binary inspection confirms a floating-point cooperative result and
  a Workgroup pointer to `vec4<u32>`. No module is submitted to a device.

Source tracing locates the mismatch: WGSL lowering retains dimensions/role
but not the requested scalar; `CooperativeLoad` IR has no scalar field and
the typifier recovers it from the pointer. This is separate from the
[unavailable Kernel-only async-copy opcode](2026-10-15-categorical-gpu.md#what-remains-fundamental).
Wide transfers are legal; exposing OpenCL's async-copy spelling is not the fix.
Neither source vector width nor valid SPIR-V proves emitted128-bit instructions,
asynchronous copies or a speedup on this driver.

## Next: prove transfer width before compiler changes

Use one shared typed, fixed-geometry convolution implementation as an isolated
Vulkan control: reproduce the current14 packed occurrences, then change only
operand transfer representation from8-byte typed vectors to16-byte bit-pattern
vectors. Retain64×64/K32, grids/splits, layouts, compensation, output precision,
all transforms and reduction costs. No three-channel padding change, larger
tile selection, async claim or resurrected matrix2 family. Require an actual
native timing/numerical control before interpreting the variant.

Only a meaningful complete-operation benefit would justify the minimal Naga
change retaining the requested matrix component type independently of its
storage pointer, followed by native whole-update validation. No Naga edits or
commits have occurred; committing there still requires user permission.
Larger role-specific reuse, multistage feeding, dependency execution and host
targets remain separate unresolved mechanisms. The root performance goal is open.

Artifacts: `runs/convolution-contract-20261015.rRE9xjXd`. One-CPU/2GiB/no-swap
audit/probes finish; zero GPU calls/actions/updates. Preserve the initial
reserved-WGSL-keyword probe failure and corrected version separately. Upstream
Meganeura99e3de27/Bladee7c3f7b is unchanged. CI418 passes preceding head1f0d6d4;
the PR dashboard carries subsequent CI status. No neighboring work changes.
