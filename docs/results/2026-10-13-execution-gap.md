# The JAX gap is an execution-design gap

October 13, 2026. [Declaration](../experiments/2026-10-13-execution-gap.md)
· [Data](2026-10-13-execution-gap.json)
· [Implementation priorities](../backend_training_performance_plan.md).

**The dominant issue is the training implementation, not Dreamer's model size
or a blanket Vulkan penalty.** Native RGB12M/B8 remains approximately53ms
versus JAX12.42ms. We have basic cooperative kernels and several independently
executed learner sessions; JAX combines training-oriented library algorithms
with a compiled learner. Adding a dtype or changing one tile does not supply
that execution machinery. The earlier operation-group accounting located the
cost, but calling it complete root-cause closure overstated what it established.

This investigation separates precision coverage, command scheduling,
kernel algorithms and compiler/API effects. It does **not** deliver JAX speed
parity, a production optimization or game-quality evidence. Production pins
and training defaults remain unchanged; no Atari work resumes.

**Follow-up:** the [shared tiled prototype](2026-10-13-tiled-gap.md) reduces a
real transposed product168.487→57.747µs, but complete RGB only52.987→51.616ms
and CDP126.203→118.089ms. All reports/tensors remain exact; world training is
unchanged. This validates one useful algorithm path, not whole-gap closure.
The active investigation next targets convolution-backward/grouped coverage.
The [subsequent isolated backward test](2026-10-13-conv-backward-gap.md) reaches
48.90ms (8.01% less) with90.51% nominal matrix/convolution BF16 coverage, still
3.94x JAX. This strengthens the algorithm/dataflow diagnosis; coverage is not
efficient execution. The numerical failure/correction and changed trajectories
remain explicit. Next inspect convolution execution counters, not more dtype
flags or an additive forecast from separate prototypes.
The [counter-guided follow-up](2026-10-13-conv-kernel-cause.md) demonstrates
layout/iterator and data-reuse causes directly: two controlled CUDA-twin pairs
reach629→343µs, then348→211µs, including packing and exact outputs. Executed
instructions fall84% and tensor activity rises while occupancy falls. Native
integration/full updates, not more microbenchmarks, are now the next proof.

## 1. The same simple kernel stays slow under CUDA

Translate our checked cooperative loop into standalone CUDA: same32×32 output
tile, K32 staging, four16×16 warp products, shared arrays, two barriers/stage,
coalesced transposed loads, split boundaries and resident BF16 operands.
Both produce F32 outputs. This is a diagnostic, not a new Kindle backend.

| Actual workload shape | Native Vulkan | Same algorithm, CUDA | JAX/cuBLAS |
| --- | ---: | ---: | ---: |
| Dense8×256×2048, eight splits |10.705µs|7.272µs|3.397µs|
| Dense128×256×2560, four splits |18.699µs|17.492µs|4.849µs|
| Transposed2560×256×1920 |168.380µs|178.720µs|70.481µs|
| Forward convolution, N128/C32→48/HW32/5×5 |667.699µs|625.669µs|not precision-matched|

**6,981,632 complete outputs match exactly across Vulkan/CUDA; all512
independent F64 probes pass.** The three dense library cases also match every
output. The two substantial matrices and convolution differ by roughly−6%
to+7% between the two implementations of our algorithm, not several-fold.
The smallest case retains a3.43µs boundary/implementation difference.

Native numbers are128-call GPU pass intervals; CUDA uses an uploaded graph
batch and event timing; library numbers sum every captured kernel, including
split reductions. Launch/barrier boundaries differ, particularly for small
cases. CUDA12.4 emits compute80 PTX, JIT-compiled by the580 driver; this is
not identical machine code or a universal API-equivalence claim.

The library's captured choices are materially different:16×16/K64/two-stage
for the smallest case,64×64/K32/six-stage for the wider one, and a32×32/K64
transposed WMMA kernel. Merely widening our K stage is not software pipelining.
The direct CUDA translation reproduces our substantial-kernel slowness;
using CUDA syntax alone does not reproduce cuBLAS's algorithms.

The attempted F32-output convolution library control is retained separately:
JAX promotes its BF16 operands to F32 before cuDNN and takes1.461ms. It is
**not** a BF16-library comparison. Do not substitute the earlier BF16-output
0.139ms result into this table or quietly rerun a more favorable output type.

## 2. BF16 coverage is still incomplete, but is not the whole explanation

The retained complete RGB graph contains95.07G nominal matrix/convolution
MACs. **48.76% remain scalar F32**: convolution derivatives37.33G and grouped
RSSM products9.03G. BF16 dense/forward-convolution work is48.72G. These are
source-level arithmetic counts, including convolution padding and duplicate
posterior work—not issued FLOPs, measured memory traffic or optimizer totals.

On unchanged upstream Dreamer/JAX, independently control the two missing
precision regions and CUDA command buffers:

| Complete RGB12M/B8 update | Ordinary time |
| --- | ---: |
| JAX BF16 control |12.417ms|
| F32 convolution derivatives |17.091ms|
| F32 grouped products |14.596ms|
| Both F32 regions |19.236ms|
| BF16, command buffers disabled |16.163ms|
| Both F32 regions, command buffers disabled |22.775ms|
| Unchanged native, refreshed compiler |52.980ms|

Each arm includes16 warmup+256 measured updates. All288 initial JAX tensors
match. Disabling command buffers preserves **all272 reports and288 final
tensors**, both with BF16 and with the combined F32 intervention. CPU F64
tests check982 custom forward/derivative values; HLO confirms the intended
precision/capture changes. Precision interventions change later stochastic
trajectories and are not exact native-arithmetic or learning-parity controls.

**Even handicapping those JAX mechanisms together leaves a30.20ms difference.**
This is a controlled residual, not an additive allocation of native savings:
graphs, layouts and algorithms still differ. Neither BF16 coverage nor launch
overhead alone explains our result.

## 3. The missing machinery is visible in source and captures

- **Kernel-family coverage:** the new shared typed path selects the checked
  32×32 loop. Meganeura's larger vector-loaded, padded, prefetched cooperative
  implementation is F32-specific; the5080 exposes no native-F32 cooperative
  shape for that path. Supporting a dtype did not generalize the optimized
  algorithm family. Grouped products still use scalar F32 tiles. This is why
  type-agnostic generation must include efficient layouts/scheduling, not just
  substitution of scalar names in the simplest template.
- **Convolution algorithm/layout:** native uses direct implicit GEMM and NCHW;
  JAX/cuDNN chooses channel-last library kernels and, in the F32-derivative
  control, Winograd transforms and complex GEMMs. The new CUDA capture confirms
  these kernels; legacy backward-input algorithm5 is Winograd in the installed
  cuDNN header. Equal F32 dtype does not imply equal arithmetic work.
  [cuDNN's algorithm documentation](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.20.0/api/cudnn-ops-library.html)
  describes the transformed alternative. This does not establish Winograd as
  our preferred implementation; efficient direct tensor kernels are also missing.
- **Whole-learner dataflow:** Kindle separately executes posterior, imagination,
  slow value, world gradient and behavior gradient sessions, with host RNG,
  targets and EMA. Encoder/posterior work is recomputed for differentiation;
  JAX reuses its forward outputs. The current native extra posterior stage is
  about6.3ms; the prior marked control measured CPU preparation2.33ms,
  targets1.61ms and slow-critic synchronization about1.04ms. These are different
  fixtures/nested boundaries, not independently removable amounts.
- **Materialization and scheduling:** thousands of small layout, cast,
  normalization and pointwise dispatches remain. The retained zero-work
  command stream costs5.89ms. Its accounting includes barriers/issue overhead,
  not5.89ms of removable pure synchronization. A single submission previously
  regressed; preserve the useful four-chunk schedule until dataflow improves.

The [prior matched controls](2026-10-12-jax-gap-accounting.md) already rule out
model-size mismatch and show only0.262ms for host-refreshing JAX inputs. Native
has no accidental fourfold convolution-resolution change. Shared allocations
do not by themselves imply PCIe/system-memory execution; no such placement
fault was established. Historical counter windows suggest staging/MIO costs,
but their failed Graphics exits remain failed and are not ordinary utilization.

## 4. Small hypotheses tested, not promoted

RGB is12M/B8/BF16; CDP is12M/B32/mixed with exploration. These are separate
whole-update controls, not interchangeable workloads or a126ms/12ms gap claim.

| Isolated diagnostic | RGB full update | CDP full update | Decision |
| --- | ---: | ---: | --- |
| K staging32→128 |52.876→56.831ms|126.157→134.720ms|reject|
| Latest Blade/Naga, unchanged kernels |52.907→52.980ms|126.227→126.295ms|no speedup|
| Coalesced transposed dense staging |52.923→52.550ms|126.121→124.382ms|small; diagnostic only|

Every pair preserves272 metric reports, first outputs and all292/346
initial/first/final tensors exactly. K128 regresses7.48%/6.79%; coalescing saves
only0.70%/1.38%. Neither is the fundamental fix. These are single matched
pairs, not confidence intervals or a tuning sweep.

Upstream was rechecked before diagnosis. Meganeura main2214412 merges the tile
fix already in our056562b. Bladefe5f5b5/Nagaa4b4183 add memory-model/rectangular
cooperative support beyond production663afd2/c9fd6a85a. The unchanged-kernel
control qualifies those dependencies for these fixtures, with no speedup;
subsequent diagnostics use them. Embedded revision strings remain historical
in the diagnostic binary; effective pins/lockfiles are preserved explicitly.

## Implementation priority

Build a **training-grade, type-agnostic matrix/convolution path**, not more
dtype-only kernels or a collection of local tile tweaks. Reuse existing
code-generation, capability, split-reduction and reference machinery. Efficient
staging, register reuse, pipelining and shape selection must apply to the actual
BF16/F16 dense/grouped/derivative workloads. The same-kernel CUDA control is a
useful regression reference; CUDA is not the proposed production backend.

Then connect reduced-precision grouped and convolution derivatives without
unnecessary materialization, and consolidate learner dataflow: reuse posterior
outputs, keep RNG/targets/EMA on device, fuse cheap intermediates. CDP's
imagination is already the largest stage; decoder removal cannot fix it.

Prove each useful mechanism on complete RGB and CDP updates with appropriate
numerical checks. Do not predict a final speed by adding isolated savings,
relax strict-F32 fallback, or restart learning before an explicit learning/
time-to-score comparison. The achievable residual after those changes is
unmeasured; no assertion that these four kernels prove all compiler paths fast.

## Receipts and retained failures

Artifacts: `runs/execution-gap-20261013.EYeKOvtY`. Thirty-five GPU guards pass;
two fail on diagnostic CPU graph-construction assertions, with no API/numerical
fault observed. Six unique allocation warnings remain. No wedge, recovery,
NVML polling, game actions or active worker. There are4,920 synthetic learner
updates,1,152 completed same-algorithm calls,432 retained calls before the
convolution-shape assertion,1,088 library calls and13 backend numerical
processes. Compilation/autotuning work is separate, not game experience.

Retain the nested-workspace preparation failure, missing Python-alias/controller
failure, malformed build-script line, unsupported BF16-input assertion, flat
convolution-shape assertion and CPU reporter formatting failure. Corrections
use new outputs/services and do not alter numerical gates. The library reporter
fix reproduces its previously saved data exactly. All original source snapshots,
binaries, plans, HLO, traces, outputs and failed guards remain; no repeated
learner timing after a numerical failure or production pin change.
