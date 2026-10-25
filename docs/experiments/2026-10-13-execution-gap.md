# Diagnose the execution-model gap, not another timing bucket

The user reopens the native/JAX root-cause investigation. The preceding report
accounted for operation groups but did not establish the fundamental reasons
equivalent work executes several times more slowly. No Atari allocation,
production optimization, driver changes or profiler-injection retry.

Use the retained RGB12M/B8/T16/H15 native and upstream Dreamer fixtures. Model
size matches, but graphs, states and precision coverage do not. Current native
056562b measures53.047ms; retained JAX BF1612.423ms (13.185ms with broader host
boundaries). Do not compare CDP/B32 to that JAX recipe.

## Questions and first bounded tests

1. Count useful matrix/convolution arithmetic by execution precision, including
   duplicate native posterior work, using retained plans and source. Separate
   useful arithmetic, padding, staging and intermediate traffic; compiler cost
   estimates are not measured DRAM bytes or achieved FLOPs.
2. On unchanged upstream JAX, run one2x2 diagnostic: ordinary BF16, F32
   convolution derivatives, F32 grouped products, both. Maintain BF16 forward
   convolution and operation outputs, F32 parameter/optimizer storage, model,
   inputs and seeds. The interventions model arithmetic coverage, not exact
   native numerics or a proposed training recipe. Inspect compiled HLO to prove
   that each intended operation changes and no unsupported shortcut appears.
   Check the custom derivative against independent small F64 examples first.
3. One additional ordinary-BF16 arm disables XLA command buffers. Retain the
   full compiled graph, all reports and checkpoints to distinguish command
   scheduling from kernel/precision differences. Exact equality is expected
   for this scheduling-only comparison; investigate discrepancies.
4. Use existing valid CUDA captures and diagnostic native Graphics counters
   to inspect dispatch granularity, arithmetic hardware, tile reuse and memory
   layout. Old Graphics failed exits remain failures. No repeated broad capture.

Each JAX arm is16 warmup+256 measured synthetic updates, with a600-second
process bound. One serial service,60-minute bound, Restart=no,
KillMode=control-group; existing host guard, expected5080 and sampled Vulkan
budget headroom>=2GiB. No CPU quota on GPU jobs. Record allocation warnings;
other failures stop for review without retries. Preparation/analysis uses
one CPU,2GiB,zero swap. No game actions. Retain every failed check and all arms,
not just favorable timings. No claim that differences from separate arms add
linearly to the native gap.

Review results before declaring further targeted tests. Completion requires a
causal explanation of the dominant structural differences, explicit residuals
and an actionable implementation priority—not relabelling convolution/matrix
time as a root cause or promising unmeasured gains.

Artifacts: `runs/execution-gap-20261013.EYeKOvtY`.
Upstream recheck: Meganeura main2214412 merges the already-present PR255;
Blade fe5f5b5 merges existing BF16 support plus an unrelated ray-query format
fix. Its updated Naga pin also contains Vulkan-memory-model corrections and
rectangular cooperative-matrix support absent from the tested compiler; the
compiler refresh control below keeps this distinct from the merged BF16 work.
Preserve current qualified pins and neighboring user work.

## Reviewed follow-up: staging granularity

The five JAX arms and independent CPU derivative check pass. Ordinary BF16
12.417ms becomes17.091ms with F32 convolution derivatives,14.596ms with F32
grouped products,19.236ms with both. Disabling command buffers alone costs
16.163ms and preserves all272 reports/288 final state tensors. Compiled HLO
confirms the intended precision/capture changes. These causes alone do not
explain native53.047ms; do not add their effects as independent savings.

Retained CUDA kernels include32x32/K128/two-stage GEMMs and128x128/K64/three-stage
convolutions. Our shared cooperative loop is32x32/K32, one stage, with two
workgroup barriers each K32; transposed dense staging also uses strided global
loads. Distinguish these source facts from demonstrated performance effects.

One isolated native diagnostic changes only shared-loop K staging32->128,
retaining output tiles, arithmetic, split boundaries and accumulation order.
Mask the partial last stage at each split's own end. This changes shared memory
and occupancy as well as barrier frequency, so it is a staging-granularity test,
not a pure barrier-time measurement or a software-pipeline implementation.
No tile sweep or production adoption. First CPU shader checks, then existing
full F64 partition/gradient/convolution and F16 checks; then one ordinary
control/candidate pair each for RGB/B8 and CDP/B32 (16+256 updates/arm).
Exact fixture outputs/reports/tensors are expected; failures stop for review.
Backend checks120 seconds each, whole updates600 seconds, service60 minutes.
Same guard/headroom requirements. No builds during timings. Preserve patches,
binary hashes, failures and negative results.

Also run one separately declared combined JAX F32-coverage/no-command-buffer
arm (same272 updates/600-second bound). This measures interaction rather than
adding effects from separate tests. It is still not exact native arithmetic.

The first native preparation fails before compilation because the nested
Meganeura checkout is discovered as part of Kindle's workspace. Retain the
failed service log; explicitly isolate the diagnostic package with an empty
workspace, then use a new preparation service. No GPU attempt or retry occurred.

## Compiler/backend freshness control

Review of Blade's merged patch finds a material compiler difference:
Naga a4b4183 adds Vulkan-memory-model fixes and rectangular cooperative types
over c9fd6a8. Do not attribute an already-fixed compiler issue to our kernels.
After the staging queue finishes and its numerical audit passes, build a
separate unchanged-K32 native diagnostic with Blade fe5f5b5/Naga a4b4183.
Meganeura056562b already contains main2214412's tile change. Only dependency
pins and necessary API compatibility may change; preserve old/new source and
lockfiles. No new shader algorithm or production pin change in this control.

Run the same four backend numerical checks followed by one old/new ordinary
pair per RGB/B8 and CDP/B32 (16+256 updates). Bounds/guards as above. Review
any failure before follow-up; do not relax numerical or validation gates.
Retain compiler-induced numerical changes explicitly; no blanket parity or
hardware-safety conclusion. CPU preparation waits for timing to end.

## Inspect actual algorithms under the F32-coverage intervention

The K128 diagnostic preserves every report/first output and292/346 tensors but
regresses RGB52.876->56.831ms and CDP126.157->134.720ms. Keep it unadopted.
The combined JAX F32-coverage/no-command-buffer arm takes22.775ms, still far
below native; separate effects must not be added.

Compiled HLO reveals five legacy cuDNN backward-input algorithm5 selections in
the F32-coverage arm: Winograd, not native's implicit GEMM. Thus even matching
arithmetic width does not match the algorithm. One CUDA-only Nsight capture of
the already-measured `both-f32` arm will identify the actual selected matrix/
convolution kernels and durations:16 warmup+8 captured updates,600-second bound,
exact first24 reports/initial state against its ordinary control. Reuse the
validated Systems CUDA-only capture setup, no GPU-metric/NVML polling or native
Graphics injection. All eight captured samples remain; do not substitute their
wall time for ordinary timing. This is a targeted algorithm check, not a sweep.

## Transposed-load coalescing control

The compiler refresh passes all eight guards, exact272 reports and292/346
initial/first/final tensors. RGB52.907->52.980ms and CDP approximately126ms
remain unchanged; no speedup or production adoption. Further kernel diagnosis
uses this reviewed Blade/Naga pair, retaining the earlier compiler evidence.

One diagnostic changes only which threads stage transposed matrix operands.
Currently an AT warp reads `A[(k+lane)*M+row]`, and a BT warp similarly strides
through rows of B; each halfword can occupy a separate memory transaction.
Transpose the shared-memory staging index so consecutive lanes read consecutive
global elements. Preserve tile/split sizes, shared layout, precision, arithmetic
order and every barrier; this may increase shared-store conflicts, so a win is
not assumed. Forward non-transposed and convolution staging remain unchanged.

CPU shader validation, then the four existing backend checks plus typed F16/
BF16 edge/transposition checks precede one full-update old/new pair per recipe.
Same budgets, fixtures and exact expected outputs as the previous controls.
No new tile/precision choice, profiler injection, learner allocation or production
code change. Whole-update effect, including a negative result, is the outcome.

The first coalescing queue launch fails before any worker starts: this shell
has no `python` alias, so its declaration was absent. Retain that controller
failure; create the declaration with explicit `python3` and use a new service.
This is not a GPU/test failure or a retry of a started measurement.

## Same-algorithm CUDA/Vulkan diagnostic

Coalescing preserves all reports/tensors, but ordinary RGB52.923->52.550ms and
CDP126.121->124.382ms improve only0.70%/1.38%. It cannot explain the large gap;
no production promotion. The captured F32 JAX arm confirms SIMT grouped GEMMs
and Winograd transforms/complex products for several convolution derivatives.

Distinguish algorithm quality from a shader/compiler/API limitation. Implement
the current coalesced checked BF16 loop in a standalone CUDA diagnostic, not
a Kindle backend: same32x32 output tiles, K32 staging, four16x16 warp products,
two barriers/stage, shared staging/results, precision, split boundaries and
resident packed operands. Compare dense8x256x2048,128x256x2560,
AT2560x256x1920 and the retained N128/C32->48/HW32/5x5 forward convolution.
All outputs remain F32 for this isolated comparison. Use native graph inputs
already BF16, with no casts/optimizer/learner; CUDA uses identical scalar
indexing and the explicit reciprocal helper for image row decoding.

One16-warmup/128-call recorded batch per case/API, plus GPU timestamps of those
batches and driver-reported native pipeline statistics. CUDA may graph-record
the batch to avoid Python/per-launch host starvation. Retain all setup, code,
compiler versions and outputs. Require exact full native/CUDA outputs and128
independent F64 probes per native case before interpreting times; failures stop.
CUDA12.4 emits compute80 PTX for the580 driver to JIT, rather than claiming a
native sm120 compiler comparison. Compilation is CPU-only,1CPU/2GiB/no swap;
no compile overlaps timing. Each guarded process has600 seconds, serial service
30 minutes, Vulkan headroom>=2GiB, same device and warning policy. No external
profiler injection, production code/pin change, kernel sweep or game action.

Different compiler lowering and barriers remain part of the comparison: equal
source algorithms do not imply identical instructions. A large CUDA advantage
would justify inspecting generated code; similar slowness would implicate the
algorithm rather than a blanket Vulkan limitation. Do not extrapolate these
isolated timings into a full-learner speedup.

CUDA compilation succeeds, but a malformed line joins `cd` and the formatter
in the first CPU build script. Retain script/log/binary, correct that line and
build the Rust diagnostic in a new CPU service; no GPU work has started.

The first native process initializes the GPU then hits the compiler's explicit
prohibition on BF16 graph inputs, before dispatch/timing. The failed guard,
source and binary remain. Do not weaken the compiler invariant: compile the
supported F32->BF16 graph, then feed its two conversion-output buffers as
prepacked diagnostic inputs and remove only those casts from this isolated
plan. Retain before/after plans, rebuild conservative per-dispatch barriers,
and keep the exact-output/probe gates. New output/binary/service names;
CUDA remains unstarted. No evidence of a device fault or a numerical failure.

The corrected process passes all three matrix outputs/probes, then the CPU
convolution constructor rejects a2D weight shape instead of its required flat
shape. Retain that failed guard and three outputs; their timing summary was
not saved before the panic and is not used. Correct the shape, persist each
case's progress and re-declare the four-case native batch under new names;
no tolerance/kernel change. Driver pipeline statistics return empty in the
ordinary context and stay unavailable, not zero-register/spill measurements.

The corrected same-algorithm comparison passes6,981,632 exact output values
and512 independent F64 probes. Native/CUDA: small10.705/7.272us,
wide18.699/17.492us, AT168.380/178.720us, convolution667.699/625.669us.
This does not support a multi-fold compiler/API penalty for the substantial
BF16 kernels; the smallest case retains a3.43us difference. No whole-update
speedup follows from these kernel-only timings.

Finish the comparison with one JAX library arm using exactly those resident
BF16 operands and explicitly requested F32 outputs, not the retained BF16-
output fixture. Same four shapes,16 warmups+128 ordinary calls+128 CUDA-only
captured calls each;600-second guarded process. Inspect actual HLO/kernel
choices and require full exact outputs/probes against native/CUDA. Use all
captured kernels per case, retaining host-call timings separately. No repeated
native/CUDA timing, precision/tile search or production change. This tests
library algorithm quality on the same arithmetic rather than attributing its
advantage to the API or output rounding.

All library outputs match, but HLO shows the requested F32 convolution output
promotes its BF16 operands to F32 compute. Retain its1.461ms measurement as an
unmatched-precision control, not BF16-library performance. The three dense
cases remain BF16 operands/F32 outputs and show library kernel sums3.397,
4.849 and70.481us. No replacement convolution run or output-precision sweep.
The CPU auditor saves correct data then fails formatting its printed kernel
names; retain the failed source/JSON and re-run only that reporter with an
exact-data comparison. No GPU measurement changes.
