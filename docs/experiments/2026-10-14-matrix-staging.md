# Shared checked matrix vector staging — October14

Completed: [results and decision](../results/2026-10-14-matrix-staging.md).

The [matched matrix study](../results/2026-10-14-matrix-library.md) finds the
same tensor instruction count/occupancy but3.40× instructions in our checked
CUDA translation, which reproduces native grouped-forward timing within.32%.
Scalar16-bit staging versus vector transfers is a concrete intervention target.
This does not explain the full24.340/12.836ms native/JAX update gap.

Implement one shared typed checked-matrix staging path. Keep32×32 output tiles,
32-element K stages,128 threads, two16-wide products/stage, partition boundaries,
accumulation, output rounding, grouping and real dependencies. Vectorize four
contiguous stored elements/load,64 bits for F16/BF16 and128 for F32, with vector
shared storage. Transposed operands use the matching cooperative shared-layout
load instead of scalar scatter. No BF16-specific engine or geometry sweep.

Select only explicit typed dense/grouped default kernels whose contiguous
operand strides are divisible by four. Prove whole-vector bounds from dimensions;
retain scalar checked fallback otherwise. Odd non-contiguous dimensions and
short reductions still need correct zero-fill. Preserve aligned/tiled kernels,
convolution, prologues/epilogues and measured variants. Selection belongs in the
existing pipeline key and must not silently override an explicitly selected
implementation. Verify allocation/binding alignment and pipeline identity.

First CPU selection/address/typed-shader/SPIR-V checks, including dense/grouped
normal/AT/BT, narrow rows, row/column/K tails, partitioned reductions, F32/F16/
BF16 storage and output combinations. Then one guarded independent numerical
fixture compares every output to F64 at existing dtype-appropriate gates, with
aliasing and changed-input/restore checks; no tolerance changes. Native F32
cooperative multiplication is unsupported on this5080: CPU shader coverage is
not GPU qualification for it. F32 storage with reduced multiplication is in scope.

After numerical review, one existing four-contract native control/candidate
timing,16 warmups+128 calls each, exact outputs and descriptors except the
declared pipeline implementation. No CUDA/Nsight sweep. If the measured grouped
forward saving is at least10% and other contracts do not regress more than5%,
review one complete RGB12M/B8/T16/H15/BF16 control/candidate pair:16 warmups+
256 measured synthetic updates, cache enabled, same initial state/RNG and full
272 reports/initial-first-final292 tensors. Short8-update production smokes
precede timing. Keep only if complete update time improves at least3%, with
all numerical/finite/state checks passing. No CDP or learner allocation follows
automatically, and no micro result substitutes for complete-update evidence.

Artifacts `runs/matrix-staging-20261014.iRAlLKXg`; isolated056562b plus the exact
retained iterator patch, Blade96c1dc7/Naga a97559b. Latest upstream7574ad1/e7c3f7b
rechecked unchanged before work. Production pins/defaults and neighboring edits
stay untouched. Existing serialized persistent guards, expected5080/580.178.04/
boot,>=2GiB sampled Vulkan budget headroom, recorded allocation warnings; other
failures stop for review. Numerical/micro jobs<=600s; complete jobs<=900s.
CPU preparation one CPU/2GiB/zero swap; no CPU quota on GPU jobs. No NVML polling,
driver action, profiler injection, game actions or training restart.

CPU review:243 generated type/layout variants validate and emit SPIR-V;16
selection cases retain alignment/explicit-implementation guards;17,350,704
independent address/shared-layout checks in18 cases pass, including partitioned
tails. Existing scalar/convolution shader and256-byte arena-alignment regressions
pass. Initial scalar template-token collision/source/log remain under
`failed-cpu-v1`; corrected before any GPU work. Numerical fixture fixes416
cases across eight storage/multiply tuples, dense/grouped normal/AT/BT, three
output types, both alias modes, odd non-contiguous dimensions and partitioned K.
Every baseline/changed/restored value must equal its independently rounded F64
sum. Logged validation; no optimizer. This fixture deliberately uses exact
binary fractions to preserve the unchanged arithmetic order's exact gate.
