# Large matrix kernels still stage through scalar shared stores

October16,2026. [Declaration](../experiments/2026-10-16-shared-store-contract.md)
· [Compact evidence](2026-10-16-shared-store-contract.json).

**The convolution word-staging fix does not cover the large tiled matrix path.**
All nine currently selected tiled matrix shader keys load global `vec4<bf16>`
values, then emit32 scalar16-bit shared stores per K-stage. This is explicit in
both generated WGSL and emitted SPIR-V, not an inferred driver instruction count.
The checked-vector matrix path already uses four vector stores per its smaller
stage. Different tile sizes prevent comparing those static counts as a speedup.

| Actual complete programs | RGB | CDP |
| --- | ---: | ---: |
| Tiled matrix occurrences / all matrix occurrences |342/716|387/769|
| Tiled share of nominal matrix MACs |91.04%|93.24%|
| Tiled share of imagination matrix MACs |98.19%|99.10%|
| Distinct tiled parameter/launch contracts |41|45|

The union is54 contracts, not86; repeated occurrences and shared RGB/CDP shapes
are accounted for. Arithmetic coverage is **not time coverage, measured bank
conflicts or achievable savings**. No matrix hardware counters were collected
in this CPU study. Complete RGB remains46.199ms versus inclusive JAX27.223ms,
1.70×; CDP43.939ms is its separate workload. Keep isolated58aff452.

## What the convolution audit does—and does not—explain

All20 retained convolution contracts reconcile measured shared-store wavefronts
minus conflicts with the existing operand/output payload. Under an explicitly
conditional model—32 lanes, consecutive local invocation membership and128-byte
transactions—the word-staged BF16/F16 operand stores predict32 extra wavefronts
per K-stage. This matches the fitted slope in all nine word contracts.

But an equally legal permutation of subgroup membership preserves every stored
value while changing the model to zero extra staging wavefronts. Vulkan does
not guarantee the local-to-subgroup mapping. [Vulkan specification](https://docs.vulkan.org/spec/latest/chapters/interfaces.html#interfaces-builtin-variables-sglid).
The bank model uses NVIDIA's32-bank/32-bit-word organization, not a measurement
of the actual instructions or their transaction grouping. [NVIDIA profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#memory).

Moreover, simply scaling that model to the **unchanged packed family fails all11
contracts**. For example, packed CNN1 forward predicts16,252,928 write conflicts
versus16,699,938 measured. The discrepancy is retained, not fitted away. Thus
neither source vector width nor this affine relation establishes the driver
lowering. The96-per-workgroup intercept remains unattributed.

Naga retains cooperative stores as `OpCooperativeMatrixStoreKHR`; all explicit
shared variables/stores reconcile with source. This excludes an extra explicit
Naga scratch allocation in these emitted modules, **not driver-managed storage**
or hidden lowering. No native assembly query, padding sweep or output-store
optimization is justified by this audit alone.

## Decision and verification

Next prepare one bounded, unchanged-kernel counter diagnostic covering the54
actual tiled matrix contracts, with checked-path controls. Use the existing
native SDK observer, exact source/parameter/launch membership and independent
output checks. Measure instruction work, shared reads/writes, local traffic and
stall behavior before choosing a staging change. This is a new diagnostic
declaration, not a geometry search or automatic GPU allocation. Do not retime
the failed prior-batching candidate or reopen rejected convolution layouts.

All1,519 actual heavy-product occurrences map to28 selected source modules;
21 keys regenerate byte-for-byte through the existing linked backend, seven
checked keys match independent source signatures.29 catalog modules validate
and emit SPIR-V, including one explicitly unused module. The audit also reads
152 retained typed F32/F16/BF16 modules; this is not new GPU type qualification.
6,144 unique ownership values,14 signature-negative tests and three bank-model/
ownership negatives pass. No backend rebuild/edit, GPU call, update or action.

The first independent reviewer fails because a forbidden substring also matches
the legitimate `CooperativeMatrixLayout` enum. Preserve its source/log; the
construction-specific CPU repair passes without rerunning preparation or any GPU
work. All services finish. CI454 passes preceding0e3d944. Production and Atari
remain stopped; the full parity/root-cause goal remains open.

Artifacts: `runs/shared-store-contract-20261016.Mi1tGL3k`.
