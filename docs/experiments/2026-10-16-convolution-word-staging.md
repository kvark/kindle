# Counter-directed convolution word staging

October 16, 2026. Follow the [native counter result](../results/2026-10-16-native-counters.md),
not a new tile search. Production and Atari stay unchanged.

Test one typed staging contract for all nine current scalar-staged convolution
occurrences (three forward and six derivative contracts). Each of 128 lanes
owns two four-element packets instead of eight separately addressed scalars.
Share each packet's reduction cursor across its four columns. Store typed
`vec4` packets in shared memory, with row pitch 40 for BF16/F16 and 36 for F32;
these preserve 16-byte row alignment while rotating row starts across banks.
Keep 32x32x32 tiles, subgroup execution, original barriers, K order, dW splits
and compensation, explicit BF16 rounding, storage interfaces and selections.
The 14 packed occurrences remain unchanged. This is not a global-vector-load
or asynchronous-copy claim. Source bank arithmetic is conditional on hardware
lowering and must be checked with counters, not used as a latency prediction.

Before GPU use, prove packet ownership, cooperative-load/output addresses,
incremental source addresses and masked tails on actual and small ragged,
rectangular/strided cases. Validate all generated typed variants using the
existing Naga compiler and offline SPIR-V tools. Use a separate source copy and
Cargo cache; assert unchanged packed/non-convolution modules and plan selections.
No compiler change or type-specific algorithm is authorized by this declaration.

Then qualify the complete 20-contract/23-occurrence family, retaining existing
independent F64 fixtures and gates, exact paired full outputs and changed-input
restoration. Run controls and candidate in fixed order, including all input
transforms and dW reductions. Retain every sample, failure and warning. Collect
the existing 41-metric/34-pass native configuration for all contracts; compare
instructions, shared conflicts and useful tensor work, not occupancy alone.
Use separate ordinary 16-warmup/128-sample timings; profiling is not timing.
Require at least 5% occurrence-weighted complete-operation improvement before
proposing a full-update trial. An aggregate gain does not imply every shape
improves. A failure does not authorize a selector, layout/tile sweep or retest.

CPU preparation/audit: 1 CPU, 2GiB, no swap. GPU workers: persistent serialized
host-guarded services, no CPU quota, Restart=no, KillMode=control-group,
120-second per-process limits and bounded queue deadlines. Expected 5080,
driver580.178.04 and >=2GiB sampled Vulkan estimated headroom. Record allocation
warnings; API/numerical failures, hard faults and deadlines stop for review.
No clock/driver changes, separate NVML polling, learning or game actions.
Artifacts: `runs/convolution-word-staging-20261016.ssN5oC6N`.
