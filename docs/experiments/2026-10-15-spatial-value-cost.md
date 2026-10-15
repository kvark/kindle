# Valid-value attribution of current non-convolution image training

Declared October15,2026 at10:04UTC, after CPU inventory and before backend
edits/build/GPU work. Latest native/JAX inclusive54.113/27.223ms remains1.99x;
JAX without command buffers29.505ms rules out graph execution as the dominant
residual. Isolate the current image branch before another implementation.

Use the unchanged current076f3135 RGB12M/B32/T16/BF16 graph, device means,
same input/checkpoint/first update, kernels, layouts, arithmetic and barriers.
Upstream99e3de27/e7c3f7b rechecked unchanged at09:55UTC. Clone its sources;
extend only the existing diagnostic saved-output copier to support every
primary/extra output. No new learning algorithm or optimized kernel.

CPU dataflow selects286 non-heavy dispatches: ancestors and descendants of
the16 fused image operations and23 convolutions, stopping at every convolution
or matrix product. Walk ancestors and descendants independently; do not flood
from ancestors into unrelated branches. Include the30 image parameter-gradient
reduction tails, layout/casts, image/sequence packing, upsampling and pixel-loss
arithmetic. This is the image-associated branch, not just spatial kernels or
all non-heavy world work. Exact indices/contracts remain in preflight.json.
No tensor product belongs to that286. Together with23 convolution dispatches,
309 slots write344 distinct logical buffers,3,154,096,284 snapshot bytes.
All selected horizontal batches are zero. Preserve actual alias mappings.

One independent integer-copy smoke precedes one fixed-world factorial study:
normal,23 convolutions substituted,286 non-heavy image operations substituted,
both substituted. Each replacement copies exact saved words into *all* of
the original primary/extra outputs at the original slot. Multiple outputs
require multiple disjoint copy dispatches; original barriers/grouping remain.
No zero-work masks, invalid intermediates, per-operation timestamps or
algorithm/geometry/precision sweep. Optimizer/AGC/accumulation are disabled
after one original initializer update; no acting or game actions.

Capture and verify each selected output at its original live group boundary,
before alias reuse. Require exact full outputs at capture, every condition,
every timing block and final normal restoration. Initial/first/before/after292
tensors, first targets/report and complete plan/pipelines/groups must match
the current retained control. Check independent copy values including signed
zeros/NaN payloads/packed words and changed/restored sources; never interpret
those bit-pattern probes as model outputs. CPU selector/dependency tests,
copy shader validation, formatting and Clippy precede GPU.

Measure an additional16-warm+64-measured full-world control before snapshot
allocation. Four conditions use fixed order0,1,2,3,3,2,1,0, each16 warm+64
measured replays. Keep all samples:576 measured/727 full-world replay
equivalents, one initializer update, plus the bounded copy smoke. The resident
normal condition must remain within5% of pre-snapshot normal and retained
34.191ms fixed world to support current-cost attribution. No repeats to pass.
Copies include their own execution cost; conditional differences/interactions
are not pure kernel costs, DRAM traffic or additive full-update savings.

Guarded persistent systemd: Restart=no,KillMode=control-group,120s copy smoke,
600s world worker,840s service. Expected5080/580.178.04/boot
`3e89d55c-a9e5-472f-a18a-06508c5bafa7`,>=2GiB sampled Vulkan estimated headroom
including snapshots. Record allocation warnings. API/numerical/deadline/hard
failures stop for review; no blind retry/recovery. No concurrent compile/
analysis or separate NVML polling. CPU work1CPU/2GiB/no swap, distinct Cargo
cache with old source-crate fingerprints excluded. No production promotion,
JAX rerun, learning restart or automatic follow-up allocation.
