# Exact-value convolution substitution in the intact fixed world graph

Declared October15,2026, before new GPU work. The fixed library stage control
is unsupported; no stage/geometry/iterator sweep follows. Earlier zero-work
masks changed intermediate values and produced contradictory cost allocations.
This diagnostic removes convolution computation while retaining its exact
recorded outputs, to determine how much of the fixed graph cost remains.

Use the current isolated RGB12M/B32/T16 source: categorical-GPU Kindle with
the general-ordering backend. One deterministic initializer update, then disable
optimizers, clipping and gradient accumulation. Keep the same inputs, parameters,
random draws, allocations, pipelines and every existing dependency boundary.
This is optimizer-free replay, not ongoing learning or ordinary full updates.

CPU preflight identifies all23 convolution products in4,023 world dispatches/
1,258 actual runtime groups, with23 unique logical outputs in17 groups.
Snapshots total855,506,944bytes, maximum268,435,456bytes. The original preflight
incorrectly read stale plan groups; retain its failure and corrected separate
runtime-group audit. At runtime verify physical lifetimes/non-overlap too.

Capture each selected product output into separate device storage immediately
after its original independent group, before alias reuse. Substitute a generic
integer-word compute copy at that product's original dispatch slot. Preserve
original convolution/non-convolution barrier classification, all groups and
nonselected kernels; no removal of true dependencies or altered graph allocation.
Retain original and copy pipelines simultaneously. No dtype/rounding change,
new mathematical algorithm, output hint, library call or production path.

Require exact initial/first/before/after292 saved tensors, original first outputs/
report/plan/pipeline inventory, every selected output at its live capture and
substituted-validation boundary, and all full-graph output fingerprints across
normal/substituted replays. Uninitialized allocation padding is not a numerical
value claim; compare the copied bytes exactly and distinguish logical capacity.
Any mismatch stops before timing. Check changed input/restore on a small copy
fixture independently, including non-vector-aligned tails, before the full graph.

One validation-enabled small GPU smoke, then one ordinary fixed-world process.
Inside the latter, capture and exact checks precede one counterbalanced ABBA
comparison:16 warmups then64 measured cached replays per block, all samples
retained. Record GPU spans, queued wall and recording separately. Also time the
23 replacement copies together with retained dependency ordering once, to
disclose their cost; do not subtract it into a promised speedup. No profiler,
concurrent build, timing repeat, kernel selector or automatic follow-up sweep.
Check full outputs after every measured block and all292 tensors at the end.

Whole-plan GPU intervals include command/dependency costs. Copy substitution
changes cache traffic and is not a pure convolution-latency measurement, a
strict zero-cost lower bound or a production optimization. It answers a
family-level importance question with valid downstream data; do not add its
result to previous masked allocations or compare it directly to JAX kernel sums.
Latest ordinary complete54.224/26.632ms remains retained, not remeasured here.

Persistent serialized host-guarded service,Restart=no,KillMode=control-group;
120s smoke,600s fixed-world process,15min service bound. Expected5080/driver
580.178.04/current boot,>=2GiB sampled Vulkan estimated headroom including
snapshots, recorded standalone allocation warnings. API/numerical/hard/deadline
faults stop for review; no blind retry, separate NVML polling or recovery.
CPU preparation/review1CPU/2GiB/zero swap; GPU workers have no CPU quota.
Seal exact sources, immutable binaries, plans and declaration before execution.
Production pins, learning defaults and Atari remain unchanged.
