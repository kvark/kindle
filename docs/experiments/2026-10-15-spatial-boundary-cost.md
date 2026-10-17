# Closed image-region materialization attribution

Declared October15,2026 at10:26UTC, before source changes/build/GPU work.
Follow the completed all-output substitution, not a retry. Its exact copies
retain3.154GB of logical outputs; the independent CPU version proof finds
only7.014MB escape the combined image/convolution region. Logical bytes are
not measured DRAM traffic. Test boundary-only substitution to distinguish
computation from intermediate materialization, not an achievable speedup.

Keep current076f3135 RGB12M/B32/T16/BF16 graph/checkpoint, device means,
precision, original allocations/groups/barriers and the previous344 snapshots.
Same four regions: none,23 convolutions,286 image-associated non-heavy
dispatches,both309. Copy every externally consumed/protected output at its
original slot; omit internal outputs only if every consumer is also replaced.
The cuts retain0/23/86/46 output buffers,0/855506944/705121428/7014164 bytes.
Do not change production, shaders, model semantics or ordinary scheduling.

Independently verify the Python cuts against actual runtime reads, including
matrix metadata operands, multiple writers, all primary/extra outputs and
protected graph/parameter/gradient/input/constant/derived/LSE values. Require
every still-executed read to observe the original logical version. CPU tests
must reject a missing escaping output and wrong producer version.

Before timing each nonempty region, poison only its discarded snapshot values
with0x7fc17fc1 and execute the all-output copier. Verify all live retained and
unselected outputs, plus finite exact complete outputs; this guards against
accidentally passing on stale valid intermediates. Restore snapshots by one
original execution/capture and require exact equality with the initial capture.
Then verify closed substitution at original live boundaries and complete outputs.
The integer-copy shader is unchanged from its3906-value GPU qualification.

One initializer update, then optimizer/AGC/accumulation disabled. Same16-warm+
64-measured pre-snapshot control and eight16-warm+64-measured blocks in order
0,1,2,3,3,2,1,0:576 measured and733 full-world replay equivalents, one synthetic
update, zero actions. Require normal mean within5% of pre-snapshot and retained
34.2065ms; no repeats to pass. Exact292 initial/first/before-after tensors,
first report/targets and entire plans/pipelines/groups match the retained control.
Retain samples, source/build failures and every guard. No per-dispatch timing.

One GPU worker600s/service720s under existing host guard/systemd, Restart=no,
KillMode=control-group. Expected5080/580.178.04/boot3e89d55c-a9e5-472f-a18a-06508c5bafa7,
>=2GiB sampled Vulkan budget headroom; retain allocation warnings. API/numerical/
deadline/hard faults stop for review. No blind retry or driver recovery. CPU
preparation/build/review1CPU/2GiB/no swap, no concurrent build or analysis during
timing, no separate NVML polling. No new JAX run, kernel/geometry/precision sweep,
production promotion, learning restart or automatic follow-up allocation.
