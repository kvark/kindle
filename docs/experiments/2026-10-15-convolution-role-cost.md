# Encoder versus decoder with exact-value substitution

Declared October15,2026, before GPU work. The
[whole-family intervention](../results/2026-10-15-convolution-value-bypass.md)
reduces fixed-world GPU34.191→14.282ms with valid consumer values. It does not
identify which costs transfer to CDP, which removes RGB reconstruction.

Use the qualified binding-repair source/binary lineage, same RGB12M/B32/T16
fixture and one initializer update. Preserve every original kernel, allocation,
group, barrier, input and snapshot. All23 outputs remain captured in855,506,944
device-local bytes for every condition, including unsubstituted control.
Only select which original dispatch slots replay their exact saved values.
The copy shader/bindings/arithmetic are unchanged; no new kernel or precision
path. Optimizers, clipping and gradient accumulation stay disabled after setup.

Classify by the audited actual weight parameter:11 encoder products versus12
decoder products, disjoint and covering all23. Reject unknown names, missing
products and duplicate selector entries. Nine checked/14 packed variants are
retained as descriptive metadata, not additional conditions or a sweep.

Four conditions: no substitution, encoder only, decoder only, both. Before
timing, each must preserve all23 live-boundary fingerprints, full finite graph
outputs, initial/first/before292 tensors, first targets/reports and the original
plan/pipelines/groups. Keep snapshots and both pipeline sets resident. CPU
source/selector checks and unchanged shader validation precede the GPU process;
the previously passing3,906-word validation smoke remains historical evidence.

One fixed order0,1,2,3,3,2,1,0;16 warmups then64 measured cached full-world replays
per block. All512 measured samples remain,128/condition. Check complete outputs
after every block and all292 model tensors afterward. Record GPU span, queued
wall and recording separately, with no external/per-dispatch profiler,
compilation or CPU analysis concurrently. No copies-only timing repeat.

Report both conditional differences per role and their symmetric interaction
allocation, retaining any negative/contradictory results. These are valid-value
interventions including replacement copies/cache/dependency effects, not pure
kernel costs or independently removable learner savings. Compare fresh control/
both means against retained34.19129625/14.28209775ms; if either shifts by>3%,
do not combine allocations across runs or hide the shift. Within-process results
still require exact values; no timing retry or changed gate.

No full-update or CDP throughput claim. A decoder-only finding is not a CDP win;
even encoder evidence needs the actual CDP workload checked before promotion.
This is one role-level diagnosis, not a tile/iterator/dtype/engine sweep or a
revival of the rejected spatial family. No automatic implementation or learning
allocation follows.

One serialized persistent host-guarded GPU process,600s deadline/15min service,
Restart=no,KillMode=control-group. Expected5080/driver580.178.04/current boot,
>=2GiB sampled Vulkan estimated budget headroom with snapshots resident,
record standalone allocation warnings. Numerical/API/hard/deadline failures
stop for review; no blind retry, NVML polling or recovery. CPU preparation/
review1CPU/2GiB/no swap; GPU workers have no CPU quota. Retain sources/binaries,
all data/failures/tails and historical pins. Production/Atari remain unchanged.
