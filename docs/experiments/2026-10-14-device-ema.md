# Remove slow-critic parameter readback and upload

Keep the qualified exact-sequence GPU RNG candidate as the control. This
experiment changes only slow-critic exponential averaging: the current
implementation reads source and destination parameters, blends them on the
host, then uploads each destination. Native RGB still takes30.053ms versus
retained complete JAX12.836ms. Do not treat this smaller boundary as the entire
remaining gap or resume Atari.

Artifacts `runs/device-ema-20261014.PuHZB9mx`.
Upstream Meganeura2214412/Bladefe5f5b5 remain unchanged at00:52 UTC. Reapply
the immutable RNG source snapshots in isolated worktrees; no installed pins,
neighboring work, model/batch/precision/RNG/targets/optimizer changes.

Add a general same-context dense-F32 parameter blend to Meganeura, not a
Kindle-specific CPU workaround. Preserve selected-parameter semantics and
refresh derived vertical/horizontal concatenations on-device. Validate storage,
logical extents, shape/layout, duplicates, incompatible aliases and prepared
step boundaries before submission. Retain unselected values. Track completion
on both sessions so later uploads/readbacks and destruction remain safe.
Unsupported reduced/packed master storage fails explicitly, never silently
reinterprets bits or switches to a CPU path.

CPU geometry/rejection tests and bounded GPU cases precede full updates.
Compare against independent F64 weighted sums and the actual F32 host formula,
including cancellation, irregular tails, multiple parameters, partial
concatenation refresh, source immutability, immediate source destruction,
queued consumer use and a subsequent host update. F64 pointwise error must be
at most4*2^-24*(abs(rate*source)+abs(decay*old))+2^-126, where decay is
F32(1-rate). This accommodates ordinary F32 rounding/contraction/underflow,
not arbitrary relative error at cancellation. Refresh/canary/untouched values
must be exact. Record F32-host differences rather than assuming contraction
is identical.

After review, two24-update RGB/CDP smokes, then one ordinary
16-warmup+256-measured control/candidate pair per workload. Same initial
292/346 tensors, first pre-EMA outputs, compiled learner plans and every
non-EMA first-step tensor must match exactly. Compare all later metrics/tensors,
require finite states, and retain any trajectory differences without claiming
learning equivalence. Only source/destination EMA execution may differ;
no tolerance relaxation after a failure or automatic retries.

Guard all GPU work in serialized bounded systemd user services, Restart=no,
KillMode=control-group, expected5080 and≥2GiB sampled Vulkan headroom.
Allocation warnings are retained; numerical/API faults and deadlines stop the
job for review. No separate NVML polling or driver recovery. CPU builds/analysis
use one CPU/2GiB/no swap, never during matched timing. Keep whole-update/stage
costs, not an isolated blend benchmark or utilization claim. Targets and
row/layout fusion remain separate subsequent work; the full goal stays active.
