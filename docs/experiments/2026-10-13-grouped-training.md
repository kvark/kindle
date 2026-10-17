# Shared typed grouped training

Investigate the remaining grouped RSSM execution gap using the existing checked
and tiled matrix generators with group-strided addressing. No dedicated BF16
arithmetic shader, CUDA learner or geometry sweep. Both complete-update arms
include the retained dense/native-forward-layout prototypes; the separate
convolution-backward prototype is not silently stacked.

Extend existing block-matmul operations and their two derivatives to explicit
typed storage/multiplication/output contracts. F32 defaults/strict gradients
remain available; the already opt-in BF16 graph/gradient policy gains grouped
coverage. Preserve row-interleaved activations and group-major weights without
host slicing or per-group launches. Bound dispatch/index/scratch extents, share
checked edges and use tiled reuse only with proved shape/shared-memory fit.

CPU checks cover graph types, strict versus reduced differentiation, generated
F32/F16/BF16 modules and fallbacks. GPU qualification uses independent F64
grouped forward/dX/dW with distinct group data, irregular dimensions, all three
orientations, ordinary/tiny incoming gradients, checked/tiled paths and bounded
long reductions. Preserve failures and unchanged format-appropriate tolerances.
Check production aliasing as well as poison/non-alias diagnostics. No timing
until numerical results are reviewed.

Then one ordinary complete RGB12M/B8/BF16 and CDP12M/B32/mixed pair each,
16 warmup+256 measured updates, retained initial checkpoints, same dense and
convolution algorithms on both arms. Include casts/reductions/host work and
audit actual selected grouped paths, finite reports, exact initials and saved
states. Changed reduced arithmetic is not bitwise trajectory or learning parity.
Both timing arms enable identical existing host NVTX annotations solely to
export complete execution plans; no profiler or GPU timestamp instrumentation.
No profiler, compile overlap, game actions, production pin/default change or
automatic Atari allocation. The main goal remains the whole JAX gap.

Artifacts: `runs/grouped-training-20261013.Qakwj7LK`.
Upstream Meganeura2214412/Bladefe5f5b5 rechecked unchanged. Base056562b plus
the retained native-forward candidate; Bladefe5f5b5/Nagaa4b4183. Other worktrees
and historical artifacts stay untouched. CPU preparation/analysis: one CPU,
2GiB, zero swap. Serialize GPU work under host guards and bounded persistent
systemd services, Restart=no/KillMode=control-group, no CPU quota/NVML polling.
Require5080 and≥2GiB sampled estimated headroom, record allocation warnings;
other failures stop for review, no blind retries or driver recovery.

The first CPU validation rejects WGSL's reserved `partition` identifier.
Retain `cpu-first.patch`/journal; rename the local to `part_id` before GPU work.
Backend lockfile matches the retained control exactly. The graph/API addition
is F32/BF16; F16 is generator validation, not a new public F16 training promise.
The next CPU check catches the emitter's BF16 operation allowlist still rejecting
BlockMatMul. Retain `cpu-admission-first.patch`/journal and add only the three
newly supported grouped orientations; do not relax the general dtype guard.
One corrective patch initially hits the dense emitter instead of that allowlist.
The CPU-only build is stopped, the source retained, and the exact context fixed;
no GPU work or numerical tolerance change is involved.
