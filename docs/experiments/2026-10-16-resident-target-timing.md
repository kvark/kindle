# Complete resident-target timing

The [revised full-model encoding contract](2026-10-16-resident-target-observer-contract.md)
passes at22:40:53UTC on the entire retained cohort. Keep the original exact-slow-
encoding smoke failed. The revised contract applies the already-qualified
component envelope consistently to all dense encodings; copied arrays, control
reproduction and each arm's checkpoint restoration remain exact. No component
tolerance, runtime code or GPU cohort changed for that CPU observer correction.
Later trajectories differ; this does not qualify learning parity.

Run one matched complete comparison on unchanged observere45b9c43/backend:
RGB control/candidate then CDP control/candidate, four guarded processes. Same
B32/T16/H15/12M BF16 recipes/checkpoints and selected backend flags as the smoke.
Only the candidate enables `KINDLE_DEVICE_TARGETS=1`.16 warmup+256 measured
updates per process:1,088 synthetic updates, zero actions. No GPU tracing,
compilation during the measured updates, per-layer retiming or tuning.

Retain every sample. Require each arm's first24 metric reports and initial/first
tensors/targets to match its own smoke exactly, retained control272 reports and
initial/first/final tensors, unchanged complete plans/pipelines/groups/configs,
first target/metric gates and finite later states. Repeat checkpoint save/drop/
restore outside the timer, preserving every tensor/counter/normalizer exactly.

Performance gate: RGB complete mean at least3% lower, CDP no more than2% slower;
fresh control within5% of retained46.199419/43.938776ms. Compare candidate RGB
only to inclusive JAX27.222569ms. No sample trimming, repeat or automatic
production/learning promotion if a gate fails. Stage boundaries may move because
target execution moves; complete update is the primary result.

Each GPU guard:360seconds,12GiB host memory,no swap,no CPU quota, expected
device0x2c02/driver580.178.04/boot3e89d55c-a9e5-472f-a18a-06508c5bafa7 and >=2GiB
sampled Vulkan budget headroom. Record allocation warnings; API/numerical/hard-
fault/deadline failures stop. Controller deadline2400seconds. CPU review is
separate: one CPU,2GiB,no swap,720seconds, one BLAS thread. No polling NVML,
GPU recovery, backend repin or Atari restart. The broader root-cause goal stays
open regardless of this bounded host-target improvement.
