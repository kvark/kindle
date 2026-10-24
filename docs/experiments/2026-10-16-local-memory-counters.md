# Distinguish local-memory traffic from convolution feeding

October 16, 2026. Follow-up to the CPU [instruction audit](2026-10-16-kernel-instructions.md).
The existing 41-metric configuration lacks the raw dependencies for local-memory
reads/writes; uncollected values cannot establish absence of spilling.

One bounded diagnostic allocation, not an optimization candidate:

- Preserve all 41 existing metrics. Add local load/store bytes and sectors,
  shared load/store bank conflicts and wavefronts, and the SDK's global-load
  sector proportion. Query support/construct configurations on CPU first.
- First run one positive-control process: runtime-indexed private 512-word
  arrays, alongside a scalar register-only shader and an empty range. Exact
  independent CPU output checks before/after capture; positive local reads
  and writes, no local traffic in the two negative controls. Stop and review
  if this instrumentation contract fails.
- Then one process for each of all 20 convolution contracts/23 RGB occurrences,
  same word-staging executable and shaders. No arithmetic, precision, geometry,
  synchronization or model changes. Reuse the retained 16-warmup/32-observation
  preparation, then at most64 counter passes/process. Those observations are
  calibration, not another optimization timing or new complete-update claim.
- Use the existing counter bridge; change only its metric list and configuration.
  Prove loaded-library identity, identical plans/pipelines/shaders/full outputs,
  changed-input restoration, exact launch counts and no dropped ranges. Preserve
  every pass/sample. Independently reevaluate saved raw counter images on CPU.
- Reconcile old instruction/tensor counts; report any mismatch rather than
  silently comparing different execution. A local-memory observation is not an
  instruction-level source attribution or proof that all traffic is spilling.

Existing guard/queue, expected5080/580.178.04, >=2GiB sampled Vulkan headroom,
record allocation warnings, other API/numerical/hard faults fatal. Processes
<=120s, overall service<=3600s, no restart and serialized GPU use. Preparation/
review stay one CPU/2GiB/zero swap. No assembly query, invalid compiler-local-
memory statistic reuse, clock changes, independent NVML polling or recovery.
No source sweep, timing promotion, learner update, game action or Atari restart.

Artifacts: `runs/kernel-instructions-20261016.evva2SX8`.
