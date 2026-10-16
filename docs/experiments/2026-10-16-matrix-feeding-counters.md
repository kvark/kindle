# Measure the remaining tiled matrix feeding

October16,2026. Follow the [shared-store audit](../results/2026-10-16-shared-store-contract.md).
This is an unchanged-kernel diagnostic, not a new optimization or learner run.
Keep isolated58aff452 and complete RGB46.199ms/JAX27.223ms; production/Atari stay stopped.

- Capture all54 distinct actual tiled shader/parameter/workgroup contracts in
  complete RGB/CDP programs, preserving their729 occurrence memberships. Add
  one actual checked contract per each of11 checked shader keys, choosing the
  lexicographically first parameter/workgroup tuple before execution. No favorable
  shape selection, tile/precision change or subset speed claim.
- Use exact retained WGSL with the same linked Meganeura/Blade/Naga libraries,
  workgroup dimensions, splits, prefetch, rounding, output layout and barriers.
  A small direct-Blade observer binds these shaders; no backend rebuild. Audit
  this binding/launch boundary against actual plans and generated sources first.
- Use device-local inputs/outputs. Per-contract structured integer-valued BF16
  inputs vary with row, column, reduction and group. Independent exact integer
  sums qualify every F32 partial/output and explicit BF16 nearest-even rounding;
  no float tolerance. Check initial, sign-changed A, restored and final outputs,
  plus poisoned output guards. Retain inputs and outputs for independent CPU
  rechecking. This fixture is not a learning trajectory or full matrix oracle.
-16 warmups and32 observations are calibration only, followed by the existing
  SDK50-metric/37-pass configuration, at most64 passes. Retain empty ranges,
  launches, all samples, raw counter images and loaded-library receipts. Reevaluate
  every metric offline. Counters cover one unchanged matrix dispatch, not its
  surrounding graph, packing, split reduction, optimizer or host work.
- Compare instruction/shared read/write/local-memory/stall behavior by complete
  contract membership. Checked controls have their own original shapes; they
  are not matched alternative implementations. Correlation is not causal latency
  attribution; an output-store slope/intercept is not automatically identified.

CPU preparation/review: one CPU,2GiB,zero swap,<=720s, no restart. GPU jobs use
the existing serialized guard/queue in one persistent service without CPU quota;
each process<=180s, whole allocation<=3600s, no restart. Expected5080/580.178.04,
>=2GiB sampled Vulkan headroom, record allocation warnings. Other numerical/API/
hard-fault/deadline failures stop for review. No NVML polling, clock change,
assembly retry, recovery, compiler feature, relaxed gate or automatic promotion.

Artifacts: `runs/matrix-feeding-counters-20261016.dKgyoaOO`.
