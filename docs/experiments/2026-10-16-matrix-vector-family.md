# Compare the complete selected matrix family with vector staging

October16,2026. The [CPU storage proof](../results/2026-10-16-matrix-vector-contract.md)
passes. Keep native58aff452 and complete RGB46.199ms/JAX27.223ms unchanged.
This is a standalone matrix comparison, not backend integration or learning.

Run all54 actual tiled contracts in both original and proposed forms,108 jobs.
Keep their729 actual RGB/CDP occurrence memberships, geometry, types, prefetch,
K splits, global loads, rounding, outputs and barriers. Only the CPU-proven
physical-major vector shared storage and cooperative consumer layout differ.
Even-numbered contracts run control then candidate; odd-numbered contracts run
candidate then control. No favorable subset, geometry/precision/padding search,
new compiler feature, backend rebuild or changed runtime selector.

Use the unchanged buffered direct-Blade observer and50-metric/37-pass native SDK
configuration. Each job keeps three numerical dispatches,16 warmups,32 queued
observations and37 counter dispatches, with final output verification:88 matrix
calls. Require exact integer-oracle outputs for initial/sign-changed/restored/
final states and128-byte guards. Initial/changed bytes also match the retained
original F64-reviewed outputs. Independently verify every new saved operand/
output against those sealed references and all raw-counter metrics offline.
These BF16 integer fixtures are not real-input gradient or F16/F32 GPU qualification.

Retain every sample and all raw images. Report each shared metric independently;
the preceding exact total=read+write counter identity remains failed, not relaxed.
Expected launches/tensor counts, empty controls, source/binding contracts and
counter-pass integrity stay exact. Local traffic and all regressions remain visible.

Decision gates, fixed before execution:

- Both fresh original workload-weighted GPU means and queued-wall means must
  calibrate within10% of the retained original54-contract observations. This is
  an aggregate gate, not per-kernel calibration or a steady-state assertion.
- Candidate weighted GPU and queued-wall means must each improve by at least5%
  for both RGB and CDP occurrence sets. Use all32 observations per contract;
  queued wall is the whole32-dispatch batch divided by32, including recording.
- Both workloads must reduce total warp instructions and must not increase the
  sum of their independently observed shared read/write wavefront counts. This
  tests useful feeding work without merely moving the burden to reads. These
  sums are observations across replay passes, not an exact combined counter.
- Every numerical/source/capture check passes. A failed gate remains failed;
  no retime, subset selector or automatic follow-up allocation.

The observer excludes surrounding graph/packing/split reduction. A passing
family result only justifies a later orientation-aware backend implementation
and real-input/full-update qualification; it is not an attainable whole-update
saving or learning claim. The two unselected over-limit geometries remain an
integration requirement, not new test shapes or an excuse to change padding.

Use the existing serialized host guard/queue, expected5080/580.178.04 and boot,
>=2GiB sampled Vulkan headroom. Record allocation warnings; other API/numerical/
hard-fault/deadline failures stop for review. Each GPU process<=180s, service
<=3600s,12GiB/zero swap, no CPU quota, no restart. Preparation/review use one CPU,
2GiB,zero swap,<=720s. No NVML polling, clock change or recovery. Expected total:
9,504 dispatches,3,996 counter passes,216 ranges,zero updates/actions.

Artifacts: `runs/matrix-vector-family-20261016.LZ3Z1Zo0`.
