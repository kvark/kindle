# Packed transfer width: one fixed-contract native proof

Declared October15,2026, before any build or GPU run for this intervention.
Motivation: the [whole-family contract audit](../results/2026-10-15-convolution-contract.md)
finds8-byte typed operand expressions in current packed convolution,16-byte
aligned library accesses, and a legal packed-pointer cooperative-load contract
that Naga cannot currently represent. This tests transfer representation, not
async staging or a claim that it explains the full2.04× learner gap.

Do not modify Naga/Blade or production yet. Use a small isolated Vulkan harness
and one shared GLSL source with F32/F16/BF16 matrix/storage parameters. Its
two variants differ only in typed4-component versus16-byte bit-pattern operand
loads/shared stores. Preserve source bits; this is not numeric conversion.
Use ordinary subgroup cooperative matrices, not the rejected workgroup-matrix2
prototype. All barrier participation and valid address ranges stay intact.

Cover all14 currently packed occurrences from the sealed B32 world plan:
five forward, five dX, four dW (11 distinct contracts). Reuse exact actual
shapes,64×64/K32 geometry,128 threads, channel padding, NHWC/KRSC transforms,
existing output scatter and dW split/reduction schedule. Keep256-position
F32 compensation,4,096-position partials and existing BF16-forward/F32-derivative
outputs. No layer selection, geometry/dtype/prefetch sweep, halo expansion,
full im2col buffer, altered final rounding or changed learning workload.

Before GPU timing:

- CPU check both variants' complete address/stride/vector bounds, tail masks
  and exact output coverage. All three scalar variants compile/validate;
  F32/F16 compilation is not hardware qualification.
- GPU BF16 fixed/generic/native outputs: independently generated random,
  cancellation and dynamic-range inputs, F64 direct convolution references
  and long dW reductions. F32 derivatives retain pointwise
  `1e-4 + 1e-5*abs(reference)` and relative L2`2e-5`. BF16 forward uses the
  existing independently BF16-rounded F64 reference/envelope: pointwise
  `1e-4 + .008*abs(reference)` and relative L2`.004`, as in the family check.
  Preserve every failed value and stop on any failure.
  Input/output canaries and changed/restored inputs must pass. Compare all
  native/generic/packed outputs, with exact agreement expected when reduction
  order is unchanged; any difference requires review before timing.
- Count all input transforms, output scatter/conversion and dW final reductions
  in complete-operation time and scratch. No kernel-only promotion.

One ordinary native/generic8-byte/packed16-byte timing set, each16 warmup and
128 measured calls per contract, with queued wall and GPU intervals retained.
No profiling counters or concurrent compilation. The generic8-byte control
must reproduce native within10% weighted complete-operation time, and disclose
every per-contract difference, before attributing its16-byte comparison to
the native stack. Failure stops interpretation; no calibration sweep.

A useful candidate needs at least10% less weighted complete-operation time
than the fresh generic control and fresh native, retaining every occurrence
and all transforms. Keep all128 samples, including transitions/outliers;
no faster-subset selector or automatic repeat. Even success is not a full-update
claim. Only then propose the minimal typed-pointer Naga lowering change and
one qualified complete RGB B32 update pair; commit permission is separate.

Use serialized guarded GPU jobs in a persistent systemd user service,
Restart=no,KillMode=control-group. Numerical process<=180s, timing<=180s per
contract, overall<=45min after CPU preparation. Expected5080/driver580.178.04/
boot3e89d55c-a9e5-472f-a18a-06508c5bafa7 and>=2GiB sampled Vulkan budget headroom.
Record standalone allocation warnings; stop affected stages for API/numerical/
hard/deadline failures and review before follow-up. No recovery operation,
separate NVML polling or game action. CPU build/review uses one CPU,2GiB,no swap.
Seal source/binaries/inputs before GPU use and recheck upstream first. Keep all
failures and original controls. No learning restart or root-goal completion.
