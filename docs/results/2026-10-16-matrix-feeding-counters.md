# Tiled matrices have substantial non-tensor work and shared-memory conflicts

October16,2026. [Declaration](../experiments/2026-10-16-matrix-feeding-counters.md)
· [Observer repair](../experiments/2026-10-16-matrix-counter-observer-repair.md)
· [Counter review](../experiments/2026-10-16-matrix-counter-review.md)
· [Data](2026-10-16-matrix-feeding-counters.json).

**Native counters now cover all54 actual tiled matrix contracts**, not just
representative shapes. All65 captures, including11 checked-path controls, have
zero local-memory traffic and the exact tensor-instruction count predicted by
their tiles and K partitions. Their output/operand checks and independent raw
counter-image decoding pass. The original exact shared-counter additivity check
fails and stays failed; the three counters are reported independently below.

Complete RGB remains **46.199ms versus inclusive JAX27.223ms,1.70×**; separate
CDP remains43.939ms. This diagnostic changes no backend or complete-update time.
Keep isolated58aff452; production and Atari remain stopped.

## What the counters establish

Weight each captured tiled contract by its actual complete-program occurrence
count, without extrapolating the checked controls to unmeasured shapes:

| Tiled products only | RGB | CDP |
| --- | ---: | ---: |
| Actual occurrences / distinct contracts |342 /41|387 /45|
| Warp instructions |597.413M|1,095.952M|
| Tensor instructions |50.024M|86.135M|
| Warp instructions per tensor instruction |11.94|12.72|
| ALU / LSU share of warp instructions |43.34% /16.50%|44.94% /15.52%|
| Shared read conflicts / read wavefronts |50.449M /155.153M|87.252M /270.280M|
| Shared write conflicts / write wavefronts |106.826M /166.472M|156.600M /259.311M|
| Local-memory read / write bytes |0 /0|0 /0|

These are measured native instruction/transaction counts under standalone
execution, **not time shares, wasted-compute percentages, a JAX instruction
ratio or a forecast speedup**. Different instruction classes have different
costs. Read conflicts account for32.52%/32.28% of their measured wavefronts;
write conflicts account for64.17%/60.39%. Unlike the convolution capture, matrix
reads matter too. The validated private-array positive control in the preceding
[memory study](2026-10-16-kernel-memory.md) establishes that the SDK can observe
local traffic; it is not a native register-count measurement.

The [source audit](2026-10-16-shared-store-contract.md) identifies32 explicit
scalar16-bit shared stores per K-stage despite vector global loads. That is a
concrete implementation candidate, but these counters do not assign instructions
or conflicts to source lines. Driver-managed lowering remains opaque.

The seven checked-vector controls have12.000–12.015 read conflicts per tensor
instruction, versus1.000–1.040 in the54 tiled cases. They have different tiles,
boundary handling and shapes: **this is not a matched alternative comparison**.
It warns against blindly copying a smaller kernel's shared layout merely
because its source stores are vectorized. All four checked-scalar controls,
all timings and all tails remain in the data; no favorable subset is selected.

## Numerical and capture boundary

The direct-Blade observer uses unchanged retained WGSL and the same linked
Meganeura/Blade/Naga libraries. All65 binding layouts match the backend's
production layout function. Actual launch geometry, prefetch, K partitions,
addends, output precision/layout and barriers remain unchanged. Device-local
buffers hold structured integer BF16 operands varying across row/column/K/group;
exact scalar F64 products and independent exponent/grid BF16 rounding verify
the saved values. These fixtures are not general real-input or learning parity.

Initial, sign-changed, restored and final checks cover335,646,720 output values,
plus128-byte guards;161,773,024 saved operand values reconcile. Restored/final
files match initial bytes exactly. All65 expected tensor counts, launch counts,
empty ranges and no-drop pass records match. All6,500 metrics across130 ranges
reproduce exactly from raw counter images. No backend rebuild occurs.

There are65 passing GPU guards and one retained failed guard, zero new allocation
warnings,2,405 counter passes and5,720 completed matrix dispatches, plus at least
two/at most three dispatches in the failed observer. Zero learner updates/actions.
Counters exclude packing, split reduction and surrounding graph execution.
Sixteen warmups/32 observation samples per case remain untrimmed; this is not an
optimization timing or steady-state qualification. No NVML polling, clock change
or recovery occurs. GPU work finishes13:26:37UTC; no worker remains.

## Failures remain visible

- Original tiled-023 exceeds180s before counter collection. The checker reads
  mapped staging element by element. Its70,778,880 partial output values pass
  an independent CPU check. A separately declared observer bulk-copy repair
  passes under the same deadline and matches both original partial files exactly;
  only this failed contract and41 unstarted contracts run. No successful capture
  repeats. The repaired70.8MB readback copy takes1.27–1.30s; checking takes94ms.
  This supports an observer bottleneck, not a learner optimization or proven
  cacheability diagnosis. The original truncated whole-history failure snapshot
  remains; incremental/time-bounded logs record no new warning or fault.
- The first failure-review reader uses the wrong kernel-query result field;
  the dependent preparation starts before a valid receipt exists. Both failures
  and their CPU-only repairs remain, without raising resource limits.
- The complete review's exact total=read+write conflict assertion fails44/65
  captures, up to.12737144%. Its source/log/controller remain failed. The
  separate CPU review preserves every difference and does not introduce a
  tolerance. Offline decoding still reproduces every original metric exactly.
  NVIDIA documents variability between replay passes; that is a plausible cause,
  not proof of the discrepancy or an accounting of other shared-operation
  classes. [NVIDIA profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#range-and-precision).

The independent numerical/decoding review finishes in4.235s at992.4MiB under
one CPU/2GiB/no swap. The publication audit verifies1,249 sealed files and all729
tiled memberships. No GPU retry is used for the counter-review correction.

## Next decision

CPU-prove one type-generic tiled operand-storage change using existing vector
storage/cooperative-load operations: preserve each operand's physical major
order when staging, and make the cooperative consumer express its logical
orientation. Cover all54 actual contracts and existing typed generator cases;
retain K order, prefetch, partitions, rounding, outputs, tapes and barriers.
Account for both read and write interfaces and all shared storage, not just
fewer source stores. This is not the rejected convolution packet-axis scheme.

Only a passing CPU proof can justify a separately bounded complete-family
numerical/counter/time comparison. No padding/tile/precision search, favorable
selector, compiler feature project, prior-batching timing or automatic promotion.
Convolution feeding, host/graph costs and the complete1.70× parity goal stay open.

Artifacts: `runs/matrix-feeding-counters-20261016.dKgyoaOO`.
