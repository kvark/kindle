# Complete fixed workgroup-matrix policy comparison

October 17, 2026. The [CPU contract](../results/2026-10-17-workgroup-matrix-contract.md)
and independent observer preflight pass. Keep qualified `e45b9c43` and the
43.197/27.223 ms, 1.59× complete-update gap. No production or Atari change.

Run **all 96 current distinct matrix-operation contracts**, retaining all 716
RGB/769 CDP memberships. The 61 eligible dense contracts use exactly the
CPU-qualified upstream 32×64/K32/64-thread policy. The other 35 retain their
original grouped/unaligned implementations. No selector or geometry search.
Include every attached split reduction and direct cast: 142 `SumRows` and
130 `ToBF16` occurrences in the combined model inventory.

For every contract, compare three conditions, rotating first position by index:

1. Current sealed Blade/Naga libraries, original shader and invocation grid.
2. Upstream Blade `4845d652`/Naga `13f582dd`, original shader and grid.
3. The same upstream observer, fixed workgroup shader/grid when eligible;
   otherwise the same original implementation.

This separates compiler/context refresh from the new matrix implementation.
It is an isolated observer, not a native Kindle refresh. Upstream heads remain
Meganeura `5d6d9a36` and Blade `4845d652` at the final preflight check.
No compiler feature, whole-backend rebuild or SDK counter capture is added.

All 288 binding checks, 4,737 buffer extents and 2,136 small-layout inverses pass.
Original matrix operands and integer-dot oracle are unchanged from the retained
buffered observer. Check initial/sign-changed/restored/final raw partials and
completed reduction/cast outputs with 128-byte guards. Independent CPU review
uses actual stored operands, F64 references and BF16 rounding; paired bytes must
match exactly. These structured fixtures are not real-input gradient/learning
qualification. Current tail shaders match retained model shader bytes.

Per condition: three numerical operations, 16 warmups, 64 timed operations and
final output verification. **288 guarded jobs, 23,904 operations and 26,643
dispatches** total, zero actions or optimizer updates. Retain every sample and
tail. Operation GPU time includes its matrix/reduction/cast passes; queued wall
includes recording, submission and waiting. These standalone weighted totals
are not ordinary full-update performance or additive model savings.

Fixed gates for both RGB and CDP occurrence sets:

- Upstream-original GPU and queued-wall totals calibrate within 10% of the
  current-stack originals. This is aggregate, not per-kernel calibration.
- Candidate GPU and queued-wall totals improve by at least 5% relative to
  **both** originals, including unchanged fallback operations and tails.
- Every numerical, binding, source, lifecycle and guard check passes.

A failed gate remains failed: no retime, favorable subset or automatic next
allocation. Review before any real-input/full-model trial; no automatic adoption.

Use the existing serialized host guard/queue in a persistent user service:
12 GiB, zero swap, no CPU quota, no restart, KillMode=control-group, ≤3600 s;
each GPU process ≤180 s. Require the expected 5080/580.178.04/boot and ≥2 GiB
sampled Vulkan budget headroom. Record allocation warnings; API/numerical/hard
faults and deadlines stop for review. No NVML polling, recovery or clock changes.
Preparation/review uses one CPU, 2 GiB, zero swap and ≤720 s; no compilation
during timing. Independent preflight finishes **04:04:19 UTC**.

Artifacts: `runs/workgroup-matrix-family-20261017.JHLPwCPE`.
