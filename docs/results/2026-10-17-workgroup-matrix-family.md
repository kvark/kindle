# Fixed upstream workgroup-matrix policy is slower

October 17, 2026. [Evidence](2026-10-17-workgroup-matrix-family.json) ·
[Declaration](../experiments/2026-10-17-workgroup-matrix-family.md) ·
[CPU contract](2026-10-17-workgroup-matrix-contract.md).

**Reject this policy.** All numerical checks pass, but complete weighted GPU
cost increases **7.25% for RGB and 17.56% for CDP**. The compiler/context controls
calibrate closely. No full-model trial, favorable subset, geometry search or
timing repeat follows. Keep qualified `e45b9c43`: complete RGB **43.197 ms versus
inclusive JAX 27.223 ms, 1.59×**. Production and Atari remain unchanged/stopped.

## Complete policy, including fallbacks and tails

Compare all **96 distinct operations**, preserving all **716 RGB/769 CDP**
matrix occurrences. The one upstream 32×64/K32/64-thread policy changes 61
contracts, covering 504/552 occurrences; grouped and unaligned operations retain
their original implementation. All original split partitions, partial buffers,
142 attached reduction occurrences and 130 direct cast occurrences stay.

Three arms use the same observer source and fixtures: current sealed libraries
with original shaders; upstream Blade `4845d652`/Naga `13f582dd` with original
shaders; and that upstream stack with the fixed policy. The unchanged generator
comes from Meganeura `5d6d9a36`. First position rotates by operation index.

| Workload / complete weighted operation cost | Current | Upstream, original shaders | Fixed policy |
| --- | ---: | ---: | ---: |
| RGB GPU | 11.932 ms | 11.939 ms | 12.797 ms |
| RGB queued wall | 12.973 ms | 12.997 ms | 13.824 ms |
| CDP GPU | 15.605 ms | 15.614 ms | 18.345 ms |
| CDP queued wall | 16.716 ms | 16.768 ms | 19.473 ms |

Upstream-original calibration is +0.058% GPU for both workloads and
+0.184%/+0.312% queued wall, within the declared 10% aggregate gate. Candidate
GPU regressions versus that control are 7.18%/17.49%; queued wall regresses
6.37%/16.13%. Thus both workloads fail the required 5% improvement against
both controls. Aggregate calibration does not establish per-operation equality.

Each condition retains all 64 timed samples after 16 warmups. GPU operation
cost includes matrix, reduction and cast passes; queued wall includes command
recording, submission and waiting. These standalone occurrence-weighted totals
are **not whole-update latency, additive savings, hardware utilization or DRAM
traffic**. No samples, tails or unchanged fallbacks are dropped.

## Correctness and lifecycle

Logical-device feature enablement, all 288 native binding checks, source/build
receipts and output guards pass. Initial, sign-changed, restored and final
buffers reconcile across all three arms. Independent CPU review checks actual
stored BF16 operands against F64 dot products and independent BF16 rounding:
**190,859,328 reference comparisons and 763,437,312 exact paired values** pass.
All 190,370,560 input values satisfy the declared fixture patterns. These
structured integer fixtures are not real-input/full-model gradient or learning
qualification.

All **288 GPU guards pass**, recording **20 new allocation warnings**. No
unfinished child, recovery action or NVML query is recorded. There are
**23,904 complete operations / 26,643 dispatches**, zero counter passes,
optimizer updates or game actions. GPU work finishes **04:30:34 UTC**; the
independent CPU review finishes **04:30:53 UTC**, checking 3,308 receipts.
No worker remains. CPU analysis keeps the one-CPU/2-GiB/no-swap limit; GPU work
was serialized with the declared time/resource limits. The preceding CPU
contract's four reader failures remain retained; this GPU queue and postreview
have no failed jobs or retries.

## What this rules out—and what remains

Enabling the newer workgroup interface with this fixed complete policy does
not resolve matrix execution cost. This is not a rejection of every upstream
geometry or its measured-search policy, and does not establish the hardware
cause of the regression. Existing convolution feeding and non-matrix recurrence
costs remain distinct; instruction-count parity is not latency parity.

Next use the retained native/JAX recurrence evidence to compare the remaining
non-matrix operators' actual reduction, dtype and materialization contracts on
CPU. Identify a concrete mechanism before proposing another allocation. Do not
reopen the rejected matrix policy, closed tile/precision searches or the failed
prior-batching qualification. No compiler project or learning promotion follows.

Artifacts: `runs/workgroup-matrix-family-20261017.JHLPwCPE`.
