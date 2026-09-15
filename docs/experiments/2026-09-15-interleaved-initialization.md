# Interleaved initialization on current upstream

Status: **the distinct candidate passes the combined initialization diagnostic**.
This follows the completed [combined initialization control](2026-09-15-recovery-and-initialization-control.md),
not a retry of the quarantined pixel bundle. Training and held Pong roots remain
stopped pending full dependency and same-backend throughput qualification.

## Why this hypothesis

The failed 0a98775 candidate restored physical allocation order but still created
all buffers before zeroing Shared memory. The qualified ce80 control instead
zeroes each Shared buffer immediately after creating it. Its combined frontend/
world initialization now passes with matched flushed observations on the recovered
host. Restore that entire sequence on current upstream as the next narrow test.

This does not establish the historical fault origin. In the third incident the
kernel source event predates the first world host-zero breadcrumb; neither that
operation nor a stale NVML row identifies the failing hardware operation. An
initialization pass would make this candidate eligible for further qualification,
not prove that deferred zeroing caused the fault or that the driver is fixed.

## Source boundary

The [fresh host-only/source capture](../../runs/initialization-hypothesis-cpu-20260915.twWdPt/source-capture/result.json)
finds the same recovered boot efe90b23 and loaded/on-disk driver 595.91.07 without
querying the GPU. Meganeura main advances to **09f7c410**; Blade stays **6ab5fcec**.
The backend updates add GGUF/packed-weight fixes, including subnormal decoding,
storage sizing and packed concatenation/restaging. They do not change the
initialization allocation/zeroing schedule. Keep them rather than rediscovering
or discarding upstream work.

The isolated candidate is Kindle **51ba190c**, Meganeura **7db0d05c**, shared Blade
**100bb813**. The backend restores alias-order create/immediate Shared-zero
sequencing, retaining parameter placement and the explicit zeroing opt-out. It
keeps current upstream math, defaults and cooperative policies. Its flushed
initialization/upload observations and three checked waits match the control.
Blade retains current upstream shader-validation fixes and the exact existing
allocation observations, with no allocator policy change or extra GPU query.
The native fixture and production Kindle bodies remain identical to control;
only dependencies/locks, reported identity and instructions change. All three
branches are committed and pushed. Main remains ce80.

The old constant-upload instrumentation patch initially fails context validation
because upstream added a surrounding tracing span. The explicit adaptation puts
the same records inside that span; there is no partial failed edit or old-writer
rerun. The [development notes](../../runs/initialization-hypothesis-cpu-20260915.twWdPt/development.md)
preserve that preparation issue.

## Validation boundary

The [CPU preparation](../../runs/initialization-hypothesis-cpu-20260915.twWdPt/declaration.md)
uses a new private release target, one job / 100% CPU quota, 2 GiB cap and zero
swap. Six source/command fixtures pass. All 83 Kindle tests pass with 23 GPU
tests ignored, thirteen Blade tests pass, and the focused backend modules pass
9 initialization + 4 packed + 4 checkpoint + 20 GGUF tests. Formatting and release
Clippy pass for all three libraries. The independent read-only audit verifies
all twenty command lifecycles, 760 inputs and 308 artifact/source pins after
commit. Complete cached-source equality covers 107 backend files including
68 WGSL files and 24 Blade source files. No Python wheel is built here.
The trace test's mocked DeviceLost stderr is a CPU fixture, not a GPU fault.

The source review finds an unignored GPU test in Meganeura's library suite.
Only the explicitly inspected initialization, packed-concat/quantization,
checkpoint and GGUF modules are selected for backend CPU execution. Full-library
execution is not a CPU-only check. The eventual reader also requires Git source
identity, shared Blade edges, complete source/cache equality including WGSL,
actual compiler artifacts/depfiles and both locks.

The [one-job boundary](../../runs/interleaved-init-runtime-20260915.WeUPsV/declaration.md)
has ten passing CPU refusal checks and a separate 931-pin declaration at
14:57:52 UTC. It reverifies complete CPU evidence and the original corrected
control audit, then fresh source/host checks. The same recovered boot is clean,
matching 595.91.07, recovery None, 0% activity and 15,841 MiB directly free.
No native invocation starts as part of declaration.
The unchanged fixture retains N6 LeVJEPA, all eleven production CPU
graphs and only the first full world session. It performs no acting, D3 parameter
initialization, update, checkpoint, restore or later GPU session. Exact plans,
interleaved zeroing, complete allocation/bind/upload/wait traces, actual device,
direct-child containment and the 2 GiB directly-free margin remain required.

## Completed candidate result

The sole native invocation runs at **14:59:13–15:00:16 UTC**, PID 55770. It exits
zero, releases both sessions and is reaped by the guard. The complete independent
read-only [audit](../../runs/interleaved-init-runtime-20260915.WeUPsV/result.json)
reverifies all 931 inputs and raw results. Unlike the old control controller,
this declared reader uses the already tested exact-F32 interpretation from the
start; no post-execution reader change or native rerun is needed.

Both exact plans match: 632 frontend and 9,439 world physical slots, with 611 and
846 interleaved Shared zero operations. All 10,081 buffer/allocator pairs,
339 frontend and 14,306 world constant-upload pairs, checked waits and 143,121
trace records pass. The observed memory types are 1 and 4, as in the control;
this is not proof of identical physical placement or allocation lifetime safety.
All **207 fresh health samples** pass, maximum gap **0.5729 s**, minimum directly
free **4,973 MiB**, with no kernel fault or unfinished child. There are exactly
two initialized sessions and **zero actions or updates**.

This completes the diagnostic; preserve its writer and use only `run.py audit`.
It tests the previously failing combined initialization context successfully,
not the remaining hardware, full-state, pixel/restore/memory or throughput gates.
The initialization-only fixture is additional to the original nineteen required
hardware tests; it does not substitute for the production loss/gradient test.
The next boundary is source-matched release preparation and separately guarded
qualification of those requirements. No new hardware group or successor is
declared by this pass. Do not modify the completed CPU target in place; preserve
its pinned artifacts if a separately declared fixture build reuses a copied cache.

No retry, automatic successor, pixel qualification, speedup or adoption follows.
Keep 0a98775/native 02b600a1 and its block carry quarantined, all completed/failed
writers immutable, and the original Pong hold intact. No host recovery is
performed or authorized by this work.
