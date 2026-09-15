# Fourth PMU halt: combined initialization remains unsafe

Status: **GPU work stopped; the latest candidate failed before world training**.
The sole invocation in [7MdNse](../../runs/interleaved-latest-init-20260915.7MdNse/declaration.md)
faults on the recovered efe90b23 boot. The guard terminates and reaps its direct
child with SIGTERM; no training, retry or successor follows. The earlier passing
[combined control](2026-09-15-recovery-and-initialization-control.md) and
[interleaved candidate](2026-09-15-interleaved-initialization.md) remain valid
single executions, not evidence that this initialization change is reliably safe.

Keep 070f4b51 and 7db0d05c quarantined from further GPU work, alongside the earlier
0a98775/native 02b600a1 bundle and its block carry. Main stays ce80. All hardware,
state/pixel/memory and same-backend throughput gates remain ahead of held Pong.
No recovery or further GPU query is authorized by this report.

## Latest upstream was included

The first release-fixture preparation, [6hLrlv](../../runs/interleaved-fixtures-cpu-20260915.6hLrlv/build/result.json),
stopped on a historical reader's boot assertion. Its separate continuation,
[hduqEv](../../runs/interleaved-fixtures-cpu-v2-20260915.hduqEv/build/result.json),
stopped when upstream advanced. Neither copied a target, compiled fixtures or
ran a GPU job. Preserve both writers and their failures. The corrected historical
reader needs the existing NumPy-equipped virtualenv with `-I`; its standalone
raw Freeway/Pong/hold check passes without changing the original launch guards.

Meganeura **5a570099** adds Q5_K/Q3_K support, packed-concat regression fixes and
extracted dequantization shaders after 09f7c410. These changes leave the
initialization body unchanged. The exact interleaved initialization patch is
carried forward, producing:

| Role | Kindle | Meganeura | Shared Blade | Combined initialization |
| --- | --- | --- | --- | --- |
| Historical control | ae7699ad | 9b9e7ee7 / ce80 | c96a9a87 / published 0.9.0 | Pass; corrected read-only F32 reader |
| Earlier upstream candidate | 51ba190c | 7db0d05c / 09f7c410 | 100bb813 / 6ab5fcec | One pass in WeUPsV |
| Latest upstream candidate | 80d2d2f9 | 070f4b51 / 5a570099 | Same 100bb813 | Failed in 7MdNse |

All three Kindle fixtures have the same production/test bodies and settings.
The latest source changes only dependency/locks, reported identity and
instructions. Its native executable is SHA256
`0379649d073bf3a172f3fe71f2969d629eb5436b4be08e0dd064e34ebc957469`.
The [HEURED CPU build](../../runs/interleaved-latest-cpu-20260915.HEURED/build/result.json)
passes 138 Rust tests (83 Kindle, 13 Blade, 42 focused backend), six source
fixtures, formatting and release Clippy. All twenty command lifecycles, 781
inputs and 324 artifact/source pins reverify. GPU tests stay ignored; no Python
wheel or full hardware group is produced. Preserve the private target.

The ten-test/960-pin initialization declaration and its fresh same-boot health
checks pass before the single native launch. The later **15:39 UTC host-only
upstream recheck** still finds 5a570099 and Blade 6ab5fcec; there is no missed
new upstream initialization fix in that check. The differences between snapshots
do not prove that packed-format changes caused the crash. There is only one
execution of each candidate, with other runtime/cache/timing state not isolated.

## Terminal evidence and clocks

The direct native child is PID **69243**, spawned at **15:29:35.815 UTC**.
It loads the N6 LeVJEPA frontend and constructs only the first production world
session, retaining all eleven CPU graphs. It never acts, initializes D3 weights,
updates, saves, restores or constructs later learner sessions.

| Event | September 15 UTC | Interpretation |
| --- | --- | --- |
| First Xid 62, projected from source monotonic time | 15:30:37.539482 | Hardware-reported event, not its originating GPU command |
| Journal receipt of Xid 62 | 15:30:38.529019 | 989.537 ms after the source event |
| Guard detects NVML recovery `Reset` | 15:30:38.566489 | 1.027 s after source; 37.47 ms after receipt |
| First retained GSP task watchdog receipt | 15:30:43.528088 | Driver remains unhealthy after direct-child termination |

All four incidents retain the same Xid 62 payload:

```text
324b06ec 0000c4c8 00000000 206db166 206da31e 206da48c 206d879e 206d8f8e
```

PMU halt and Xid 154 GPU Reset Required follow; delayed Xid 45/109 and GSP
watchdogs continue in the later host-only snapshot. The GSP build remains
`4afb346010dfe4194bb348767b732373eaf867a7`. The current-boot capture contains
no OOM, AER, machine-check or driver API-mismatch message. Absence of these
messages does not exclude hardware or driver faults.

This time **NVML detects failure before the next kernel poll**, not a
`kernel_fault` event in the guard. Three accepted health rows occur after the
projected source event while still reporting recovery None. Four NVML calls
start after that source event, including the final Reset response. **None
starts after detection.** The 204 accepted rows reach a minimum reported
4,977 MiB free, maximum 32°C and 45.42 W; they are not a completed health/memory
gate or evidence against an unobserved transient. The child exits -15, no child
remains unfinished, and the top-level success result stays absent.

The clock conversion uses the same journal record's realtime-minus-monotonic
offset. It correlates host records; it does not synchronize a CPU breadcrumb
with GPU execution. Do not call the last CPU operation the fault origin.

## What the added observations establish

The independent reader verifies **137,607 complete trace records**. A final
122-byte JSON fragment is preserved but not interpreted as a completed event.
All **46,905 complete Kindle/Meganeura records** exactly match the corresponding
prefix of both passing runs after removing only process/clock/line metadata.
Both exact allocation plans match, and all actual traced bindings are checked.

- Frontend initialization completes: 632 physical slots, 611 Shared zeros and
  339 constant uploads, with its checked wait passing.
- World cooperative-probe wait completes. All 9,439 physical buffer creations
  and 846 immediate Shared zeros are eventually logged. There are 11,571 complete
  constant-upload pairs; the next upload's completion is the truncated record.
- At the projected source event, only 1,129 world creations have completed, all
  846 Shared zeros have completed, and constant uploading has not begun.
  The source time falls between allocator observations for `buf_1129`; this
  is correlation, not evidence that this buffer or allocation caused the halt.
- No world pipeline creation, device-local zero submission, optimizer allocation,
  world-ready record or learning follows. This is not a gradient-value mismatch.

The separate [placement reader](../../runs/interleaved-init-incident-20260915.Xd7nhF/placement-cpu/result.json)
checks the 10,071 simultaneously retained frontend/world physical buffers in
all three runs. Their recorded extents do not overlap within any of the 227
observed memory handles. After renaming opaque handles by first appearance,
all offsets, extents, memory types and sharing relationships match exactly.
This rejects a recorded overlapping-suballocation explanation for these retained
buffers. It does **not** check every temporary's lifetime, underlying Vulkan
allocation capacity, driver state or synchronization. `block_bytes` comes from
the allocator's `MemoryBlock::size()`, not the capacity of its parent
`VkDeviceMemory`; request counts are not `vkAllocateMemory` call counts.

Immediate zeroing, exact logical plans and matching recorded placements are
therefore insufficient safety evidence. The failure is narrower than the old
full Atari constructor, but its cause is still unresolved.

## Preserved analysis and next boundary

[Xd7nhF](../../runs/interleaved-init-incident-20260915.Xd7nhF/declaration.md)
contains the guard's original evidence, a separate later host-only snapshot,
fresh source comparison and independent terminal analysis. It verifies all 960
declared inputs, 125 direct evidence pins, 76,108 retained kernel records and
both complete passing controls through their original readers. The terminal
reader's 23 CPU tests and placement reader's nine CPU tests pass; both complete
analyses independently re-audit. No GPU query is made by either reader.
Completed writers remain immutable; reusable read-only commands are:

```bash
python3 -B runs/interleaved-init-incident-20260915.Xd7nhF/capture_analysis.py --audit
python3 -B runs/interleaved-init-incident-20260915.Xd7nhF/placement.py --audit
```

The [local vendor brief](../../runs/interleaved-init-incident-20260915.Xd7nhF/vendor-brief.md)
is prepared, not submitted. NVIDIA classifies Xid 62 as a microcontroller halt
and recommends reporting it; Xid 154 describes the required recovery action,
not a separate cause. [Xid catalog](https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html),
[reporting guidance](https://docs.nvidia.com/deploy/gpu-debug-guidelines/gpu-node-triage.html#reporting-a-gpu-issue).
Those data-center recovery tables do not override this GeForce card's observed
`GPU reset: Not Supported` result.

Pause broad backend qualification. The next decision is an **operator-approved
driver/runtime investigation**, potentially a controlled driver comparison or
vendor-guided reduced reproducer, not another attempt to accumulate passing
initializations. A driver comparison must first verify supported package/kernel
compatibility and retain the same application build/fixture; it is not authorized
here. The current fixture hard-codes driver 595.91.07: a cross-driver comparison
needs a newly declared, driver-bound fixture, with the same new binary in both
arms. Do not edit old fixtures or falsify reported driver identity. Any reduced
reproducer must explicitly retain or isolate the resident
frontend, cooperative probe and allocation/upload history. No Vulkan reproducer
is launched or declared by the current CPU readers.

Follow the [recovery boundary](../gpu_incident_response.md#recovery-boundary).
The agent has not reset/reloaded/rebooted, changed drivers, run a vendor ioctl
collector or sent an external report. Recovery requires renewed user approval.
Even recovered health will not release the quarantines, full qualification
requirements or Pong hold. The five-game goal remains incomplete.
