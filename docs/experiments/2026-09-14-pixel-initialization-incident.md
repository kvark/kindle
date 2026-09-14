# Third PMU-halt incident: combined LeVJEPA/world initialization

Status: **GPU work stopped; candidate quarantined; root cause unresolved**.
At **September 14 01:49:19 UTC**, the first candidate N6 pixel window caused
another Xid 62 / PMU halt / Xid 154 Reset Required. The guard terminated and
reaped its direct child, preserved evidence and started no successor. The driver
continued emitting watchdog failures afterward. No reset, reload, reboot or
post-detection NVML query was performed by the agent.

This is not unstable reinforcement learning, a gradient mismatch or a failed
memory-margin assertion. It happens inside construction, before an actor is
ready to act or train. The source change restored allocation order but retained
deferred host zeroing; **allocation-order restoration alone was insufficient**.
Neither this candidate nor its block-matmul carry is eligible for new GPU work.

## Exact comparison and boundary

The [pinned ten-window declaration](../../runs/native-f32-alias-pixels-20260913.m6kNer/declaration.md)
retains N6/R256/B16/T64, full recurrence, all numerical/state/trace/memory gates,
and individually reviewed windows. The preceding qualified ce80 train/frozen
control passes: 3,840 training actions / 610 updates, then 768 frozen actions /
zero updates. Its complete checkpoint, all moments and non-timing traces match
the retained control exactly; minimum directly free memory is 3,303 MiB.

The failed index is **2, `pixel1-candidate-train`**, using Kindle **7728d8d**,
Meganeura **0a98775**, Blade **f6f2729e**, native extension **02b600a1** and driver
**595.91.07**, on boot **372a5604-b508-4f42-b109-ecab2084ffec**. The guarded direct
Python process hosts the Rust extension and synchronous ALE adapter; no GPU
worker descendants are used. This is **not the block-matmul package**.

The latest-source hardware group had passed all nineteen requirements, and all
six full-state canaries had passed exact weights, moments and reports. Those
remain valid isolated results, not qualification of the combined pixel runtime.
LeVJEPA still uses `Disabled`, the learner `Auto`; **NativeF32 was not selected**.
The [post-fault upstream recheck](../../runs/native-f32-pixel-incident-20260914.4ZQG26/upstream-recheck.json)
still finds Meganeura **428fc2d** and Blade **68a23e49**. The candidate contains
the former's exact policy patch; the latter changes only the unused renderer.
No newer initialization fix was available at that check.

| UTC event | Observation |
| --- | --- |
| 01:48:14.597 | Guard spawns direct child 99692 after complete predecessor/import/host checks. |
| 01:48:16.816 | Session 1, LeVJEPA, records initialization ready. Its loading returns before world construction. |
| 01:49:16.466 | Session 2, world training, starts: 55,247 dispatches / 83,295 logical buffers. |
| 01:49:17.487 | Its cooperative self-test wait records complete, no error. |
| About 01:49:19.448 | Kernel source timestamp projected into host realtime; world buffer creation is still in progress. This is host-clock correlation, not the originating GPU command. |
| 01:49:19.632–.667 | CPU finishes all physical-buffer creations and all Shared zeroing; no world pipeline-construction breadcrumb follows. |
| 01:49:19.695 | Journal receives the first Xid 62, then PMU-halted / Xid 154 records. |
| 01:49:19.761 | Guard detects the kernel fault, then signals its own child. |
| 01:49:22.172 | Guard command exits 1; child is reaped with SIGTERM (-15), no unfinished children. |

All **86 command-history records** reverify: 42 successful commands and the
terminal failed guard command. There is no candidate checkpoint, completed
phase-2 result, action/update log entry or later phase start. The empty Atari
output and native stdout are preserved, not filled in or called a learning run.
Never restart this comparison. The original Pong pair and seed-2017 output
reservation remain unchanged.

## What the new logs narrow down

The [retained-log analysis](../../runs/native-f32-pixel-incident-20260914.4ZQG26/result.json)
checks **46,520 kernel records**, including 1,893 in the new incident boot.
Exactly three retained Xid 62 incidents, across three boots, share the complete
payload from the [September 13 investigation](2026-09-13-gpu-forensics.md):

```text
324b06ec 0000c4c8 00000000 206db166 206da31e 206da48c 206d879e 206d8f8e
```

This boot adds one Xid 154, three channel-cleanup Xid 45 records, four Xid 109
context-switch timeouts and 47 locked-GPU watchdog messages through the later
01:56 host-only snapshot. Five GSP task-watchdog reports retain firmware build
**4afb346010dfe4194bb348767b732373eaf867a7**. No concurrent kernel OOM, PCIe AER,
machine-check or driver API-mismatch record appears in this boot. That absence
does not certify hardware health. NVIDIA categorizes Xid 62 as an internal
microcontroller halt; Xid 154 reports the recovery action, not another root cause.
[NVIDIA Xid catalog](https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html)

The **23,077 flushed initialization records** contain:

- Complete frontend initialization: 632 physical buffers, all 611 Shared zeros,
  pipelines and checked device-zero wait.
- Partial world initialization: all 9,439 buffer creations and 846 Shared zeros
  finish, but no world pipelines, device-zero submission, optimizer initialization
  or session-ready marker appears. Constant uploads lie in the uninstrumented
  interval after the final zeroing marker; that interval is not the first fault.
- The world allocation plan and **all 20,570 observed creation/zeroing events**
  match the corresponding prefix of each of the **three successful same-backend
  canaries** exactly. Logical shapes and slot order alone do not explain the
  difference. Frontend residency and the shared context's preceding work matter
  to the next diagnostic; they are not proof of cause.

Constructor ordering is explicit in
[vector.rs](../../kindle/src/dreamer/agent/vector.rs),
[LeVJEPA loading](../../kindle/src/vision/levjepa.rs) and
[core construction](../../kindle/src/dreamer/agent.rs); the experiment pins the
candidate copies. No world learning shader or optimizer update is reached.
The cooperative probe is GPU work and precedes the allocations; its completed
wait does not retrospectively prove every possible driver/firmware side effect
was harmless.

## Clock and telemetry correction

The [separate clock reader](../../runs/native-f32-pixel-incident-20260914.4ZQG26/clock-result.json)
finds **247.318 ms** between kernel source timestamp and journal receipt, then
**65.438 ms** until guard detection: about **312.758 ms source-to-detection**.
The source event is approximately **184.572 ms before world host zeroing begins**.
Do not blame the final host-zero slot for an already-recorded fault.

The last NVML request **straddles the source event**, yet returns 0% activity,
recovery `None`, 6,385 MiB free, 32°C and 13.46 W. All 211 retained samples precede
guard detection; their maximum sampled temperature/power is 33°C / 45.75 W.
These are sampled readings, not proof against a transient or current health.
The first analysis's `pre_fault_health` field means *before journal receipt*,
not necessarily before hardware failure; its [label clarification](../../runs/native-f32-pixel-incident-20260914.4ZQG26/interpretation.md)
is preserved alongside the unchanged reader. There are **zero NVML queries after
fault detection**. The incident cannot supply a memory-gate pass or idle-time
measurement.

## Containment, evidence and next decision

The guard's automatic capture verifies **27 sealed files**; the subsequent
host-only capture verifies **21**. Twelve incident-reader and six clock fixtures
pass. Independent read-only analysis reproduces all **89,308 original input pins**,
both complete raw control windows, 86 command records, **224 direct evidence pins**
and three exact canary-prefix comparisons. The clock supplement independently
reverifies its five pins. Completed readers and raw captures remain immutable;
only their documented `--audit` modes may be reused.

The [allocation input extraction](../../runs/native-f32-pixel-incident-20260914.4ZQG26/allocation-inputs.json)
prepares the exact two session plans without constructing graphs or touching the
GPU. It is data for a smaller reproducer, **not a runnable GPU test**. Actual
Vulkan memory type/heap/offset and allocator suballocation identities are missing;
9,439 Blade buffers do not necessarily mean 9,439 Vulkan memory allocations.

The next diagnostic should retain the N6 frontend's loading/residency and stop
after world initialization, before any training. First add matched allocation
and constant-upload observability, including actual Vulkan memory placement.
Compare one explicit schedule hypothesis against a qualified control; restoring
the old immediate create-and-zero schedule is an available hypothesis, not an
established fix. Do not change driver, precision, batch size and initialization
policy together. A local vendor-report brief is prepared; nothing is uploaded.

**No further GPU stage is declared.** The CPU-only block helper LK2cCI is stopped
on its failed dependency prerequisite. No backend adoption, throughput gain or
new Atari competence result follows. Main remains ce80; full hardware/state/
pixel/memory and same-backend block timing gates still precede held Pong work.

Recovery requires the user's approval. Earlier reset was unsupported and normal
module reload failed GSP initialization; do not repeat either or improvise PCI
resets. Prefer a user-controlled full shutdown/power-on, then fresh health and
identity verification, without automatically restarting old jobs. The
[runbook](../gpu_incident_response.md) now includes delayed host-only capture,
terminal-result-first monitoring and the clock-domain caveat.
