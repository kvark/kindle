# B2/T64 smoke: training completes, restore stops on a real GPU fault

The longer-context smoke executes4096 actions/867 updates in45.245s, first
update at action632 and zero debt. Independent CPU accounting, finite346-tensor
checkpoint checks and exact346 initial tensors against the B8/T16 control pass.
This does **not** qualify the complete smoke: frozen restore fails before
run_start and before the first constructed-memory record. No2M training starts.

## Incident

On driver615.71.09, boot`76297c0a-a477-4dbb-8426-2188b4d558f7`:

- 05:16:55UTC October23: training guard passes and exits normally.
- 05:16:55: the separate frozen-restore process starts.
- 05:17:15: current-boot kernel records Xid62 on PCI0000:01:00, followed by
  `PMU has halted` and Xid154 changing recovery action to `PF FLR`.
- 05:17:17: guard detects the fault and stops the child. It exits−15; no
  unfinished children remain. Service terminates05:17:25.
- Subsequent logs contain `NV_ERR_RESET_REQUIRED`, Xid45 and repeated Xid109
  `CTX SWITCH TIMEOUT`, continuing through05:20:30 in the bounded supplement.

Both restore output logs are empty. The exact initialization substage and cause
are unknown; do not blame sequence length, a shader or NVML from chronology.
The job and guard make no NVML calls. Other processes have not been excluded as
contributors. This is a real firmware/driver failure, not the tolerated
allocation warning or an observer-only assertion.

The original guard's broad kernel snapshot exceeds its output limit and mixes
historical boots; retain that incomplete capture. The sealed incremental
current-boot events contain the primary fault. A separate bounded current-boot
time-window capture retains36 matching records, and the CPU-only review makes
zero GPU calls. No guard bypass, GPU retry, reset, driver reload or reboot was
performed. No learner remains active.

## Restore fails again after recovery onto 595.91.07

Boot `dddae8da-ad5c-4cd0-a9d8-8105d2eafb1d` starts with no recorded kernel
fault in the guard's preflight. Only the unfinished frozen restore is launched,
using the unchanged qualified binary and the completed4096-action checkpoint.
No completed training is repeated and no full2M run starts.

- 06:39:00UTC: the restore child starts on driver595.91.07.
- 06:39:20: Xid62 and `PMU has halted`; Xid154 requests GPU reset.
- 06:39:25: the guard detects the fault and stops the child; exit−15.
- The service terminates06:39:28. No unfinished children remain. Subsequent
  reset-required errors and Xid109 context-switch timeouts persist.

The CPU review verifies the binary,329 source hashes, controller and all sealed
guard artifacts. Both run and memory logs remain empty, with no frozen export.
A bounded06:39–06:45 current-boot supplement retains546 kernel records. The
first supplement command rejected a hyphenated journal boot ID; removing the
hyphens repairs only log collection, with no GPU retry. Original guard records
remain unchanged. No job/guard NVML calls or agent-initiated recovery.

This repeats the failure across two drivers and boots, but does not establish
whether the cause is application code, the driver/firmware, or hardware.

### CPU inspection and next diagnostic boundary

Inspection uses the exact qualified source, not the current upstream backend:

- Fresh `learned_rgb` constructs the core before the pixel preprocessor;
  `restore_rgb` constructs the preprocessor before the core. The preprocessor
  constructor creates a pipeline and command encoder but submits no frame work.
- Restore constructs the ordinary core, loads world/behavior/slow-value tensors,
  restores normalization, then synchronizes inference weights. The loader
  validates tensor layouts before writing, waits for its session, and allocates
  Adam moments before restoring them. Device writes use bounded staging copies
  with waits. This inspection finds no proven bad transfer or missing wait.
- Crucially, an empty `constructed` memory log does **not** prove that the native
  constructor failed: device identity and the first Vulkan memory-budget query
  also precede that record. No existing trace distinguishes these stages.

After recovery, localize those stages with flushed host markers before another
restore attempt: constructor entry/exit, core construction, each checkpoint
load, weight synchronization, and initial device/budget queries. Keep GPU
operations and model semantics unchanged; no bypass of the failed restore gate.
Do not repeat the same uninstrumented launch or begin full training blindly.

The future guard's snapshot now selects only the current boot's last4096 kernel
records and marks a reached limit as incomplete. Previously it requested100,000
records across boots, exceeding64MiB in both incidents. All133 guard CPU tests
pass, including current-boot scoping and truncation reporting. This is incident
capture maintenance, not a runtime/backend change or GPU qualification.

- [Second incident CPU review](../../runs/cdp-context64-restore-20261023.a2He3BM1/fault-review.json)
- [Second incident kernel supplement](../../runs/cdp-context64-restore-20261023.a2He3BM1/fault-kernel-window.jsonl)
- [Second failed restore guard](../../runs/cdp-context64-restore-20261023.a2He3BM1/queue/breakout-final-recovery/result.json)

## Resume direction (updated after the second fault)

GPU recovery is needed before further dispatch. After recovery is confirmed,
review the new boot/driver and localize initialization as described above.
The unfinished frozen restore is from `smoke/checkpoints/4096`; do not repeat
completed training or restart old queues. Require exact frozen tensors before
declaring smoke success. Then
review the evidence before the fresh2M learning experiment. Backend parity
remains parked; the successful Freeway cohort is unaffected.

- [CPU review and retained fault events](../../runs/cdp-breakout-context64-20261023.5OA15Iew/fault-review.json)
- [Failed restore guard](../../runs/cdp-breakout-context64-20261023.5OA15Iew/smoke/queue/breakout-final/result.json)
- [Bounded kernel supplement](../../runs/cdp-breakout-context64-20261023.5OA15Iew/fault-kernel-window.jsonl)
- [Fixed learning plan](../experiments/2026-10-23-cdp-breakout-context64.md)
