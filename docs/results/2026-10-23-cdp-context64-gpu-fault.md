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

## Resume direction

GPU recovery is needed before further dispatch. After recovery is confirmed,
review the new boot/driver and declare **only the unfinished frozen restore**
from `smoke/checkpoints/4096`; do not repeat completed training or restart old
queues. Require exact frozen tensors before declaring smoke success. Then
review the evidence before the fresh2M learning experiment. Backend parity
remains parked; the successful Freeway cohort is unaffected.

- [CPU review and retained fault events](../../runs/cdp-breakout-context64-20261023.5OA15Iew/fault-review.json)
- [Failed restore guard](../../runs/cdp-breakout-context64-20261023.5OA15Iew/smoke/queue/breakout-final/result.json)
- [Bounded kernel supplement](../../runs/cdp-breakout-context64-20261023.5OA15Iew/fault-kernel-window.jsonl)
- [Fixed learning plan](../experiments/2026-10-23-cdp-breakout-context64.md)
