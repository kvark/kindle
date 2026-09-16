# Driver-580 initialization comparison without NVML

The user approved exactly two initialization-only executions: ce80 control,
then the quarantined candidate only after the complete control passes, stopping
on any failure. This is a narrow diagnostic exception, not a quarantine lift,
runtime adoption, training restart or authorization to recover the host.

The new declaration is
[`bT2bAx`](../../runs/driver580-host-init-20260916.bT2bAx/declaration.md), with
[10,125 source/artifact/experiment pins](../../runs/driver580-host-init-20260916.bT2bAx/manifest.json).
Every arm has its own exclusive attempt, direct native child, result and explicit
invocation. There is no run-all mode, retry or automatic successor.

## Fixed comparison

| Arm | Kindle fixture | Meganeura | Shared Blade | Native SHA256 prefix |
| --- | --- | --- | --- | --- |
| Historical ce80 control | `f7b1914` | `9b9e7ee7` | `c96a9a87` | `85a49d6e` |
| Historical quarantined candidate | `eebbf7c` | `070f4b51` | `100bb813` | `20cbdd11` |

The control backend is ce80 with matched initialization observations and checked
waits; its Blade is published 0.9.0 with observations. The candidate retains
Meganeura main 5a570099 plus interleaved alias-order allocation/immediate Shared
zeroing and observations; Blade retains 6ab5fcec plus observations. Neither
runtime was changed to obtain this driver's result.

Both ignored fixtures are byte-identical. The candidate's only changes from
80d2d2f9 are the driver's explicit selection/assertions, their CPU tests and
worktree instructions. Its shared selection tag contains `control` for fixture
equality, not as a claim about the runtime role. The actual native device must
report RTX 5080, NVIDIA, 580.178.04, non-software and requested ID 0x2c02.

N6 LeVJEPA stays resident while all eleven production CPU graphs and the first
B16/T64 world session are constructed and subsequently dropped. Full recurrence,
R256, microbatch 16, F32, seed 7301, eighteen actions, reconstruction zero and
future prediction .25 remain fixed. There are no actions, D3 initialization,
learning updates, checkpoint/restore, later GPU session or Atari environment.

Require both exact allocation plans, interleaved Shared zeroing, full matched
Blade allocations/binds, all constant uploads and successful checked waits.
Only expected F32 configuration values are widened; actual traces are unchanged.

## CPU preparation and upstream check

The completed control build ZkxGRu is reused without modifying its target.
The fresh [candidate preparation ITVpsF](../../runs/driver-bound-candidate-cpu-v2-20260916.ITVpsF/cpu/result.json)
passes 84 Kindle CPU tests, formatting and release Clippy; all 23 GPU tests remain
ignored. Its eight commands, 5,023 input pins and five native/artifact pins
independently re-audit. The build uses a private copy of HEURED's cache in a
one-CPU / 2 GiB / zero-swap scope. HEURED's unchanged CPU evidence is checked
locally; its full old-boot GPU/host audit is not relabeled new-driver evidence.
No Python wheel or broader hardware fixture is produced.

Preserve fHY59j's failed pre-compilation newline expectation, with no cache copy,
build or GPU work there. ITVpsF corrects that checker only. Also preserve the
pre-declaration test mock correction documented in bT2bAx/development.md.
The completed [launcher/trace review](../../runs/driver580-host-init-20260916.bT2bAx/cpu/result.json)
passes 48 CPU fixtures, re-audits both builds, the 105-test host guard preparation
and the old failed control. All five command lifecycles and 10,119 inputs verify.

Fresh upstream reads find Meganeura **5a570099** and Blade **bbf5bf5**. Blade's
new **1c2e06bd** changes optional timestamp collection/API to resolve the submission
just waited on. Its query-pool changes are conditional on timing, which is
disabled here; the other new changes concern rendering and XR presentation.
Keep the historical runtimes fixed for this diagnostic. Before later throughput
qualification, pick up the relevant timing API fix with a compatible Meganeura
integration; do not call this frozen candidate the latest Blade runtime.

## Monitoring and limits

Boot is `4f5152d1-e5fd-46cf-a0c4-06534c430d26`. Host preflight binds loaded/on-disk
580.178.04, the kernel module bytes, NVIDIA ICD/userspace library and upstream
tips. Its current-boot kernel baseline has no recorded fault. No visible device
file holder is found, but 352 process FD directories are inaccessible at the
declaration check: this is not complete process visibility or measured GPU idle.

The [host-only guard](../gpu_incident_response.md#prepared-host-only-guard) reads
kernel logs and boot/loaded-driver identity. It owns only the direct native
child, with a 900-second child budget and .25-second polling; cleanup and host
reads can extend elapsed wall time. Native assertions independently bind the
actual adapter. A separate host-only check follows the full raw CPU audit to
catch later-arriving kernel messages. No NVML call or telemetry worker is used.

GPU recovery action, utilization and directly free/reserved memory are explicitly
**unmeasured**. A clean kernel log is not proof of hardware health or memory
headroom. This diagnostic does not satisfy the historical memory/cadence gates,
production gradients, full state/pixels, throughput or long-run reliability.
It cannot prevent the first wedge or guarantee termination of a kernel-blocked
child. No driver/module/service change, reset, reboot or external report occurs.

## Execution

The control completes as direct PID **40606**, exit zero and reaped. Its full
independent audit passes: 632/9,439 physical slots, 10,081 buffer/allocation pairs,
all 14,645 constant uploads and **143,121** complete initialization records.
All 235 host checks pass; maximum gap is .288666 seconds. These are kernel/host
checks, not GPU-health samples. The post-audit host check also passes, with no
unfinished child and zero actions/updates.
[Complete control result](../../runs/driver580-host-init-20260916.bT2bAx/control-result.json).

After that raw result was inspected and independently re-audited, the candidate's
single separately invoked initialization also completes: direct PID **41120**,
exit zero and reaped. Both exact plans and all counts above match the control,
including 143,121 complete records and every checked wait. All **240** host
checks and the separate post-audit kernel check pass; maximum host-check gap is
**.288863 seconds**. No fault is recorded, no child remains, and actions/updates
are zero. A separate read-only invocation independently re-audits the complete
candidate result and all 10,125 declared inputs.
[Complete candidate result](../../runs/driver580-host-init-20260916.bT2bAx/candidate-result.json).

Control and candidate run approximately 14:52:36–14:53:39 and 14:54:46–14:55:51
UTC, respectively. The control completes and is reviewed before the candidate
starts. These initialization durations are not a training-throughput comparison.
The post-completion upstream check still finds 5a570099/bbf5bf5.

The prior [3O90H9 control](2026-09-16-driver580-control-preparation.md) remains a
failed declared gate because its NVML sample gaps exceeded 1.5 seconds. Do not
overwrite its absent success result or treat this changed monitoring protocol
as a retroactive pass. Earlier 595 faults and passing initializations also stay
intact. Comparing these runs cannot isolate a driver fix: monitoring and the
driver-binding fixture also differ from those historical executions.

Main remains ce80; all adoption/throughput gates and the Pong hold stay fixed.
Both approved invocations are consumed. Two successful initializations do not
establish candidate reliability or complete any of the nineteen original
hardware requirements. Further GPU work needs a new bounded declaration and
authorization, not another automatic
initialization retry or a training queue.

Read-only rechecks after completion (never rerun writers):

```bash
/usr/bin/python3 -B runs/driver-bound-candidate-cpu-v2-20260916.ITVpsF/prepare.py --audit
/usr/bin/python3 -B runs/driver580-host-init-20260916.bT2bAx/capture_cpu.py --audit
/usr/bin/python3 -B runs/driver580-host-init-20260916.bT2bAx/run.py audit --role control
/usr/bin/python3 -B runs/driver580-host-init-20260916.bT2bAx/run.py audit --role candidate
```
