# B2/T64 smoke: training completes, restore stops on a real GPU fault

The longer-context smoke executes4096 actions/867 updates in45.245s, first
update at action632 and zero debt. Independent CPU accounting, finite346-tensor
checkpoint checks and exact346 initial tensors against the B8/T16 control pass.
This does **not** qualify the complete smoke: frozen restore fails before
run_start and before the first constructed-memory record. No2M training starts.

**Latest diagnosis (October24):** ASLR-preserving tracing reproduces the halt
on a fourth boot and captures the main thread waiting for the first world
session's device-local buffer fills. No checkpoint load is reached. This
localizes the waiting operation, not the underlying firmware defect. No learner
is running; boot24f26399 now requires recovery. See the fourth-boot result below.

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

### Startup tracing prepared without rebuilding the qualified binary

`runs/context64-startup-trace-20261023` contains a locally extracted Ubuntu GDB
17.1 (no system installation), symbol inventory and entry/return tracer. All ten
native stage symbols resolve in the qualified binary: vector restore, context,
pixels, core, restored core, checkpoint wrapper/backend, world synchronization,
device identity and budget query. The debugger reads the object file without
loading or executing it. A dynamically loaded CPU-only fixture verifies nested
entry/return events across repeated calls and process completion: nine events.
No GPU calls, native rebuild or runtime/source changes. Binary SHA-256 remains
`ecd69601c7930fd1e797f949be2f199727af2d0c90abc692f92fe312b1a77d4c`.

This is prepared instrumentation, not a native restore result. After recovery,
declare the diagnostic with containment for both debugger and inferior; the
existing guard's direct-child assumption alone does not cover a debugger tree.
Breakpoints perturb host timing, so diagnostic timing is not training throughput.
The qualified ordinary frozen-restore acceptance still remains required.

- [CPU tracing check](../../runs/context64-startup-trace-20261023/review.json)
- [Tracer](../../runs/context64-startup-trace-20261023/trace.py)

- [Second incident CPU review](../../runs/cdp-context64-restore-20261023.a2He3BM1/fault-review.json)
- [Second incident kernel supplement](../../runs/cdp-context64-restore-20261023.a2He3BM1/fault-kernel-window.jsonl)
- [Second failed restore guard](../../runs/cdp-context64-restore-20261023.a2He3BM1/queue/breakout-final-recovery/result.json)

## Same-boot traced success and ordinary failure

On October23 the host has rebooted onto595.91.07,
boot`c1fd11cc-57ba-4ca2-8d75-8f38dbbdec89`. Its preflight kernel journal has
no GPU faults. One instrumented restore runs, followed only after its CPU
review by one ordinary restore. Both use the same qualified native binary,
checkpoint, model config, action budget and GPU environment. No backend change,
driver setting change, reset or training. A CPU timeout fixture first verifies
that the systemd containment stops both debugger and inferior.

| Attempt | Outcome | Time from child launch to PMU halt |
| --- | --- | ---: |
|615.71.09, first boot | Ordinary restore fails |19.630s |
|595.91.07, second boot | Ordinary restore fails |20.415s |
|595.91.07, third boot | GDB-traced restore passes | No fault |
|595.91.07, same third boot | Ordinary restore fails |19.632s |

The successful trace covers all ten stages. Core construction takes19.785s;
the first checkpoint wrapper runs for0.529s. All three checkpoint loads,
weight synchronization, device queries and memory-budget queries return.
The run completes1024 frozen actions/zero updates, learner counter867, and
all346 exported tensors match the smoke checkpoint. Its complete run identity
matches the original training except the expected driver/evaluation fields.
This proves the saved model can restore and act on595; it does not qualify
ordinary restore reliability. Tracing overhead is not a throughput measurement.

The subsequent ordinary control faults at16:12:42UTC, before either run_start
or constructed-memory output. **Its eight-word Xid62 payload exactly matches
the second incident.** The guard stops the child and the service ends16:12:43;
no unfinished children or learner remain. All three failed guards retain their
failure status. The new bounded kernel snapshot completes without the earlier
cross-boot output overflow. A separate224-record window preserves the third
incident; comparison audits all saved guard manifests and traced frozen tensors.

### What the logs establish

PMU halt is the first reported failure in all three incidents. Reset-required
errors, resource-cleanup assertions and Xid109 timeouts follow. On both595
incidents, a subsequent GSP watchdog identifies partition4#0/task3. Neither
original incident's bounded two-minute lead-in contains allocation/OOM,
thermal or PCIe AER errors. Absence of those logs does not exclude hardware,
temperature or power problems; failure-time telemetry was not recorded.

NVIDIA's [Xid guidance](https://docs.nvidia.com/deploy/gpu-debug-guidelines/gpu-node-triage.html)
identifies62 as a PMU halt. Its
[open driver handler](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/595.91.07/src/nvidia/src/kernel/gpu/gsp/kernel_gsp.c)
marks the GSP fatal after receiving that notification and avoids further RPCs.
This explains why stopping Python does not restore the GPU; it does not identify
the original trigger.

A [firsthand Blackwell report](https://github.com/NVIDIA/open-gpu-kernel-modules/issues/1307)
describes PMU halt followed by the same GSP watchdog task during desktop idle,
without Kindle. That supports considering a shared firmware failure class,
but is not confirmation that our cause is identical or that the hardware is sound.

### What the successful debugger run changes

GDB's default `disable-randomization` setting is **on**, and this run did not
override it. It also introduces startup delay, thread stops and breakpoint
overhead. Therefore this is not a controlled timing-only experiment. Host
virtual-memory layout, concurrency/timing, and intervening device state differ.
One success does not justify disabling ASLR, adding sleeps, rearranging the
constructor, or blaming a particular shader. Checkpoint corruption is a weaker
hypothesis after an exact346-tensor successful restore, but the failing native
operation remains unlocalized.

After recovery, the next diagnostic should retain normal ASLR explicitly with
`set disable-randomization off`, retain flushed stage markers and capture the
last host stack if it stalls. Keep the same binary/checkpoint. No further blind
ordinary retry, driver cycling or training while the device requires recovery.

- [Comparison audit](../../runs/gpu-fault-comparison-20261023/comparison.json)
- [Successful traced restore](../../runs/gpu-fault-comparison-20261023/restore/review.json)
- [Third failed guard](../../runs/gpu-fault-comparison-20261023/ordinary/guard/result.json)
- [Third kernel window](../../runs/gpu-fault-comparison-20261023/595-third-kernel.jsonl)

### October24: ASLR-preserving tracer passes CPU checks

The host remains on bootc1fd11cc with Xid109 timeouts; no GPU test runs.
`runs/context64-aslr-trace-20261024` retains the next tracer separately from
the original evidence. It explicitly keeps ASLR enabled and checks the child's
personality flag. Two CPU fixture runs retain ASLR and use different library
addresses. Two negative cases verify nonzero exit propagation and capture a
stopped inferior's thread stack before killing it; a stopped process can no
longer be mislabeled finished. All four checks pass. Original traced success
also had an explicit normal-exit message and independently verified outputs;
its result is unchanged. Native stack capture/restore remains unrun.

- [ASLR and stop-handling CPU checks](../../runs/context64-aslr-trace-20261024/review.json)

## Fourth boot: fault localized before checkpoint loading

Boot`24f26399-efd1-442a-9d22-45c0fc1aaf53`, driver595.91.07, starts with no
recorded GPU faults. One bounded trace retains normal ASLR explicitly. A CPU
timeout fixture first verifies that SIGINT lets GDB capture the inferior stack,
then both processes exit under the existing systemd containment. On a real fault
the controller allows at most two seconds for that stack before normal forced
termination. No retry or training follows this diagnostic.

At02:35:49UTC October24, Xid62/PMU halted recurs with the identical595 payload.
The guard interrupts GDB, captures all thread stacks, and stops both processes;
the service ends02:35:50. No checkpoint export, adapter memory record, run_start,
gameplay or learner update is reached. A216-record bounded kernel window and
all sealed guard artifacts pass CPU review. No unfinished process remains.

The trace observes context and pixel-pipeline constructors returning, then
enters `DreamerCore::with_gpu` without returning. It never enters the checkpoint
loader. The main host stack is:

```text
poll / NVIDIA user driver
Blade Context::wait_for
Meganeura wait_for_timed_encoder
Meganeura Session::build_session_impl
Meganeura make_session / build
Kindle build_session_with_policy
Kindle DreamerCore::with_gpu / restore_with_gpu / restore_rgb
```

Offline disassembly of the unchanged binary maps the saved return addresses
to offsets`0x4532f7` and`0x3929e3`, with the same load base. The former is the
first of the two initialization waits in `build_session_impl`: the variable-fill
loop named `zero_device_local`, not the later optimizer-buffer clearing loop.
The latter is the first session construction in `with_gpu`, the world training
session. Qualified source at Meganeura `runtime.rs:5086` and Blade's transfer
implementation identifies `vkCmdFillBuffer` calls followed by a submission wait.
No model forward/backward or checkpoint tensor upload is reached on this path.
Internal allocation budget checks occur earlier; the absent memory record only
excludes the adapter's post-construction memory query.

This excludes checkpoint tensor loading as the immediate operation in this
incident and provides a much smaller reproduction target: initialization fills
of the world's device-local allocations. A stack captured after the halt cannot
prove that a particular fill caused the firmware failure. Actual fill sizes,
handles and memory bindings were not captured, and the source's aligned sizing
alone does not validate all runtime values. Do not declare the driver or backend
exonerated. The earlier ASLR-off success is on a different boot and remains only
a correlation, not proof that ASLR is the cause or a workaround.

After recovery, target the allocation/fill submission boundary: retain normal
ASLR and capture actual buffer creation/binding/fill sizes and order for a
minimal reproduction. Avoid another full uninstrumented restore, model/kernel
optimization, or disabling ASLR as a supposed fix. Keep the qualified binary
and all prior evidence. Ordinary restore and full2M training remain unqualified.

- [Fourth incident CPU review and stack](../../runs/cdp-context64-aslr-native-20261024/review.json)
- [Bounded kernel window](../../runs/cdp-context64-aslr-native-20261024/kernel-window.jsonl)
- [Session initialization disassembly](../../runs/cdp-context64-aslr-native-20261024/session-init.asm)
- [Core initialization disassembly](../../runs/cdp-context64-aslr-native-20261024/core-init.asm)

## Resume direction (updated after the fourth fault)

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
