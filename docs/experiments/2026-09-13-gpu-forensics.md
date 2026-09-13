# RTX 5080 device-loss investigation and containment

Status: **two matching PMU-halt incidents; a later external reboot restores
observable health**. The latest backend remains quarantined. The
[initialization follow-up](2026-09-13-initialization-diagnostic.md) records the
17:21 boot and passing guarded production control and allocation-order candidate.
The candidate result narrows the investigation without establishing root cause
or full runtime qualification.
No reset, driver change or host recovery was performed by the agent.
Pong's completed pair and the remaining-root hold are unchanged.

This is a device/firmware failure triggered during backend qualification, not
ordinary unstable learning. The same-boot qualified control passes; the candidate
fails before session initialization completes. That is strong evidence for a
candidate-sensitive initialization problem, but does not yet distinguish a
runtime defect, driver/firmware interaction or hardware fault.

## What the logs establish

The [sealed host capture](../../runs/gpu-forensics-20260913.O0en46/manifest.json)
preserves **38,153 kernel records**, their boot IDs, monotonic/realtime timestamps
and journal cursors, plus service, PCI, driver and process evidence. The
[recomputed analysis](../../runs/gpu-forensics-analysis-20260913.ru9tzy/result.json)
joins the first fault in each boot to the original process lifecycles and GPU CSVs.
All times below are UTC on September 13.

| Event | Evidence and outcome |
| --- | --- |
| 13:18:36 → 13:20:34 | Latest-backend production world-gradient test, PID 313333; first Xid 62 at **13:20:03.840535**, exit 101 about 30.6 seconds later. Fourteen earlier tests passed. |
| 14:45 | External reboot restores observable health, with the same 595.91.07 driver. This is not runtime qualification. |
| 15:37:58 → 15:40:23 | Qualified ce80e9cd control, PID 10870; complete production scalar losses and all parameter gradients pass. Post-test health passes. |
| 15:40:24 → 15:42:19 | Latest candidate, PID 11267; first Xid 62 at **15:41:48.753147**, exit 101 about 30.4 seconds later. Graph ready, but no session ready, D3 weight initialization, training step or gradient comparison. |
| Before 16:26 | User's privileged device-owner check prints no owners; their reset request explicitly returns **Not Supported**. |
| 16:26:13 → 16:26:26 | User unloads/reloads modules. The reloaded driver cannot initialize GSP firmware: **`RmInitAdapter failed! (0x62:0x40:2168)`**. |
| 16:27–16:32 | NVML exits 6, **`No devices were found`**. PCI still enumerates the RTX 5080 at `0000:01:00.0`, bound to `nvidia`. Repeated queries produce more initialization failures, not recovery. |

The [first fault window](../../runs/gpu-forensics-analysis-20260913.ru9tzy/fault-1.jsonl)
and [second fault window](../../runs/gpu-forensics-analysis-20260913.ru9tzy/fault-2.jsonl)
contain the **same complete Xid 62 payload**:

```text
324b06ec 0000c4c8 00000000 206db166 206da31e 206da48c 206d879e 206d8f8e
```

Both proceed to Xid 154 / reset required, PMU-halted/GSP records, channel cleanup
and context-switch timeouts. The first faults occur **87.44 and 84.55 seconds**
after their respective test starts. This is a repeatable signature across two
boots, not a decoded firmware cause. NVIDIA classifies Xid 62 as an internal
microcontroller halt and recommends GPU reset plus support investigation;
Xid 154 summarizes the required recovery action rather than identifying another
independent cause. [NVIDIA Xid catalog](https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html)

The second test's backtrace and CPU disassembly locate **error reporting** at
`zero_optimizer` submission. They do not locate the first bad GPU command. Keep
the [original control/candidate report](2026-09-13-world-gradient-recovery.md)
and its exact executable, source and tolerance bindings.

## Why the previous telemetry was insufficient

Both failed tests cross the first fault between sampled memory usage of
**34 MiB and 4,530 MiB**. Sampled free memory never falls below **11,292 MiB**;
the successful control uses more memory and retains at least **6,545 MiB**.
The sampled initialization transition is a useful lead, not proof of which
allocation, host write or submission failed, nor an exclusion of transient faults.

More importantly, the existing long-lived logger keeps emitting numeric memory
and **zero utilization for 122 and 121 samples after the first fault**, right
until the failed tests exit about 30 seconds later. A later fresh NVML process
reports reset required and unavailable activity. Numeric CSV rows and a running
logger therefore were not an adequate health guard. No source of GPU-idle or
throughput truth follows from these post-fault zeros.

The original telemetry has no continuous temperature/power trace. The second
candidate's prelaunch XML records **33°C, 49.01 W instantaneous draw, a 360 W
limit**, matching GSP 595.91.07 and no accumulated thermal slowdown. These are
prelaunch readings, not measurements during the fault; they cannot rule out a
power/thermal transient. Neither incident boot has a recorded kernel OOM, PCIe
AER error, machine-check error or NVIDIA API-version mismatch. Absence of these
messages does not certify the board, power supply, RAM or PCIe link.

## What changed relative to the long stable period

The retained kernel **journal** begins **August 26**, across four boots, with
no Xid before September 13. A [supplemental rotated-log scan](../../runs/gpu-rotated-logs-20260913.vMfjdM/result.json)
preserves five complete kernel-log rotations, six boot dmesg files and five
driver/package logs. It extends the checked kernel records to **August 16**;
all 1,170 Xid lines in those files are from September 13. These are overlapping
log sources, not 1,170 independent GPU failures or a count to add to the journal.
Older boot-list entries do not imply their kernel logs survived; neither source
establishes months of continuous, fault-free operation. Historical August OOM
messages are not concurrent with these failures. `/var/crash` has no GPU crash
dump, only kdump bookkeeping; `/sys/fs/pstore` is not readable by this user.
That access limit is retained, not bypassed or interpreted as an empty store.

The package logs confirm an **unattended upgrade at September 11 06:42 UTC**
from **595.71.05 to 595.91.07**, including NVIDIA firmware. That earlier boot
has user/kernel API-mismatch messages, unlike both incident boots. It is a
confound when comparing against weeks of earlier training.
However, the qualified ce80 control passes immediately before the second
candidate on the **same boot, driver and GPU**, making the candidate comparison
more informative than a historical success alone.

The [source inspection](../../runs/gpu-initialization-source-20260913.ZMUpSF/result.json)
freshly checks both remote `main` heads at **17:01 UTC**: Meganeura remains
**75dfe901**, Blade **f6f2729e**. No newer upstream fix was available at that
check. It preserves actual old/new runtime, build and Blade Vulkan source,
not just commit titles.

Two concrete findings guide the next diagnostic:

1. Meganeura's loading change **43384d9** changes the default initialization
   schedule: formerly each physical slot was allocated in alias order and each
   Shared buffer zeroed immediately; now all Shared allocations precede all
   device-local allocations, with host zeroing afterward. Logical slot mapping
   remains intact. Allocation/initialization scheduling is a suspect to isolate,
   **not a demonstrated bug**.
2. Both old and new runtimes discard errors from the initialization waits.
   Latest `wait_for_timed_encoder` propagates Blade's device-loss error, but its
   callers at `zero_device_local` and `zero_optimizer` ignore the result. The
   old code similarly ignored `gpu.wait_for`. An earlier failure can therefore
   be reported only at the next submission. This is a **pre-existing
   failure-handling weakness**, not proof of what caused the PMU halt.

The fixture supplies its GPU context explicitly and leaves session tuning,
skipped parameter zeroing, capture and timestamps off. The new default shared
context helper is not used by that construction path. Blade's Vulkan resource
implementation is byte-identical between these two versions; the new submit
calibration is conditional on timing being enabled. These narrow the inspection,
but do not establish that Blade or firmware is innocent. No source patch or
learning-arithmetic change is adopted from this review.

## Prepared for another incident

The standalone [GPU guard](../../python/examples/gpu_guard.py) and
[operator runbook](../gpu_incident_response.md) are now available for **newly
declared direct native jobs**. They do not edit or restart old launchers.

The guard checks kernel history before launch, tracks new kernel records as
well as fresh NVML health, binds boot/driver/GPU identity and requires at least
2,048 MiB directly free. On a fault, missing telemetry, failed child or timeout,
it stops only its own direct child, captures bounded host evidence and seals
the files. It never retries, resets hardware, reloads drivers, uploads logs or
starts another stage. A kernel-blocked process may survive TERM/KILL; that is
reported explicitly rather than mistaken for completed cleanup. This is not a
process-tree supervisor or a global GPU scheduler.

The [CPU qualification](../../runs/gpu-guard-cpu-v2-20260913.dn1CAS/result.json)
passes **65 tests**, including real CPU-child termination with an unrelated
process left alive, and an **actual current-host kernel-fault refusal** that
spawns neither the `/usr/bin/true` sentinel nor an NVML query. Its six command
lifecycles and 102 pins reverify. Preserve the first qualification attempt:
its journal boot argument was malformed, so it failed closed before detecting
the fault; the corrected command binds an undashed ID with `--boot=...`.
The initial development assertion correction is also retained. Simulated faults
and real refusal are **not hardware qualification, measured guard overhead or
a guarantee against the first wedge**.

The capture audit verifies 60 pins; analysis recomputes 38,153 records and its
nine direct pins; the rotated-log reader recomputes 16 files and 20 pins;
source inspection verifies 49 pins. The latter retains a
pre-writer Python module-shadowing failure and uses `python3 -P` for its completed
invocation and read-only audit. The rotated-log reader also retains its
pre-writer relative-path launch failure and absolute-path continuation.
The complete historical diagnostic/Pong/Freeway
audit also passes unchanged. Reuse only documented read-only `--audit` modes;
never rerun completed writers.

## Next diagnostic, not another blind retry

The host subsequently rebooted externally; fresh 17:24 checks find health
restored, not runtime qualification. During the preceding failed recovery,
a reset was explicitly unsupported, and normal module reload failed to boot GSP.
Do not escalate to
forced unload, PCI/bus resets, firmware changes or repeated NVML probing.
The recommended escalation was a user-controlled full shutdown and power-on,
followed by fresh health checks; NVIDIA likewise requires health verification
and recommends power cycling if reset leaves the GPU unhealthy.
[NVIDIA reset guidance](https://docs.nvidia.com/deploy/nvidia-smi/#-r---gpu-reset)

Before another candidate execution:

1. Recheck upstream and preserve this failing package. Stage diagnostics in an
   isolated source tree; do not change qualified main or historical executables.
2. Add immediately flushed initialization breadcrumbs before/after allocation,
   host zeroing, pipeline construction, device-zero submit/wait and optimizer
   submit/wait. Record allocation counts/bytes and the actual device. Make a
   failed wait stop before another submission. Keep instrumentation separate
   from a putative allocation-order fix and apply matched observation to controls.
3. Declare a bounded initialization comparison, with a healthy qualified control
   first and one explicit hypothesis per candidate. Allocation order is the first
   local hypothesis; do not change it together with driver, precision, batch
   size or zero-initialization policy. A smaller allocation-only probe can help
   localize the fault, but cannot replace the production test or pass its gate.
4. Use the guard around each direct executable, inspect its evidence before
   any next stage and stop on the first failure. Do not execute the unchanged
   known-failing candidate merely to obtain a third identical crash. A vendor
   report is worthwhile before further reproduction; collection/upload is
   an explicit operator decision, not automatic.

This forensic report launches no GPU diagnostic or follower; the separately
declared [initialization follow-up](2026-09-13-initialization-diagnostic.md) records
the passing control and separately tested allocation-order hypothesis.
Even an initialization fix leaves full hardware gradients, update-1/eight-update
state and moments, N6 pixel/restore/memory and same-backend block throughput
qualification before the held Pong roots. No speedup, adoption or learning
result is claimed.
