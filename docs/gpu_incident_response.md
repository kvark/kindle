# GPU operation and incident response

Use the GPU for learning and inference. On driver **580.178.04**, the user has
temporarily prohibited **all NVML calls**: no `nvidia-smi`, bindings, legacy health
logger, recovery-action polling or vendor diagnostic tools. No CPU learning fallback.

## Bounded execution

Use [gpu_host_guard.py](../python/examples/gpu_host_guard.py) around the direct
process hosting native GPU work, never Cargo, a scheduler or a process tree.
Its declaration binds boot ID, driver, absolute command, executable SHA256,
timeout and polling interval. Review each result before the next individual
invocation. There is no automatic retry or successor.

The guard checks kernel logs and boot/driver identity before, during and after
execution. It stops/reaps only its own direct child on faults or timeout. An
uninterruptible child may remain unfinished; preserve that result and its logs.
It cannot prevent a first wedge or prove hardware health/numerical correctness.

Retain native adapter assertions and the job's correctness gates. Current jobs
require >=2 GiB **Vulkan estimated budget minus usage** after GPU stages. This
is not physical free/reserved or peak VRAM. Utilization and recovery action
remain unmeasured, not zero or healthy.

Do not edit pinned launchers/helpers or completed evidence. Historical
[gpu_guard.py](../python/examples/gpu_guard.py) remains for host capture and
auditing; its NVML `run`/health paths are not permitted. Changed boot/driver
identity needs a distinct matched declaration, not modified historical inputs.

## On a fault

Stop scheduling GPU work. Inspect the guard result and process identity; do not
kill unrelated processes by name or reuse old PIDs. Preserve stdout, stderr,
declaration, input identities and child lifecycle. Capture into a fresh directory:

```sh
gpu_incident_dir=$(mktemp -d /x/Code/kindle/runs/gpu-incident-XXXXXXXX)
python3 python/examples/gpu_guard.py snapshot "$gpu_incident_dir/host"
python3 python/examples/gpu_guard.py audit "$gpu_incident_dir/host"
```

Snapshot mode reads journal/boot/PCI/module/driver/process information without
NVML or recovery. Inspect incomplete-capture markers. A later fresh host-only
capture may record delayed teardown; never overwrite an earlier capture.

Keep kernel source time distinct from journal receipt time. Logs can arrive
late; the final CPU breadcrumb or nominal health row does not locate the fault.
Never infer GPU idleness or recovery from missing/zero telemetry.

## Recovery boundary and retained incidents

**New user approval is required** for resets, module/service reloads, reboot,
power cycling, PCI changes or driver/package installation. Do not repeatedly
query an unhealthy device or retry failed candidates as recovery.

Four September 13–15 incidents share Xid 62/154 and PMU/GSP failure symptoms.
The user's reset returned `Not Supported`; module reload did not recover it.
Failed 75dfe, 0a98775/02b600a1 and interleaved 070f4b51/7db0d05c bundles remain
quarantined outside their already consumed diagnostics. Successful driver 580 /
no-NVML runs do not establish a causal fix or general safety.

Full reports and local vendor briefs are preserved, not submitted:
[September 13](https://github.com/kvark/kindle/blob/9ef9a169d52ed36a912d578a89690f992c5982fc/docs/experiments/2026-09-13-gpu-forensics.md),
[September 14](https://github.com/kvark/kindle/blob/9ef9a169d52ed36a912d578a89690f992c5982fc/docs/experiments/2026-09-14-pixel-initialization-incident.md),
[September 15](https://github.com/kvark/kindle/blob/9ef9a169d52ed36a912d578a89690f992c5982fc/docs/experiments/2026-09-15-interleaved-initialization-incident.md).
Do not rerun terminal invocations or reinterpret their absent success files.
