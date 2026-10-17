# GPU operation and incident response

Use the GPU for learning and inference on driver **580.178.04**. The September26
user direction permits normal JAX/CUDA initialization, including its internal
NVML calls, for the bounded Dreamer control. The earlier blanket restriction
does not require an NVML-free backend fork. Separate `nvidia-smi`/binding polling,
legacy health loggers and vendor diagnostics remain off for current declarations.
No CPU learning fallback. The retained incidents never established NVML causality.

## Bounded execution

Use [gpu_host_guard.py](../python/examples/gpu_host_guard.py) around the direct
process hosting native GPU work, never Cargo, a scheduler or a process tree.
Its declaration binds boot ID, driver, absolute command, executable SHA256,
timeout and polling interval. Audit each result before the next invocation.
Successful stages may advance within a declared serial study; failed stages
stop for review, without automatic retry.

**October4 user direction supersedes the warning-only GPU stop:** proceed
unless the GPU is wedged. Fresh declarations set `record_allocation_warnings`
to true. Standalone allocation warnings are retained as `allocation_warning`
events, including baseline records, but do not abort work or require another
approval. No first10-second window or occurrence cap is required. Actual API/
numerical failures, hard GPU faults and deadlines still fail the affected job;
review and fix ordinary job failures without treating them as a wedged driver.
A hang/unreaped native process or wedge remains a recovery boundary.

The known `VUID-StandaloneSpirv-None-10684` remains explicitly non-blocking.
Other emitted Vulkan validation errors still fail a job even if it exits zero.
The following older mechanisms remain for auditing historical declarations;
they are not requirements for fresh training. A declaration may identify exact reviewed
historical allocation-warning cursor/message pairs for its baseline only;
new occurrences and hard faults remained stops. This was not a warning-class
waiver or proof of hardware health. The October 2
[ordinary-compute canary](results/2026-10-02-joint-tiny-qualification.md) stopped;
do not add its new warning to an exception list and automatically retry.

On October 3 the user explicitly authorized an instrumented initialization
diagnostic and classified `VUID-StandaloneSpirv-None-10684` as non-blocking.
Fresh declarations may list that exact VUID in `reviewed_validation_vuids`;
its output and events remain retained. Other validation errors still stop.
For the authorized short probe only, `allocation_diagnostic` names one exact
allocation-warning message and a maximum of one or two new occurrences.
The guard enforces a 120-second maximum; baseline records still require exact
review, and other messages, excess occurrences, native failures and hard faults
still stop. A successful diagnostic is not a clean-warning result, proof of
GPU health or authorization for a training campaign. No recovery is involved.

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
