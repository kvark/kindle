# GPU incident response

For newly declared Kindle native jobs on Linux/NVIDIA. This is containment and
evidence capture, not hardware recovery or numerical qualification. The
[September 13 investigation](experiments/2026-09-13-gpu-forensics.md) records
the actual RTX 5080 failures and failed module-reload recovery.

## Capture before recovery

Stop scheduling new GPU work. Do not relaunch a failed experiment or remove a
queue hold. Use the existing controller's documented stop procedure for any
active work; do not kill unrelated processes by name or recycle an old PID.

The standalone tool uses only the Python standard library, imports no native
Kindle module and needs readable kernel journal access. Its snapshot mode does
not query the GPU, reset it, alter services or change drivers:

```bash
gpu_incident_dir=$(mktemp -d /x/Code/kindle/runs/gpu-incident-XXXXXXXX)
python3 /x/Code/kindle/python/examples/gpu_guard.py snapshot "$gpu_incident_dir/host"
python3 /x/Code/kindle/python/examples/gpu_guard.py audit "$gpu_incident_dir/host"
```

Keep the original job declaration, native executable/source identities, stdout,
stderr, GPU logger output and exact process lifecycle alongside this snapshot.
It saves retained kernel JSON with boot/cursor/timestamps, boot history, PCI,
modules, driver metadata and process states. Each probe has a time/output limit;
`capture_incomplete` events describe missing evidence. Audit checks retained
bytes, not whether all requested diagnostics succeeded. Do not overwrite or
rerun a completed capture root.

An optional vendor bundle requires an explicit operator decision and root:

```bash
sudo timeout --signal=TERM --kill-after=5s 45s /usr/bin/nvidia-bug-report.sh \
  --safe-mode --output-file "$gpu_incident_dir/nvidia-bug-report.log"
```

Use the directory created above. This command was **not run** during the
investigation. The installed script's safe mode still calls
`nvidia-debugdump --ioctl --nvlogonly`; it is not a host-only command or a promise
against hangs. A kernel-blocked task can outlive the timeout. Do not add
`--extra-system-data` casually: the installed script can trigger CPU backtraces
through sysrq. Review the bundle for private host/process information before
sharing; nothing uploads automatically. NVIDIA requests full logs, configuration,
reproduction details and its bug-report output for investigation.
[NVIDIA reporting guidance](https://docs.nvidia.com/deploy/gpu-debug-guidelines/gpu-node-triage.html#reporting-a-gpu-issue)

## Guard a new direct native job

Pin the guard, declaration, executable, environment and predecessor proof in a
new experiment. Substitute those paths below; this example is not a new GPU
declaration or authorization to run the quarantined candidate:

```bash
python3 /x/Code/kindle/python/examples/gpu_guard.py run /absolute/path/to/NEW-guard-output \
  --uuid GPU-6869e50d-83aa-bec7-6169-adc413f49b32 --driver 595.91.07 \
  --declaration /absolute/path/to/declaration.json --timeout 1800 \
  -- /absolute/path/to/pinned-kindle-test \
  dreamer::world::tests::temporal_batching_matches_serial_losses_and_gradients \
  --exact --ignored --nocapture --test-threads=1
```

Use a **direct native executable**, or a separately declared Python process
that executes the native extension in that same process. The latter requires
pinning and checking the interpreter, actual imports, extension, adapter and
environment; it is not permission to wrap an arbitrary Python launcher. The
[guarded N6 protocol](../runs/native-f32-alias-pixels-20260913.m6kNer/declaration.md)
binds the specific synchronous ALE adapter, with no GPU worker descendants.
The earlier dckRs2 declaration stopped before GPU work on an upstream change;
never restart that attempt.
Do not wrap a scheduler, Cargo, shell pipeline or controller that spawns GPU
descendants: the guard owns and stops only its direct child. It is not a
machine-wide lock; retain serialized GPU scheduling. Existing declared
packages and launchers remain immutable.

The guard refuses a faulted or unreadable kernel baseline, driver/boot/GPU
identity changes, reset-required/N/A health, less than 2,048 MiB directly free,
nonzero child exit and time-budget exhaustion. It tracks the kernel cursor and
fresh NVML responses while the child runs, retaining optional temperature/power
readings without turning N/A into zero. Successful repeated probes are stored
in one event stream, not thousands of separate files.

On failure it signals only its own still-unreaped child, bounds TERM/KILL waits,
then captures host evidence. `unfinished_children` means cleanup did not finish;
it does not authorize broader kills. A kernel fault earlier in the same boot
also causes refusal, even if a later memory query looks normal. There is no
automatic override, retry, hardware reset or next job.

Always inspect `result.json`, `events.jsonl` and `audit` output. A guard pass
does not prove the job used the intended Vulkan adapter or computed correctly;
retain actual native device assertions and all existing numerical/state/memory
gates. Validation includes CPU tests, real unhealthy-host refusal and a
[healthy CPU-sentinel launch](experiments/2026-09-13-initialization-diagnostic.md).
That sentinel performs no GPU computation. The separately declared
[instrumented production control](experiments/2026-09-13-initialization-diagnostic.md#completed-guarded-control)
also passes under the guard with clean kernel/health evidence and complete
initialization traces. This is a control result, not qualification of the
quarantined candidate or prevention of a future first fault.
The separately declared [allocation-order candidate](experiments/2026-09-13-initialization-diagnostic.md#completed-candidate-result)
also passes the production diagnostic under the guard. That does not qualify
the original failing backend, establish root cause or replace the remaining
hardware/state/pixel/memory gates.
Before timed adoption, measure guard overhead with a matched healthy control;
do not treat guarded and historical unguarded timings as an identical benchmark.

## Recovery boundary

Capture first, then obtain the user's approval before resets, module/service
changes, reboot or power cycling. Stopping an application is not guaranteed to
recover a halted GPU microcontroller. Do not repeatedly query a GPU after it is
known to be unhealthy merely to see whether numeric counters return.

For this machine on September 13, the user already stopped clients and received
**`GPU ... Not Supported`** from reset. Normal module unload/reload then failed
GSP initialization. The card remained on PCI, so NVML's **`No devices were found`**
was not proof it had fallen off the bus. Do not repeat those attempts, force
module unload or improvise PCI/bus resets. The recommended escalation was a
user-controlled full shutdown and power-on. A later external boot at 17:21
restored observable health; the agent did not perform recovery. NVIDIA says
reset may not recover all components and recommends power cycling when
post-reset health fails.
[NVIDIA reset guidance](https://docs.nvidia.com/deploy/nvidia-smi/#-r---gpu-reset)

After recovery, freshly bind the boot, loaded/on-disk/userspace driver, GPU UUID
and native executing adapter; require readable numeric health, no recovery
action, clean kernel evidence and the direct free-memory margin. Recovery is
not runtime qualification. Keep the original failing candidate quarantined;
test changes only through separately declared guarded diagnostics. Inspect
each result before continuing, and do not resume old queues or training
automatically.
