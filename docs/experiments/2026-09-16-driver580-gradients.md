# GPU gradient comparison passes without NVML

The user's direction to resume GPU work supersedes the earlier approval stop.
The prepared **ce80 control and current-Meganeura candidate both pass** their
production world-model comparison on driver **580.178.04**, without NVML calls.
This is GPU execution, not another CPU-only preparation.

The [separate declaration](../../runs/driver580-gradients-20260916.P34QyD/declaration.md)
reuses the existing host guard and the [already built fixtures](2026-09-16-driver-gradient-preparation.md).
It adds no production learning code, fallback backend or alternative arithmetic.
Each direct native invocation is independently limited to 900 seconds; control
was reviewed before launching candidate. Both invocations are terminal: no retry.

| Arm | Native SHA256 prefix / PID | Native result | Host checks |
| --- | --- | --- | --- |
| Control: afb973a / Meganeura 9b9e7ee7 / Blade c96a9a87 | b069df58 / 46858 | Exit zero, reaped; 9 losses, 51 nonzero gradients | 546, maximum gap 0.288739 s |
| Candidate: 0d0a831 / Meganeura 070f4b51 / Blade 100bb813 | 7b4b9a8c / 48099 | Exit zero, reaped; 9 losses, 51 nonzero gradients | 546, maximum gap 0.288731 s |

Both execute the original **12M/B16/T64/full-BPTT/F32** synthetic world test.
All 62 parameter entries agree between serial and batched sessions; the eleven
critic entries remain frozen, while every compared gradient is finite and nonzero.
The worst per-parameter relative L2 error is **0.0007456096368231737** in both
arms, below the unchanged outer .003 limit. The native loss and gradient
tolerances are unchanged. All nine loss values and all 51 per-parameter gradient
statistics match exactly between arms; raw cross-arm gradient vectors are not
retained, so this is not a separate bitwise cross-arm tensor comparison.

Each run verifies both complete historical allocation plans: 11,319 and 9,439
physical slots, 29,602 constant uploads, 20,978 buffer/allocation pairs and
292,932 initialization/placement records. Control/candidate parameter and upload
inventories and all initialization counts match exactly. Checked initialization
waits pass. `step.wait_returned` itself remains only a breadcrumb: the pinned
runtime still discards that wait error, as documented in the preparation report.

The direct-child guards, independent read-only audits and separate post-audit
kernel check pass; no kernel fault or unfinished child is recorded. The host
checks are **not GPU-health samples**.
Recovery action, utilization and directly free/reserved VRAM remain unmeasured.
The roughly 148-second test durations include graph compilation and construction;
they are not a gameplay throughput measurement or a claimed speedup.

The declaration binds 18,133 inputs, including the completed CPU build and native
artifacts. Eight focused reader tests reject missing stages/gradients, incorrect
device/configuration, nonfinite values, frozen-critic violations, all-zero
gradients and violations of the original numerical gates. The retained historical
trace provides expected plans only; it is not relabeled as a driver-580 run.

```bash
/usr/bin/python3 -B runs/driver580-gradients-20260916.P34QyD/run.py audit control
/usr/bin/python3 -B runs/driver580-gradients-20260916.P34QyD/run.py audit candidate
```

Use only those audit modes after completion. Preserve the declaration, guard
outputs and earlier initialization incidents. The abandoned CPU-reader directory
was left empty when the user redirected work to GPU execution.

## Small profiler change

The current [vector profiler](../../python/examples/profile_atari_vector.py) no
longer starts `nvidia-smi`, creates a live GPU CSV or takes a monitor-only `--gpu`
argument. It still measures actions/sec, updates/sec and stage times. Missing
activity, power and VRAM are JSON null, with `gpu_telemetry: "unmeasured"`;
historical CSV analysis remains read-only. All **360 Python tests pass**, including
six profiler checks. The historical guard and pinned profiler copies are unchanged
and must not be launched through their NVML paths.

## Continue toward throughput

These passes establish the bounded production-gradient result, not long-run
reliability, a causal driver/NVML fix, full backend qualification or new Atari wins.
Main remains ce80. Continue bounded GPU qualification and performance work under
the user's resumed authorization; no further blanket GPU-approval wait is needed.
Stop on a new fault and do not perform host recovery without approval.

Keep the remaining hardware/state/pixel/restore gates and same-backend throughput
comparison ahead of Pong. The old NVML direct-memory gate is not fulfilled by
these host-only runs; any replacement measurement must be explicit in its new
declaration. Latest upstream is Meganeura 5a570099 and Blade bbf5bf54. The candidate
contains the former; its frozen Blade predates the new optional-timing API fix.
Integrate that fix compatibly before relying on GPU pass timing, and distinguish
that new timing candidate from this fixed-runtime comparison.
