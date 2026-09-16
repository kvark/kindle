# Monitoring cadence: CPU-only persistent-reader preparation

**Superseded by the user's no-NVML direction on driver 580.** The proposed
hardware cadence/parity test is withdrawn, not awaiting approval. Preserve the
CPU prototype and evidence; neither is adopted or to be executed against NVML.

Status: **173 CPU tests pass; no new GPU query or hardware test occurs.**
The [approved ce80 control](2026-09-16-driver580-control-preparation.md)
initialized correctly on 580.178.04, but its declared monitoring-coverage gate
failed. That terminal result, its absent top-level success file, and the 1.5 s
limit remain unchanged. The single-execution approval is consumed.

## What the retained timings show

The [read-only analysis](../../runs/persistent-health-cpu-20260916.Yn3D4H/summary.json)
matches all 210 health rows to their raw successful query responses and partitions
query durations using the recorded native phase marks:

| Query interval relative to native phases | Count | Median | Maximum |
| --- | ---: | ---: | ---: |
| Starts before frontend initialization | 2 | 1.3242 s | 1.4248 s |
| Wholly inside frontend/world initialization | 205 | 0.0405 s | 0.0605 s |
| Overlaps teardown | 2 | 0.8030 s | 1.4851 s |
| Starts after native completion | 1 | 1.4055 s | 1.4055 s |

The two failed health intervals are still exactly 1.756196 and 1.656448 s.
The expensive query calls, plus polling delay, dominate those intervals. This
identifies where time went; it does not establish a driver mechanism or explain
the four earlier Xid faults.

NVIDIA documents initialization/deinitialization associated with GPU client
lifetime and possible startup costs. Keeping a client open is therefore a
testable hypothesis for these timings, **not a measured fix**. See NVIDIA's
[persistence overview](https://docs.nvidia.com/deploy/driver-persistence/overview.html)
and [NVML initialization/cleanup contract](https://docs.nvidia.com/deploy/nvml-api/group__nvmlInitializationAndCleanup.html).
No persistence mode or daemon/service setting has been changed.

## Isolated prototype

Branch `exp/persistent-health-cpu-20260916`, commit **3ebdf2e**, adds a small
request-driven NVML worker and bounded subprocess client. Main's runtime,
dependencies and original `gpu_guard.py` are unchanged. No launcher uses this
prototype yet.

- Importing it or leaving an idle helper open does not load/initialize NVML.
  The first valid request opens one session; later requests reuse it. There is
  no autonomous polling, prefetch, retry or successor job.
- Recovery is checked first. API/field errors, nonzero recovery, changed
  driver/UUID/PCI, invalid counters and insufficient directly free memory stop
  sampling. Only unsupported optional temperature/power may become `[N/A]`.
- The parent checks host state before and after each response. Ordered sequence
  numbers, monotonic request/response clocks, exact fields and byte-counter
  conversions reject stale protocol responses or altered data. Duplicate JSON
  keys, booleans masquerading as integers, extra output and late responses fail.
- Subsequent responses, including the post-response host check, must complete
  within 1.5 s of the previous received sample. First bootstrap has the existing
  three-second probe bound, before any prospective native job. No live job
  exists here, and no prior gate is relaxed.
- Cleanup addresses only the owned helper. An unkillable helper remains
  unfinished, not successfully reaped. Shutdown releases its NVML session; it
  is not another health query.

The worker retains raw memory bytes and floors MiB counters, avoiding rounding
up directly free memory. CLI field/rounding parity is still unmeasured. The local
SDK predates recovery field 230; the identifier is supported by NVIDIA's
[field documentation](https://docs.nvidia.com/deploy/nvml-api/group__nvmlFieldValueEnums.html),
not inferred from the older header. Successful API returns and field timestamps
cannot prove that internal hardware status never lags a fault.

Keeping an NVML client alive changes the runtime's client lifetime and may change
the expression of a fault. It is **not a neutral observer** for a matched driver
comparison. A future diagnostic must disclose that change. CPU fakes, header
layout and exported symbols prove neither hardware compatibility nor cadence,
GPU safety, overhead, throughput or a fix for initialization faults.

## Completed CPU evidence and next decision

The [Yn3D4H capture](../../runs/persistent-health-cpu-20260916.Yn3D4H/cpu/result.json)
and separate derived summary independently re-audit **eight commands and 38
direct inputs**. Tests comprise 102 reader/client fixtures, all 65 unchanged
guard fixtures and six retained-timing fixtures. A compiled C oracle checks all
five ctypes layouts against the installed header without linking or calling
NVML. ELF reads find all eleven required exports in the installed library.
The old terminal reader also reverifies its 5,024 inputs and 57 terminal pins,
retaining the original coverage failure.

Preserve the first development collection failure (pytest reserves the argument
name `request`) and its exact archived source. The subsequent 149-test development
pass is also preserved before the additional refusal tests. Do not rerun any
completed capture writer. The final reusable CPU-only audit is:

```bash
python/.venv/bin/python -B runs/persistent-health-cpu-20260916.Yn3D4H/capture_cpu.py --audit
```

The proposed NVML-only hardware diagnostic is now withdrawn at the user's
request. Do not keep seeking approval for it. Future diagnostics must use
non-NVML monitoring with their requirements declared explicitly; absence of
memory/utilization telemetry is not a healthy measurement. Historical failures
and CPU results remain unchanged. This change starts no GPU job and lifts no
candidate quarantine, qualification gate or Pong hold.
