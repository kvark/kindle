# Production world-gradient diagnostic after external recovery

Status: **qualified control passes; latest candidate reproduces device loss
during initialization**. The GPU again requires recovery, subject to user
approval. Both failed attempts and Pong root 1009's complete successful pair
are preserved; roots 2017/3019 stay held until throughput qualification.

## Recovery and scope

After an external reboot at **14:45 UTC**, read-only checks observe boot
`80351da1-04bb-4547-aab9-0b538ca01418`, the same RTX 5080 and kernel, matching
**595.91.07** drivers, **no recovery action** and **no current-boot Xid**. The
agent performed no reset, reboot or driver change. Current health is not runtime
qualification. The [raw NVML](../../runs/world-gradient-recovery-v2-20260913.CfqW0Y/declaration-nvml.stdout)
and [kernel record](../../runs/world-gradient-recovery-v2-20260913.CfqW0Y/declaration-kernel.stdout)
are preserved. The **15:30 UTC** recheck still finds upstream Meganeura
**75dfe901** and Blade **f6f2729e**; no newer fix was found.

The [declaration](../../runs/world-gradient-recovery-v2-20260913.CfqW0Y/declaration.md)
runs one qualified **ce80e9cd** control test, then one latest **75dfe901** test
only after a successful control and health checks. Both retain production
**B16/T64/F32**, full recurrence, eighteen actions, original initialization and
inputs, complete scalar losses and all parameter gradients. Loss tolerance
remains **3e-4**, gradient relative L2 **3e-3 + 1e-7 absolute**.

Isolated fixtures add the same **26 test-only lines**: stage progress, actual
RTX 5080 identity reporting/assertions. Production bodies are unchanged.
Backtraces are enabled; GPU capture, timestamps, tuning and block matmul are not.
Each test has a 1,800-second limit and NVIDIA-only loader selection. Retain
250-ms direct free/reserved memory sampling, maximum 1.5-second coverage gaps
and at least 2,048 MiB directly free. No retry or automatic follow-up is allowed.

## Prepared evidence and historical-host separation

The [completed release preparation](../../runs/world-gradient-recovery-20260913.LKYbiO/prepare-result.json)
passes **78 CPU tests per arm**, formatting, ten command lifecycles and **5,806
pins**, using private copied caches under one core/2 GiB/zero swap. GPU tests
were listed, not run. The original source/packages/executables are unchanged.

The first declaration [refuses the changed boot](../../runs/world-gradient-recovery-20260913.LKYbiO/boundary.stderr)
before any GPU test or hardware manifest. Preserve it; never rerun its writer.
The new read-only adapter explicitly supplies **recorded host context only to
historical raw-data validators**, while checking the actual new host. Native
commands through the shared runner are forbidden in that audit process. Old
scripts and their live launch guards stay unchanged.

The [full raw-pair recheck](../../runs/world-gradient-recovery-v2-20260913.CfqW0Y/boundary.stdout)
matches the preserved preflight except its verification timestamp: eight Pong
commands, four GPU windows, complete state/moments, replays/videos, the full
Freeway predecessor and exact seed-2017 no-spawn hold. It explicitly labels the
old and current host contexts and establishes no new runtime qualification.
The [new manifest](../../runs/world-gradient-recovery-v2-20260913.CfqW0Y/manifest.json)
binds **46,419 inputs** and **49 passing CPU guard tests**, including wrong
device/host, missing progress, memory, environment and changed-boundary negatives.

The [independent preflight](../../runs/world-gradient-recovery-v2-20260913.CfqW0Y/preflight-audit.json)
reverifies those inputs before the once-only controller starts. It repeats the
full historical proof and fresh upstream/host checks before GPU execution.

## Same-boot result

The qualified ce80 control runs **15:37:58–15:40:23 UTC**. Both serial and
temporally grouped production sessions complete. All losses and parameter
gradients pass; worst relative L2 is **0.000745721**. Its **581 GPU samples**
have a maximum gap of **0.267 seconds**, with at least **6,545 MiB directly
free**. Actual executing-device identity and post-test NVML/kernel health pass.
This is a successful diagnostic control, not full current-boot runtime qualification.

The latest candidate starts at **15:40:24 UTC**, on the same verified GPU/boot.
It reaches graph construction for the first, serial session but never emits
`session-ready`. Its [backtrace](../../runs/world-gradient-recovery-v2-20260913.CfqW0Y/candidate.stderr)
reports device loss in `Session::build_session_impl -> Context::submit`; the
process exits **101 at 15:42:19 UTC**. It performs **no D3 weight initialization,
training step or gradient comparison**. This is not a measured gradient mismatch.

The [bounded kernel record](../../runs/world-gradient-gpu-incident-20260913.fjNcOg/kernel-window.stdout)
first records **Xid 62 at 15:41:48.753147**, followed by **Xid 154 / GPU Reset
Required**, PMU-halted/GSP crash records and watchdog warnings. The candidate
window retains **459 samples**, maximum gap **0.253 seconds**, minimum sampled
free memory **11,292 MiB**. Memory samples do not establish recovered health
or exclude transient failure causes. The [post-failure NVML](../../runs/world-gradient-gpu-incident-20260913.fjNcOg/nvml.stdout)
again reports recovery action `Reset` and unavailable utilization.

CPU inspection of the pinned executable maps the recorded return PC to the
`zero_optimizer` transfer submission: [disassembly](../../runs/world-gradient-gpu-incident-20260913.fjNcOg/panic-callsite.stdout),
[literal bytes](../../runs/world-gradient-gpu-incident-20260913.fjNcOg/panic-strings.stdout),
[source](../../runs/world-gradient-gpu-incident-20260913.fjNcOg/runtime-source.stdout).
This mapping derives the PIE base from a narrow caller symbol; it is not a
captured GPU trace. **The reporting submission is not proven to originate the
fault**: the preceding device-buffer initialization/wait may already have failed.
The same-boot control narrows the comparison, but does not distinguish changes
in Meganeura from Blade, driver interaction or hardware causes.

The [read-only incident result](../../runs/world-gradient-gpu-incident-20260913.fjNcOg/result.json)
reverifies **46,419 inputs**, eleven diagnostic command histories, complete
control gradient/progress output and both sampled windows. Its
[independent read-only checker](../../runs/world-gradient-gpu-audit-20260913.lvdFst/audit.py)
also rechecks all **69 incident pins**, normalizing only the candidate-stage
tuple/list serialization boundary; the sealed collector is unchanged. All workers and the
logger have exited. The **15:47 UTC** upstream recheck still finds the tested
Meganeura/Blade revisions. No new upstream fix or root cause is established.

## Next decision

Do not restart either failed attempt, reset hardware or change/reload drivers
without approval. After recovery, investigate initialization using a new bounded
comparison; retain this successful control and the actual failing case. Do not
shrink the production test or skip initialization to turn this failure into a pass.
The full hardware suite, complete update-1/eight-update state and moments,
N6 pixel/restore traces, combined memory and same-backend block AB/BA gates
remain unqualified. No later GPU stage or learning job is launched. Pong stays
held, and Boxing remains the only confirmed three-root game.
