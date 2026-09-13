# Latest-backend qualification: GPU device loss

Status: **hardware gate failed; post-reboot diagnostic also fails on the candidate**. No new
backend adoption, speedup or subsequent learning is established. Pong root
1009's complete original paired result remains valid and preserved. All
experiment processes have exited; do not restart any stopped queue.
The [full retained-log investigation](2026-09-13-gpu-forensics.md) now ties both
incidents to an identical PMU-halt signature, records failed GSP initialization
after the user's module reload, and prepares a kernel-aware guard and runbook.
The GPU is still unrecovered; the candidate remains quarantined.

## Completed pair and report-format repair

The [Pong handoff](2026-09-13-throughput-priority.md#completed-pong-pair-and-qualification-handoff)
finishes the user's requested active pair before attempting throughput work.
The original three throughput followers then stop before GPU work: the old
read-only score helper omits the scoring CLI's explicit
`campaign_declaration: null` field. The
[actual-data diagnostic](../../runs/meganeura-conv-runtime-v2-20260913.tyHbhU/diagnostic-execution.json)
finds no differing common field. No score, checkpoint or acceptance rule is changed.

The isolated replacement
[declaration](../../runs/meganeura-conv-runtime-v2-20260913.tyHbhU/declaration.md)
adds that field in its own adapter and retains full equality checks. Its **62
CPU checks and 43,233 pins** pass. The
[complete raw-pair preflight](../../runs/meganeura-conv-runtime-v2-20260913.tyHbhU/boundary-preflight.json)
verifies eight completed commands, four covered GPU windows, complete trained
and untrained state/moments, scores, replays/videos, Freeway's predecessor and
the exact seed-2017 no-spawn boundary. The
[independent handoff audit](../../runs/meganeura-conv-runtime-v2-20260913.tyHbhU/handoff-audit.json)
rechecks every pin and actual follower/child identities. No old input is edited.

## Failed native gate

The unchanged nineteen-test suite uses prepared Kindle **58f328a / native
fa6bdd2a**, Meganeura **75dfe901** and shared Blade **f6f2729e** on driver
**595.91.07**. Hardware begins at **13:16 UTC** on September 13. Fourteen named
tests return success. The fifteenth,
`dreamer::world::tests::temporal_batching_matches_serial_losses_and_gradients`,
runs with the declared production B16/T64/F32 scope and exits **101** at
**13:20:34 UTC**. Its [stderr](../../runs/meganeura-conv-runtime-v2-20260913.tyHbhU/hardware-14.stderr)
reports Blade's `ERROR_DEVICE_LOST` panic, not a numerical comparison failure.
The trace has no debug markers or backtrace identifying the failing dispatch.

The [bounded kernel record](../../runs/meganeura-gpu-incident-20260913.5yCpTz/kernel-window.txt)
first reports **Xid 62 at 13:20:03**, followed immediately by **Xid 154: GPU Reset
Required**. Later records bind channel teardown to test PID **313333**, then
report **Xid 109 / CTX SWITCH TIMEOUT** and watchdog warnings. This identifies
the device-loss episode, not whether its root cause is in generated code,
the runtime, driver or hardware. Do not infer a gradient bug or a wrong adapter.

The [read-only inspection](../../runs/meganeura-gpu-incident-20260913.5yCpTz/result.json)
reverifies all 43,233 inputs, the fifteen command histories and their raw outputs.
The hardware CSV has **1,080 samples**, maximum gap **0.270 seconds**, minimum
sampled free memory **11,292 MiB** and maximum usage **4,550 MiB**. These samples
do not establish successful qualification or exclude unsampled failure causes.
The [13:26 NVML readout](../../runs/meganeura-gpu-incident-20260913.5yCpTz/nvml-after-failure.txt)
still returns the correct UUID/driver and free-memory values, but activity is
`[N/A]`. Readable memory counters are not evidence that the GPU recovered.

Both [Meganeura](../../runs/meganeura-gpu-incident-20260913.5yCpTz/meganeura-upstream.txt)
and [Blade](../../runs/meganeura-gpu-incident-20260913.5yCpTz/blade-upstream.txt)
are freshly rechecked after the incident: the tested revisions are still the
latest upstream heads. No newer fix is known from that check.

## Safe stop and next decision

The hardware worker, logger and follower exit. The separately carried
full-state/pixel declaration in `runs/meganeura-conv-learning-v2-20260913.wGOgE7`
has **55 passing CPU checks and 43,251 pins**, but its
[launch refuses](../../runs/meganeura-conv-learning-v2-20260913.wGOgE7/start-refusal-execution.json)
the missing successful hardware result before creating a follower or GPU work.
No replacement block declaration or GPU job is launched. Preserve all original
pre-GPU failures, the corrected hardware failure and the downstream refusal.

Do not reset the GPU, reload/change drivers or reboot without user approval.
After recovery, recheck device/driver health and separately declare a
bounded diagnostic with the qualified ce80 control and the latest candidate.
Locate the failing execution before proceeding to full-state/pixel and matched
block throughput gates. Retain all tolerances, memory margins and learning gates;
do not turn a hardware failure into a pass by shrinking the production case.
Pong roots 2017/3019 stay held, and Boxing remains the only confirmed three-root game.

The host was subsequently rebooted externally at **14:45 UTC**. Read-only checks
observe matching drivers, no requested recovery and no current-boot Xid. No host
recovery was performed by this agent. The [new bounded diagnostic](2026-09-13-world-gradient-recovery.md)
passes its same-boot qualified control, then reproduces the candidate's device
loss during session initialization. The GPU again reports Reset Required.
That separate declaration and historical/current-host separation do not restart
or rehabilitate the failed attempt described here; both failures are preserved.
