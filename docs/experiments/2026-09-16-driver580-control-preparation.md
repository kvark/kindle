# Driver 580 control: initialization succeeds, monitoring coverage fails

Status: **the sole approved control initializes successfully without a recorded
GPU fault, but fails the declared monitoring-coverage gate**. The user explicitly
approved health checks and this one guarded initialization. No candidate,
training, retry or host recovery runs. The
[fourth-fault quarantine](2026-09-15-interleaved-initialization-incident.md)
and all throughput/adoption gates remain in force.

## What changed on the host

The [Yz0Vmu capture](../../runs/driver580-host-20260916.Yz0Vmu/host/result.json)
retains thirteen host-only command lifecycles and independently re-audits.
APT history records installation of 580-server-open at September 16 04:26–04:27
UTC, replacing the 595-server-open stack. The current boot is
`4f5152d1-e5fd-46cf-a0c4-06534c430d26`, first journal entry 04:31:42 UTC.
The agent performs neither change.

Loaded module, on-disk module and NVML library path all identify **580.178.04**.
DKMS lists it installed for kernels 7.0.0-30 and the running 7.0.0-31. None of
the 1,444 captured current-boot kernel records matches the retained fault
patterns. This is not an NVML health/memory check, a loaded Vulkan-device check,
or evidence that the driver fixes the initialization fault. That capture makes
no GPU query.
Fresh upstream main reads still find Meganeura **5a570099** and Blade **6ab5fcec**;
there is no newer merged fix in this observation.

## Prepared historical control

The isolated branch `exp/driver-bound-control-20260916`, Kindle **f7b1914**,
starts from ae7699ad and retains exact control Meganeura **9b9e7ee7 / ce80** and
shared Blade **c96a9a87 / published 0.9.0**. Only its ignored initialization
fixture and worktree instructions change. No production or dependency change
lands on the main working branch.

The fixture requires the new selection `combined-driver-control-20260916` and
mandatory `KINDLE_INIT_EXPECTED_DRIVER`, accepting only 580.178.04 or 595.91.07.
Missing/unlisted values are rejected before GPU-context creation; each session's
actual device must match the declared version, not merely either allowed version.
This lets one new binary bind either driver without rewriting metadata. Only
the 580 arm has now executed; there is no 595 execution of this new binary and
no same-binary driver comparison or parity claim.

All original N6/B16/T64/R256/full-recurrence settings, eleven CPU graphs,
frontend/world initialization, observed allocation/upload ordering and checked
waits are retained. A future invocation would initialize only the resident
frontend and first world session, with no actions, D3 initialization, learning,
checkpoint, restore or later session.

The [ZkxGRu CPU preparation](../../runs/driver-bound-control-cpu-20260916.ZkxGRu/cpu/result.json)
passes **84 Kindle CPU tests**, formatting and release Clippy. During that CPU
preparation all **23 GPU tests stay ignored**; the selected fixture is only listed.
Eight command lifecycles, 4,914 input pins and five compiled artifacts independently
re-audit. The private cache is copied from, never written into, the completed
historical target. The command requests one CPU, 2 GiB memory and zero swap;
the short-lived scope's peak-memory measurement was not retained.

The new executable's SHA256 is
`85a49d6e3c503f0b6b8adc3f98b3698ada8c6000de3f5c763beb7795696afa93`.
Its filename matches the old target's name, but its path and bytes differ;
the old executable remains unchanged. No Python wheel or full hardware group
is produced. Preserve both completed writers and their caches; reusable reads:

```bash
python3 -B runs/driver580-host-20260916.Yz0Vmu/capture.py --audit
python3 -B runs/driver-bound-control-cpu-20260916.ZkxGRu/prepare.py --audit
```

## Sole approved initialization: terminal result

The user's subsequent **“Approve health check + control test”** authorizes the
[separate 3O90H9 declaration](../../runs/driver580-control-20260916.3O90H9/declaration.md).
Forty launcher/reader CPU fixtures pass, including the exact F32 expected-value
correction and explicit driver checks. The complete archived 595 control trace
replays exactly. The new declaration binds 5,024 inputs; both fresh preflights
pass on the expected boot, loaded/on-disk/reported driver and actual adapter.
Initial recovery is None, utilization 0% and directly free memory 15,840 MiB.

The fixture records completion at **05:31:13 UTC**. Its sole native child
**7873** exits zero and is reaped. The unchanged guard reports success and no
unfinished child.
The independent complete trace reader verifies both exact plans, 632/9,439
physical buffers, 611/846 immediate Shared zeros, 339/14,306 constant uploads,
10,081 buffer/allocation pairs and 143,121 records. All checked waits pass;
actions and updates are zero. No kernel fault or recovery request is recorded.

However, the controller **correctly rejects the declared <=1.5-second maximum
health-sample gap**. All 210 guard health rows report recovery None and at least
4,975 MiB directly free, but two intervals exceed the coverage limit:

| Sample interval (UTC) | Gap | Corresponding NVML call |
| --- | --- | --- |
| 05:31:12.857–14.613 | 1.756196 s | 1.485066 s; starts during native resource teardown |
| 05:31:14.613–16.270 | 1.656448 s | 1.405500 s; final post-exit query |

World initialization is ready at 05:31:12.839; resource teardown completes at
05:31:13.304. Both over-limit intervals begin after world.ready. The expensive
NVML calls account for most of these intervals, alongside the configured poll
delay. This is timing evidence, not proof of a particular driver teardown or
power-management mechanism. Do not discard the intervals or retroactively relax
the bound. Native/guard success and observed healthy samples do **not** establish
a fully passing declared control.

Preserve the original controller's failure and absent top-level `result.json`.
Its original audit refuses that failed result; it is not a reason to retry.
The [separate terminal reader](../../runs/driver580-control-20260916.3O90H9/terminal-analysis/result.json)
reproduces the exact refusal, checks the complete traces and reports the gap
failure explicitly. Six CPU fixtures pass; all 5,024 declared inputs and 57
terminal pins independently re-audit. It makes no GPU query. The reusable
complete interpretation is:

```bash
python3 -B runs/driver580-control-20260916.3O90H9/terminal.py --audit
```

The authorized single execution is complete; no hardware successor is declared.
Investigate monitoring cadence before requesting a new diagnostic, preserving
the current guard and the unchanged coverage limit. This ce80 control also
initialized successfully on 595, so its success on 580 does not distinguish the
candidate fault's cause or prove a driver fix. Candidates stay quarantined,
Pong stays held, and full dependency/state/pixel/memory/throughput gates remain.
