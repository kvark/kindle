# Driver 580 observed; control prepared on CPU only

Status: **external driver change verified from host records; GPU health and
initialization remain unverified**. No GPU query, job or recovery is performed
by this work. The [fourth-fault quarantine](2026-09-15-interleaved-initialization-incident.md)
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
or evidence that the driver fixes the initialization fault. No GPU is queried.
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
This lets one new binary bind either driver without rewriting metadata. Neither
driver arm has been executed with this binary; no same-binary driver comparison
or parity is claimed.

All original N6/B16/T64/R256/full-recurrence settings, eleven CPU graphs,
frontend/world initialization, observed allocation/upload ordering and checked
waits are retained. A future invocation would initialize only the resident
frontend and first world session, with no actions, D3 initialization, learning,
checkpoint, restore or later session.

The [ZkxGRu CPU preparation](../../runs/driver-bound-control-cpu-20260916.ZkxGRu/cpu/result.json)
passes **84 Kindle CPU tests**, formatting and release Clippy. All **23 GPU
tests remain ignored**; the selected ignored fixture is listed, not run.
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

## Required next decision

The user's “resume” permits this host-only/CPU work, not an implicit lifting of
the explicit GPU stop. Approval has been requested for health queries and
**one separately declared, guarded historical-control initialization**. No
answer is presumed, and no GPU declaration or launcher is created here.

After approval, bind the new boot, driver, actual adapter, current kernel and
numeric health, exact new executable/source/environment, original encoder and
plans, direct-child guard and >=2 GiB directly free. Inspect the complete result
before any successor. A control pass would not qualify a candidate or prove a
driver fix. Failed candidates remain quarantined; no training or Pong restart,
driver installation, recovery action or vendor submission is authorized here.
