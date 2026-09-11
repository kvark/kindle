# Host NVIDIA update during Breakout training

At 06:42 UTC on September 11, an unattended host update replaced NVIDIA
user-space 595.71.05 with 595.91.07 while the loaded kernel module remained
595.71.05. Newly launched `nvidia-smi` exits 18 with
`Failed to initialize NVML: Driver/library version mismatch`.

This is an external runtime incident, not a changed learning recipe or a
completed Breakout result. No driver, kernel, service, experiment input or
running job was changed by this investigation.

## Evidence captured during training

- [Host evidence](../../runs/host-driver-incident-20260911.1KckCc/host-evidence.json)
  preserves the failed observation and `/usr/bin/unattended-upgrade` history:
  NVIDIA upgrade 06:42:13–06:43:03, old firmware removal at 06:43:08.
  The installed packages are `595.91.07-0ubuntu0.26.04.2`; the inspected APT
  version tables do not offer the previously running 595.71.05 package.
- [Initial independent snapshot](../../runs/host-driver-incident-20260911.1KckCc/initial-observation.json)
  at 06:48:39 verifies the original trainer 2455288/start ticks 116275878,
  parent 2454804, logger 2455284 and serial follower 2318785. Training has
  reached 158,328/200,004 actions and 39,233 updates; recent learner reports
  are finite. No native command has ended or started a replacement phase.
- `/proc` maps show the trainer still using the original 595.71.05 NVIDIA
  libraries and the logger still using original NVML and `nvidia-smi`.
  Their files were unlinked by the package update but remain mapped in those
  existing processes. The loaded kernel module is also still 595.71.05.
- The original, declared 250-ms GPU logger was still live and current. Its
  snapshot directly reports 3,303 MiB free, 462 MiB reserved and 12,540 MiB
  used on the same RTX 5080 UUID. This is an actual logger sample, not a
  total-minus-used estimate. Fresh NVML queries still fail; do not report
  restored host health or a new runtime gate pass.
- [Input checks](../../runs/host-driver-incident-20260911.1KckCc/checks.json)
  reverify all 756 serial-handoff pins, including the active and queued
  experiment inputs. This does not qualify the newly installed driver.

The separate [read-only observer](../../runs/host-driver-incident-20260911.1KckCc/observe.py)
retains actual process/command identities, finite recent metrics and fresh
GPU-log checks. It explicitly records the fresh NVML error and labels memory
as coming from the original pinned logger. It does not modify or replace
that logger, weaken the experiment's device guard, or launch a GPU workload.
Other query errors, missing processes or stale logger samples still need
independent rechecking; an observer exit never authorizes a training restart.

## Actual terminal boundary

The original training finished with exit 0 at **08:11:08 UTC**, completing
200,004 actions and 49,652 updates. The
[independent final-training audit](../../runs/breakout-final-training-20260911.6Mnq8o/result.json)
passes full state, optimizer/encoder/ledger and whole-training GPU checks;
see the [pilot report](2026-09-10-breakout-qbert-pilots.md#completed-training-frozen-evaluation-blocked).
All 94,518 raw GPU samples pass coverage with at least 3,303 MiB directly free.

The pinned pilot's next `memory.gpu_guard()` failed at **08:11:10 UTC**, before
frozen evaluation. The serial follower stopped at **08:11:11** with child exit
1. All original trainer, logger, controller and follower PIDs are absent;
the observer correctly stopped on that terminal process change. Do not wait
on those old handles, restart the original queue or bypass the device guard.
The logged memory above describes completed training, not current free VRAM.

The final checkpoint is now saved and independently verified. Fresh NVML still
fails with the driver/library mismatch; host repair remains unapproved. Do not
reboot, reload the driver or downgrade host packages without user approval.
A reboot would terminate all host processes, not just this experiment.

After host recovery, record the actual loaded driver/library versions and
re-establish the required GPU numerical, state, memory and runtime evidence
before long work on a changed driver. Any needed continuation must be declared
separately and preserve completed work, all game criteria and fresh-seed
requirements. Do not rewrite the original queue or assume it succeeded.

Boxing's completed three-root result remains preserved. Breakout is still a
seed-0 pilot, and Qbert, fresh Freeway and fresh Pong remain unstarted successors.
The five-game objective is not complete.
