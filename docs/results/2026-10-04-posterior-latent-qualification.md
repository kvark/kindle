# Posterior-target implementation ready; native qualification stopped

[Declared experiment](../experiments/2026-10-04-posterior-latent-targets.md)
· [Configuration and retained evidence](2026-10-04-posterior-latent-qualification.json).

The optional current-feature objective is implemented in `e34ac5d`. It reuses
the full-posterior decoder with training-only target statistics and coefficient
.25; it predicts Tiny features, not pixels. The future objective and defaults
remain unchanged. Shared initial control tensors are checked before learning.
**109 Rust CPU tests, 1,053 Python tests, formatting and strict workspace/binding
Clippy pass.** CI274 is still running at the recorded review.

There are **no new learning results**: no candidates, production smokes,
gameplay actions or learner updates have run. Qualification stopped before a
numerical result. Historical controls are unchanged.

## Retained attempts and review

1. The first numerical test selected the AMD integrated GPU because its launch
   omitted `MEGANEURA_DEVICE_ID`. The RTX5080 assertion failed before creating
   the numerical session. This was a launch mistake, not a driver failure.
   The child exited101 and was reaped; no kernel event was recorded.
2. A distinct corrected declaration explicitly selected `0x2c02`. At
   **05:35:17.200198 UTC**, the journal received another
   `NV_ERR_NO_MEMORY` from `_memdescAllocInternal`, `mem_desc.c:1359`.
   The guard stopped and reaped its child with SIGTERM. There was no reported
   application allocation failure or completed numerical result before that
   stop. The exact internal request and caller remain unidentified.

This is the same warning text as the October3 initialization diagnostic, which
completed allocation and computation despite it. That precedent does not
automatically waive new occurrences. There is **no recorded new Xid, hang,
host OOM-kill or host recovery** in the reviewed interval; a warning alone is
not evidence of a wedge. No new GPU memory-budget sample was obtained. Host
available RAM was about25GiB; this is not a measurement of GPU free memory.

Both failed runs' evidence seals verify27 pinned files, with no unsealed files;
neither guard passed. Their broad kernel snapshots hit the output-size limit.
The bounded journal review retains the new record separately and found no
other kernel event from05:33 through the05:38 review. Source and journal receipt
timestamps are retained separately, without treating their difference as an
allocation duration. All experiment/native services are stopped, children
reaped, and no NVML polling, reset, reboot or driver change was performed.

The first CPU preparation service also exited127 before compilation; a fresh
service with an explicit tool PATH completed successfully. No failure or
unfinished candidate is relabeled as a completed experiment. Raw evidence:
`runs/posterior-latent-targets-20261004.7jsp3t`.

## Next action

Approval requested for one initialization-only check, at most120 seconds and
two occurrences of this exact warning, retaining all records. Other warnings,
faults and native failures would still stop it. Training remains stopped until
that check is reviewed and the declared numerical qualification and control
parity pass. Do not add this new cursor to an exception list and automatically
retry the stopped job. No CPU learner fallback or broader campaign.
