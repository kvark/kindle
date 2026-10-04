# Five-game study: first launch stopped before training

The selected panel is **Boxing, Pong, Freeway, Breakout and Qbert**, comparing
CDP, faithful Dreamer RGB and pretrained frozen Tiny over three learner seeds.
The [declaration](../experiments/2026-10-04-cdp-atari-comparison.md) is unchanged.
**0/45 training runs are complete.** This failed initialization is retained;
it is not a completed run or a reason to discard the study's negative results.

At16:37 UTC the first Freeway CDP seed1009 launch was stopped by the ordinary
host guard on one new exact `_memdescAllocInternal / NV_ERR_NO_MEMORY` warning.
The worker was terminated with SIGTERM and reaped. Training and memory logs
are empty: zero recorded gameplay actions/learner updates and no checkpoint.
The two preceding excluded Tiny qualification smokes still pass; neither
qualifies this stopped launch. No successor, retry or GPU recovery was started.

The warning's kernel source timestamp precedes child spawn by5.956 seconds;
journal receipt follows spawn by1.170 seconds. Preserve both clocks; these
timestamps do not establish whether this process caused the warning. The
message alone does not establish physical VRAM exhaustion. No Xid/hang is
recorded in the scoped host review, and no GPU utilization/health claim follows.

All27 failed-guard evidence pins verify, with no unsealed files. Its broad
kernel snapshot reached the64MiB output limit; the streaming warning record
is intact. A separate bounded host-only capture since16:35 UTC contains one
allocation warning and zero hard faults; its nine pins verify. This repairs
the review's scoped coverage without overwriting or calling the broad capture
complete. No NVML or native context was used for the review.

The status dashboard initially said training was running after service launch;
the later completion check found the initialization stop and corrected it.
The ordinary guard had already stopped the worker at16:37:31, independent of
that delayed status check. CPU-only audit development continued; no further
GPU process ran. All1,104 Python tests pass for the frozen export/comparison/
replay tooling. The native implementation and learning math are unchanged.

**Next:** user approval is requested for at most two occurrences of this exact
warning within the first10 seconds of each native process in the declared
study. Later/other warnings, API/numerical failures and Xids would stay fatal;
no automatic retries or recovery. This proposed startup-only allowance is
**not implemented or enabled**. Existing ordinary guards still stop on any new
occurrence; historical120-second numerical permission does not cover training.

[Compact evidence](2026-10-04-cdp-atari-initialization-stop.json) ·
[Failed guard](../../runs/cdp-atari-comparison-20261004.KnROnx/freeway-cdp-1009-queue/freeway-cdp-1009) ·
[Scoped host review](../../runs/cdp-atari-comparison-20261004.KnROnx/initialization-stop-review)
