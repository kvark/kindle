# CDP machine handover

Read [AGENTS.md](../AGENTS.md) for working constraints, the
[roadmap](kindle_single_life_dreamer_plan.md) for retained results and gates,
and [PR31](https://github.com/kvark/kindle/pull/31) for the live dashboard.
The five-game goal remains unfinished. Detailed experiment history belongs in
those reports; historical “next” paragraphs do not authorize restarting jobs.

The [latest old-run/worktree cleanup](results/2026-11-10-storage-cleanup.md)
increases available space39.41 GB to316.20 GB(294.48 GiB) at completion.
Remove Meganeura's6.93 GB Cargo cache,three obsolete source worktrees,two stale
registrations,the unused plotting environment and393 empty old output directories.
All dirty/untracked worktree source has verified archives and tested recovery;
commits/branches remain reachable. Retain the patched Naga dependency.

The2,478 older logs now live in the verified4.48 GB archive at
`/x/Artifacts/kindle-log-archive-20261110`;restore before checking an old original
pathname. Models,shared checkpoints,current runtimes/data and280 protected input
hashes remain intact. The learner and both monitors stay frozen. Cleanup and
recovery receipts: `/x/Artifacts/kindle-storage-cleanup-20261110`.
Earlier137 GB and671 MB cleanup passes remain recorded in their linked reports.
Retained artifacts can share hardlinks/sparse blocks;keep their bytes immutable.

## Current work — November 10

Kindle now builds against **upstream Meganeura main** (`a0eff600`, Blade
`d9112e26`) instead of the October handover fork; see the
[switch report](results/2026-11-10-meganeura-main-switch.md). The fork's
`MEGANEURA_COMMAND_CACHE`/`IMAGE_VIEWS`/`CONVOLUTION_*`/`COPY_COMPOSITION` flags
are inert. A one-method upstream patch (`Session::slot_buffer`) waits for the
user to push from `/x/Temp/kindle-meganeura-slot-buffer-20261110/`; until then
the retained-off actor-dynamics/expected-value snapshots cross the host.
Run every heavy job as a memory-capped `systemd-run --user` unit; a bare
`cargo test` peaked at 17.6 GB and the OOM killer took the session down.

The goal **prove the world model training on Breakout** is active and has
its first positive mechanism. Three feedback-side screens
([16x more play](results/2026-11-10-breakout-world-scale.md), [96x96
frames](results/2026-11-10-breakout-world-rgb96.md), [palette-committed
feedback](results/2026-11-10-breakout-world-palette-feedback.md)) left
eight-step development ball tracking at 38/107, 36/112 and 23/112 against the
64x64 control's 53/107, and a [linear state probe](results/2026-11-10-breakout-world-state-probe.md)
showed why: the spatial-history recipes never encode the ball in the recurrent
state, only in fed-back frames that fade. [Removing the image history](results/2026-11-10-breakout-world-state-only.md)
makes the state learn the ball (prior-state readout 13/7 native pixels at one
step, 22/11 at sixteen, baselines 40/28) and, at the 4M preset, gives the best
forecasts so far on the shared origins: observed 113/113, one-step 105/112,
four-step 78/112 (control 66/109), eight-step 45/112 (control 53/107),
sixteen-step 15/115, at 97.8 ms per update on 96x96 frames. The 1M state-only
run blurs the brick rows, so the active run gives the spatial decoder a
[fixed observed context frame](experiments/2026-11-10-breakout-world-context-frame.md)
(`rgb_context_lag` 8: scenery from a real frame, ball and paddle from the
state, nothing fed back), followed by the 65,536-update state-only run
(`kindle-world-scale-20261110.service`, `guards-after-4m`, outputs
`train-96-context`, `train-96-plain-65k-v2`). Root
`/x/Temp/kindle-world-scale-20261110`; driver `python/examples/breakout_world_scale.py`
(`probe` mode runs the state readout on any checkpoint). Actor/critic, CDP and
the five-game campaign stay parked.

The previous [cumulative-survival learner](results/2026-11-10-breakout-world-survival-loss-learning.md)
remains frozen by systemd at the user's request at update 9,964/16,384
(`kindle-world-survival-loss-learning-20261110.service`, PID 2287733); it holds
about 1 GB of host memory and 0.5 GB of VRAM. Do not thaw, restart or stop it
without the user. Its fixed protocol, qualification and the closed RGB-event
reviews stay as recorded in their reports; its runtime is the fork's.

## Preserved source and fresh-machine setup

The working branch is **`phase2-levjepa-evaluation`**, continuing PR31.
It now contains the qualified c8aeba28 library implementation directly, with
portable Git dependencies. There is no separate takeover branch to switch to.
The previous356-commit work history and detailed experiment corpus are retained
on [archive/phase2-before-cleanup-20261025](https://github.com/kvark/kindle/tree/archive/phase2-before-cleanup-20261025).
The working branch keeps current results, source and tests; no `runs/` tree,
raw experiment data or checkpoint history is committed.

| Dependency | Revision |
| --- | --- |
| [Meganeura](https://github.com/kvark/meganeura) | `a0eff600d0aaf877a6a6ba02c2de1eb8804583ed` (main) |
| [Blade](https://github.com/kvark/blade) | `d9112e26ab1244d9e206e7d27387295a28cb3fa3` (main) |
| Naga (wgpu trunk, dev-dependency) | `9964733300924e724b6d41ef268e7c2933f3dcdb` |

Since November 10 the source builds against upstream Meganeura main instead of
the preserved `handover/cdp-qualified-20261025` fork (`423d3736`/Blade `bbbc93b7`).
The fork's command cache, image views, convolution reuse, pooled tapes and
diagnostic-group APIs are gone; its `MEGANEURA_COMMAND_CACHE`,
`MEGANEURA_IMAGE_VIEWS`, `MEGANEURA_CONVOLUTION_WORDS`, `MEGANEURA_COPY_COMPOSITION`
and `MEGANEURA_CONVOLUTION_REUSE` flags are inert. Main has no device view of
parameter storage, so the optional actor-dynamics/expected-value parameter
snapshots cross the host until a `Session::slot_buffer` accessor lands upstream.
Retained runtimes under
`/x/Temp/*/runtime` keep their own fork binaries; the checkpoints they wrote
still load, but new learners are not byte-comparable to the fork's qualifiers.
Only the user merges PRs. Cargo fetches the dependency revisions without sibling
checkouts or old absolute paths. The prior handover verified329 library source files
against the measured implementation and passed CPU workspace/Python compilation.
`tools/handover/verify_sources.py` and its manifest preserve the source receipt
and eight explicit cleanup differences: Rust formatting, actual dependency
revision reporting, and two stale CPU test repairs. Learning operations and
checkpoint architecture validation are unchanged. A separate Meganeura
portability repair records GPU commands directly on Metal/GLES, whose Blade
backends do not expose Vulkan secondary encoders; the Vulkan path is unchanged. The full CPU Rust suite passes
(127 tests;78 GPU tests ignored), as does formatting.
No GPU qualification or speed claim follows from CPU checks. Rebuilt binary
bytes can differ with paths, compiler and dependency metadata. On a new machine, rebuild instead of importing the original Linux/CPython3.14
binary. The current AMD machine already has qualified CPython3.13 packages; use
the matching retained runtime above and do not repeat its completed qualification.

Example setup (Python3.11+; original environment was3.14.4, Rust1.98.0):

```sh
git clone --branch phase2-levjepa-evaluation https://github.com/kvark/kindle.git
cd kindle
python3 -m venv python/.venv
. python/.venv/bin/activate
python -m pip install -r tools/handover/requirements.txt
cd python
maturin develop --release --locked
cd ..
```

The requirements record the source machine's versions, rather than silently
upgrading the environment. Install the target's Vulkan driver and obtain Atari
ROMs separately; ROMs and the Python virtual environment are not in the transfer.
A release build is not a successful GPU initialization check.

The original NVIDIA runtime enabled the flags below. This is historical
configuration, not a command for the current AMD machine: in particular,
`KINDLE_DEVICE_TARGETS=1` is unavailable here. Use each active declaration's exact
environment instead of copying this block.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MEGANEURA_COMMAND_CACHE=1 KINDLE_DEVICE_MEANS=1
export MEGANEURA_IMAGE_VIEWS=1 MEGANEURA_CONVOLUTION_WORDS=1
export MEGANEURA_COPY_COMPOSITION=1 KINDLE_DEVICE_TARGETS=1
export MEGANEURA_CONVOLUTION_REUSE=1
```

Select the target GPU/ICD using that machine's actual configuration. Do not copy
`MEGANEURA_DEVICE_ID=0x2c02`,5080 name assertions, old driver versions or boot IDs
onto another GPU. Declare actual device/driver expectations for the host guard.
These flags retain the qualified implementation; capability fallbacks and
performance on other GPUs still need checking.

## Transfer package

Historical shared destination: **`/mnt/data/kindle-handover-20261025`**.
That pathname is absent at the November10 cleanup check. The retained local
extraction is `/x/Temp/kindle-amd-20261025/handover`; all eight files of its
two original checkpoint sets still match the October25 availability receipt.
This cleanup does not modify `/mnt/data`. The archive/bundle details below
describe the original transfer package, rather than a new verification that
the historical shared directory is present.

`manifest.json` identifies every included artifact by original path, size and
SHA256. `archive-review.json` records independent verification of all304 members.
The artifact archive is392,172,204 bytes (about374MiB); it contains no build caches.
`README.md` there lists package contents and source revisions. Source Git bundles
provide an additional fallback to the pushed branches.

Extract into a new local directory, after checking the archive SHA256 against
the manifest. Never extract over an existing run. The archive includes:

- `context64/frozen-checkpoint`: recovered867-update B2/T64 weights, and the
  successful frozen-restore review. See the recovery note below.
- `breakout-b8/2m-frozen`: recovered source-control export from the warm-start
  evaluation, learner counter499939 and environment counter2027152. All346
  tensors are finite. It is a frozen derivative of the2M control; the complete
  original training logs and initial/500k checkpoints are not included.
- `diagnostics`: retained PMU captures and prepared pre-submission collector.
- `original-runtime`: original native package, build receipt and example helpers.
  Historical scripts contain absolute paths and host pins: treat as evidence,
  not ready-to-run controllers on another machine.

**Originals located:** the shared-drive cleanup moved the runs to
`/x/Artifacts/shared-drive-cleanup-20261025` on the original machine; they were
not lost. Verified original checkpoints are:

- `kindle-cdp-breakout-context64-20261023.5OA15Iew/smoke/checkpoints/4096`:
  learner867, environment4096.
- `kindle-cdp-breakout-budget-20261021.wqYChxy8/checkpoints/2000000`:
  learner499939, environment2000000.

Both originals match all346 tensors in their corresponding shared-package
frozen exports exactly. The exports have advanced environment counters5120 and
2027152; preserve their metadata rather than relabeling them as original saves.
The shared package remains usable without transferring the whole local archive.
Use the saved originals when their original counters or full run evidence are
needed. Neither version preserves replay, games, RNG or live belief.

For historical artifact paths, look under the relocated run directory above.
The package is deliberately not a copy of every profiling dump or checkpoint.

## Original arrival procedure — completed on this AMD machine

The following describes the October25 transfer protocol. Its initialization,
restore and fresh context experiments are complete. Do not execute this list
again on the current machine or treat its BF16 recipe as current AMD settings.
Use the active work above and Auto precision. A genuinely new machine needs
its own bounded host qualification, not a repetition of completed learning.

1. Build the preserved runtime and record source, compiler, Python, GPU and driver
   identity. Check available memory/disk. No backend port or optimization first.
2. Run a bounded initialization/frozen restore under `gpu_host_guard.py` or
   `run_gpu_queue.py`, using the recovered867-update checkpoint and an output
   directory created for this check. Use eight environments, native observation
   size, the published Atari protocol and sticky actions.25. Frozen evaluation
   must produce zero updates and preserve all346 tensors. A1024-action check
   suffices for this startup boundary; it is not a learning evaluation.
3. If initialization succeeds, finish the smoke review and run the declared
   fresh Breakout1009 B2/T64 life, with500k and2M checkpoints and the actual
   initial control. Eight streams;128 replay positions/update;ratio32;12M/BF16;
   centered CDP and action-effects coefficient1; no Freeway action aid. Expect
   first update at action632,124843 updates at500k,499843 at2M. Full life is
   fresh, not a continuation from the smoke or prior warm-start8M model.
4. Evaluate fixed milestones with24 natural sampled episodes each (three per
   stream), zero updates and full replay/videos. Compare to retained B8/T16
   scores58.83/98.21/1.42. Disclose different hardware/build, and the coupled
   sequence-length/independent-batch change. Review before any extension or seed
   replication. Stronger Qbert learning follows useful Breakout evidence.

See the existing runner's `--help` and the archived declarations for the complete
protocol. Do not launch the old absolute-path `run.py train` unchanged. Retain
ordinary guard checks, accounting and actual-initial/frozen comparisons; do not
turn relocation into another weeks-long qualification matrix.

## GPU fault scope

On the original host, ordinary restore repeatedly produced Xid62/PMU halt on
615.71.09 and595.91.07. One traced restore succeeded; an ASLR-preserving trace
later failed while waiting for the first world's device-local buffer fills,
before checkpoint loading/model execution. IOMMU errors followed30s later.
This localizes the wait, not the root cause. Neither hardware, driver/firmware
nor invalid runtime arguments are conclusively excluded.

No blind retries on a faulted device, and no automatic reset/reload/reboot.
The prepared collector records fill handles/capacities/offsets/sizes/values and
stops before Blade submit. Six CPU cases passed; native capture remains unrun.
Its ABI and symbols apply only to the original binary. On another healthy GPU,
start with the bounded restore above; do not inherit a historical machine-wide
stop or presume the failure reproduces there. Stop and collect evidence if it
actually faults. No training was launched during this handover.

## What not to redo

Do not restart Pong2M, the rejected B32 Freeway recipe, unchanged Breakout
warm-start extensions, the45-run old matrix, kernel geometry sweeps or JAX parity.
No concurrent/swarm learner project. Current five-game scores and limitations
are in the roadmap. User-authored `TASK.md` and `docs/cdp-claude-feedback.md`
are now tracked unchanged as historical intent/review, not current measurements.
