# CDP handover — October25,2026

The goal remains strong, reliable learning on **Boxing, Pong, Freeway, Breakout
and Qbert**, then native games and transfer. Training is the priority; another
backend optimization/JAX-parity campaign is not the next step.

Read this file, the [short roadmap](kindle_single_life_dreamer_plan.md), and the
[Breakout experiment](experiments/2026-10-23-cdp-breakout-context64.md). The
[PR31 description](https://github.com/kvark/kindle/pull/31) is the live dashboard.
Historical reports retain failed attempts and limitations; their old next steps
are not instructions to restart those queues.

## Source you actually need

Recent learning used an isolated c8aeba28 implementation under ignored `runs/`,
not the normal native build on `phase2-levjepa-evaluation`. That gap is now
preserved explicitly in Git. Use these branches, which require no old paths:

| Repository | Branch | Dependency revision |
| --- | --- | --- |
| [Kindle](https://github.com/kvark/kindle/tree/handover/cdp-qualified-runtime-20261025) | `handover/cdp-qualified-runtime-20261025` | Exact library source used for recent learning; portable manifests |
| [Meganeura](https://github.com/kvark/meganeura/tree/handover/cdp-qualified-20261025) | `handover/cdp-qualified-20261025` | `49a192bfbb5f248041c281bfaea5c3a41d826b24` |
| [Blade](https://github.com/kvark/blade/tree/handover/cdp-qualified-20261025) | `handover/cdp-qualified-20261025` | `bbbc93b7d26b8196d9143944fe7d8a61d6da4583` |
| Naga fork | Existing published pin; no new compiler change | `13f582ddd2792befb46952498010248bd65818e6` |

The Kindle runtime branch pins both dependencies, so Cargo fetches them without
manual sibling checkouts. The takeover Kindle branch is one commit above the existing main base; the
original PR history remains intact. These preserve the measured implementation; they are
not an upstream port or a claim that all experimental code should be merged.
Only the user merges PRs. The PR branch retains the historical normal runtime;
its cleanup commit contains documentation, source receipts and handover tools.

**Verified locally:** all329 preserved library files match the measured source;
CPU `cargo check` succeeds for the workspace and Python extension with the new
Git dependencies. The source verifier is `tools/handover/verify_sources.py` and
its manifest includes the original native SHA256. With sibling source checkouts,
run `python tools/handover/verify_sources.py --kindle . --meganeura ../meganeura --blade ../blade`. No GPU qualification or new
performance claim follows from these CPU checks. Rebuilt binary bytes can differ
with paths, compiler and dependency metadata. The original native binary was
Linux x86-64 / CPython3.14; use a rebuild on the new machine.

Example setup (Python3.11+; original environment was3.14.4, Rust1.98.0):

```sh
git clone --branch handover/cdp-qualified-runtime-20261025 https://github.com/kvark/kindle.git
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

The measured runtime enabled:

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

Shared destination: **`/mnt/data/kindle-handover-20261025`**.
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

**Recovery during handover:** the original shared B2/T64 run directory disappeared
while this package was being assembled. The local export from the successful
October23 traced evaluation was secured instead. Its saved audit reports all
346 tensors unchanged and zero learning updates. It has learner counter867 and
environment counter5120 (4096 training actions +1024 frozen actions). Preserve
that metadata; do not relabel it as the original4096 snapshot or an uninterrupted
life. It is suitable as the saved trained-state input to a frozen restore check.
The original controller/training directory is not claimed to have been recovered.
No deleted shared run directories were recreated.

Historical result reports link to old `runs/` paths that may no longer exist
after the independent shared cleanup. The compact reports remain in Git, but
full replay/video re-audits may be impossible where their raw files were removed.
Use the verified package inventory, not old paths, to determine what survived. The handover is deliberately
not a copy of every profiling dump or checkpoint generation.

## First action on the new machine

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
