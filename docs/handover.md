# CDP handover — October25,2026

**Receiving-host update:** [AMD takeover](results/2026-10-25-cdp-amd-takeover.md).
RX7900XT/RADV cannot run the saved BF16 configuration. A labelled F16-input copy
passes restore with all346 tensors unchanged. The fresh2M B2/T64 life and audits
are complete: frozen500k/final/initial scores14.71/10.04/1.42, all0/24 two-wall
successes. Training took11h40m; that service has exited. Paired early updates
show precision-dependent differences, and Auto's short training/restore passes.
The fresh [500k Auto comparison](experiments/2026-10-25-cdp-amd-auto.md) also
completed: frozen15.67 versus F16's14.71, all0/24 two-wall successes, in2h53m.
All audits pass. Both services have exited; do not restart or extend these lives.
The user now prioritizes cheaper game-specific recipes; use the roadmap.
[Cheap Freeway screens](results/2026-10-25-cdp-cheap-freeway.md) finish200k actions
in about8 minutes but remain weak (unassisted/aided frozen0.00/0.25, initial0.00).
[Smaller native recipes](results/2026-10-25-cdp-cheap-runtime.md) add4M capacity
and32/48 image geometry; GPU training/frozen checks and retained64 restore pass.
The [200k Boxing screen](results/2026-10-25-cdp-cheap-boxing.md) completes in6m03s
but fails: frozen−3.875 versus initial0.00, both10/24 wins, all audits pass.
The [Pong4M/RGB48 screen](results/2026-10-25-cdp-cheap-pong.md) completes in10m22s:
frozen−17.125 versus initial−20.375, both0/24 wins, all audits pass.
Fresh Boxing2017,1M/RGB32/500k/R32 now passes: frozen76.8333 versus initial2.875,
24/24 wins,124939 updates in28m22s; all audits pass and the service exits. The fixed
recipe also passes fresh confirmation4013:79.1667 versus initial−0.625,24/24 wins,
500k/124939 updates in28m13s, all audits pass and service exits. Two useful learners
do not establish broad reliability. Details and curves are in the Boxing report.
[Qbert4M/RGB48/R32](results/2026-10-25-cdp-cheap-qbert.md) finishes200k in19m45s:
frozen809.375 versus180.2083,0/24 first pyramids. Mean coverage improves5.83→16.63
of21 cubes; all audits pass, but the task gate fails. No extension.
[Breakout4M/RGB64/R32](results/2026-10-26-cdp-cheap-breakout.md) finishes200k in
22m30s: frozen6.2083 versus1.375,0/24 two-wall successes, all audits pass. A CPU
replay verifies one12061-action delay before FIRE after a lost life. No extension.
[Breakout's1% native mixture](results/2026-10-26-cdp-breakout-unimix.md) completes
200k in22m29s: frozen7.0833 versus initial1.375, longest selected no-FIRE gap447
versus control12115. The baseline weights with only frozen mixture score6.0833
with gap724 in9.8s and zero updates. All three policies remain0/24 two-wall
successes. All audits pass and services exit; no extension.
[Compact aided Freeway](results/2026-10-25-cdp-cheap-freeway.md#compact-rgb32-with-the-successful-update-budget)
completes500k/124939 updates in28m17s: frozen31.0417 versus actual initial0.00,
24/24 versus0/24 crossing successes, aid disabled in evaluation. All audits pass
and service exits. Constant-UP CPU reference scores21.6667,0/24. The successful
1M/RGB32/R32 recipe's focused confirmation follows below; no isolated attribution.
[Pong500k/R32](results/2026-10-25-cdp-cheap-pong.md#r32-and500k-actions) completes
in48m53s: frozen11.0 versus initial−20.375,23/24 versus0/24 wins. All audits pass
and service exits; the mean15 gate fails. No unchanged extension.
[Freeway seed4013 confirmation](results/2026-10-25-cdp-cheap-freeway.md#fixed-recipe-seed4013-confirmation)
also passes:32.000 versus initial0.00,24/24 crossing successes,500k/124939 updates
in28m50s. All audits pass and service exits. Two compact learners now pass within
30 minutes each. The [retained Breakout2M frozen check](results/2026-10-26-cdp-retained-breakout-auto.md)
scores67.5833 with0/24 two-wall successes,21,976 actions in14.127s and zero updates.
All346 source/copied/frozen tensors match; all audits pass and service exits.
Only precision metadata changes to Auto; the warm start below is separately declared.
[Qbert coefficient100](results/2026-10-25-cdp-cheap-qbert.md#coefficient100-does-not-improve-the-fixed-budget-result)
completes200k/49939 updates in20m12s: frozen809.375,0/24 pyramids. Score matches
the control but mean coverage falls16.625→13.708 of21 cubes. All audits pass and
service exits; no extension.
The [H5 Breakout warm start](results/2026-10-26-cdp-breakout-short-imagination.md)
completes50k new actions/12439 updates in8m02s but regresses67.5833→27.2083,
both0/24 two-wall successes. The frozen source trace matched H15 before learning;
all audits pass and service exits. Retain the source; no extension.
The [Pong4013 confirmation](results/2026-10-25-cdp-cheap-pong.md#fixed-recipe-seed4013-confirmation)
completes500k/124939 updates in50m14s: frozen7.333 and22/24 wins versus own
initial−20.333 and0/24. All audits and curves pass; service exits successfully.
Both compact Pong learners win most matches but miss mean15. No extension.
The [six-action Qbert test](results/2026-10-25-cdp-cheap-qbert.md#six-action-choices-run-faster-but-reduce-frozen-coverage)
completes200k/49939 updates in18m21s: frozen605.208 versus own initial148.958,
0/24 pyramids. Coverage10.167/21 and score are lower than the full18 control.
All audits/curves pass and service exits; no extension. The separate CPU Breakout
pixel follower scores11.333,0/24 two-wall successes, with complete replay; no aid.
The [compact Breakout H5 result](results/2026-10-26-cdp-breakout-compact-h5-budget.md)
retains a45-minute cutoff at487,736 actions/121,873 updates: frozen8.458 versus
initial1.375,0/24 two-wall successes. Its shared prefix costs18.06% less wall time
than H15, but the larger-budget recipe still fails the task. Numerical,
checkpoint/accounting and frozen replay checks pass; the original controller/
guard retain their time-budget failure. Both services exit; no extension.
Current allocation: [Pong coefficient.01](experiments/2026-10-26-cdp-pong-small-intrinsic.md),
otherwise fixed4M/RGB48/Auto/N8/B8/T16/H15/R32,500k and55-minute cap,
fresh1009/eval4000001009, no aid. Own initial/fixed final controls and existing
mean15/90%-win gate remain. Artifacts/controller:
`pong-4m-48-r32-intrinsic001-500k`, `run_pong_intrinsic001_500k.py`; owner
`kindle-cheap-pong-intrinsic001-500k-20261026`. Check live state before GPU work.
Started06:36:15 UTC, source`12a6553`. Full config/counters, finite initial weights
matching all346 control tensors and first updates pass;3,578,225 parameters.
Fixed evaluations/full audit follow; no automatic extension or next allocation.
Current native source`bfb8c31`, runner`1ed07af`, native SHA256
`f2606ac28f5f293b73ccdb3f408278c6c584e50ccc73d52fd605a39b231609a5`.
Evidence, wheels and build receipt: `/x/Temp/kindle-cheap-20261025`;
`original-native` preserves the prior AMD binary. Both installed and source-tree
extensions now use the new native build; old completed queues are historical.
Use these reports for current host, precision, results and evidence details.
The historical handover steps below describe its origin.

The goal remains strong, reliable learning on **Boxing, Pong, Freeway, Breakout
and Qbert**, then native games and transfer. Training is the priority; another
backend optimization/JAX-parity campaign is not the next step.

Read this file, the [short roadmap](kindle_single_life_dreamer_plan.md), and the
[Breakout experiment](experiments/2026-10-23-cdp-breakout-context64.md). The
[PR31 description](https://github.com/kvark/kindle/pull/31) is the live dashboard.
Historical reports retain failed attempts and limitations; their old next steps
are not instructions to restart those queues.

## Source you actually need

The working branch is **`phase2-levjepa-evaluation`**, continuing PR31.
It now contains the qualified c8aeba28 library implementation directly, with
portable Git dependencies. There is no separate takeover branch to switch to.
The previous356-commit work history and detailed experiment corpus are retained
on [archive/phase2-before-cleanup-20261025](https://github.com/kvark/kindle/tree/archive/phase2-before-cleanup-20261025).
The working branch keeps current results, source and tests; no `runs/` tree,
raw experiment data or checkpoint history is committed.

| Dependency | Revision |
| --- | --- |
| [Meganeura](https://github.com/kvark/meganeura/tree/handover/cdp-qualified-20261025) | `423d373634a1b1f04ff623db2b419757f1e8d96d` |
| [Blade](https://github.com/kvark/blade/tree/handover/cdp-qualified-20261025) | `bbbc93b7d26b8196d9143944fe7d8a61d6da4583` |
| Naga fork | `13f582ddd2792befb46952498010248bd65818e6` |

These preserve the measured implementation, not a fresh upstream port. Only the
user merges PRs. Cargo fetches the dependency branches without sibling checkouts
or old absolute paths. The prior handover verified329 library source files
against the measured implementation and passed CPU workspace/Python compilation.
`tools/handover/verify_sources.py` and its manifest preserve the source receipt
and eight explicit cleanup differences: Rust formatting, actual dependency
revision reporting, and two stale CPU test repairs. Learning operations and
checkpoint architecture validation are unchanged. A separate Meganeura
portability repair records GPU commands directly on Metal/GLES, whose Blade
backends do not expose Vulkan secondary encoders; the Vulkan path is unchanged. The full CPU Rust suite passes
(127 tests;78 GPU tests ignored), as does formatting.
No GPU qualification or speed claim follows from CPU checks. Rebuilt binary
bytes can differ with paths, compiler and dependency metadata. Rebuild on the
new machine rather than importing the original Linux/CPython3.14 binary.

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
