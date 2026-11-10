# CDP machine handover

Read [AGENTS.md](../AGENTS.md) for working constraints, the
[roadmap](kindle_single_life_dreamer_plan.md) for retained results and gates,
and [PR31](https://github.com/kvark/kindle/pull/31) for the live dashboard.
The five-game goal remains unfinished. Detailed experiment history belongs in
those reports; historical “next” paragraphs do not authorize restarting jobs.

The [November 9 storage cleanup](results/2026-11-09-temp-cleanup.md) recovers
about 137 GB across five passes. The [November 10 pass](results/2026-11-10-temp-cleanup.md)
recovers another 671 MB from duplicates and regenerable bytecode, preserving
all checkpoints and evidence. All 8,054 large artifact paths and 670 current/
retained-runtime hashes verify; about 268.7 GiB remains available on the root
filesystem. Retained artifacts may share hardlinks and contain sparse zero
blocks; keep their bytes immutable. Archiving older logs could save roughly
29 GB, but the user's archive choice is pending and all original logs remain.

## Current work — November 10

The goal **prove the world model training on Breakout** remains active and
unproven. Select the [decoded-RGB event-head prototype](experiments/2026-11-10-breakout-world-rgb-event-heads.md)
for implementation/CPU checks,then separately declared native qualification.
Default-off standalone heads readh/z plusshared-CNN features oftheir own decoded
image. No posterior correction/extra transition orfuture actual image inbelief;
keep local width16,two-image history,RGB3 replay andencoder/RSSM geometry. At1M64,
expectedtwo head matrices640→896,+32,768 scalars,80/636,683/240;derive before
allocation. Explicitv28 wraps v25/v18;reject whole-agent/unqualified compositions.
No implementation orGPU allocation yet,no main learner selected. Root
`/x/Temp/kindle-world-rgb-event-heads-20261110`. Hypothesis only;h/z information
absence andcause ofcalibration/physics failures are not established.

The [wider-motion learner](results/2026-11-09-breakout-world-wide-motion-learning.md)
closes3/3 dense:107/112,105/109,77/109,58/107,31/114,11/95. Native3,888.344s/
loop4,065.276s versusearly physical1,841.013s/1,998.783s;no efficiency gain.
215/172/41/180 pins,426 distinct closed inputs,240 frozen/18 restore,all80 weights
changed,two guards/18 sealed files andall four terminal services pass. All five
final figures reviewed;original image-launcher observation failure remains with
separate same-invocation recovery,no restart. Earlier packaging errors remain.
Its [broader assessment](results/2026-11-10-breakout-world-wide-motion-branches.md)
fails:energy.160386921843 vsrequired<=.127126664317,ending Brier16/32.0426034/
.263400,RIGHT16.302606,276/304 false alarms. Native185.711s/CPU16.006s,
963/238/242 pins,240 exact arrays,210 oracle joins/168 feature checks,one guard/
nine sealed files andboth terminal services pass. No modelquery repeated.

The [survival decomposition](results/2026-11-10-breakout-world-wide-motion-survival.md)
closes:ordinary.160387→oracle-assisted.127843,wall/serve residuals.496435/.276916;
old edge-rebound residue improves.426554→.023331.312 pins/6,720 cases/840 groups/
420 checks,max4.44e-16,4.688s CPU. These are score substitutions,not passing
scores,learned repairs orcausal fractions. All original gates remain failed.
The [saved-case review](results/2026-11-10-breakout-world-wide-motion-error-cases.md)
checks209 inputs,22 plans/44 history groups/1,760 rows/ending joins,4.041s CPU,
five fixed grids reviewed. Serve FIRE8 has0/512 detected pairs;wall NOOP16
0/512 within1px,LEFT16 RMSE9.859px. Unsupported detector regions are not physical
ball absence,andstarting wall labels do not locate later error. System-Python/
NumPy ABI preparation failure remains;project-Python inspection/declared analysis
passes. No model/GPU/emulator query orupdate in these CPU stages;all sealed.

Earlier image-feedback/posterior-history,position/color,encoder coordinates,
wider-denoiser andother negatives stay closed. This prototype changes onlyhead
visual readout,not prior recurrence orlosses;no old failed comparison isrepaired.
Full physics,serve/collisions,natural events,brick memory,late/above-top coverage,
untouched families andfresh confirmation remain. Actor/critic,CDP andfive-game
work stay parked;no learner,collector,extension orautomatic successor isselected.

The [frozen-state diagnostic](results/2026-11-09-breakout-world-state-readouts.md)
stays closed:30 inputs,42 fits,144 metric rows,1.918s CPU,zero model queries.
Partial h/z probes omit24,576 image-history values;they provide no precise
transferable forecast rescue andcannot prove information absence. All original
image scores,plots,folds and27,648 per-origin rows remain retained. Wider denoiser
failure andall earlier architecture failures stay closed;this is a distinct
local-decoder branch,not a repeat orassumed fix.

The [posterior-history RGB100 learner](results/2026-11-09-breakout-world-posterior-natural100.md)
stays closed/failed at2/3 dense gates: observed108/112,one99/109,eight19/107,
sixteen5/114,thirty-two0/95 versus prior-suffix103/112,102/109,53/107,22/114,8/95.
Fitting8 falls84/176→41/176;current reconstruction/paddle scores improve,but
long ball forecasts regress. Native1,820.531s/loop1,951.302s,182/174/38/182 pins,
two guards/18 sealed files andexact18-array restore pass. All three services are
inactive/successful;all five final figures reviewed. No extension orbroader
assessment. Bundled data/loss/budget/source changes do not isolate a cause.

The [local forward comparison](results/2026-11-09-breakout-world-local-forward-contract.md)
stays strictly failed on small observed/sampling differences,with exact forced-
sample recurrent/generated suffix values;it does not prove free-running physics.
The [late learner](results/2026-11-09-breakout-world-late-action-learning.md)
and[earlier posterior-history run](results/2026-11-09-breakout-world-posterior-history.md)
remain closed/failed,including original source-specific failures/repairs.
The [early feature assessment](results/2026-11-09-breakout-world-action-features-branches.md)
stays failed at.224881 versus one-hot.172366,296/304 nonending alarms. Full
physics/events,brick memory,late/above-top coverage,untouched families andfresh
confirmation remain required. Actor/critic,CDP andfive-game work stay parked.

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
