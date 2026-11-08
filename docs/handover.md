# CDP machine handover

Read [AGENTS.md](../AGENTS.md) for working constraints, the
[roadmap](kindle_single_life_dreamer_plan.md) for retained results and gates,
and [PR31](https://github.com/kvark/kindle/pull/31) for the live dashboard.
The five-game goal remains unfinished. Detailed experiment history belongs in
those reports; historical “next” paragraphs do not authorize restarting jobs.

## Current work — November 8

The goal **prove the world model training on Breakout** remains active and unproven.
Actor/critic, CDP comparisons and the five-game campaign stay parked. Use native
pixel supervision and H0 as the inspectable reference. No pretrained encoder or
object/RAM training labels. The [local-action protocol](experiments/2026-11-08-breakout-world-local-action.md)
has passed native qualification and started one fresh8,192-update learning check.
Direct incoming action planes and a16-channel image-history branch test local
control alongside the RSSM. Source/runner7f797a2b,native SHA38680c73;140 CPU/273
isolated packaged Python checks and strict core/test/binding Clippy pass. Geometry
80 logical/8,893,203 scalars/240 tensors. Qualification passes1,250 pins/31.256s:
F64/native values/derivatives,lane/causal target controls,one finite13-report
update,210 common initial tensors/16 nonimage arrays and18 initial/trained
restores. Host guard passes nine sealed files. Cache836,613,824 bytes released.
Learning pins1,257 inputs:original4/2/1/1 plan,RGB64/12M/B8/T32/H16/full BPTT/Auto,
seed41001,rates4e-4,mixing.01,RGB100/events1. Retain actual initial/2048/4096/final,
384 dense origins and exact final restore. Unit
`kindle-world-local-action-learning-20261108`;observer audits completion without
restart. After reviewed learning,declare only its84 final frozen groups. Root
`/mnt/data/kindle-breakout-world-local-action-20261108`;consult the protocol.
All original gates stay;the change does not establish a global-state cause.

The [pixel-transport experiment](results/2026-11-08-breakout-world-transport.md)
is complete and fails retention. Default-off595-weight horizontal/vertical shift
head plus RGB fallback;RSSM and13-report event objective stay. Fresh8192 updates,
RGB64/12M/B8/T32/H16/full BPTT/Auto,seed41001,rates4e-4,mixing.01,RGB100/events1,
original data/4/2/1/1 plan. Source/runner80dc8c76,native SHA013ae7a7;82 logical/
8,872,550 scalars/246 tensors. Dense dev observed/one/eight94/112,102/109,31/107;
fitting eight54/176. One/eight gates pass,observed fails. All commands retain64/64
freeflight balls,but primary16/32 energy.200277 is worse than H0.169502 and misses
.127127. Ending16/32 Brier.017587/.337100,RIGHT16 risk.068080,183/304 nonending
alarms>5%. Presence improves;precise physics/events remain inaccurate.

Qualification1,247 pins/115.674s,learning1,254 input/170 CPU pins and84 final-only
frozen groups/1,345 pins pass their implementation/audit checks. All8192 finite
reports,82 changed weights,240 cached common initial tensors/16 nonimage arrays,
246 own initial/frozen tensors,384 dense origins and18 exact final restore arrays.
Four successful guards/36 sealed files,no recovery/NVML/unfinished child.122 CPU/
249 packaged Python checks and strict core/test/binding Clippy pass. Preserve the
fixture compile failure and separate repair;invalid first uint8-truth display
preview and separate CPU-only display repair remain. No model query is repeated.
Native1562.699s/loop1734.273s;assessment220.168s,CPU comparison74.807s. Cache
836,444,204 counted bytes released;runtime,weights,arrays,figures and full causal
clips retained. Root `/mnt/data/kindle-breakout-world-transport-20261108`.

The [fixed-filter/future-event phase](results/2026-11-08-breakout-world-fixed-events.md)
and earlier linked experiments remain closed with their original failures. Keep
H0 as reference. Do not repeat completed learners,assessments or collectors. No
unchanged extension,sweep or automatic successor. Precise physics,event timing/
calibration,persistent brick memory,late/tunnel/second-wall coverage,untouched
families and fresh confirmation remain required. Existing oracle has no above-
top-brick endpoints. Full world-model and five-game goals stay unfinished.

## Current AMD runtime and retained position

Use Auto precision on RX7900XT/RADV. BF16 and `KINDLE_DEVICE_TARGETS` are
unavailable; see the [paired precision evidence](results/2026-10-25-cdp-amd-precision.md).
The completed F16/Auto context experiments do not justify extensions.
Native learning remains on Meganeura + Blade; backend porting, optimization
and JAX parity are parked. Retained agents explicitly select centered CDP; the
new isolated-world baseline selects pixel supervision. The CLI default is learned
RGB. Keep native pooled frames and one GPU resize.

Current learning uses native447fe5013b6442c477fc7bfef7eab4122082bc54,
SHA256 `35ad0db42fc7ed5dbaf6dacddff613fdc04d83ae6e0f38548532f6ab2c7812a2`.
Breakout uses `/x/Temp/kindle-breakout-corrective-20261028/runtime` with its
exact qualified runner. Qbert uses
`/x/Temp/kindle-qbert-color-masks-20261101/runtime` and its qualified faster,
decision-equivalent teacher. Python is `/x/Code/kindle/python/.venv/bin/python`
(CPython3.13). Declarations pin exact sources, binaries, runners, teachers,
configuration and environment. Do not mix optional runtime overlays.

The latest completed guard records boot `a8648671-8577-4f42-9b46-a4345d3ac297` and kernel
`7.0.0-38-generic`, with no host recovery performed. Check live state before
GPU work; preserve guard stops on actual faults or nonfinite values. An observer
timeout does not prove the learner stopped. Do not reset/reload/reboot the GPU.

Boxing and aided Freeway pass on two compact learners; Pong passes on both
retained learners with greedy evaluation. Qbert's fixed240k/R20 recipe passes
its15k/90% gate on the development learner and
[one fresh greedy confirmation](results/2026-11-02-cdp-qbert-r20-seed18013.md),
in about26 minutes each. Its sampled means still miss15k. Breakout retains
[70.750 sampled/143.667 greedy](results/2026-11-02-cdp-breakout-replay-value1.md)
from sampled collection, and a distinct
[78.542 sampled/90.042 greedy](results/2026-11-02-cdp-breakout-greedy-training.md)
point from greedy collection. The [fresh-seed/common-cohort check](results/2026-11-06-cdp-breakout-seed22003.md)
scores94.625/75.250 greedy and70.500/44.708 sampled on1009/22003; the historical
greedy point varies materially with cohort. All remain zero of24 two-wall successes.
The [lower-dynamics/value1 comparison](results/2026-11-02-cdp-breakout-value1-dynamics1e4.md)
regresses both modes and is closed. No completed stage should repeat.

The full 3,188-line handover and 376-line working guide before the November2 condensation
remain in Git at `52c2174e2bc6e3cd9ff6ae5ae9b598308e9a8f51`; the roadmap and
linked results preserve experiment details. PR31's earlier expanded dashboard
is retained at `/x/Temp/kindle-pr31-dashboard-20261102/pr-before-condensation.json`.
The November3 dashboard before condensation is separately saved at
`/x/Temp/kindle-pr31-dashboard-20261103/pr-before-condensation.json`.
No artifact, checkpoint or experiment source is removed by this documentation
change. The retained original transfer/build instructions below are historical;
current active declarations take precedence.

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
