# CDP machine handover

Read [AGENTS.md](../AGENTS.md) for working constraints, the
[roadmap](kindle_single_life_dreamer_plan.md) for retained results and gates,
and [PR31](https://github.com/kvark/kindle/pull/31) for the live dashboard.
The five-game goal remains unfinished. Detailed experiment history belongs in
those reports; historical “next” paragraphs do not authorize restarting jobs.

## Current work — November 7

The active goal is **prove the world model training on Breakout**. Actor/
critic and the five-game campaign are parked. Use the native pixel-supervised
CNN/categorical RSSM/RGB decoder as the inspectable reference; CDP is a later
efficiency comparison. No pretrained encoder or RAM/object/phase labels enter
training. Full competence remains unproven.

The [motion learner](results/2026-11-07-breakout-world-motion-attention.md)
solves most observed ball reconstruction. Retain the [coverage trajectory
control](results/2026-11-07-breakout-world-coverage.md): observed108/112,
one-step94/109,eight-step14/107. Source/native3fa37b6d,SHAeafe8359,matching runtime
`/mnt/data/kindle-breakout-world-motion-20261107/runtime`. Accurate long futures,
reward/event timing and brick memory remain weak; do not promote on reconstruction.

The completed [loss experiments](results/2026-11-07-breakout-world-motion-rollout.md)
remain closed and failed retention. Motion-weighted H16 RGB targets improve
recorded eight-step tracking 14→25/107 but regress observed/one-step tracking,
bricks, ending calibration and primary 16/32 energy. Retain the coverage control;
do not repeat or extend completed learners. Linked reports preserve all earlier
ST/score-function, latent/recurrent-energy, readout and state-substitution results,
including original failures and separate repairs.
The [gradient-noise audit](results/2026-11-07-breakout-world-gradient-noise.md),
[future-informed state substitutions](results/2026-11-07-breakout-world-state-swap.md)
and [CPU drift decomposition](results/2026-11-07-breakout-world-drift-components.md)
remain diagnostics, not learned competence or grounds for a repeat.

The [CPU action-gap audit](results/2026-11-07-breakout-world-action-gap.md)
is closed. Matching metrics/horizons shows some short gains also transfer to fixed
plans: supported visible-ball pair accuracy at 8/16 improves 22.45→27.51% /
20.22→39.63%; at 32 it falls 5.90→0.29%. These are related model/oracle pairs,
not independent episodes or calibrated distribution scores. In eight verified
freeflight histories, all four oracle plans have identical ball paths through 8,
but the new model renders a ball in 64/64 NOOP versus 41/64 RIGHT images. Fixed
commands have nonzero but uneven fitting support. No checked source/data layout
bug is identified; native argument buffers were not retained. This narrows the
previous recorded-control/fixed-action framing without changing any failed gate.
877 input pins, 210 endpoint identities, 336 cached-feature checks and 140 energy
cells pass. Three full causal APNGs use retained arrays, with no new queries.

The [paired-data learner](results/2026-11-07-breakout-world-paired-actions-learning.md)
is closed and fails retention. Fresh 8,192 updates regress recorded observed
99→76/112, one-step 84→46/109, eight-step 25→5/107 versus motion rollout. New
paired fitting also underfits: 76/251 at eight, 0/102 at 32; freeflight-origin
subset 4/64 at eight. Fixed-plan primary energy .349371 is 106.12% worse than
control; freeflight detections NOOP/LEFT/RIGHT/REVERSE are 8/32/32/43 of 64, all
below 90%. Ending32 Brier .378178 and 134/304 non-ending cases >5% (max 96.24%) fail.
Some fixed-plan one-step accuracy improves; no broad/per-action competence claim.
Keep coverage; no unchanged extension, repeat or sweep.

Data and sampling changed together; do not infer paired data is generally useless
or isolate an architecture/optimizer cause. The
[collection](results/2026-11-07-breakout-world-paired-actions-data.md)
remains qualified: 56 fitting-only parents/448 branches, 13,068 new actions /
52,122 frames in 228.179s, source/restore replay 503,280 actions /2,012,229 frames separately.
Preserve parent NaN/112-file CPU repair and base-interpreter NumPy launch failure.
No completed stage repeats; all native learning/assessment stages pass first try.

Root `/mnt/data/kindle-breakout-world-paired-actions-20261107`; Python runner
`e751a883`, unchanged native `85dbe150` / SHA `5b112866`, matching motion runtime.
Native calls 769.167s / loop 1,128.611s; separate frozen 84 groups 299.786s. All 24 CPU
checks, 517 qualification/524 learning/857 branch/1,078 CPU comparison pins,
204 initial/frozen tensors, 108 initial evaluation arrays, eighteen restore arrays
and three current guards/27 sealed pins pass. Original evaluation cohorts and
all 989,057 suffix targets/10,562 terminal rows match. All 448 paired fitting
origins and three full causal APNGs remain retained. That allocation is closed.

The [CPU linear-reference diagnosis](results/2026-11-07-breakout-world-linear-reference.md)
is closed: at8 on32 common-control development origins, coverage/motion/paired
track3/9/1, versus last-two/history-line2/0. None has a fully linear observed
sixteen-frame history. Strict4-step fit/dev have only6/2 origins; line fit gets
6/6 and2/2, but support is too small for broad claims. No32-step eligible case.
Five unique cached histories underlie40 paired fitting8-step origins. All500
pins,21 source joins,56 export matches,42 instrument controls and an independent
505-input/260-cell/448-parent review pass. CPU4.418s +.255s, no native/emulator
imports or queries; no acceptance-gate change. Preserve the first closure
self-stdout hash failure and separate CPU-only seal repair; analysis/recount
remain unchanged and are not rerun.

The [frozen-visual staged world experiment](results/2026-11-07-breakout-world-frozen-visual.md)
is closed and fails retention. Warm-start8192+4096 updates preserves development
observed108/112 and one-step94/109, improves four-step33→51/109 and eight-step
14→24/107, but misses28/107; sixteen4/114 and thirty-two0/95 remain weak.
Primary16/32 energy0.204741 versus.169502 H0;
all existing event/per-plan gates remain in the linked result. Keep coverage;
no unchanged extension, repeat, collector, sweep or automatic successor.

Root `/mnt/data/kindle-breakout-world-frozen-visual-20261107`, native7ed07b2b/
SHA3032c0a6,runner442a2b45. Native phase calls383.411s/loop535.628s plus inherited
529.515/796.900s; final-only84 branch groups and CPU diagnostics are complete.
The experiment combines warm initialization,H16 and freezing; no isolated causal
claim. All36 visual weights stay exact,all32 trainable weights/72 frozen-weight
moments change. All375 learning/1129 branch pins,4096 finite loss rows,204 inherited/
frozen tensors,108 initial arrays,18 restore arrays and four successful guards pass.
Preserve first CPU fixture/native-config failures and separately qualified exact-F32
repair (92 CPU checks,353 pins,two native verification updates); failures remain
failures. No old stage repeats. No GPU worker or new allocation is selected.
Actor/critic,CDP comparisons and five-game work remain parked; full world goal open.

Remaining requirements include precise causal1/4/8/16/32 forecasts,reward/event
calibration,removed-brick memory,late-game/tunnel/second-wall coverage,untouched
families and fresh confirmation. The retained oracle has no above-top-brick
endpoints. A privileged state substitution is not world-model competence.

The [prior-reward comparison](results/2026-11-06-cdp-breakout-prior-reward.md#full200k-comparison)
remains closed/inconclusive; keep prior reward off and posterior reward1 for the
retained whole agent,with original failures/repairs intact. Expanded November3
working guide/handover remain in Git at0cf70be8d8cc9d4e6cab32bf1b8e97195fd1d0ab.
Keep current work focused on the active allocation; link completed details.

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
