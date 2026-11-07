# CDP machine handover

Read [AGENTS.md](../AGENTS.md) for working constraints, the
[roadmap](kindle_single_life_dreamer_plan.md) for retained results and gates,
and [PR31](https://github.com/kvark/kindle/pull/31) for the live dashboard.
The five-game goal remains unfinished. Detailed experiment history belongs in
those reports; historical “next” paragraphs do not authorize restarting jobs.

## Current work — November 7

The active goal is **prove Breakout world-model training in isolation**. Actor/
critic and the five-game campaign are parked. Use the native pixel-supervised
CNN/categorical RSSM/RGB decoder as the inspectable reference; CDP is a later
efficiency comparison. No pretrained encoder or RAM/object/phase labels enter
model inputs/objectives. Full competence remains unproven.

The [motion learner](results/2026-11-07-breakout-world-motion-attention.md)
solves most observed ball reconstruction. The [coverage comparison](results/2026-11-07-breakout-world-coverage.md)
retains observed108/112,one-step94/109,eight-step14/107; waiting/contacts improve,
but long forecasts remain weak. Its final8,192 checkpoint is the common anchor
for the new diagnosis. Exact native/source3fa37b6d,SHAeafe8359,matching runtime
`/mnt/data/kindle-breakout-world-motion-20261107/runtime`.

The [ST energy comparison](results/2026-11-07-breakout-world-energy.md)
learns ending(last-life RIGHT97.182%,Brier.006129 versus50.211%/.144566),but
regresses one/eight-step tracking81/109,4/107. The [conditional score-function
comparison](results/2026-11-07-breakout-world-score-function.md)
passes exact/native derivatives yet fails learning: one/eight-step71/109,4/107;
NOOP8 false balls29/64; ending RIGHT8.591%,Brier.433024. Primary16/32 ball energy
.393950 versus.151784 ST/.169502 coverage. Both learners are closed; retain their
distinct capabilities without replacement,extension or seed-reliability claims.
Their roots/exact runtimes,all original failures and repaired evidence remain
linked in the reports. Earlier ST Python-count claim41/41 was wrong(actual27/27);
preserve its pinned records and separate correction. No old checks rerun.

The [CPU frozen-feature diagnosis](results/2026-11-07-breakout-world-readout-diagnosis.md)
finds useful short-term motion but no simple position-head rescue. Failed linear
probes do not prove information absent. Its eight feature and30 source/action
pins,63 coefficient/prediction files and interpretation limits remain retained.

The [fixed-weight gradient diagnostic](results/2026-11-07-breakout-world-gradient-noise.md)
is **closed**:108 raw forward/backward steps at the coverage anchor,three fixed
fitting batches,16 paired predictive draws,zero optimizer/model updates. Corrected
sampling variance is550–893× ST in the prior,1,250–11,699× reward,57–207×
continuation. Repeated-input gradients are bitwise exact; paired decoder relative
differences<6.15e-7. Prior noise394–460 versus observed-gradient norm7.12–8.01;
even16 samples poorly resolve corrected means. CPU AGC/LaProp sensitivity shows
strong nonlinear effects. This is a concrete estimator-noise finding,not a full
cause of fresh-run regression or a successful variance-reduction method. Earlier
`adam_m` norms included LaProp RMS normalization and did not prove AGC saturation.

Root `/mnt/data/kindle-breakout-world-gradient-noise-20261107`,source7c66bf12,
diagnostic executable SHA636bf7e5. Native43.282s/CPU analysis77.774s;50 CPU world
checks,strict Clippy,five CPU optimizer checks,28 data/21 execution/123 analysis
pins,one host guard/nine sealed pins and408 exact snapshot tensors pass. Initial
compile/tiny-fixture failures and pre-GPU output-index correction remain recorded.
Test-only measurement; retained packages/production graphs are unchanged.
`closure-review.json` verifies closure. No Kindle worker,learner,collector or
successor is active/selected. Do not repeat completed stages or old-model runs.

Remaining requirements include precise causal1/4/8/16/32 forecasts,reward/
uncertainty calibration,removed-brick memory,late-game/tunnel/second-wall coverage,
untouched tests and fresh confirmation. The retained oracle has no above-top-
brick endpoints; detector-support limits and duplicate contexts remain. Investigate
variance/objective coupling before more capacity or longer training; separately
justify a new experiment rather than selecting a scalar from this audit alone.

The [prior-reward comparison](results/2026-11-06-cdp-breakout-prior-reward.md#full200k-comparison)
remains closed and inconclusive:79.167/47.000 greedy/sampled versus matched
75.250/44.708,intervals crossing0,all0/24 two-wall. Retain prior reward off and
posterior reward1 for the retained agent,with original failures/repairs intact.

Completed details remain in the [roadmap](kindle_single_life_dreamer_plan.md)
and linked reports. Expanded November3 guide/handover remain in Git at
`0cf70be8d8cc9d4e6cab32bf1b8e97195fd1d0ab`. This condensation changes status prose,
not checkpoints,logs,runtimes,protocols or experiment sources. Keep current work
focused on the active allocation; link completed results rather than a job ledger.

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
