# CDP machine handover

Read [AGENTS.md](../AGENTS.md) for working constraints, the
[roadmap](kindle_single_life_dreamer_plan.md) for retained results and gates,
and [PR31](https://github.com/kvark/kindle/pull/31) for the live dashboard.
The five-game goal remains unfinished. Detailed experiment history belongs in
those reports; historical “next” paragraphs do not authorize restarting jobs.

## Current work — November 7

The user selected [Breakout world-model learning in isolation](kindle_single_life_dreamer_plan.md#selected-direction-breakout-world-model-in-isolation) after
pausing the five-game campaign. Start with a pixel-supervised latent RSSM
baseline using Kindle's native CNN/RSSM/RGB decoder. This deliberately supersedes
the preceding CDP-only direction for this experiment. Remove actor/critic,
replay-value, imitation and intrinsic-reward training dependencies. Use fixed
trajectory data, episode-family splits and explicit open-loop physical tests;
gameplay score is a later integration test. CDP remains a subsequent efficiency
comparison after the pixel baseline establishes competence.

The active goal is to prove Breakout world-model training; the five-game
campaign remains parked. The [closed motion-weighted pixel learner](results/2026-11-07-breakout-world-motion-attention.md)
passes observed development ball reconstruction:107/112 within1 resized pixel
(95.54%), versus5/112 for plain H0. One-step forecasts reach85.32%, falling to
32.11%/7.48% at4/8 decisions. Unseen bricks remain weak. No actor/critic learns.

All8,192 updates,307 pins,204 frozen tensors and18 restore arrays pass. Native
training530.143s; total734.272s. Source/native3fa37b6d, SHA prefixeafe8359,
matching runtime `/mnt/data/kindle-breakout-world-motion-20261107/runtime`.
This learner/restore is closed; do not repeat or extend it. Earlier negative
results and separate CPU repairs remain in linked reports.

The [closed fixed-plan challenge](results/2026-11-07-breakout-world-branches.md)
finds short paddle responses but drifting ball forecasts,wrong serving and severe
last-life ending underprediction(.581% versus100% for RIGHT). Its431 input pins,
both guards and204 invariant tensors pass; preserve its three original failures
and separate repairs. No completed phase repeats. Full world competence is unproven.

The [closed coverage experiment](results/2026-11-07-breakout-world-coverage.md)
adds16,384 fitting actions/65,492 frames and a4/2/1/1 old/new/waiting/terminal
sampler,without changing model/loss/native. At8,192 updates(529.515s native),
observed balls108/112 remain precise; NOOP8 false balls95.3%→0%,paddle-approach32
ball energy.8510→.2563,last-life RIGHT ending probability.581%→50.211%(oracle100%).
Ending Brier.144566 still fails.10; long forecasts/reward/bricks remain mixed.

Root `/mnt/data/kindle-breakout-world-coverage-20261107`. All35/80/482 phase
input pins,204 initial/frozen tensors,18 initial-development/final-restore arrays
and four native guards pass. Original training queue is protocol-failed on a
wrong total-arrival expectation(32,849 sampled versus32,850 available); separate
CPU audit and only the unstarted restore complete evidence. Preserve that failure,
the CPU comparison SyntaxError/repair,and the invalid FIRE gate:90% pair matching
is unattainable on its finite oracle(max78.125%); do not relabel it passed.
Use proper forecast scores/oracle uncertainty in future protocols.

The [pointwise predictive prototype](experiments/2026-11-07-breakout-world-predictive.md#closed-prototype-before-full-learning)
is parked without a full learner: per-sample target error can average uncertain
futures. Its one-update core completed; original final68-change assertion failed
(actual65, three reward-hidden tensors receive no gradient with a zero output
weight). Separate CPU-only70-pin repair verifies evidence, preserves failure,
and does not rerun GPU. Root `/mnt/data/kindle-breakout-world-predictive-20261107`.

The selected [joint energy experiment](experiments/2026-11-07-breakout-world-energy.md)
compares two sampled prior worlds to the arrival and to each other. RGB metric
weights use real past frames only; reward/continuation are sampled from their
existing heads/encoded targets. Native categorical gradients remain straight-through
surrogates. No new parameters/inputs; unchanged posterior losses. Explicit
`predictive_motion_energy`,13 reports; no actor/critic or collector.
Root `/mnt/data/kindle-breakout-world-energy-20261107`. Native/source161839d9,
SHA8d5552e5,matching root/runtime.43 CPU world tests,41 affected and packaged
Python checks,strict workspace/bindings Clippy pass. Preserve initial CPU fixture/
path failures and repairs. New native F64 energy fixture and full-size preflight
pass,77 input pins,two clean guards/sealed audits,204 initial/frozen tensors,
18 initial/restore arrays and65 expected first-update changes. Old observed
components match; no GPU fault/recovery. No completed qualification repeats.

The fixed fresh matched-init8,192-update learner and restore are closed.
89 learning pins,all68 parameters changed,8,192 finite13-report rows,204 initial/
frozen tensors,18 exact initial-development/restore arrays and both sealed guards
pass. Native987.968s/loop1,338.229s,versus529.515/796.900s coverage. Dense observed
ball109/112 versus108/112;one-step81/109 versus94/109,eight-step4/107 versus14/107.
Brick balanced error.207707 improves.223954. No blanket promotion.

Learning reviewed; only this final checkpoint's frozen oracle evaluation is
active: history16/32,eight draws,84 groups,648 input pins. No learning,collector
or old-model rerun. Do not start another GPU job or mistake an observer timeout
for a stopped worker. A CPU three-bin counterexample verifies that sampled
head-coordinate ST gradients can miss variance correction; it does not attribute
this full learner's regression. Full goal,untouched tests,late-game/second-wall
coverage,brick memory and fresh confirmation stay unfinished.

The [prior-reward comparison](results/2026-11-06-cdp-breakout-prior-reward.md#full200k-comparison) remains closed and inconclusive:
79.167/47.000 greedy/sampled versus matched75.250/44.708, with paired intervals
crossing zero, all0/24 two-wall. Retain prior reward off and posterior reward1
for the retained agent. Its original failures, repairs and closure remain intact.

Completed experiment details and their failures remain in the [roadmap](kindle_single_life_dreamer_plan.md)
and [diagnosis](results/2026-10-29-cdp-learning-diagnosis.md), with links to exact protocols and artifacts.
The expanded November3 working guide and handover remain in Git at
`0cf70be8d8cc9d4e6cab32bf1b8e97195fd1d0ab`. This condensation removes duplicated status prose only;
no checkpoint, log, runtime, protocol or experiment source is changed.
Keep this current-work section focused on the active allocation; replace its
status in place and link completed results instead of accumulating a job ledger.

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
