# Kindle working direction

Kindle learns while acting. Games are the first testbed; sparse explicit rewards
and human guidance are allowed. Favor minimalism, expressiveness, safety and
speed. Keep learning/inference native on Meganeura + Blade. Python is for game
adapters, reference controls and analysis. Follow `/mnt/data/GUIDELINES.md` when
available. Only the user merges PRs; commits and pushes are allowed.

## Start here

- [Machine handover](docs/handover.md): source branches, build instructions,
  shared checkpoint/evidence package and unfinished work.
- [Roadmap](docs/kindle_single_life_dreamer_plan.md): results, priorities and gates.
- [PR31](https://github.com/kvark/kindle/pull/31): live status dashboard. Update its
  description when status changes; do not create a separate STATUS.md.
- [Learning diagnosis](docs/results/2026-10-29-cdp-learning-diagnosis.md):
  representation, forecasting, actor/critic and training-coverage evidence.
- [GPU incident](docs/results/2026-10-23-cdp-context64-gpu-fault.md):
  initialization failures on the original machine and their limits.

## Current work — November 7

The active goal is **prove the world model training on Breakout**. Actor/
critic and the five-game campaign are parked. Use the native pixel-supervised
CNN/categorical RSSM/RGB decoder as the inspectable reference; CDP is a later
efficiency comparison. No pretrained encoder or RAM/object/phase labels enter
training. Full competence remains unproven.

The [motion learner](docs/results/2026-11-07-breakout-world-motion-attention.md)
solves most observed ball reconstruction. Retain the [coverage trajectory
control](docs/results/2026-11-07-breakout-world-coverage.md): observed108/112,
one-step94/109,eight-step14/107. Source/native3fa37b6d,SHAeafe8359,matching runtime
`/mnt/data/kindle-breakout-world-motion-20261107/runtime`. Accurate long futures,
reward/event timing and brick memory remain weak; do not promote on reconstruction.

The completed [loss experiments](docs/results/2026-11-07-breakout-world-motion-rollout.md)
remain closed and failed retention. Motion-weighted H16 RGB targets improve
recorded eight-step tracking 14→25/107 but regress observed/one-step tracking,
bricks, ending calibration and primary 16/32 energy. Retain the coverage control;
do not repeat or extend completed learners. Linked reports preserve all earlier
ST/score-function, latent/recurrent-energy, readout and state-substitution results,
including original failures and separate repairs.
The [gradient-noise audit](docs/results/2026-11-07-breakout-world-gradient-noise.md),
[future-informed state substitutions](docs/results/2026-11-07-breakout-world-state-swap.md)
and [CPU drift decomposition](docs/results/2026-11-07-breakout-world-drift-components.md)
remain diagnostics, not learned competence or grounds for a repeat.

The [CPU action-gap audit](docs/results/2026-11-07-breakout-world-action-gap.md)
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

The [paired-data learner](docs/results/2026-11-07-breakout-world-paired-actions-learning.md)
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
[collection](docs/results/2026-11-07-breakout-world-paired-actions-data.md)
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

The [CPU linear-reference diagnosis](docs/results/2026-11-07-breakout-world-linear-reference.md)
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

The [frozen-visual staged world experiment](docs/results/2026-11-07-breakout-world-frozen-visual.md)
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

The [prior-reward comparison](docs/results/2026-11-06-cdp-breakout-prior-reward.md#full200k-comparison)
remains closed/inconclusive; keep prior reward off and posterior reward1 for the
retained whole agent,with original failures/repairs intact. Expanded November3
working guide/handover remain in Git at0cf70be8d8cc9d4e6cab32bf1b8e97195fd1d0ab.
Keep current work focused on the active allocation; link completed details.

## Direction and retained results

Training is the priority. Backend optimization,upstream porting and JAX parity
are parked. The retained agent uses **centered CDP** with a jointly learned CNN,
categorical RSSM and imagined actor/critic. The newly selected isolated-world
baseline uses pixel supervision as specified above. No pretrained DINO or Tiny
JEPA encoder is required. Explicitly select the intended objective; learned RGB
remains the CLI default.

The priority is compute-constrained success on this game set. Game-specific
model sizes,resolutions and learning settings are allowed; one universal recipe
is not required. Prefer useful frozen learning per GPU-hour. Start fromT16;
T64 is an experiment,not the default. Native32/48/64 geometry is supported.
Size1M works well for Boxing/Freeway but was much weaker for Breakout/Qbert.
Changing several settings to find a cheap useful recipe is allowed; disclose
them without claiming isolated causes or broad reliability.

Work on `phase2-levjepa-evaluation` and PR31. The branch includes the qualified
c8aeba28 implementation with portable dependencies. Do not switch to a separate
takeover branch or modify neighboring worktrees. Keep `runs/`,large data and
archived experiment ledgers out of Git.

The **five-game goal is unfinished**:

| Game | Retained evidence | Remaining limitation |
|---|---|---|
| [Boxing](docs/results/2026-10-25-cdp-cheap-boxing.md) |1M32:76.833/79.167,two learners with24/24 wins,in28m22s/28m13s|Both pass; broad reliability open|
| [Freeway](docs/results/2026-10-25-cdp-cheap-freeway.md) |Aided1M32:31.042/32.000,two learners with24/24 successes,in28m17s/28m50s|Aid off during evaluation; broad reliability open|
| [Pong](docs/results/2026-10-25-cdp-cheap-pong.md) |12M48/B16/R16:greedy15.208 on both learners,24/24 wins,in41m28s/41m31s|Greedy passes narrowly; sampled passes one of two|
| [Breakout](docs/results/2026-11-02-cdp-breakout-replay-value1.md) | Retain 200k sampled70.750/greedy143.667 in21m28s; [400k budget point](docs/results/2026-11-03-cdp-breakout-budget400k.md) scores92.792/173.708 in43m37s | [New cohort](docs/results/2026-11-06-cdp-breakout-seed22003.md):greedy94.625/75.250 andsampled70.500/44.708 on1009/22003; material seed/cohort variation. All0/24 two-wall; original864/90% gate unfinished |
| [Qbert](docs/results/2026-11-02-cdp-qbert-r20-seed18013.md) |Fixed240k/R20:greedy15,279.167/15,931.250 on16001/18013,24/24 first pyramids each,in25m58s/26m23s|One passing development learner andone fresh confirmation;sampled12,972.917/14,976.042 both fail15k. Earlier R24 failure remains; broad reliability open|

Published long-run scores are aspirations,not matched-budget comparisons.
Finite losses,short runs and backend speed do not establish mastery or CDP
superiority. Guided online returns and replay imitation losses are not frozen
competence. Keep detailed results in the roadmap and linked reports,not a
growing history in this file.

Useful Breakout diagnostic references:
- [Value consistency](docs/results/2026-11-02-cdp-breakout-value-consistency.md):
  sampled one-step RMSE.732/.941 versus complete-return6.682/6.691. Short targets
  use the same critic's bootstrap; terminal slices are outcome-conditioned.
  All32 diagnostic/73 source/912 active pins pass;6.366s CPU-only,original Qbert
  identity preserved. Import failure/repair retained; no new loss orlearner.
- [Encoder4e-5](docs/results/2026-11-01-cdp-breakout-encoder4e5.md):
  sampled/greedy56.292/104.667 versus25.292/42.625 at2e-5;21m28s versus22m27s.
  All227 pins,recovery,initial/frozen tensors,replays,guards and original
  services pass. Retain4e-5; earlier4M/expert10 rate failure remains relevant.
- [Representation and own-policy replay](docs/results/2026-11-01-cdp-encoder4e5-representation.md):
  ball/motion readouts improve but remain less precise than pixel diagnostics.
  All replacement action readouts trail the learned actor. Own episodes last
  25.36% longer and earn95.88% more reward/action; state distributions differ.
- [Teacher takeovers](docs/results/2026-11-01-cdp-breakout-recoverability.md):
  earlier control avoids108/120 recorded losses versus26/120 at the last
  near-descending state. This supports earlier-control investigation,not exact
  causal error attribution or optimal irrecoverability. CPU-only,293 pins pass.
- [Delayed-takeover states](docs/results/2026-11-01-cdp-breakout-takeover-states.md):
  onlytwo of21 earlier-only rescues project contact withinfive actions at delay16;
  14 already match the teacher command. Timing estimates/imperfect teacher do not
  isolate the cause. All480 joins/478 saved-branch recounts,nine inputs and524
  active Qbert pins pass;CPU-only2.766s,complete15:23:52 UTC November1. Do not rerun.

Completed negative results and protocol failures remain retained. Do not restart
completed learners,old queues,failed assessments or collectors; no unchanged
long extensions,broad seed campaigns or automatic successors. Consult the linked
protocol before a separately justified new experiment. Preserve original failures
when a separate CPU-only repair completes missing evidence; never relabel a
protocol-failed comparison as passed. Single-actor learning precedes swarms or
concurrent learners.

## Runtime and precision

Use **Auto** on the current AMD RX7900XT/RADV. BF16 and
`KINDLE_DEVICE_TARGETS` remain unavailable. Three paired early updates leave the
first reward projection unchanged underF16 while Auto changes it; derivatives
and mechanism remain unmeasured. Completed F16/Auto B2/T64 lives do not justify
extensions. [Precision evidence](docs/results/2026-10-25-cdp-amd-precision.md).

Current retained learning uses native447fe50/SHA256 prefix35ad0db and the exact
matching runner. Breakout retains `/x/Temp/kindle-breakout-corrective-20261028/runtime`;
current Qbert uses `/x/Temp/kindle-qbert-color-masks-20261101/runtime` with the
qualified faster,decision-equivalent teacher.
Each declaration pins exact native,runner,teacher and source bytes. Optional
posterior,temperature,centroid,open-loop and imitation-decay runtimes have
separate receipts; do not mix overlays. Preserve source-specific validation
failures and limits. No rebuild is needed for an exposed learning-rate change.

## Experiment practice

- Use small development runs and meaningful checks; avoid turning preparation
  into another backend qualification project. Automate monitoring and audits.
- Keep batched inference with eight independent Atari environments. Report
  aggregate actions and emulator frames separately,plus wall-clock time.
- Retain config,seeds,source/native identity,curves and frozen natural-episode
  evaluations against actual-initial controls. Record aids,budgets,cutoffs and
  failures. Confirm promising changes across seeds before reliability claims;
  selected/reused development seeds are not held-out tests.
- Checkpoints omit replay,live belief,games and RNG. A restored extension is a
  warm start,not uninterrupted learning or an independent seed. Restore preserves
  architecture/geometry and rejects fresh-training overrides.
- Preserve native max-pooled frames and one GPU resize; verify small-object
  visibility. Do not reintroduce CPU downscale followed by upscale.
- Put compact results in docs/results and update PR31. Logs,weights,videos,
  build caches and profiling dumps belong outside Git. Keep useful checkpoint
  milestones; release disposable build caches after packaging.

## GPU operation

GPU work is authorized unless the device is genuinely faulted. Known allocation
warnings alone are not an approval gate. Use the existing host guard and stop on
real faults or numerical failures. No automatic driver reset/reload/reboot.
Separate NVML polling is unnecessary for the current experiment. Check live
process/service identity before work; an observer timeout is not a stopped job.

The current AMD host is separate from the original NVIDIA PMU incident. Older
595/615 drivers repeatedly halted during first-world initialization,before
checkpoint loading. An ASLR-disabled debugger success was not a fix; localization
does not establish a buffer,firmware or hardware cause. On that host,use the
prepared pre-submission capture after recovery,not another blind restore. The
ABI-specific GDB collector applies only to its original binary. On a new healthy
GPU,qualify bounded initialization/restore and record device/driver identity
without inheriting old boot assertions.

## Historical material

The shared transfer `/mnt/data/kindle-handover-20261025` contains the4096-action
context smoke and2M Breakout originals,matching all346 tensors. Other historical
game weights were not transferred. New AMD checkpoints remain locally retained.
Original archives moved to `/x/Artifacts/shared-drive-cleanup-20261025` on the
original machine; exact counters and paths are in the handover.

Old incident/optimization ledgers and detailed data remain on
`archive/phase2-before-cleanup-20261025`; current reports stay here.
Earlier expanded instructions remain in Git at34b1e74; the earlier experiment
ledger is preserved at52c2174 (and earlier113ecde/8b1ba49) and in the roadmap/reports. Historical “next”
paragraphs are not active instructions. Preserve user-authored TASK.md and
docs/cdp-claude-feedback.md as context; subsequent CDP decisions and measured
results supersede their older architecture/performance assumptions. Do not
merge this or upstream PRs on the user's behalf.
