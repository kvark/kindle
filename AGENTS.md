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

## Current work — November 9

The goal **prove the world model training on Breakout** remains active and
unproven. Select the [fixed wide-motion learning comparison](docs/experiments/2026-11-09-breakout-world-wide-motion-learning.md):
one fresh16,384-update1M/RGB64/B8/T32/H16 physical-feature learner with
`rgb_wide_motion=true`. Local motion/fusion widths16→32;80 parameters/675,643
scalars/240 saved arrays,13 changed local shapes,+71,728 scalars. Encoder/RSSM/
event geometry stays. This is a capacity hypothesis,not a diagnosed cause.
Same original two-cycle plan/data/RGB100/event losses/rates4e-4/Auto/41001;
new initialization/native andfirst-update checks differ. Fixed final decision,
unchanged dense gates,6,600s training/600s restore caps,no automatic successor.
The main learner is active under one host guard;215 execution/41 CPU-plan pins,
CPU model-construction trap andexact first-batch/plan checks pass. The first actual
update matches13 reports/240 qualifiedage1 tensors exactly;77 required-family
parameters change,finite240 tensors andtotal/support pass before update2.
Source56611db4,
main invocationc993d779a66f46d6a58daee8078ecfb1/controller2231604,observer2231609,
CPU finisher2231611;root `/x/Temp/kindle-world-wide-motion-learning-20261109`.
Restore/CPU final audits are attached;no broader assessment is allocated.
The [provisional2,048 review](docs/results/2026-11-09-breakout-world-wide-motion-learning.md)
shows mixed small-curve development:observed17/26 vs23/26,one17/27 vs21/27,
four6/26 vs3/26,eight1/27 vs0/27. Native483.300s vs228.781s (2.113x).
34-input/60-join CPU review passes,all six groups/four draws/four fixed grids
retained;long drift/ball loss remains. These are not final dense gates orproof
of useful learning/hour. Continue the fixed endpoint. Original requirements remain.

The [native qualification](docs/results/2026-11-09-breakout-world-wide-motion-native.md)
passes in81.157s:small-fixture native values/derivatives/lane isolation,
32 cached default-off forecasts/240 tensors,two exact equivalent-command
updates/13 reports/240 tensors and32 exact restored forecasts.77/80 first-update
parameters change in required families. Explicitv27 wraps v18/v25;unsupported
scopes/whole-agent use reject.130 execution/71 closure pins andone guard/nine
sealed files pass;both services inactive/successful. Sourcee3d18841/native
SHA02feea6e at `/x/Temp/kindle-world-wide-motion-20261109`;reuse exact runtime/
runner without rebuild. Initial dev-profile wheel/log andseparate optimized
release selection remain retained. The3,580,907,520-byte compiler cache is
released;retained artifact hashes pass. This qualifies implementation,not
learning orphysics. Existing runtimes/scripts stay immutable.

The [frozen-state diagnostic](docs/results/2026-11-09-breakout-world-state-readouts.md)
stays closed:30 inputs,42 fits,144 metric rows,1.918s CPU,zero model queries.
Partial h/z probes omit24,576 image-history values;they provide no precise
transferable forecast rescue andcannot prove information absence. All original
image scores,plots,folds and27,648 per-origin rows remain retained. Wider denoiser
failure andall earlier architecture failures stay closed;this is a distinct
local-decoder branch,not a repeat orassumed fix.

The [posterior-history RGB100 learner](docs/results/2026-11-09-breakout-world-posterior-natural100.md)
stays closed/failed at2/3 dense gates: observed108/112,one99/109,eight19/107,
sixteen5/114,thirty-two0/95 versus prior-suffix103/112,102/109,53/107,22/114,8/95.
Fitting8 falls84/176→41/176;current reconstruction/paddle scores improve,but
long ball forecasts regress. Native1,820.531s/loop1,951.302s,182/174/38/182 pins,
two guards/18 sealed files andexact18-array restore pass. All three services are
inactive/successful;all five final figures reviewed. No extension orbroader
assessment. Bundled data/loss/budget/source changes do not isolate a cause.

The [local forward comparison](docs/results/2026-11-09-breakout-world-local-forward-contract.md)
stays strictly failed on small observed/sampling differences,with exact forced-
sample recurrent/generated suffix values;it does not prove free-running physics.
The [late learner](docs/results/2026-11-09-breakout-world-late-action-learning.md)
and[earlier posterior-history run](docs/results/2026-11-09-breakout-world-posterior-history.md)
remain closed/failed,including original source-specific failures/repairs.
The [early feature assessment](docs/results/2026-11-09-breakout-world-action-features-branches.md)
stays failed at.224881 versus one-hot.172366,296/304 nonending alarms. Full
physics/events,brick memory,late/above-top coverage,untouched families andfresh
confirmation remain required. Actor/critic,CDP andfive-game work stay parked.

## Direction and retained results

Training is the priority. Backend optimization,upstream porting and JAX parity
are parked. The retained agent uses **centered CDP** with a jointly learned CNN,
categorical RSSM and imagined actor/critic. The isolated-world track
uses explicit pixel supervision; the denoiser still fails physics gates, with
bounded reward support retained as a development improvement. No pretrained DINO or Tiny
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

- Treat every declaration-pinned script as immutable, including CPU analysis.
  Write changes under a new filename and declare them separately.
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
