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

## Active work — November 2

The [fresh Breakout replay-value1 comparison](docs/experiments/2026-11-02-cdp-breakout-replay-value1-200k.md) launches
03:14:00.615 UTC November2/source45cd1ab, mainPID887827/invocation
`4b67c053d030409f8734e6bfbdab6707`. Original100k/6,227 recovery succeeds
03:25:06.808 with357 finite tensors; all338 pins andretained hashes pass a
separate3.650s confirmation03:25:45.442. Same main remains active; other original
observers continue. No observer,checkpoint copy orGPU stage repeats.
Startup passes338 pins,357 control/bounded matching initial tensors,47 equal
pre-update transitions and111 finite updates through2136 actions. Train200k/
12,477 updates within30min; require final/actual-initial
sampled/greedy24-episode assessments against59.000/117.292. Sampled remains
primary; original90% two-wall gate stays fixed. No full frozen result exists.

Only replay-value0.3→1 changes at12M64/Auto/N8/B16/T16/H15/R16/base-encoder4e-5/
behavior8e-5/dynamics4e-4/expert100,sampled collection. Native447fe50/35ad0db and
teacher stay fixed; runnercf4ea238 adds ten fresh-CLI lines to control6c1b69fb.
Critic and representation gradients change together; gameplay benefit remains
unmeasured. The47.690s bounded check and1.917s full qualification pass. Original
in-memory generator and declaration-filename failures remain preserved; separate
CPU repairs pass before the first GPU launch. No GPU stage has repeated.

Root`/x/Temp/kindle-breakout-replay-value1-20261102/full`; main
`kindle-breakout-replay-value1-200k-20261102`; declarationSHAe3b25b65. Check live
state; do not overlap, restart, extend or reuse check experience. Original progress,
recovery, postprocessing and collection observers are responsible for this run.
Terminal readerSHA04ce2848 passes its separate.106s CPU preparation; wait for
original completion before terminal mode. No automatic successor is selected.
The five-game goal remains unfinished.

The [Qbert replay-value1 phase diagnosis](docs/results/2026-11-02-cdp-qbert-replay-value1-phases.md) locates the entire
frozen score gap after identical550-point first-pyramid prefixes inall96 episodes.
Both learners collect about76% later actions; late native movement agreement is
77.50%/77.07%. Greedy median gap150 is smaller than mean1611.458; fresh scores
range11175–15650 versus14800–22025. These are descriptive cohort differences,
not an isolated cause or changed gate. All27 diagnostic/338 active pins and
fixtures pass in4.038s CPU at03:19:44.467/source305d145; original Breakout identity
stays unchanged. No GPU work,replay repeat,loss change or new learner. Do not rerun.

The [completed Qbert replay-value1 confirmation](docs/results/2026-11-02-cdp-qbert-replay-value1-seed16001.md) scores
**14,913.542 greedy /14,745.833 sampled**, each24/24 first pyramids. Both means
miss15,000. Fresh16001 completes200k actions/799,817 frames/18,715 updates in
**24m45.754s**; own initials86.458/148.958 clear none. Development14009 remains
16,525/14,882.292 in25m36.935s. The fixed recipe passes primary greedy onone of
two learners; it has no successful fresh confirmation or broad reliability.

All912 pins,100k/9,340 recovery,357 finite/invariant tensors,48 initial tensor
and47 pre-update differences,full replays/videos,guards,curves,coverage,exposure
andcomparison pass. Main/post/collection succeed02:49:22.758/02:49:37.668/
02:50:08.343 UTC November2/sourcebff6eb4. Allfive original services succeed;
no training,frozen,replay,audit orcollector stage fails orrepeats. Separate
5.166s terminal reader passes02:50:48.666;completionSHA587f591e/summarySHA8d825bdf.
CPU preparation failures andseparate repairs remain preserved.

Only learner/evaluation seeds change. Keep12M64/Auto/N8/B16/T16/H5/R24/dynamics
1e-4/base-encoder-behavior4e-5/replay-value1/greedy/expert100,native447fe50/35ad0db,
runnere304af89 andteacherfa261c53. Actual overrides24.0885%;online15,187 and
late replay CE.068976 are not frozen competence. CPU co-host costs remain
uncorrected. Root`/x/Temp/kindle-qbert-replay-value1-seed16001-20261102` isclosed.
No extension,retry,third learner,tuning,seed substitution orautomatic successor
is selected. Do not repeat completed stages orreaders. No active allocation
remains atclosure;check live state. The five-game goal remains unfinished.

Breakout's latest behavior8e-5 comparison scores59.000 sampled/117.292 greedy
in21m45s,allzero two-wall successes. Its original stages are also closed.
The saved-data value diagnostic below finds smaller short-horizon residuals
than complete-return errors;it selects no new loss orlearner.

## Direction and retained results

Training is the priority. Backend optimization,upstream porting and JAX parity
are parked. Use **centered CDP** with a jointly learned CNN,categorical RSSM and
imagined actor/critic. No pretrained DINO or Tiny JEPA encoder is required.
Explicitly select CDP; learned RGB remains the CLI default.

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
| Boxing |1M32:76.833/79.167,two learners with24/24 wins,in28m22s/28m13s|Both pass; broad reliability open|
| Freeway |Aided1M32:31.042/32.000,two learners with24/24 successes,in28m17s/28m50s|Aid off during evaluation; broad reliability open|
| Pong |12M48/B16/R16:greedy15.208 on both learners,24/24 wins,in41m28s/41m31s|Greedy passes narrowly; sampled passes one of two|
| Breakout |Behavior8e-5/dynamics4e-4:sampled59.000/greedy117.292 in21m45s; behavior4e-5/dynamics1e-4:50.625/138.750 in21m48s|Small sampled gain; all0/24 two-wall; reused development seed|
| Qbert |RGB64/H5/R24/dynamics1e-4/replay-value1:greedy16,525.000,sampled14,882.292,both24/24 first pyramids,in25m36.935s|Fresh16001 scores14,913.542/14,745.833 in24m45.754s,both miss15k;scale1 passes greedy one of two. Scale0.3 also passed one of two earlier learners|

Published long-run scores are aspirations,not matched-budget comparisons.
Finite losses,short runs and backend speed do not establish mastery or CDP
superiority. Guided online returns and replay imitation losses are not frozen
competence. Keep detailed results in the roadmap and linked reports,not a
growing history in this file.

Latest qualified Breakout evidence:
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
Earlier expanded instructions remain in Git at34b1e74; the recent experiment
ledger is preserved at113ecde (and earlier8b1ba49) and in the handover/roadmap. Historical “next”
paragraphs are not active instructions. Preserve user-authored TASK.md and
docs/cdp-claude-feedback.md as context; subsequent CDP decisions and measured
results supersede their older architecture/performance assumptions. Do not
merge this or upstream PRs on the user's behalf.
