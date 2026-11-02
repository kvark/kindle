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

## Current work — November 2

The [Breakout greedy-collection check](docs/results/2026-11-02-cdp-breakout-greedy-training.md) completes
8,192 actions /32,738 frames /489 updates in47.418s. Frozen1,024 actions take
.869s with zero updates and no natural episodes. Conservative200k projection
21m13.430s passes30min; gameplay benefit remains unmeasured. All387 pins,357
control-matching initial/invariant frozen tensors,47 recorded pre-update
semantic differences,finite reports,full guided replay and host guards pass.
Only live collection changes; native447fe50,12M64/H15/R16/replay-value1,rates,
actor unimix .01 and teacher stay fixed. Original CPU wrapper-sentinel failure
and separate qualification remain;45 tests pass with no repeat. Main/collector
succeed05:03:51.035/05:03:56.103 UTC November2, sourceb2105b9; terminal reader
05:04:16.073. Completion SHAfc5ef001, summary SHA86262978. Root
`/x/Temp/kindle-breakout-greedy-training-20261102/native-check` is closed.
No active allocation, full learner or automatic successor is selected. Do not
repeat completed stages/readers. Check live state; the five-game goal remains
unfinished.

The [Qbert 240k/R20 comparison](docs/results/2026-11-02-cdp-qbert-r20-240k.md) completes 240,000 actions,
959,778 frames and 18,721 updates in 25m57.740s. Greedy 15,279.167 with 24/24
first pyramids passes the 15k/90% gate, versus retained 14,913.542/24. Sampled
12,972.917/24 falls from 14,745.833/24 and fails 15k. Greedy score rises 2.45%
and training time rises 4.85%, so initial-subtracted return/min falls 2.27%
(sampled: −16.21%). Retain this passing development budget point with its measured efficiency cost;
independent confirmation remains open. Seed 16001 and its original
cohort are reused. R24→20 and 200k→240k change experience, cadence and sampling
together. Native 447fe50, runner e304af89, teacher and other settings stay fixed.
The earlier R24 confirmation remains failed.

All 991 pins, both recoveries, 357 initial/invariant frozen tensors, paired
traces, replays/videos, guards, curves, coverage and original services pass.
Main/post/collection succeed at 04:48:55.980/04:49:14.933/04:49:46.075 UTC
November 2, source 55ead06. The terminal reader completes at 04:50:26.168;
completion SHA7423739f, summary SHA7f721710. Original CPU preparation failures
and corrections remain preserved; no GPU stage repeats. Root
`/x/Temp/kindle-qbert-r20-240k-20261102/full` is closed. No allocation is active,
and no extension, confirmation, sweep or successor is selected. Do not repeat
completed stages/readers. Check live state; the five-game goal remains unfinished.

The [saved Breakout score-coverage diagnostic](docs/results/2026-11-02-cdp-breakout-score-coverage.md)
finds97.03% ofcandidate training actions below216 points andzero at432+,with
maximum408 versus control418. Of5947 candidate actions at216–431,4685 areteacher
overrides; native exposure there is1262 actions (0.63% oftraining). Frozen greedy
means117.292→143.667 still havezero432+ episodes. Score fractions are not verified
wall boundaries; this locates scarce experience,not an isolated cause orselected
change. All400k training/175144 frozen actions,seven fixtures,31 original/six
repair/991 active pins reconcile. Original cumulative-counter failure remains;
separate9.051s CPU analysis completes04:37:06.430/sourceb608d35 withno GPU work.
Root`/x/Temp/kindle-breakout-score-coverage-20261102` iscomplete; do not rerun.

The earlier [Qbert R20 bounded check](docs/results/2026-11-02-cdp-qbert-r20-240k.md)
completes8192 actions/32760 frames/611 updates in50.566s;1024 frozen/.868s has
zero updates/no natural episodes. Conservative240k projection27m09.558s passes
30min. All940 pins,357 matched initial/invariant frozen tensors,47 pre-update
semantic events,finite reports,replay andguards pass. Main/collector succeed
04:05:09.369/04:05:13.602 UTC/source8f7888a; terminal reader04:05:26.383.
No stage fails orrepeats. OnlyR24→20 changes onreused16001, with native447fe50,
runnere304af89,teacher andcurrent12M64/H5/dynamics1e-4/replay-value1 fixed.
This prerequisite preceded the separately selected full comparison above.
Its early cost projection did not establish gameplay benefit. Earlier
larger-budget regressions remain relevant. Root`/x/Temp/kindle-qbert-r20-240k-20261102/native-check` isclosed;
completionSHAdf02e12d/summarySHA15fb8ccd. No active allocation atclosure; check
live state. Do not repeat the check,collector orreader.

The [Qbert replay-value1 crossed assessment](docs/results/2026-11-02-cdp-qbert-replay-value1-cross-cohort.md)
is complete. Learner14009 stays ahead on both cohorts:16525.000/15906.250 versus
16001's15163.542/14913.542,all24/24 first pyramids. Cohort choice changes16001's
near-threshold gate; its original fresh confirmation remains failed. Do not
substitute the favorable cohort or count reused checkpoints as new learners.
All957 pins,357 finite/invariant tensors,replays,guards,matrix andoriginal services
pass; four new arms take115.406s withzero updates. Main/collector succeed
03:46:51.221/03:47:07.244 UTC/sourceb883162; terminal reader passes03:47:46.833.
Root`/x/Temp/kindle-qbert-replay-value1-cross-cohort-20261102` isclosed;
completionSHAcd59f0ff/summarySHAf92533f6. Do not repeat its stages orreader.

The [Breakout replay-value1 comparison](docs/results/2026-11-02-cdp-breakout-replay-value1.md)
completes200k/12477 updates in21m27.910s. Sampled70.750/greedy143.667 improve
59.000/117.292 by19.92%/22.49%; initial-subtracted return/min improves21.96%/
24.50%. Retain scale1 as stronger development; allzero of24 two-wall successes.
Only replay-value0.3→1 changes onreused1009, affecting critic andrepresentation
jointly; no isolated mechanism orreliability claim. All338 pins,recovery,initial/
frozen tensors,replays,guards andfive original services pass. Main/post/collection
succeed03:39:28.406/03:39:34.988/03:39:37.809 UTC/source45cd1ab. Terminal reader
passes03:40:38.280; completionSHA6254f729/summarySHAb55b2016. Original CPU
preparation failures/repairs remain; no GPU stage repeats. Root
`/x/Temp/kindle-breakout-replay-value1-20261102/full` isclosed; do not restart.

All current allocations are closed. No full learner or automatic successor is
selected.
Check live state before work. The five-game goal remains unfinished. Qbert has
passing development points on two seeds across different recipes, without
independent R20 confirmation. Breakout remains below the two-wall gate.
The [Qbert phase diagnosis](docs/results/2026-11-02-cdp-qbert-replay-value1-phases.md)
places its score gap after identical550-point first-pyramid prefixes and finds
substantial later-play exposure. This does not identify an architectural cause.

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
| Breakout |Replay-value1/behavior8e-5/dynamics4e-4:sampled70.750/greedy143.667 in21m28s; scale0.3 control59.000/117.292 in21m45s|21.96%/24.50% higher learning/min; all0/24 two-wall; reused development seed|
| Qbert |R24/14009 greedy16,525;R20/240k/16001 greedy15,279.167 in25m58s,each24/24 first pyramids|R24/16001 remains failed;R20 is reused-seed development,not independent confirmation. Its greedy learning/min falls2.27%;sampled12,972.917 fails15k|

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
