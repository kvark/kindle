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

The [fresh Breakout H30/R12 comparison](docs/experiments/2026-11-02-cdp-breakout-h30-r12-200k.md) isselected afterthe bounded
check closes successfully:8,192 actions/367 updates in51.413s,1,024 frozen/.892s
withzero updates/natural episodes;all382 pins,tensors,traces,replay andguards
pass. Terminal reader07:26:54.448/completionSHA697e58e0. Do not repeat thecheck.
Conservative200k projection23m00.725s passes30min; no experience isreused.

Full CPU qualification passes07:30:14.885 in2.378s:18 scripts,800 declaration
paths,five commands,36 config/nine actual preflight/20 startup/13 recovery-body
cases andretention fixtures. All382 bounded/338 control pins stayunchanged.
Train200k/9,358 updates with100k/4,670 recovery andfour final/actual-initial
24-episode frozen arms against70.750 sampled/143.667 greedy;sampled staysprimary.
H15/R16→H30/R12 changes rollout length,cadence,sampling,warmup andrelative work.
Keep12M64/Auto/N8/B16/T16/encoder4e-5/behavior8e-5/dynamics4e-4/replay-value1/
expert100,sampled collection,exact runnercf4ea238/native447fe50/teacher.
Prior short-Breakout/long-Qbert negatives remain;no isolated causal claim.
Root `/x/Temp/kindle-breakout-h30-r12-20261102/full`;service
`kindle-breakout-h30-r12-200k-20261102`. Checklive state;no overlap,restart,
retry,cap increase,extension,confirmation orautomatic successor.

The [completed lower-dynamics/replay-value1 comparison](docs/results/2026-11-02-cdp-breakout-value1-dynamics1e4.md)
scores **42.917 sampled /112.125 greedy**, below70.750/143.667 by39.34%/21.95%,
in21m36.378s (+0.66%). Initial-subtracted return/min falls40.54%/22.81%; all0/24
two-wall. Retain dynamics4e-4 withbehavior8e-5/replay-value1 andsampled collection.
Onlydynamics4e-4→1e-4 changes at12M64/Auto/N8/B16/T16/H15/R16/expert100,
native447fe50/runner ea2bf8c6/teacher fixed. Reused1009/cohort isdevelopment
evidence,not independent reliability or an isolated architecture cause.

All422 pins,100k/6,227 recovery,357 matching initial/invariant frozen tensors,
47 pre-update events,800/795 initial semantic events,replays/videos,guards,
curves,exposure andcomparison pass. Allfive original services succeed; main/
post/collection finish07:10:06.890/07:10:34.218/07:11:08.882 UTC November2,
launch sourcec691416. Terminal reader completes once07:11:52.749 in2.877s;
completionSHA5f2cd92c/summarySHA8ca4e947. Original bounded preparation failure
andseparate correction remain; no full stage fails orrepeats.

The saved-log coverage follow-up completes07:12:03.388 in2.969s with48 diagnostic/
422 full pins: native actions at216–431 fall1,262→927 (.631%→.4635% oftraining),
none at432+. All283,208 new actions/1,132,245 frames reconcile; the completed
control isreused without recount. Score fractions are not verified walls or
an isolated causal test. Coverage reviewSHA65b130da. Actual overrides48.841%,
late expert CE.662474 andguided71.18 are separate from frozen competence.

Root `/x/Temp/kindle-breakout-value1-dynamics1e4-20261102/full` isclosed. Recovery,
terminal andcoverage readers arecomplete; do not repeat stages,collectors or
readers. No Kindle allocation isactive atclosure; no extension,confirmation,
rate sweep orautomatic successor isselected. CPU co-host costs remain uncorrected.
Check live state; Breakout andthe five-game goal remain unfinished.

The [completed Qbert R20 fresh-seed confirmation](docs/results/2026-11-02-cdp-qbert-r20-seed18013.md)
scores15,931.250 greedy/24 first pyramids on18013 in26m23.114s,versus development
16001's15,279.167/24 in25m57.740s. Both pass the original15k/90% gate within30min:
one passing development learner andone fresh confirmation,not broad reliability.
Sampled14,976.042/24 and12,972.917/24 both fail15k; own fresh initials142.708
greedy/198.958 sampled clear none. Retain240k/R20 withgreedy evaluation.
Earlier R24 confirmations remain failed; Breakout andthe five-game goal remain
unfinished. No Qbert extension,third learner orautomatic successor isselected.

Fresh240k actions/959,784 frames/18,721 updates keep12M64/Auto/N8/B16/T16/H5/R20/
dynamics1e-4/replay-value1/expert100/greedy collection,native447fe50,runnere304af89
andteacher fixed; onlylearner/evaluation seeds change. All1,046 pins,both100k/7,784
and200k/15,596 recoveries,357 finite/invariant frozen tensors,48 recorded initial
differences,full replays/videos,guards,curves,exposure,coverage andcomparison pass.
Allfive original services succeed; main/post/collection finish06:25:18.233 /
06:25:49.125 /06:26:21.455 UTC November2/source0a4892a. Original main901816 /
976a1ed5007f4ce8b343732f91c38cf9 isterminal. Reader completes once06:27:04.671 in
9.806s; completionSHA3fb594a7/summarySHAc41ca062. No full-run stage fails orrepeats;
original CPU preparation failure andseparate correction remain preserved.

Root `/x/Temp/kindle-qbert-r20-seed18013-20261102/full` isclosed. Both recovery
readers andterminal reader arecomplete; do not repeat stages,collectors orreaders.
Actual overrides25.3096% andguided16,043.5 are not frozen competence. CPU co-host
cost staysuncorrected. No Kindle allocation isactive atclosure; checklive state.

The [Breakout ascending-centering reference](docs/results/2026-11-02-cdp-breakout-ascending-center.md) fails its expanded24-episode
screen:233.208 versus225.083 (+3.61%),four ofeight lane means improve,all0/24
two-wall. The earlier eight-episode gain46.93%/six lanes remains recorded;
all16,425 first-episode semantic events reproduce. Retain the original teacher;
no fresh cohort,integration,learner or sweep is selected. All31 diagnostic/1,046
active Qbert pins,full pixel/action/debug replay and original learner identity
pass. CPU collection35.990s/replay45.402s/controller87.435s completes06:06:24.338
UTC November2/sourcef8e4931,with no GPU work or updates. Root
`/x/Temp/kindle-breakout-ascending-center-20261102/standard24` is complete;
do not repeat stages. CPU co-host cost remains uncorrected.

The [completed Breakout greedy-collection comparison](docs/results/2026-11-02-cdp-breakout-greedy-training.md)
finishes200k actions /799,661 frames /12,477 updates in21m44.645s. Primary
sampled78.542 improves70.750 by11.01%, with9.81% more initial-subtracted return
per training minute. Greedy90.042 falls37.33% from143.667; all0/24 two-wall.
Retain this sampled development point and the stronger greedy control. Only live
collection changes at12M64/H15/R16/replay-value1; native447fe50,runner6e03d739,
rates,teacher and other settings stay qualified. Reused1009/cohort provides no
independent reliability claim; guided131.780 is not frozen competence.

All427 pins,100k/6,227 recovery,357 matching initial/invariant frozen tensors,
paired traces,replays/videos,guards,curves,exposure and comparison pass. Main/
post/collection succeed05:36:00.508/05:36:21.599/05:36:57.105 UTC November2,
source9247ffa. Allfive original services succeed; no GPU stage repeats. Terminal
reader completes once05:39:32.077 in1.844s; completionSHA8ceba70a, summarySHAf9ba2112.
The original bounded CPU wrapper failure and separate qualification remain.

The [coverage follow-up](docs/results/2026-11-02-cdp-breakout-score-coverage.md#greedy-collection-follow-up)
finds native actions at216–431 rising1,262→2,382 (0.631%→1.191% of training),
with zero actions at432+. Three completed unguided episodes reach216 versus
none incontrol. Training maxima408→424 still miss later score ranges. These
are not verified wall boundaries or an isolated cause. All43 diagnostic/427
full pins and292,528 new actions reconcile; the saved control is reused.
CPU reader completes once05:39:43.053 in2.967s, reviewSHA19787de1; no GPU work.

Root `/x/Temp/kindle-breakout-greedy-training-20261102/full` is closed. No learner
or GPU allocation remains active at closure. Do not repeat completed stages,
collectors or readers. No extension, confirmation, cap increase or automatic
successor is selected. Check live state; the five-game goal remains unfinished.

The earlier [greedy-collection prerequisite](docs/results/2026-11-02-cdp-breakout-greedy-training.md#earlier-bounded-prerequisite)
passes8,192 actions/489 updates in47.418s;1,024 frozen actions/.869s havezero
updates/no natural episodes. All387 pins,tensors,replay,guards andoriginal
services pass. It precedes the separately selected full comparison above; its
experience is not reused. The45 CPU tests and separate wrapper qualification
remain retained. Native-check completionSHAfc5ef001/summarySHA86262978;
do not repeat the bounded check or its readers.

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

Earlier allocations are closed; the fresh Breakout comparison above is active.
No automatic successor is selected.
Check live state before work. The five-game goal remains unfinished. Qbert's
fixed240k/R20 recipe has one passing development learner andone fresh greedy
confirmation. Breakout remains below the two-wall gate.
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
| Breakout |Greedy collection:sampled78.542/greedy90.042 in21m45s;sampled collection:70.750/143.667 in21m28s|Retain the sampled gain andstronger greedy control separately;all0/24 two-wall,reused1009/cohort|
| Qbert |Fixed240k/R20:greedy15,279.167/15,931.250 on16001/18013,24/24 first pyramids each,in25m58s/26m23s|One passing development learner andone fresh confirmation;sampled12,972.917/14,976.042 both fail15k. Earlier R24 failure remains; broad reliability open|

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
