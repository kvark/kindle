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

## Current work — November 3

The [fresh life-loss comparison](docs/experiments/2026-11-03-cdp-breakout-life-penalty-200k.md)
is running since06:16:18.414 UTC November3/sourceb80194e,453 pins. Main983163/
0191b9d7c0fe47acbe15c6db9845f519; progress,recovery,postprocessing andcollection
observers areactive. Startup passes357 initial tensors,47 pairedpre-update
events,correct cadence and113 finite reports through2,184 actions,including46
verified life-loss penalties. No100k recovery orfrozen result yet. Require the
fixed200k/12,477updates/30min andfour24-episode assessments. Root
`/x/Temp/kindle-breakout-life-penalty-20261103/full`; do not restart oroverlap GPU
work. Only the declared training reward aid changes; raw scores stay unchanged.
Read-only terminal preparation passes06:19:26.520:five score/six feedback cases,
all453 pins/five identities. Fixed coverage preparation passes06:20:09.659:
50 diagnostic/453 active pins,eight config/five retained boundary cases. Run
recovery,terminal andcoverage readers once after their original prerequisites;
do not repeat preparation oractive stages. Keep CPU co-host caveats.

The [life-loss 200k comparison](docs/experiments/2026-11-03-cdp-breakout-life-penalty-200k.md)
passes CPU preparation at06:14:37.261 UTC November3/source2eafbf2:five actual
commands,34 config/24 startup/seven prefix/12 recovery cases,observer/prelaunch/
retention checks,19 scripts and834 declaration entries. All408 bounded/338
control pins stay unchanged; no failure orGPU work. Next select andlaunch the
declared fresh200k learner; no allocation active atthis preparation milestone.

The [life-loss feedback bounded check](docs/results/2026-11-03-cdp-breakout-life-penalty.md)
passes8,192 actions/32,730 frames/489 updates in47.317s; frozen1,024/.891s has
zero updates/episodes and357 invariant tensors. All408 pins,357 matchedinitial,
47 pairedpre-update events,118 independently replayed penalties,raw scores,
full pixel/action/label replay andguards pass. Main/collector succeed06:06:19.889/
06:06:24.949 UTC November3/source2594ab5; terminal confirmation06:07:15.464.
Three CPU fixture/path/terminal-reader corrections are preserved; no original
stage repeats. Conservative200k projection21m10.713s passes30min. The bounded
root isclosed; do not repeat stages/readers. Separately select the [fresh200k comparison](docs/experiments/2026-11-03-cdp-breakout-life-penalty-200k.md);
full preparation andlaunch areunfinished. No allocation isactive atclosure;
check live state. Native447fe50/teacher unchanged; gameplay benefit unmeasured.

The [optional life-loss aid passes CPU qualification](docs/results/2026-11-03-cdp-breakout-life-penalty.md):
57 tests / 365 pins, including 338 control pins, plus both bounded commands,
nine config assertions, five prefix cases and the 357-tensor schema. Native
447fe50 stays byte-identical; runtime `/x/Temp/kindle-breakout-life-penalty-20261103/runtime`.
Two repaired CPU fixture/path failures remain preserved; production code is
unchanged across repairs. This CPU milestone preceded the completed bounded check
above; do not repeat qualification.

The [saved life-feedback analysis](docs/results/2026-11-03-cdp-breakout-life-feedback.md)
is complete: all 120 selected lost lives have zero immediate reward, including
96 nonterminal losses. It reuses 37,016 recorded actions with no emulator or GPU
work; 11 fixtures and 14 input pins pass. This does not establish a causal defect.
Next develop the [optional one-point life-loss training aid](docs/experiments/2026-11-03-cdp-breakout-life-penalty.md):
explicit raw/training reward accounting, default off, frozen shaping off, and
CPU qualification before one bounded check. The CPU implementation is now qualified;
the bounded GPU check now passes. Preserve retained runtimes and raw scores.
The diagnostic is closed; do not repeat it. No Kindle job is active at closure.

The [completed replay16k comparison](docs/results/2026-11-03-cdp-breakout-replay16k.md) lowers sampled70.750→34.000
andgreedy143.667→57.625 in21m44.359s (+1.28% training time),all0/24 two-wall.
Initial-subtracted return/min falls53.60%/61.32%. Retain shared replay100k with
12M64/Auto/N8/B16/T16/H15/R16/lambda.95/encoder4e-5/behavior8e-5/dynamics4e-4/
replay-value1/expert100 andsampled collection. Onlycapacity100k→16,384 changes;
native447fe50,runnercf4ea238 andteacher remainexact. Recency,diversity,reuse and
coverage change together; cache age andisolated cause remainunmeasured.

All445 pins,100k/6,227 recovery,357 matchedinitial/invariant frozen tensors,
16k/2,078-event semantic prefix,paired initial traces,full replays/videos,guards,
curves,exposure andcomparison pass. Main/post/collection succeed05:34:16.487/
05:34:44.611/05:35:19.248 UTC November3/source9cfd8cc; allfive originals succeed.
Terminal05:35:33.562 andcoverage05:35:39.907 pass once;50 diagnostic/445 full
pins,270,920 actions/1,083,085 frames reconciled. Native216–431 actions fall
1,262→878,none432+; score bins are not verified walls orcausal coverage. Late
replay expert CE falls.560514→.381669 onchanged replay despite worsegameplay.
Actual45.4125% overrides/online54.42 are not unaided competence.

Root`/x/Temp/kindle-breakout-replay16k-20261103/full` isclosed. Allreaders and
original stages arecomplete; do not repeat them. No full-run failure orGPU repeat.
Earlier bounded main/collector failures andseparate CPU repair remainpreserved;
original bounded services stayfailed. CompletionSHA2e48b8b0/summarySHAba7fd117/
coverageSHA25a8eb80. No allocation active atclosure; check live state. No capacity
sweep,extension,confirmation orautomatic successor. Keep CPU co-host caveats.
The five-game goal remains unfinished.

The [completed lambda1 comparison](docs/results/2026-11-03-cdp-breakout-lambda1.md) scores **45.667 sampled /
90.292 greedy**, below retained lambda0.95's70.750/143.667 by35.45%/37.15%,
in **21m52.277s** (+1.89%). Both final arms have0/24 two-wall successes.
Retain lambda0.95 with12M64/Auto/N8/B16/T16/H15/R16/encoder4e-5/behavior8e-5/
dynamics4e-4/replay-value1/expert100 and sampled collection. Lambda changes
both imagined/replay targets; no isolated mechanism or seed-reliability claim.

All448 pins,100k/6,227 recovery,357 matching initial/invariant frozen tensors,
paired initial traces,replays/videos,guards,curves,exposure andcomparison pass.
Main/postprocess/collection succeed04:39:49.589/04:40:12.443/04:40:16.846 UTC
November3/source958d622; allfive original services succeed. Terminal confirmation
04:40:46.018 andcoverage04:40:51.786 pass once;50 diagnostic/448 full pins,
282,720 actions/1,130,258 frames reconciled. Native216–431 actions rise1,262→1,434,
none432+; score bins are not verified walls or causal coverage. Expert CE falls
.560514→.536404 despite worseplay; actual overrides45.3985% are not competence.

Root`/x/Temp/kindle-breakout-lambda1-20261103/full` isclosed. Recovery,terminal
andcoverage readers arecomplete; do not repeat them orcompleted stages. Earlier
CPU import/date-wiring failures andrepair remain; no full-run failure orrepeat.
Native447fe50/teacher unchanged; qualifiedrunner9eb21cca adds10 CLI lines.
Keep CPU co-host caveats. No extension,confirmation,sweep orautomatic successor
selected. No Kindle allocation isactive atclosure; check live state beforework.
The five-game goal remains unfinished.

The [lambda1 bounded check](docs/results/2026-11-03-cdp-breakout-lambda1.md)
passes8,192 actions/32,727 frames/489 updates in50.121s; frozen1,024/.868s has
zero updates/no natural episodes. Conservative200k projection22m26.028s clears
30min. All385 pins,357 control-matching initial/invariant frozen tensors,47
pre-update events,replay andguards pass. Main/collector succeed04:07:39.518/
04:07:43.412 UTC November3/sourcef466942; terminal summary04:08:34.276.
Native447fe50/teacher stayfixed; runner9eb21cca adds10 CLI lines.515 current/
22 private Python checks,eight refusals andfour exact-source scalar Rust cases
pass; initial import failure remains. No bounded stage fails orrepeats.
Root`/x/Temp/kindle-breakout-lambda1-20261103/native-check` isclosed. No full
learner orsuccessor isselected; gameplay remains unmeasured. Lambda changes
both imagined/replay targets; no causal claim. No allocation active atclosure;
check live state anddo not repeat completed stages/readers.

The [fixed pixel-motion readout](docs/results/2026-11-03-cdp-breakout-pixel-motion-readout.md)
fails its screen: learned-plus-motion teacher agreement55.91% overall/55.88%
near descending trails actor62.56%/66.91%. One new CPU predecessor capture
matches20,960 actions/83,799 frames and2,064 saved current pixel predictions;
existing features and baseline fits are reused. Six fixtures/75 pins pass;
capture12.177s+fit.950s, completion03:56:34.670 UTC November3/sourcec91dcfa.
No failure,GPU work or updates. Root`/x/Temp/kindle-breakout-pixel-motion-20261103`
is closed; do not repeat capture,fits or qualification. No temporal-input
prototype or learner is selected. Reused older control states/readout geometry
limit interpretation; no general temporal-input or causal defect claim.
No Kindle service is active at closure; check live state before new work.

The [completed Breakout crop comparison](docs/results/2026-11-02-cdp-breakout-playfield-crop.md)
scores47.042 sampled/114.833 greedy, below full-frame70.750/143.667 by33.51%/
20.07%, in21m49.159s (+1.65%). Own initials1.375/1.917; all0/24 two-wall.
Initial-subtracted return/min falls35.20%/21.47%. Retain full-frame12M64/Auto/N8/
B16/T16/H15/R16/encoder4e-5/behavior8e-5/dynamics4e-4/replay-value1/expert100
and sampled collection. Crop0/32/160/168 removes HUD information; nativeacb175f/
runner046aa5bc differ from447fe50/cf4ea238. No isolated cause or reliability claim.

All413 pins,100k/6,227 retained recovery,357 matched initial/invariant frozen
tensors,47 bounded-crop transitions,full replays/videos,guards,curves,exposure
and comparison pass. Main/postprocess/collection succeed12:52:52.604/12:53:23.251/
12:53:27.613 UTC November2/source53a75d7; allfive original services succeed.
Terminal reader completes12:54:38.573 in2.282s; coverage12:55:05.372 in3.053s,
with54 diagnostic/413 full pins and283,824 actions/1,134,745 frames reconciled.
CompletionSHA701b9246/summarySHA26d91aa1/coverageSHA92d30ffb. No full-run failure
or stage repeat. Earlier bounded compile/packaging failures remain preserved.

Native216–431 actions fall1,262→1,057, none432+; score bins are not verified walls
or causal coverage. Late expert CE improves.560514→.543409 despite worse play.
Actual guidance41.9085% and online80.18 are not competence. Keep crop default-off;
its153 Rust/642 Python CPU checks,geometry/default compatibility and47.995s
bounded training remain qualified. CPU co-host times stay uncorrected.
Root `/x/Temp/kindle-playfield-crop-20261102/full` is closed. Recovery,terminal
and coverage readers are complete; do not repeat completed stages or collectors.
No Kindle allocation is active at closure. No extension,confirmation,crop sweep
or automatic successor is selected. Check live state before new GPU work;
Breakout and the five-game goal remain unfinished.

The [completed Breakout 25M comparison](docs/results/2026-11-02-cdp-breakout-capacity25m.md)
scores **49.958 sampled /104.292 greedy**, below12M70.750/143.667 by29.39%/
27.41%, in **36m08.795s** (+68.40% training time). Own initials1.458/1.583;
all0/24 two-wall. Initial-subtracted return/min falls58.46%/56.88%. Retain12M64/
Auto/N8/B16/T16/H15/R16/encoder4e-5/behavior8e-5/dynamics4e-4/replay-value1/
expert100 and sampled collection. Preset12M→25M expands several components
with different initial values on reused1009/cohort; no isolated capacity cause.

All420 pins,100k/6,227 retained recovery,357 matching bounded initial/invariant
frozen tensors,47 pre-update transitions,full replays/videos,guards,curves,
exposure and comparison pass. Allfive original services succeed: main/post/
collection **11:28:52.929/11:29:26.592/11:29:35.005 UTC November2**, launchsource
0e69b83. Terminal confirmation completes11:30:31.187 in3.019s; completionSHA
1988e621/summarySHAb80dc809. No original learner,observer or GPU stage repeats.

The separately corrected CPU coverage reader completes11:30:39.622 in3.214s,
with55 diagnostic/420 full pins and290,216 actions/1,160,239 frames reconciled.
Native216–431 exposure falls1,262→307,with none at432+. Score fractions are
not verified walls or causal coverage evidence. Final50k expert CE improves
.561→.499 despite lower frozen scores; actual overrides41.812%,guided online
79.7 is not competence. Original coverage fixture failure remains preserved;
no preflight,trajectory or replay repeats. CoverageSHA7f2efd01.

Root `/x/Temp/kindle-breakout-capacity25m-20261102/full` is closed. Recovery,
terminal and repaired coverage readers are complete; do not repeat them.
Native447fe50,runnercf4ea238 and teacher remain unchanged. No Kindle allocation
is active at closure; check live state before new work. No extension,confirmation,
capacity sweep or automatic successor is selected. CPU co-host times remain
uncorrected. Breakout and the five-game goal remain unfinished.

The [bounded25M check](docs/results/2026-11-02-cdp-breakout-capacity25m.md)
is complete:8,192 actions/489 updates in84.864s,357 actual/invariant frozen tensors,
380 pins and all guards/replays pass. Conservative200k projection37m59s fits45min;
short training costs77.95% more than12M. No stage repeats or gameplay claim.

The [saved crop-gap diagnostic](docs/results/2026-11-02-cdp-breakout-crop-gap.md)
finds the ball visible before all103 available loss crossings; all98 nearby
dropouts occur afterward. Retain the teacher. Its14 CPU cases/17 pins pass in
.387s with zero GPU work; the saved diagnostic is complete and must not repeat.

The [posterior-bootstrap comparison](docs/results/2026-11-02-cdp-breakout-posterior-bootstrap.md)
remains **closed_protocol_failed**: accepted sampled53.708 is below70.750 in
21m37.556s (+.75%); two greedy timeouts invalidate that arm. Retain imagined
bootstrap with12M64/Auto/N8/B16/T16/H15/R16, encoder4e-5/behavior8e-5/
dynamics4e-4/replay-value1/expert100 and sampled collection. All412 original/
553 repair pins pass. Original main,collection and first CPU closure remain
failed; the separate CPU repair completes10:11:27.827, terminal confirmation
10:12:01.748. No GPU stage or successful replay repeats. Root
`/x/Temp/kindle-replay-bootstrap-20261102/cpu-closure-repair` is closed; preserve
all failures and do not repeat training,captures,replays,collectors or readers.
The optional qualified implementation remains default-off; its2.989GiB build
cache is released, with wheel/native/test binary and evidence retained.
No teacher variant, learner, extension or automatic successor is selected.
Check live state before new work; detailed evidence stays in the linked reports.

The [earlier scalar calibration](docs/results/2026-11-02-cdp-breakout-value-calibration.md)
recovers only a small sampled-value signal (affineRMSE7.698 versus constant7.855).
It does not identify a causal critic defect or establish a gameplay correction.
Keep all original scores and completed evidence; no diagnostic should repeat.

The [completed Breakout H30/R12 comparison](docs/results/2026-11-02-cdp-breakout-h30-r12.md) scores **58.917 sampled /
124.708 greedy**, below70.750/143.667 by16.73%/13.20%, in **23m15.641s**
(+8.36%). Initial-subtracted return/min falls23.47%/20.09%; all0/24 two-wall.
Retain H15/R16 with12M64/Auto/N8/B16/T16/encoder4e-5/behavior8e-5/dynamics4e-4/
replay-value1/expert100 and sampled collection. Horizon and ratio change rollout
length,cadence,sampling,warmup and relative work together; reused1009/cohort
provides no isolated mechanism or independent reliability claim.

All422 pins,100k/4,670 recovery,357 matching initial/invariant frozen tensors,
47 pre-update and800/795 actual-initial semantic events,replays/videos,guards,
curves,exposure and comparison pass. Allfive original services succeed: main/
post/collection **07:58:54.411/07:59:16.112/07:59:20.722 UTC November2**,
launch source538ef64. Terminal reader passes once08:00:10.140 in2.623s;
completionSHAac50672c/summarySHA5d7058c4. No original stage fails or repeats.

The separate CPU coverage reader fails on its obsolete12,477-update assertion;
preserve that failure. A separately qualified9,358-update repair completes
08:03:41.018 in2.918s, with52 original/11 repair/422 full pins. All288,800 actions
and1,154,554 frames reconcile; completed control analysis is reused. Native
216–431 exposure falls1,262→1,157 (.631%→.5785%), with none at432+. Score
fractions do not establish walls or a causal mechanism. CoverageSHA41792f03.

Root `/x/Temp/kindle-breakout-h30-r12-20261102/full` is closed. Recovery,terminal
and repaired coverage readers are complete; do not repeat stages,collectors,
qualification or readers. No learner or GPU allocation is active at closure;
check live state. No extension,confirmation,sweep or automatic successor is
selected. CPU co-host costs remain uncorrected. Breakout and the five-game goal
remain unfinished. Detailed evidence stays in the linked result and protocol.

Earlier allocations are closed. The completed
[lower-dynamics comparison](docs/results/2026-11-02-cdp-breakout-value1-dynamics1e4.md)
regresses both Breakout scores; retain dynamics4e-4 with replay-value1 and
behavior8e-5. Qbert's fixed240k/R20 recipe passes on its development learner and
[one fresh greedy confirmation](docs/results/2026-11-02-cdp-qbert-r20-seed18013.md).
Breakout remains below the two-wall gate. Do not repeat completed learners,
checks, replays, observers or readers; no automatic successor is selected.
The roadmap and linked reports retain detailed outcomes, failures and limits.

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
ledger is preserved at52c2174 (and earlier113ecde/8b1ba49) and in the roadmap/reports. Historical “next”
paragraphs are not active instructions. Preserve user-authored TASK.md and
docs/cdp-claude-feedback.md as context; subsequent CDP decisions and measured
results supersede their older architecture/performance assumptions. Do not
merge this or upstream PRs on the user's behalf.
