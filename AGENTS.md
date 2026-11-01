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

## Active work — November 1

The [fresh Qbert H3/R24 comparison](docs/experiments/2026-11-01-cdp-qbert-h3-r24-200k.md)
is active from14:13:06 UTC November1/source df3a75e,506 inputs,originalPID803859/
invocation798ad208226e40a3a73fa9e43fec8066. Service`kindle-qbert-h3-r24-200k-20261101`,
root`/x/Temp/kindle-qbert-h3-r24-20261101/full`. All four observers are active.
Startup14:13:34 passes357 control/bounded-matching tensors,47 pre-update
transitions,correct R24 cadence and264 finite supervised reports through3,192
acts. Fixed200k/18,715updates,35min cap;require100k/9,340 audit with automatic
four-file retention and final/actual-initial sampled/greedy24-episode assessments.
Greedy14,339.583 is primary,sampled13,761.458 secondary;original15k/90% gate.
H5→3 only;native447fe50/runneraba4300d/teacher unchanged. No frozen result yet.
Inspect this original process;do not restart,repeat checks or allocate concurrent
GPU work. No extension,retry,horizon sweep or automatic successor.

The [bounded H3 result](docs/results/2026-11-01-cdp-qbert-h3-r24.md) passes8,192
acts/32,760frames/733updates in64.175s;1,024 frozen/.806s haszero updates/no
natural episodes. Conservative200k28m43s passes35min. All475 pins,357 matching/
invariant tensors,47 events,full replay/guards pass;main14:08:14/collector14:08:26,
source8dad414. Full CPU qualification3.225s and plot import.044s pass,including
actual recovery retention/collector checks. Original preparer wording failure
remains;no GPU repeat. Do not rerun passed checks or reuse bounded experience.

The [completed Breakout400k/R8 comparison](docs/results/2026-11-01-cdp-breakout-r8.md)
finishes400k/12,489updates in24m03.031s,+12.06% time. Sampled56.708 versus56.292
is nearly unchanged;greedy97.583 versus104.667 falls6.77%,all0/24 two-wall.
Retain200k/R16. All315 pins,357 matching/invariant tensors,three retained
recoveries,paired initial traces,full replays,guards,curves,exposure and original
collectors pass. Main/post/collection succeed13:44:22/13:44:36/13:45:11 UTC
November1/source566a2a5;all five original services terminal,no full/GPU failure
or repeat. Root`/x/Temp/kindle-breakout-r8-20261101/full`. No extension,ratio retry
or automatic successor. Do not rerun completed checks,learner or collectors.

The [completed CPU centering reference](docs/results/2026-11-01-cdp-breakout-unseen-center.md)
improves225.083→269.833 on expanded24,five of eight lane means better,zero two-wall.
The selected eight screen episodes improve185.375→333.000;the additional16
average244.938→238.250. Overall development gain does not establish independent
improvement. Keep the native teacher;no integration or learner selected. All31
pins/315 completed R8 inputs and full exact replay/video pass;68.302s CPU controller
completes13:51:07 UTC/sourcebd3ce22 after R8 closure. No GPU work/updates or repeated
gameplay. Original unreceipted preflight/tool-output loss remains;separate completed-
service preflight passes. Root`/x/Temp/kindle-breakout-unseen-center-20261101/standard24`.
Do not repeat completed stages. No GPU allocation is active;check live state.

The [completed Qbert R24 temperature assessment](docs/results/2026-11-01-cdp-qbert-h5-r24-temperature.md)
scores12,840.625/24 first pyramids at0.5 versus greedy14,339.583/24;retain greedy.
Default sampled13,761.458 reproduces exactly;own initials171.875/203.125 clear
zero pyramids. All558 pins,357 invariant tensors per arm,default semantic traces,
full replays/videos and guards pass. Four arms125,944actions/503,626frames/105.190s,
zero updates/truncations;original28m34s training stays. Main/collector succeed
13:59:10/13:59:17 UTC November1/source1308ffb,originalPID801574/invocation
6ce4cfed31e44e73aef685bb38ed64b7. Both services terminal,no failure or repeat.
Root`/x/Temp/kindle-qbert-h5-r24-temperature-20261101`;do not rerun completed
stages. No sweep,retry or automatic successor. No active GPU allocation;
check live state. The five-game goal remains unfinished.

The [completed Qbert R32 comparison](docs/results/2026-11-01-cdp-qbert-h5-r32-greedy.md)
finishes200k/24,954updates in34m59.741s,+22.53% versus R24. Greedy14,057.292/23
and sampled10,305.208/18 trail14,339.583/24 and13,761.458/24 first pyramids.
Retain R24. All490 pins,357 matching/invariant tensors,100k retained recovery,
initial semantic traces,full replays,guards,curves,coverage/exposure and original
collector pass. Main/post/collection succeed12:58:49/12:58:56/12:59:27 UTC
November1/source4a10fcc;all five services terminal,no full/GPU failure or repeat.
Do not rerun its learner or collectors. CPU co-host caveats and earlier CPU
preparation failures remain;the five-game goal is unfinished.

Recent completed diagnostics and negative comparisons are in the
[learning diagnosis](docs/results/2026-10-29-cdp-learning-diagnosis.md),roadmap
and handover. They do not select new losses,teachers or successor learners.
Keep this section compact; detailed experiment history belongs in those reports.

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
| Breakout |Encoder4e-5:sampled56.292/greedy104.667 in21m28s;encoder8e-5:55.875/125.375 in21m24s|Sampled remains primary;all0/24 two-wall;reused development seed|
| Qbert |Current H5/R24/encoder4e-5/greedy collection:15,430.208 on8039 and14,339.583 on10007,24/24 first pyramids each,in28m47s/28m34s|15k/90% gate passes one of two; H5/R32 lowers both means at higher cost; retain R24|

Published long-run scores are aspirations,not matched-budget comparisons.
Finite losses,short runs and backend speed do not establish mastery or CDP
superiority. Guided online returns and replay imitation losses are not frozen
competence. Keep detailed results in the roadmap and linked reports,not a
growing history in this file.

Latest qualified Breakout evidence:
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
matching runner at `/x/Temp/kindle-breakout-corrective-20261028/runtime`.
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
ledger is preserved at8b1ba49 and in the handover/roadmap. Historical “next”
paragraphs are not active instructions. Preserve user-authored TASK.md and
docs/cdp-claude-feedback.md as context; subsequent CDP decisions and measured
results supersede their older architecture/performance assumptions. Do not
merge this or upstream PRs on the user's behalf.
