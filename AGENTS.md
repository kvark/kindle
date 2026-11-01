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

The [fresh greedy-training Qbert comparison](docs/experiments/2026-11-01-cdp-qbert-greedy-training-200k.md)
is selected after the completed bounded check. Train fresh200k/12,477 under35min,
12M48/Auto/N8/B16/T16/H15/R16/encoder4e-5/expert100;only live collection changes
sampled→greedy. Native447fe50,losses,teacher and posterior/imagined sampling remain.
Require100k/6227 recovery and final/actual-initial sampled/greedy24-episode arms
against14,028.125 primary greedy/11,522.917 sampled and unchanged15k/90% gate.
CPU preflight1.832s passes305 bounded/276 control pins and actual CLI/audit/observer
checks withzero native/emulator construction. Root
`/x/Temp/kindle-qbert-greedy-training-20261101/full`;service
`kindle-qbert-greedy-training-200k-20261101`. Check live state;never restart or
schedule another GPU job while active. No full result or successor is selected.

The [completed greedy-training Qbert check](docs/results/2026-11-01-cdp-qbert-greedy-training.md)
finishes8192 actions/32,761 frames/489 updates in66.241s;1024 frozen/.798s haszero
updates and no natural episodes. Conservative200k projection29m39s passes35min.
Only training collection changes sampled→greedy;native447fe50,teacher,losses and
12M48/Auto/N8/B16/T16/H15/R16/encoder4e-5/expert100 remain. Explicit fresh-training
CLI mode/default compatibility pass637 source/34 private CPU checks. All305 pins,
357 matching initial/invariant frozen tensors,47 expected initial trace differences,
full guided replay and guards pass. Main succeeds06:05:29 UTC November1/sourcec806de5,
original collector06:06:03. Supplemental silent-parent journal assertion fails;
retained.638s CPU correction closes06:06:50,zero original/GPU repeats. Test-path/
old-audit CPU failures also remain preserved. Root
`/x/Temp/kindle-qbert-greedy-training-20261101/native-check`. No active allocation,
full learner or automatic promotion;execution/cost only. Five-game goal unfinished.

The [completed frozen Qbert training readout](docs/results/2026-11-01-cdp-qbert-training-readout.md)
replays200k actions/799,739 frames in128.729s,zero updates. Last50k later movement
agreement is96.92% forced,85.56% in unguided episodes and75.10% after guidance
releases;all passive83.79% versus59.31% original changing-policy agreement.
These are different model times/contexts,not a causal sampling or coverage test.
Only159 wholly unguided training actions reach age1920+,versus12,782 released
actions;all200k actions already have labels. All562 pins,357 invariant tensors,
128 F64 comparisons,98 repeated readbacks,exact smoke/full prefix,four tests,
phase counts and guards pass. Service succeeds05:56:14 UTC November1/source5d53dac,
original collector passes;zero failures/repeats. Roots
`/x/Temp/kindle-qbert-training-coverage-20261101` and
`/x/Temp/kindle-qbert-training-readout-20261101`. No active allocation or automatic
successor;greedy training actions are a candidate requiring a separate bounded
protocol. Five-game goal unfinished;check live state before work.

The [completed first-disagreement Qbert intervention](docs/results/2026-11-01-cdp-qbert-first-disagreement.md)
finds five of 25 first-command differences change survival through the original
loss endpoint: teacher-first survives 25, policy-first 20, with both then using
the same feedback-aware teacher. Four differences are movement versus movement;
one is movement versus NOOP. At the separate 64-action endpoint, paired outcomes
are 16 both, seven teacher-only and two neither. This identifies local action
effects under this controller, not an architectural cause or new game score.
All 16,613 source actions / 66,440 frames, 1,373 retained teacher suffix rows,
eight reverse-order checks, 11 tests and 552 input hashes pass. CPU analysis
65.045s / controller 65.779s completes 05:44:58 UTC November 1, source8a8b433.
No native/GPU work, learning, failures or retries; no new learner/loss or automatic
successor. Root `/x/Temp/kindle-qbert-first-disagreement-20261101`. The five-game
goal remains unfinished; check live state before new work.

The [exact actor readout](docs/results/2026-11-01-cdp-qbert-policy-readback.md)
remains qualified: 20,584 original actions reproduced, zero updates, 357 invariant
tensors and 525 pins. Failure-window movement agreement is52.34% versus97.66%
for NOOP. Read-only runtimef864fe4 passes130 Rust/585 Python checks and guards;
original CPU collector failure and.661s repair remain retained. Build cache is
released; runtime/wheels/source/test binary remain. No original stage repeats.

The [completed Breakout encoder8e-5 comparison](docs/results/2026-11-01-cdp-breakout-encoder8e5.md)
finishes200k actions/799,597 frames/12,477 updates in21m24s. Sampled55.875 falls
.74% versus56.292;greedy125.375 improves19.79% versus104.667. All0/24 two-wall;
own initials1.417/2.208 match. Retain4e-5 as sampled development point and8e-5 as
separate greedy option;sampled remains primary,no overall win or reliability.
All265 pins,100k/6227 recovery,357 matching/invariant tensors,47 pre-update and
800/795 initial semantic events,replays/videos,guards,curves,exposure and comparison
pass. Main/post/collection succeed05:14:41/05:15:03/05:15:06 UTC November1;
all five original services succeed,zero failures/repeats. Separate.872s terminal
confirmation passes05:15:35. Source3897e74;only encoder4e-5→8e-5 changes in
12M64/Auto/N8/B16/T16/H15/R16/expert100,native447fe50/exact runner/teacher fixed.
Actual guidance44.4695%;late CE.5252 versus.5581 is not frozen competence.
Reused1009/cohort and CPU timing caveats remain. Root
`/x/Temp/kindle-breakout-encoder8e5-20261101/full`;no extension,rate queue,
confirmation or automatic successor. No learner/GPU allocation remains active
at closure;check live state. The five-game goal remains unfinished.

The [completed CPU Qbert teacher takeovers](docs/results/2026-11-01-cdp-qbert-recoverability.md)
avoid25/32 selected recorded losses at their original endpoint;23 survive the
separate64-action follow-up. First episode per8 lanes,all32 losses;origins begin
up to32 actions earlier within the same life. This establishes useful teacher
continuations on many selected learner states,not an architectural cause or
new game score. All16,613 source actions/66,440 frames,recorded branches,eight
reverse-order checks,eight tests,309 diagnostic/265 protected Breakout pins and
original terminal identity pass. CPU63.910s/controller65.230s completes05:16:04
UTC November1/source1a4fffe,after Breakout training/main completion. No native/GPU
work,updates,failures or retries. Root`/x/Temp/kindle-qbert-recoverability-20261101`;
no new learner/loss or automatic successor. Preserve CPU overlap timing.

The [completed CPU Qbert life/color replay](docs/results/2026-11-01-cdp-qbert-life-replay.md)
finds all96 losses per retained2e-5/4e-5 policy occur after the first pyramid,
withzero life gains in each fixed24-episode cohort. Candidate selected duration
rises21.13%,score1.76%,and reward/action falls15.99%. Losses at episode age1920+
rise5→33 of96;this is descriptive,not matched training coverage or a causal
cap effect. Complete44,512/60,648-action replays reproduce all recorded outcomes.
Five CPU tests,293 diagnostic/265 active Breakout pins and original process
identity pass. Analysis49.272s/controller50.619s completes05:07:23 UTC November1,
sourceaa80855,withzero native/GPU work or updates. Raw cube-color histories are
retained without a general level detector. Root
`/x/Temp/kindle-qbert-life-replay-20261101`;no new learner/loss or automatic
successor. Retain CPU co-host timing caveats;the original Breakout run continues.

The [completed Qbert encoder4e-5 comparison](docs/results/2026-11-01-cdp-qbert-encoder4e5.md)
finishes200k actions/799,739 frames/12,477 updates in27m41s,8.56% less observed
training time than2e-5. Greedy14,028.125/sample11,522.917 improve1.76%/15.37%,
each24/24 first pyramids;own initials80.208/177.083 clear none. Both means still
miss15,000. Retain4e-5 as a development point,not independent reliability.
All276 pins,100k/6227 recovery,357 matching/invariant tensors,47 pre-update and
1139/1101 initial events,full replays/videos,guards,curves,coverage,exposure and
comparison pass. Main/post/collection succeed04:38:49/04:38:58/04:39:00 UTC
November1,source7f794b0;all five original services terminal success,zero repeats.
Separate.597s terminal confirmation corrects an overly strict watcher-PID check:
original collector output comes from a child in the same invocation. Preserve
that ancillary failure;the original pipeline passes unchanged. Actual overrides
38.4%;late replay CE.0893 versus.0814 and guided scores are not frozen competence.
Native447fe50,exact runner and teacher fixed;only encoder2e-5→4e-5 changes in
12M48/Auto/N8/B16/T16/H15/R16/expert100. Reused4013/cohort and CPU co-host caveats
remain. Root`/x/Temp/kindle-qbert-encoder4e5-20261101/full`. No Qbert extension,
confirmation or automatic successor;no active GPU allocation at closure.
Check live state before work. The five-game goal remains unfinished.

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
| Qbert |H5/R32:greedy14,050/15,792.708 on4013/6029,23/24 and24/24 pyramids,inabout40m each|15k/90% gate passes one of two; cheaper4e-5 H15/R16 scores14,028.125 in27m41s and still misses15k|

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
The previous expanded instructions remain in Git at34b1e74. Historical “next”
paragraphs are not active instructions. Preserve user-authored TASK.md and
docs/cdp-claude-feedback.md as context; subsequent CDP decisions and measured
results supersede their older architecture/performance assumptions. Do not
merge this or upstream PRs on the user's behalf.
