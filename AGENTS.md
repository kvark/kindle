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

The [selected frozen Qbert cross-cohort diagnostic](docs/experiments/2026-11-01-cdp-qbert-cross-cohort.md) assesses4013
final/actual-initial on4000008039 and8039 final/actual-initial on4000004013.
Reuse completed diagonal assessments;four new greedy arms,24 natural episodes
per arm,600k actions/300s,zero updates/truncations,357 invariant tensors,replays/
videos and guards. Native447fe50/runneraba4300d and checkpoint configs remain.
This tests frozen-agent/cohort dependence,not isolated weights or reliability;
original8039's failed gate stays unchanged. Breakout is fully closed and the
host idle at selection. CPU command/controller qualifications pass;their two
checker failures and co-host timing remain preserved. Root
`/x/Temp/kindle-qbert-cross-cohort-20261101`,service
`kindle-qbert-cross-cohort-20261101`. Launch08:24:54 UTC November1/source63fce09,
413 pins,originalPID770056/invocationd56c32f594864b018418ed008250835f;collector
PID770059/invocatione17b21c8b78648fb8aeccf316d7cc5df. The first crossed final
run and outcome replay finish;the full matrix remains pending. No new learner,
retry,cap increase or automatic successor. Do not rerun qualifications or completed
diagonal GPU stages;check live state before work and do not overlap assessments.

The [completed Breakout greedy-collection comparison](docs/results/2026-11-01-cdp-breakout-greedy-training.md) finishes200k/
799,639 frames/12,477 updates in21m46s. Sampled39.042/greedy99.083 fall30.64%/
5.33% below retained56.292/104.667;all0/24 two-wall. Own initials1.417/2.208 match.
Training takes1.41% longer;retain sampled collection. All272 pins,100k/6227
recovery,357 matched initial/invariant frozen tensors,paired initial semantic
traces,full training pixel/action/label and frozen replays/videos,guards,curves,
exposure and comparison pass. Main/post/collection succeed08:20:54/08:21:10/
08:21:42 UTC November1/sourcebd51be9. Independent2.048s terminal confirmation
and its observer succeed08:22:04. No original/GPU stage or confirmation repeats;
prior CPU preparation path failure and co-host timing remain. Actual overrides
40.8625%;higher aided online125.64 is not frozen competence. Reused1009/cohort;
no isolated coverage cause or reliability claim. Root
`/x/Temp/kindle-breakout-greedy-training-20261101/full`;all services terminal,
no allocation active at closure. Do not rerun learner,collectors or confirmations.
No extension,cap increase,confirmation learner or automatic successor. Check live
state before GPU work;the five-game goal remains unfinished.

The [completed bounded Breakout check](docs/results/2026-11-01-cdp-breakout-greedy-training.md) passes8192 actions/32,740 frames/
489 updates in46.633s;1024 frozen/4076 frames/.874s haszero updates and eight
zero-return natural episodes. All252 pins,357 matching initial/invariant frozen
tensors,47 expected initial semantic differences,full replay and guards pass.
Conservative200k projection20m52s passes30min;no competence inferred. Original
service succeeds07:51:36 UTC November1/sourceccb4387,collection07:51:45. No
original stage fails or repeats. Root
`/x/Temp/kindle-breakout-greedy-training-20261101/native-check`;terminal at closure,
do not rerun. The full learner is a separate selection and uses no check experience.

The [completed CPU delayed Breakout takeover](docs/results/2026-11-01-cdp-breakout-delayed-takeover.md) records survival
108/108/110/106/105/91 of120 at delays0/1/2/4/8/16 actions from first disagreement.
Most early sequences remain recoverable under the teacher;later takeover reduces
aggregate rescue,with nonmonotonic cases. No architectural cause or game score.
All478 new branches,37,016 source actions/148,038 frames,suffixes,eight reverse
checks,334 diagnostic/359 active Qbert pins and21 CPU tests pass.55.638s analysis /
57.311s controller closes07:12:23 UTC November1/sourcec9fbf9f.240 retained baseline
branches do not repeat;zero native/GPU work,updates,failures or retries. Original
QbertPID760870/invocationa0b219d7702b4c82a65e9b33498c3544 remains unchanged;
retain CPU timing. Root`/x/Temp/kindle-breakout-delayed-takeover-20261101`;
complete,do not rerun. No new learner,loss or successor is selected.

The [completed Qbert seed8039 check](docs/results/2026-11-01-cdp-qbert-greedy-seed8039.md) finishes200k/799,796 frames/
12,477 updates in27m34s. Greedy11,376.042/23 first pyramids and sampled8,292.708/24
miss15k;own initials251.042/128.125 clear none. Keep4013's15,515.625 passing point,
but this same recipe passes primary greedy on one of two learners;reliability
remains unresolved. Only learner/evaluation seeds change,no sampled-training
control for8039 or isolated collection-mode effect. All359 pins,100k/6227 recovery,
357 finite initial/invariant frozen tensors,48 initial differences,full guided
pixel/action/label and frozen replays/videos,guards,curves,coverage,exposure and
comparison pass. All five original services succeed;main/post/collection finish
07:44:45/07:45:00/07:45:02 UTC November1/sourcecbee206. Independent.878s terminal
confirmation passes07:45:29;zero failures or repeated original/GPU stages.
Actual overrides32.0105%;lower late replay CE/higher aided online returns are not
frozen competence. Root`/x/Temp/kindle-qbert-greedy-seed8039-20261101/full`;
all services terminal,no active allocation at closure. Do not rerun this learner,
collectors or confirmations. No extension,cap increase,third Qbert learner or
automatic successor. Check live state before work;the five-game goal is unfinished.

The completed8039 CPU exposure recount finds later wholly unguided actions
81,530→80,031 versus4013,but unguided age1920+5525→2977;last50k31,625→26,900
and4541→2165. Released-guidance exposure increases. This localizes a difference
to long autonomous continuations without proving a cause;both learner and frozen
cohort differ. All381 diagnostic/272 active Breakout pins,original process identity,
200k actions/799,796 frames,boundaries,labels and overrides pass. Analysis.847s /
command5.579s completes08:00:47 UTC November1/sourceafa6ae6. Prior4013 counts are
reused;no native/emulator/GPU work,updates or failures. See the result's exposure
section. Root`/x/Temp/kindle-qbert-seed8039-coverage-20261101`;do not rerun. Retain
CPU co-host timing;no new learner,loss or automatic successor is selected.

The [completed CPU Breakout first-disagreement diagnostic](docs/results/2026-11-01-cdp-breakout-first-disagreement.md)
finds102 both/six teacher-only/six policy-only/six neither survive the matched
original loss endpoint;both arms total108/120. At64 actions,86/10/11/13;not the
old window-origin endpoint. First disagreements usually remain recoverable under
the teacher;no aggregate first-command survival gain or architecture cause.
All37,016 source actions/148,038 frames,6,905 old teacher-suffix rows,eight reverse
checks,320 repair/317 active Qbert pins and22 CPU tests pass. Analysis39.271s /
controller40.843s completes06:24:01 UTC November1/sourced139b43. Original23fe57f
origin-order assertion fails before replay and remains preserved;separate
`order-repair` closure is`complete_with_cpu_order_repair`,zero completed replay/
GPU repeats. Root`/x/Temp/kindle-breakout-first-disagreement-20261101`;do not rerun.
Original QbertPID755504/invocation5fb0d30ffb5d47aca552009e586bb929 was preserved;
retain CPU co-host timing. No new learner,loss or successor is selected.

The [completed greedy-training Qbert comparison](docs/results/2026-11-01-cdp-qbert-greedy-training.md)
finishes 200k actions / 799,803 frames / 12,477 updates in 27m33s. Primary frozen
greedy 15,515.625 / 24 first pyramids passes the 15k/90% gate on this learner;
sampled 11,989.583 / 24 improves but misses the score gate. Own initials
80.208 / 177.083 clear none. Relative to sampled training, greedy improves
10.60% and sampled 4.05%, at .51% less observed training time. Keep CPU co-host
timing caveats; no causal speedup or coverage mechanism is established.
Only live collection changes sampled to greedy; native447fe50, teacher, losses
and 12M48/Auto/N8/B16/T16/H15/R16/encoder4e-5/expert100 remain unchanged.
All 317 pins, 100k/6227 recovery, 357 matching initial/invariant frozen tensors,
full guided pixel/action/label replay, frozen replays/videos, guards, curves,
coverage, exposure and comparison pass. All five original services succeed;
main/post/collection finish 06:56:22/06:56:26/06:56:28 UTC November 1, sourcec659423.
Separate .821s terminal confirmation passes 06:57:01. Preserve the ancillary
CPU observer-accounting preflight failure; zero original pipeline/GPU repeats.
Actual overrides 30.113%; guided online scores are not frozen competence.
Reused4013/cohort gives one passing learner, not seed reliability. Retain this
recipe for a separately declared fresh-seed check. Root
`/x/Temp/kindle-qbert-greedy-training-20261101/full`; all services terminal,
no allocation active at closure. Do not rerun completed stages or confirmations.
The five-game goal remains unfinished; check live state before new work.
The completed1.017s CPU coverage recount confirms later wholly unguided actions
58,487→81,530 and age1920+159→5,525,with332 diagnostic/359 active pins unchanged
at07:21:44 UTC/source234ca3a. This supports the coverage hypothesis without proving
causal mediation;all actions already have labels. See the result's coverage section.
No native/emulator/GPU work or new learner;retain CPU co-host timing.

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
