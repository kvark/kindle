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

The [fresh Qbert RGB64 seed12011 check](docs/results/2026-11-01-cdp-qbert-rgb64-seed12011.md) completes8192 actions/
32,763frames/733 updates in58.040s;frozen1024/.847s haszero updates/no natural
episodes. All568 pins,357 initial/invariant tensors,full guided replay and guards
pass. Main succeeds20:20:00/collection20:22:04 UTC November1/source314e071.
Conservative200k projection25m58.703s passes30min;no gameplay result yet.

The [separate fresh200k confirmation](docs/experiments/2026-11-01-cdp-qbert-rgb64-seed12011-200k.md) is selected after
3.864s CPU qualification:five CLI paths,30 config cases,retention/observer checks,
568 bounded/521 control pins and357 bounded tensors pass. Only seeds change
10007/4000010007→12011/4000012011;retainRGB64/12M/H5/R24/encoder4e-5/base
behavior4e-5/greedy collection/expert100/native447fe50/runneraba4300d/teacherfa261.
Fix200k/18,715 updates/30min,100k/9,340 recovery andfour final/own-initial24-episode
frozen arms. Full service launches20:23:42 UTC November1/source34c6a88 with562 pins,
PID842010/invocationd37840f258104f9caf8b6bc5c0316899. Startup passes357 own-bounded
initial tensors,47 pre-update transitions and258 finite reports through3128 actions;
all47 pre-update events differ from10007,as expected for another seed. Progress,
100k/9,340 recovery passes20:36:18 UTC:all357 finite tensors andfour retained
files (206,067,159bytes) verify;metadata17ff65f5. The recovery observer succeeds;
main/progress/postprocessing/collection continue with original identity. No frozen
result,bounded experience reuse or third learner selected. Do not restart or overlap it.
Root`/x/Temp/kindle-qbert-rgb64-seed12011-20261101/full`;check live state before
work. The five-game goal remains unfinished.

The [exact Breakout batched-policy capture](docs/experiments/2026-11-01-cdp-breakout-policy-readback.md) is CPU-qualified,
not GPU-selected or launched. Reuse qualified readoutf864fe4b/native447fe50 plus
readout only;preserve12M64/encoder4e-5/.01unimix andoriginal greedy actions.
All37,016 actions/148,038frames,31,161 selected decisions and120 life-clipped
windows/3,816 unique rows align. Six fixture groups/264 inputs pass in6.385s
at20:37:04 UTC November1,with562 active Qbert pins andoriginal identity preserved.
Require Qbert's complete terminal receipt andidle host before separate selection;
no executable GPU queue or declaration exists yet. No new learner or native build.
Root`/x/Temp/kindle-breakout-policy-readback-20261101`;retain CPU co-host timing.

The [fixed current-pixel action readout](docs/results/2026-11-01-cdp-breakout-pixel-action-readout.md) fails its screen:
learned+pixel kernel57.40% overall/55.15% near-descending teacher agreement,
versus learned-only56.91%/58.82% and actor62.56%/66.91%. Pixel-only48.25%/43.38%.
Keep the actor;no native spatial path,head,loss,teacher or learner selected.
All2064 aligned rows,51 diagnostic/562 active Qbert pins andsix CPU fixtures pass.
Analysis.760s/controller3.410s completes20:30:08 UTC November1/source6a6ce27,
preserving originalPID842010/invocationd37840f258104f9caf8b6bc5c0316899.
No replay,native/emulator/GPU work,updates,failure or repeated fit. The additive
kernel and reused teacher/cohort limit interpretation;no broad spatial-input
claim. Retain CPU co-host timing. Root`/x/Temp/kindle-breakout-pixel-action-readout-20261101`
is complete;do not rerun. Qbert confirmation continues;the five-game goal is open.

The [completed Qbert12M/RGB64 comparison](docs/results/2026-11-01-cdp-qbert-rgb64.md) reaches **15,710.417 greedy /24 first pyramids**,
passing the unchanged15k/90% gate and improving RGB48's14,339.583 by9.56%.
Sampled14,250/24 rises3.55% from13,761.458 but still misses15k. Own initials
1.042 greedy/195.833 sampled clear none;cross-geometry traces differ. Retain this
single-learner development point;reused10007/cohort does not establish reliability.
No extension,automatic confirmation or successor is selected by this result.

Training completes200k actions/799,823frames/18,715 updates in**25m09.797s**,
within30min. Total time falls11.90% versus the original slower-teacher control;
initial-subtracted gain/min rises24.35% greedy/17.38% sampled. Recorded learning
work rises16.73%;the faster teacher also changes acting cost,so no isolated
resolution/native speed claim. Keep12M64/H5/R24/encoder4e-5/base behavior4e-5/
greedy collection/expert100/native447fe50/runneraba4300d/teacherfa261c53.
Geometry/head shapes/initialization change together;older4M64 negative remains.
Late imitation CE.051060→.049491;44,600 actual overrides (22.3%) are not competence.

All521 pins,357 bounded-matching initial/invariant frozen tensors,47 pre-update
transitions,100k/9,340 retained four-file recovery,full replays/videos,guards,
curves,coverage,exposure,comparison and original collector pass. All five services
succeed:main20:11:19/post20:11:52/collection20:12:24 UTC November1/sourceb8de34f.
Four frozen arms156,312 actions/625,086frames/146.320s,zero updates/truncations.
No failure,cap change,GPU/replay/collector repeat or build. CPU co-host caveats
remain without time correction. Root `/x/Temp/kindle-qbert-rgb64-20261101/full`
is complete;do not restart it. No GPU allocation remains active at closure.
Check live state before future work;the five-game goal remains unfinished.

The [fixed Breakout one-step risk controller](docs/results/2026-11-01-cdp-breakout-one-step-risk.md) lowers mean return
225.083→184.417 (−18.07%), with three of eight lanes improved and both0/24
two-wall successes. Retain the original teacher;no fresh cohort,penalty retry,
integration or learner. The penalty changes39 of1,355 eligible decisions on
38,223 selected candidate states;this is not original-trajectory disagreement.
All36 diagnostic/521 active Qbert inputs,43,056-action/172,188-frame exact
action/debug/live-pixel/outcome replay and video pass. Six CPU fixtures pass;
nonfatal test ResourceWarning retained. Collection26.617s/replay32.202s;
controller61.624s completes19:52:07 UTC November1/source316a857, preserving
originalQbertPID836919/invocation425aae9cee2541fe9623b753225af57f.
No failure,repeat,native/GPU work or update;retain CPU co-host timing.
Root`/x/Temp/kindle-breakout-one-step-risk-20261101` is complete,do not rerun.
The error-prediction result remains;this use of it did not improve gameplay.
Qbert RGB64 continues;the five-game goal remains unfinished.

The [fixed Breakout response-error check](docs/results/2026-11-01-cdp-breakout-response-uncertainty.md) passes its
prediction screen: command-conditioned error-forecast MSE falls 5.04% overall
and 4.49% near descending balls, with lower Gaussian NLL and both scores better
in all eight lanes. Stationary-edge MSE worsens and predicted error²4.809 remains
far below observed18.383. No controller or gameplay gain is established.
All37,448 rows/control metrics,16 diagnostic and521 active Qbert inputs pass;
.772s analysis/3.262s controller completes19:46:15 UTC November1/source1bbd75e,
preserving originalPID836919/invocation425aae9cee2541fe9623b753225af57f.
Four CPU fixture groups pass;no failure,repeat,native/emulator/GPU work or update.
Keep CPU co-host timing. Root`/x/Temp/kindle-breakout-response-uncertainty-20261101`
is complete;no controller,integration,sweep or learner is selected.
Qbert RGB64 continues;the five-game goal remains unfinished.

The [command-dependent Breakout teacher comparison](docs/results/2026-11-01-cdp-breakout-command-response.md) is complete.
Fresh mean return 250.875 → 273.125 rises 8.87%, with five of eight lane means
improved, but misses the declared 10% requirement. Both arms have zero of 24
two-wall successes. Retain the current teacher; no integration, retry, further
cohort or learner is selected. Development 225.083 → 295.333 (+31.21%, six
lanes) remains a separate reused-cohort result. All 41 fresh inputs, unchanged
policy/response bytes, exact action/debug/live-pixel/outcome replays and both
videos pass. Control55,280 actions/221,090 frames/32.304s; candidate64,560/
258,195/35.377s. Full controller144.403s completes 19:26:13 UTC November 1,
sourcef24cd1a. No failure, refit, cap change, repeated stage, native/GPU work or
update. The .470s fixed prediction fit and 72.478s development comparison also
remain complete. Do not rerun these stages. Root
`/x/Temp/kindle-breakout-command-fresh-20261101`; no active allocation.
The five-game goal remains unfinished; check live state before future work.

The [completed Qbert encoder8e-5 comparison](docs/results/2026-11-01-cdp-qbert-encoder8e5.md) lowers both frozen
scores: greedy 14,339.583 → 12,435.417 (−13.28%) and sampled 13,761.458 →
11,213.542 (−18.51%), with 23/24 and 24/24 first pyramids. Both means miss
15,000. Actual-initial scores 1.042/171.875 reproduce, with zero pyramids.
Retain encoder4e-5 and the faster teacher. No rate extension, sweep,
confirmation or successor is selected. Training completes 200k actions /
799,800 frames / 18,715 updates in 20m36.093s. Time falls 27.87%; initial-
subtracted gain/minute rises 20.22% greedy and 12.64% sampled, but teacher CPU
cost also changes. This does not establish a rate-induced speed benefit.
Late imitation CE rises .051060 → .052894. Actual 44,160 overrides (22.08%)
and online guided scores are not frozen competence.

All 528 inputs, 357 matching/invariant tensors, 47 pre-update events,
100k/9,340 four-file recovery, full replays/videos, guards, curves, coverage,
exposure, comparison and the original collector pass. All five services
succeed: main 19:01:36, postprocessing 19:01:43 and collection 19:02:15 UTC on
November 1, source ea34785. Pre-launch preparation and CPU selector failures
remain preserved; one repaired CPU preflight rerun passes. No full-run or
GPU/replay/collector failure or repeat occurs. Native447fe50, runneraba4300d,
12M48/H5/R24/base behavior4e-5 remain fixed. The reused10007 cohort and teacher
implementation change limit interpretation. Retain CPU co-host timing without
correction. Root `/x/Temp/kindle-qbert-encoder8e5-20261101/full` is complete;
do not rerun it. No Kindle service or GPU allocation is active. The five-game
goal remains unfinished; check live state before future work.

The [Breakout full-width paddle diagnostic](docs/results/2026-11-01-cdp-breakout-paddle-crop.md) acceptszero of42,303
selected decisions:24 initial/reset,37,960 extra-red,4,285 image-edge and34
nonstandard exclusions. Thus old-crop bias remains unquantified;do not widen
the teacher crop on this evidence. All52,072 recorded actions/208,254frames,
original actions/debug/pixels/outcomes and208,254 prior raw annotations match.
The source stays225.083/zero two-wall;no new score,policy,loss or learner.
Eight CPU fixtures/47 diagnostic pins/528 active Qbert pins pass. Replay53.457s/
controller56.159s completes18:41:41 UTC November1/source7433689,preserving
originalPID831942/invocationbae3fe2dd4a24070ab2845a7828145fa. Retain CPU co-host
cost without correction. No GPU work,failure or repeated stage;no relaxed filters.
Root`/x/Temp/kindle-breakout-paddle-crop-20261101`;complete,do not rerun.

The [completed Qbert300k comparison](docs/results/2026-11-01-cdp-qbert-300k.md)
finishes300,000 actions/1,199,763frames/28,090 updates in31m18.803s,+9.64% time
versus200k with the faster teacher. Greedy14,339.583→13,861.458 falls3.33%;
sampled13,761.458→8,668.750 falls37.01%. First pyramids are24/24 greedy and
18/24 sampled;both means miss15k. Own initials1.042/171.875 reproduce,zero
pyramids. Final50k-action imitation CE.051060→.037849 improves without frozen
benefit. Retain200k/12M48/H5/R24/encoder4e-5/base behavior4e-5/greedy collection/
expert100 and the faster teacher;no budget extension,confirmation or successor.
All524 inputs,357 initial/invariant tensors,47 pre-update events,both100k/9,340
and200k/18,715 recoveries,full replays/videos,guards,curves,coverage and comparison
pass. Main succeeds18:22:35 UTC November1/source7bd5001. Original postprocessor
fails18:22:46 on stale200k exposure assertions;collection records failure18:22:47.
Separate6.491s CPU repair completes18:24:23,all9 repair pins/95 original artifacts
unchanged,zero GPU or completed CPU-stage repeats. Original failed services and
ancillary CPU reporting failures remain;closure`complete_with_cpu_postprocess_repair`.
Actual56,103 overrides (18.701%) and online guided scores are not frozen competence.
Extra experience and updates remain coupled on reused10007;no reliability claim.
CPU co-host caveats stay without correction. Root`/x/Temp/kindle-qbert-300k-20261101`,
authoritative`postprocess-repair/completion-review.json`. All services terminal;
no GPU allocation active,no restart. The five-game goal remains unfinished.

The [completed CPU prefix comparison](docs/results/2026-11-01-cdp-qbert-budget-prefix.md)
finds all25,000 transitions,18,715 non-timing reports and118 episode records
identical through200k actions/799,841frames,with357 elementwise-equal200k arrays.
The100k control has metadata only;serialized hashes differ at both milestones.
Original bare-key failure is preserved;the group-qualified repair completes
18:06:05 UTC in4.883s/4.921s wrapper. A separate.902s cumulative-frame-count
correction completes18:08:25 without repeating the comparison. Final closure
18:09:12 is `complete_with_cpu_reporting_repairs`. All25 repair/524 active pins
and originalPID826399/invocation stay. No native/GPU work or updates;retain CPU
co-host time without correction. Root `/x/Temp/kindle-qbert-budget-prefix-20261101`;
complete,do not rerun. Extra experience and updates remain coupled;no new frozen
score or independent reliability follows.

The [completed Qbert25M/R16 comparison](docs/results/2026-11-01-cdp-qbert-25m-r16.md)
finishes200k actions/799,786frames/12,477 updates in23m40.953s,17.08% less time
than12M/H5/R24. Primary greedy14,339.583→10,696.875 falls25.40%;sampled
13,761.458→11,967.708 falls13.03%;all24/24 first pyramids,both candidate means
miss15k. Own initials39.583/173.958 clear none and differ across capacities.
Own-initial-subtracted return/minute falls10.36% greedy and rises4.66% sampled.
Retain12M/H5/R24 and the faster teacher;no larger-recipe promotion or extension.
All521 inputs,357 bounded-matching/invariant tensors,retained100k/6,227 recovery,
full replays/videos,guards,curves,coverage,exposure,comparison and original
collector pass. All five services succeed:main17:25:24/post17:25:48/collection
17:25:55 UTC November1/source65f4cee. Original CPU selection/blocked-launch
failures and one-basename repair remain;no GPU or completed-stage repeat.
Capacity,initialization,ratio,total work and teacher CPU cost change together;
no isolated cause or reliability. Actual48,158 overrides (24.079%) are not
frozen competence. CPU co-host caveats remain without correction. Root
`/x/Temp/kindle-qbert-25m-r16-20261101/full`;complete,do not rerun. No Kindle
service or GPU allocation remains active. No retry,confirmation or automatic
successor selected;the five-game goal remains unfinished. Check live state.

The [existing centroid-target CPU reference](docs/results/2026-11-01-cdp-centroid-target-precision.md)
has ball x/y MAE.058/.225 pixels on555 shared test rows,versus saved-head error
5.461/5.206 against that target. Both target and earlier pixel decoder miss the
same2/557 reference balls;target haszero extra detections versus4. The crop
difference does not explain most of this old encoder6e-6 head error;GPU readback,
current-model accuracy and gameplay causality remain unmeasured. All20,960
recorded actions/83,799frames,2,064 prior pixel predictions,27 inputs and521 active
Qbert inputs pass;originalPID821599/invocation stays. CPU-only11.769s completes
17:00:05 UTC November1/source79701cc,zero GPU work,updates,failures or repeats.
Retain CPU co-host time;no crop,loss or learner selected. Root
`/x/Temp/kindle-centroid-target-precision-20261101`;complete,do not rerun.

The [bounded Qbert25M/R16 check](docs/results/2026-11-01-cdp-qbert-25m-r16.md)
completes8,192 actions/32,762frames/489updates in56.153s;1,024 frozen/1.159s
haszero updates/no natural episodes. All501 inputs,357 derived/invariant tensors,
finite reports,full pixel/action/label replay and guards pass. Main16:44:28/
original collection16:44:45 UTC November1/source d211ad3;no failure,GPU repeat
or repair.31,636,273 trainable parameters;no cross-capacity initial equality.
Root`/x/Temp/kindle-qbert-25m-r16-20261101/native-check`;complete,do not rerun.

The [saved Qbert training-phase diagnostic](docs/results/2026-11-01-cdp-qbert-training-phases.md)
finds76–77% of collected actions after the first pyramid across three H5/R24
learners,rising to87–89% in the final50k. Late native movement agreement is100%
before the milestone but70–72% afterward;overrides are excluded. Broad absence
of later-phase exposure is unsupported,while difficult-state coverage,fitting,
representation and teacher limitations remain unresolved. All600k actions,
357 completed episodes,24 partial tails and17 inputs reconcile. CPU-only3.440s
completes16:34:52 UTC November1/source2365121,zero replay,GPU work,updates or
failures. No learner/intervention selected;host idle. Root
`/x/Temp/kindle-qbert-training-phases-20261101`;complete,do not rerun.

The [completed Breakout B8/T32 comparison](docs/results/2026-11-01-cdp-breakout-context32.md)
finishes200k actions/799,549frames/12,469updates in23m12.128s,8.11% longer than
B16/T16. Primary sampled56.292→48.583 falls13.69%;greedy104.667→130.417 rises
24.60%,all0/24 two-wall. Retain B16/T16 for primary sampled performance and
B8/T32 as a stronger greedy development point;no overall win or reliability.
All292 inputs,357 matching/invariant tensors,retained100k/6,219 recovery,paired
initial traces,full replays/videos,guards,curves,exposure and original collector
pass. All five services succeed:main16:26:03/post16:26:09/collection16:26:47 UTC
November1/source4f7b4c4. Zero failures,GPU repeats or repairs. Native447fe50,
runner6c1b69fb and teacher stay fixed;batch,temporal gradients,diversity,target
rows and warmup change together. CPU co-host caveats remain without correction.
Root`/x/Temp/kindle-breakout-context32-20261101/full`;complete,do not rerun.
No Kindle service or GPU allocation remains active. No extension,context retry,
confirmation or automatic successor selected;the five-game goal is unfinished.

The [saved centroid-head CPU diagnostic](docs/results/2026-11-01-cdp-centroid-head-readout.md)
finds5.460/5.275-pixel ball x/y MAE versus9.267/8.064 for the earlier linear
recurrent probe on555 shared test rows. The existing head recovers more precise
position than those probes;substantial error remains. This is the old encoder6e-6
centroid checkpoint,not the current learner;its original gameplay protocol
failure stays. CPU F64 reconstruction has no independent native-forward check.
Native target and pixel-reference crops differ by one boundary row;actual target
accuracy is unmeasured. All29 diagnostic/292 then-active Breakout inputs and
original process identity pass. Analysis2.640s (2.696s wrapper) completes16:21:52
UTC November1/source5318dc9,zero new replay,fitting,GPU work,updates or failures.
Retain co-host time;no new head,loss or learner selected. Root
`/x/Temp/kindle-centroid-head-readout-20261101`;complete,do not rerun.

The [saved Qbert behavior-rate outcome diagnostic](docs/results/2026-11-01-cdp-qbert-behavior-phases.md) retains the425-point
first-pyramid failure:its slot loses15,275 points while the other23 gain15,725,
leaving only+18.75 official greedy mean. Twelve other slots improve,11 worsen;
conditional successful mean14,964.130 still misses15k. Late replay CE rises
.051060→.055250 (+8.21%). Five successful candidate episodes share the failure's
first197 commands;no simple command-level cause is established. All96 rows,
22 diagnostic/292 active Breakout inputs and originalPID812861/invocation stay.
Analysis1.719s completes16:06:19 UTC November1/sourcec197078,zero replay/native/
GPU work,updates,failures or repeats. Retain CPU co-host time;no new rate,loss
or learner selected. Root`/x/Temp/kindle-qbert-behavior-phases-20261101`;complete,
do not rerun. The Breakout comparison above subsequently completes.

The [completed Qbert behavior-rate comparison](docs/results/2026-11-01-cdp-qbert-behavior8e5.md) finishes200k actions/
799,828frames/18,715updates in20m52.376s,26.92% less training time than control.
Greedy14,358.333/23 first pyramids versus14,339.583/24 is nearly unchanged;
sampled14,173.958/24 versus13,761.458/24 rises3.00%. Both means still miss15k.
Retain the qualified faster teacher;doubled actor/critic rate does not establish
clear quality improvement or a new default. Acting/feedback falls566.324→184.264s,
accounting for82.83% of total time reduction;this includes Python teacher work,
not an isolated native speed measurement. All524 inputs,357 matching/invariant
tensors,paired initial traces,retained100k/9,340 four-file recovery,full replays/
videos,guards,curves,coverage/exposure and original collector pass. All five
services succeed:main15:45:24/post15:45:34/collection15:46:07 UTC November1,
source6089e8b. No failed or repeated GPU stage;CPU co-host caveats remain.
Root`/x/Temp/kindle-qbert-behavior8e5-20261101/full`;complete,do not rerun.
No Qbert extension,rate sweep or confirmation selected;the goal is unfinished.

The [completed Breakout B8/T32 bounded check](docs/results/2026-11-01-cdp-breakout-context32.md) passes8,192 actions/
32,730frames/481updates in53.799s;1,024 frozen/.870s haszero updates/no natural
episodes. Conservative200k projection24m04.805s passes30min. All261 inputs,357
matching initial/invariant frozen tensors,47 common-prefix events,481 finite
supervised reports,full pixel/action/label replay and both host guards pass.
Main15:52:52/original collection15:53:01 UTC November1/sourcec2f7abf;no failure,
GPU repeat,cap change or native edit. Longer gradients also change diversity,
target rows and warmup;cached recurrent history already exists. Native447fe50/
runner6c1b69fb/teacher stay fixed. Root`/x/Temp/kindle-breakout-context32-20261101/native-check`.
Complete,do not rerun. No full learner or automatic successor selected;the host
is idle at closure. Gameplay benefit remains unmeasured;the goal is unfinished.

The [qualified Python Qbert teacher](docs/results/2026-11-01-cdp-qbert-teacher-cost.md)
uses equivalent channel predicates. All130 CPU tests,472 exact snapshot decisions/
states and a full200k/799,841-frame candidate replay pass. CPU replay takes230.331s
versus retained633.898s;whole-training timing is reported in the completed comparison above. All25 declared/506
completed H3 inputs stay unchanged. Replay15:01:16/packaging15:03:33 UTC November1;
no GPU work,learning or replay failure/retry. Original CPU harness failure remains.
Prefer`/x/Temp/kindle-qbert-color-masks-20261101/runtime` for future Qbert trials,
explicitly pinning its teacherSHAfa261c53. Same native447fe50/SHA35ad0db;only teacher
and distribution RECORD change among16 package files. Old runtimes remain intact.
Do not repeat completed qualification/replay/packaging. No learner is selected
or GPU allocation active;the five-game goal remains unfinished.

The [fresh-cohort Breakout centering reference](docs/results/2026-11-01-cdp-breakout-unseen-center-fresh.md)
fails the fixed screen:209.833→212.042 (+1.05%),four of eight lane means improve,
both0/24 two-wall. Keep the native teacher;no integration,parameter retry or
cohort sweep. All31 declared/506 active Qbert inputs,original Qbert identity and
full exact replays/videos pass. CPU-only135.931s completes14:26:31 UTC November1/
sourceb7edcf8,zero updates/failures/repeats;retain co-host timing. Root
`/x/Temp/kindle-breakout-unseen-center-fresh-20261101`;complete,do not rerun.
The Qbert H3 comparison below also closes;check live state before GPU work.

The [completed Qbert H3/R24 comparison](docs/results/2026-11-01-cdp-qbert-h3-r24.md)
finishes200k/799,802frames/18,715updates in26m41s,6.56% less time than H5.
Greedy13,823.958/sample11,277.083 trail14,339.583/13,761.458;all trained arms
24/24 first pyramids but miss15k. Retain H5 for stronger absolute primary score;
H3 is a cheaper development point with slightly higher greedy return/minute.
All506 inputs,357 matching/invariant tensors,retained100k recovery,paired initial
traces,full replays/videos,guards,curves,coverage/exposure and original collector
pass. Main/post/collection succeed14:53:25/14:53:42/14:53:44 UTC November1/source
df3a75e;all five original services terminal,zero original/GPU failures or repeats.
Root`/x/Temp/kindle-qbert-h3-r24-20261101/full`. Do not rerun;no extension,horizon
sweep,confirmation or automatic successor. No GPU allocation remains active.

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
