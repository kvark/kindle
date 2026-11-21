# Fresh Breakout seed confirms learning, with lower scores and material cohort variation

The [fixed protocol](../experiments/2026-11-06-cdp-breakout-seed22003.md) selects
one fresh200k learner at seed22003 and a new shared evaluation cohort4000022003.
The strongest retained recipe was selected on learner1009 and repeatedly
assessed on its original cohort. Its400k extension uses an identical training
prefix and provides no independent learner confirmation. This test changes
only the learner seed, with all consequent training environment, sampling and
guidance randomness disclosed. It introduces no new recipe.

Retained native447fe50/SHA35ad0db, runnercf4ea238 and the exact original runtime
and teacher are reused. Actual-initial controls and four frozen arms for each
of the new and retained policies separate evaluation-cohort movement from the
combined new training-seed effect. The original864/90% two-wall gate remains;
positive learning or a new best development seed alone cannot complete the goal.
The one fresh learner and all eight frozen arms are complete. No second learner,
extension or automatic successor is selected. Useful learning appears on this
new seed; the retained score level and original task gate are not confirmed.

## Frozen results and interpretation

Each arm selects the first three natural episodes from each of eight independent
lanes:24 episodes, guidance/labels/learning off, no selected truncations and357
unchanged tensors. Both learners use new evaluation cohort4000022003. Training
seed1009 is the retained checkpoint; its original training cost is reused.

| Learner / evaluation cohort | Greedy final / initial | Sampled final / initial | Training time | Greedy / sampled gain per minute |
|---|---:|---:|---:|---:|
| Fresh22003 / new4000022003 |75.250 /2.292|44.708 /1.875|21m30.822s|3.391 /1.991|
| Retained1009 / new4000022003 |94.625 /2.375|70.500 /1.208|21m27.910s|4.298 /3.228|
| Historical1009 / old4000001009 |143.667 /2.208|70.750 /1.417|21m27.910s|6.590 /3.230|

All final arms have0/24 two-wall successes. The historical row is context, not
the matched-cohort control. Fresh training takes.226% longer.

Fresh final-minus-own-initial gains are positive on all eight lanes in both
modes:greedy72.958, paired95% interval[54.792,93.125]; sampled42.833,
[35.000,50.500]. This confirms useful learning on one fresh seed. It does not
confirm the selected baseline's score level, broad reliability or task mastery.

Against retained1009 on the same new cohort, fresh greedy final score is19.375
lower, paired interval[−38.667,−.417]. Its own-initial gain/min is21.091% lower,
difference−.906408 with interval[−1.850482,.009363], improving3/8 lanes. The
primary efficiency interval narrowly includes zero. Sampled score is25.792
lower, interval[−45.167,−7.708]; gain/min is38.323% lower, difference−1.237117,
interval[−2.143972,−.412610], improving1/8 lanes. These are10,000-resample,
seed0, eight-lane bootstrap intervals, not uncertainty across training seeds.

Retained1009 greedy falls143.667→94.625 on the new cohort (−34.136%), while
sampled changes70.750→70.500. Cohort variation therefore explains part of the
gap from the historical greedy point. The shared-cohort comparison still favors
the retained learner. Changing the training seed also changes initialization,
environment trajectories, policy sampling and guidance selection; their causes
are not isolated. One fresh seed is insufficient for a reliability rate.

Keep the retained recipe and original checkpoints, with these limitations
attached to the selected historical score. No new seed is promoted. The original
mean864/90% two-wall gate and the five-game goal remain unfinished.

## Training cost and coverage

Fresh200k aggregate actions execute799,616 emulator frames and12,477 updates in
1290.821634s. First update remains376; all weighted losses and required CNN,
reward, continuation, actor and critic parameter changes pass. The100k/6,227
recovery retains357 finite arrays and optimizer state with read-only files;
independent verification takes.412486s while the original learner continues.
It does not contain replay, games, live beliefs or full RNG state.

Learning takes1147.017s (88.859% of training time), environment stepping62.533s
(4.844%). Within learner timers, imagination takes450.524s, world training
357.969s and posterior processing185.912s. These wrapper timers describe this
run, not isolated kernel costs. More useful policy improvement per update
remains more relevant than emulator throughput alone.

Teacher overrides cover78,075 actions (39.038%);49 of267 started episodes are
selected for guidance. All-state labels supply2,989,752 repeated replay rows.
The run reaches maximum cumulative score405, including partial tails; no state
reaches432. Only1,577 native actions are in pre-action score216–431, alongside
5,249 teacher actions. Coverage and guided online returns are descriptive and
do not establish frozen competence or a causal explanation of the seed gap.

## Preserved preparation failure

Preparation initially fails copying an archived replay helper from a stale path.
`preparation-failure.json` and `failed-preparation-source.py.txt` retain the error
and exact source. `continue_preparation.py` resumes at that copy using the existing
SHAf40da642 helper under `/x/Temp/kindle-breakout-cdp1000-20261106/full`.
Earlier copied controller helpers and runner are verified without recopying;
no native/emulator construction or completed experiment stage repeats. This is
a separate preparation repair, not a relabeling of the failed original script.

Root `/x/Temp/kindle-breakout-seed22003-20261106/full` retains scripts, receipts,
queues and subsequent evidence. Logs, weights and videos remain outside Git.

## Execution and independent closure

All nine actual queued command paths pass in1.745132s (wrapper1.817601s),
including native/emulator construction guards, actual environment seed lists,
six corrupt-header cases, three wrong-training-command cases, two incorrect
weighted losses and the complete comparison entry with synthetic outcomes.
There are90 active pins plus338 verified retained input pins. Initial/final
retained schemas are checked without native construction.

Execution sourcedcc4dd5d9a89a44cfecca9deaff1643e128c2d0c launches20:36:28.653 UTC
November6, controller1546932/invocation316ee06c85004cb68a061364822fdc3b. Startup
passes at20:36:34.266:357 finite initial tensors,47 pre-update transitions and
first update376. The new seed changes2,400 encoder first-layer weights and
655,360 actor first-layer weights relative to1009; exact initial equality is
not claimed. Config and header differ only in declared seed fields and metadata.
Training finishes20:58:03.649 UTC; the full pixel/action/label/override/reward/
frame/boundary replay passes. The controller completes all eight frozen arms,
their recorded-action replays and videos, coverage, exposure and curves, then
exits successfully at21:03:54 UTC. Frozen logs omit live pixel hashes; their
replays do not establish unrecorded image equality.

Independent closure finishes21:04:53.361 UTC November6 in2.299634s
(wrapper2.346980s). All90 active/338 retained pins,69 completion artifacts,
52 checkpoint files and288 candidate/new-reference/historical selected episodes
pass. Own-initial and paired intervals, cohorts, task gates and checkpoint
invariance are independently recounted. All nine original GPU children exit0;
main/recovery/terminal services are inactive/successful, with no render holder.
No native stage, learner update, evaluation or collector is repeated. The
original preparation copy failure remains failed and preserved separately.

Execution sourcedcc4dd5d9a89a44cfecca9deaff1643e128c2d0c; retained native447fe50 /
SHA35ad0db and runnercf4ea238. Key receipts are `comparison.json`,
`publication-closure.json`, `closure-execution.json`, `terminal-summary.json`,
`recovery-verification.json` and `candidate/learning-review.json` under the root
above. All useful checkpoints, curves, logs and videos remain outside Git.
