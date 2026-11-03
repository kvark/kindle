# CDP machine handover

Read [AGENTS.md](../AGENTS.md) for working constraints, the
[roadmap](kindle_single_life_dreamer_plan.md) for retained results and gates,
and [PR31](https://github.com/kvark/kindle/pull/31) for the live dashboard.
The five-game goal remains unfinished. Detailed experiment history belongs in
those reports; historical “next” paragraphs do not authorize restarting jobs.

## Current work — November 3

The [fresh current-critic B8/T32 Breakout comparison](experiments/2026-11-03-cdp-breakout-value1-context32-200k.md) is **running**,
launched21:30:20 UTC November3/source06c873c2 with122 pins. Fixed200k actions /
12,469 updates /30min,12M64/Auto/N8/B8/T32/H15/R16,behavior8e-5/replay-value1/
expert100,full18,native447fe50/runnercf4ea238. Startup passes357 initial tensors
against both bounded andretained controls,63 bounded/47 shared pre-update events,
andfirst update504. MainPID1071673/invocation`2cecc3ab00f9498795197cf1de25639a`;
the terminal observer remains active. Read-only 100k/6,219 recovery passes
21:42:21.105 UTC: all 357 finite tensors and 122 pins, original learner identity
unchanged; independent CPU verification takes .478s. Four final/actual-initial
24-natural-episode arms remain pending; greedy primary, sampled secondary,
against retained143.667/70.750. No full result is available. Root
`/x/Temp/kindle-breakout-value1-context32-20261103/full`;service
`kindle-breakout-value1-context32-200k-20261103`. Check live state;do not restart
oroverlap this learner. Bounded CPU failures remain preserved. No extension or
automatic successor;the original864/90%-two-wall andfive-game goal remain unfinished.

The [current-critic Breakout B8/T32 check](results/2026-11-03-cdp-breakout-value1-context32.md) passes8192 actions /32,722 frames /
481 updates in51.909s; padded200k projection23m14.028s clears30min. Frozen1024/
.875s haszero updates/no natural episodes. All357 initial/invariant tensors,
47 shared prefix events,full pixel/action/label replay andboth guards pass.
Native447fe50/runnercf4ea238 remain unchanged; behavior8e-5/replay-value1 fixed.
Original main/collector fail21:16:10 UTC November3/source9de712d1 on the replay
manifest path. Separate9.605s CPU repair closes21:19:14.395,readback21:20:20.243,
with67 original/7 repair pins and39 outputs unchanged; no GPU orsuccessful replay
repeats. Earlier CPU schema-fixture failure remains preserved. Closure is
`complete_with_cpu_audit_repair` under`/x/Temp/kindle-breakout-value1-context32-20261103/cpu-repair`.
No full learner orautomatic successor isselected; no allocation remains active.
Retain70.750/143.667; gameplay benefit andthe five-game goal remain unfinished.

The [completed four-action Breakout comparison](results/2026-11-03-cdp-breakout-minimal-guided.md#full-200k-comparison) finishes
200k actions /799,580 frames /12,477 updates in **18m36.686s**,13.29% less training
time than full18. Sampled **55.958** andgreedy **91.500** fall20.91%/36.31% from
retained70.750/143.667; all0/24 two-wall. Own initials1.250/4.667 differ fromfull18.
Greedy initial-subtracted gain/min falls29.20%,with paired difference−1.925
[−3.907,−.192]. Both primary replacement andcheaper-budget selection rules fail.
**Retain full18 andthe existing200k recipe.** This is one reused development
seed/cohort,not a training-seed reliability orisolated mechanism result.

All169 current/338 retained pins,read-only100k/6,227 recovery,357 bounded-matching
initial/invariant frozen tensors,47 pre-update events,full pixel/action/label
replays/videos,guards,curves,exposure andcomparison pass. Actual overrides42.116%;
late expert CE.567343 andonly1,222 native actions at scores216–431 do not establish
a causal limitation. No training action reaches432. Main succeeds20:59:55.332 UTC
November3/source1917a4de; terminal collector succeeds21:00:26.026 in2.408s.
All three services succeed; no full-run stage fails orrepeats. Earlier bounded
preflight andCPU collector failures remain preserved. Native447fe50 unchanged;
root`/x/Temp/kindle-breakout-minimal-guided-20261103/full` isclosed. Do not restart
it orrepeat its observers. No extension,confirmation orautomatic successor is
selected. No Kindle allocation remains active; check live state before work.
The original864/90%-two-wall gate andfive-game goal remain unfinished.

The [four-action guided Breakout check](results/2026-11-03-cdp-breakout-minimal-guided.md) completes8192 actions /
32,728 frames /489 updates in **41.133s versus47.690s**,13.75% less early training
time. Including separate construction gives44.072s versus50.094s (−12.02%).
All118 inputs,357 finite/invariant tensors,correct cadence,full training pixel/
action/label replay andboth guards pass. Frozen1024/.918s haszero updates and
five natural episodes; no24-episode gameplay gain is established. Native447fe50
is unchanged; explicit adapter mapping preserves the full18 teacher internally.

Main completes20:21:17.038 UTC November3/source11023a19; terminal closure at
20:22:44.473 is `complete_with_cpu_terminal_repair`. The original preflight
accidentally initializes a native agent on synthetic pixels before failing with
zero emulator steps/updates; its unguarded work andcheckpoint remain preserved.
Repaired CPU qualification passes161 tests/zero agent constructions; source
qualification passes258 tests. A later CPU collector failure on cleared systemd
InvocationID is preserved and repaired using the original journal identity.
No declared learner/frozen/replay stage repeats. Root
`/x/Temp/kindle-breakout-minimal-guided-20261103/bounded-check` is closed; do not
rerun it. No active allocation orfull learner isselected;18m24.645s projected
cost clears30min,but competence and the original864/90%-two-wall gate remain
unmeasured/unfinished. Retain70.750/143.667 andthe existing five-game goal.

The [completed life-bounded actor comparison](results/2026-11-03-cdp-life-return-actor.md#full-200k-comparison-does-not-improve-the-retained-recipe)
finishes 200k actions / 799,589 frames / 12,477 updates in **21m41.269s** (+1.04%).
Sampled **62.417** and greedy **137.542** fall from retained70.750/143.667 by
11.78%/4.26%; all0/24 two-wall. Own initials1.417/2.208 match. Greedy gain/min
falls5.31%, with paired difference−.350 [−2.534,2.012]. The declared selection
rule fails; the interval spans zero and does not establish reliable harm.
**Retain the existing200k recipe.** Against the contextual whole-game actor,
sampled rises26.95% but primary greedy falls4.98%; that comparison also fails.

All330 current/338 retained pins, 100k/6,227 recovery, 357 initial/frozen tensors,
47 pre-update and799/794 actual-initial semantic events, full replays/videos,
guards, curves, exposure and comparisons pass. Independent replay verifies
1,288 raw life losses, including1,032 nonterminal boundaries. The loss has
613,108 positive repeated rows on11,932 updates. Late expert CE.543307 improves
without a frozen gain; native exposure at scores216–431 is only1,201 actions,
with none at432+. These are descriptive counts, not causal architecture evidence.

Original main completes **19:50:54.540 UTC November3**, source`e6e5f4ba`, and
terminal collector completes **19:51:02.459** in3.231s. Main, recovery and collector
all succeed; no full-run failure, repair, GPU/replay repeat or cap change occurs.
The preceding bounded CPU collection failure/repair remain preserved. Root
`/x/Temp/kindle-life-return-actor-20261103/full` is closed; do not restart the
learner or repeat its observers. The optional mode remains default-off in
qualified native`ec12d41e`.
No extension, coefficient sweep, confirmation or automatic successor is selected.
No Kindle allocation remains active at closure; check live state before work.
The original864/90%-two-wall gate and five-game goal remain unfinished.

The [optional life-bounded actor qualification](results/2026-11-03-cdp-life-return-actor.md)
passes 8,192 actions / 32,725 frames / 489 updates in **49.039s**, versus the
retained whole-game short check's 50.754s. Frozen 1,024 actions take .903s with
zero updates, no completed natural episodes and all 357 tensors unchanged.
All 123 raw-frame life-loss boundaries reproduce in independent pixel/action/
label replay. Targets become positive at action 568; 34,261 positive repeated
rows appear on 469 updates. These are execution results, not gameplay gains.

Default-off native `ec12d41e` passes 174 Rust / 1,044 packaged Python CPU tests,
strict Clippy and 12 F64 derivative cases. All 263 current / 338 retained pins,
214 preserved sources, 357 matched initial tensors and 47 pre-update events pass.
The original main service succeeds 19:15:26.762 UTC November 3. Its CPU collector
fails on a list/dictionary manifest mismatch; a separate 1.213s repair closes
19:16:56.676 as `complete_with_cpu_collection_repair`, preserving that failure.
No GPU or replay repeats. Runtime and evidence remain under
`/x/Temp/kindle-life-return-actor-20261103`; bounded subdirectory `check` is closed.

The padded 200k projection is 21m56.964s, below 30 minutes. No full learner,
extension or coefficient sweep is selected by this qualification. Keep retained
sampled70.750/greedy143.667 and the original 864/90%-two-wall gate. No Kindle
allocation remains active at closure; check live state before new work.
The five-game goal remains unfinished.

The [completed return/life audit](results/2026-11-03-cdp-breakout-return-life-audit.md)
finds390→143 positive targets among480 fixed retained states when returns stop
at the next life loss;70.74% of clipped weight disappears. Within16 actions of
loss,30 positives becomezero,but that subset is only3.49% of original weight.
Full-game returns remain valid for game-total reward; this arithmetic does not
identify causal errors or prove that a life-bounded target would help. These are
retained frozen states,not the conservative learner's online batches. At that
audit's closure, no native change or learner was selected; the later optional
prototype is reported above.

All252 pins,16 CPU fixtures,35,552 actions/142,169-frame annotation replay,
eight pixel-sequence hashes,512 selected pixels,480 original returns/weights
and independent recount pass.142 raw life losses,zero gains; one historical
file resolves exactly fromGit. Replay15.018s/analysis.639s/recount.273s close
18:52:33.129 UTC November3/source0dc11ba4. No failure,GPU work,model forward,
update or prior-stage repeat. Root`/x/Temp/kindle-breakout-return-life-audit-20261103`
is closed; do not rerun. Retain70.750/143.667 and the original task gate;
the five-game goal remains unfinished. Check live state before new work.

The [completed200k conservative-actor comparison](results/2026-11-03-cdp-conservative-actor.md)
finishes200k actions /799,546 frames /12,477 updates in21m46.724s (+1.46%).
Greedy144.750 versus143.667 is nearly flat; sampled49.167 versus70.750 falls30.51%.
All0/24 two-wall; own initials1.417/2.208 match. Greedy initial-subtracted score/min
falls.68%,paired lane difference−.045 [−2.574,2.589]. The declared selection rule
fails: retain the existing200k recipe. One reused seed/cohort and different
native/runner revisions do not establish an isolated effect or reliability.
The original864/90%-two-wall gate and five-game goal remain unfinished.

The loss has1,010,324 positive repeated rows on12,403 updates, with bonuses active.
Final50k teacher CE.559403 versus.560514 does not yield better frozen efficiency.
Training max377 and only939 native actions at216–431 show limited later-score
exposure; no432+ action occurs. These are score bands,not verified wall events
or causal evidence. Actual overrides46.429%; guided returns are not competence.

All308 current/338 retained pins,100k/6,227 recovery,357 initial/frozen tensors,
47 pre-update and799/794 initial semantic events,replays/videos,guards,curves,
exposure and comparison pass. Main completes18:38:09.815 UTC November3/sourcecf8c53fc;
original collector completes18:38:40.403. Both services succeed,all five GPU
stages run once; no full-pipeline failure or repair. Earlier prelaunch copy-script
failure remains. Root`/x/Temp/kindle-conservative-actor-20261103/full` is closed;
no restart,extension,confirmation or automatic successor. No Kindle allocation
is active at closure; check live state before separately selected work.

The [completed bounded check](results/2026-11-03-cdp-conservative-actor.md) passes8192/489 in47.402s control/50.754s
candidate (+7.07%),with33,588 positive targets and the bonus active. Padded200k
projection22m43s passes30min; frozen1024 has zero updates/no natural episodes.
Native3f6c6648 passes1018 Python/169 Rust CPU checks,Clippy and12 F64 cases.
All258 current/338 retained pins,215 preserved sources,357 initial/frozen tensors,
replays and five GPU stages pass. A CPU comparison matches357 final control
arrays/1065 events to the old short control. Closed17:58:41.844 UTC November3;
CPU fixture/path failures remain,no GPU repeats. Bounded root`check` is closed.
Actual full CPU postprocessing and queued-config preflights pass before launch;
one private copy-script assertion is preserved. See the result for evidence.

The [completed-return actor pair](results/2026-11-03-cdp-episodic-actor.md) closes with a useful bonus-off
result:sampled46.458→56.542 (+21.70%),greedy65.208→95.375 (+46.26%),all0/24
two-wall,in17m05.486s→17m12.959s (+.729%) for200k actions/12,477 updates each.
Greedy lane-bootstrap gain30.167 [2.167,60.001],gain/min1.726 [.095,3.462];sampled
intervals also exceedzero. One reused seed/cohort,not training-seed reliability.
Retained action-effects1 remains70.750/143.667 in21m27.910s;candidate costs19.80%
less but has lower score and gain/min point estimates. Keep the retained primary
recipe and this bonus-off development result; no extension,third learner or
automatic confirmation. Original864/90%-two-wall gate remains unmet; the
five-game goal is unfinished.

Both100k/6,227 recoveries,261 matching initial/invariant frozen tensors,47
pre-update events,799/794 initial semantic events,full pixel/action/label and
frozen replays/videos,guards,curves,exposure and comparison pass. Candidate
positive targets appear on12406 updates;1167144 positive replay rows are repeated
exposures. Neither learner reaches432 training points;maxima399/415. Guided
returns and replay fit are not competence. Same nativee3b02163/runner/teacher;
only actor scale100 differs within the bonus-off pair.

All307 original/22 first-repair/10 closure pins,69 control/154 candidate files
and338 retained pins pass. Original main fails16:45:24 UTC November3 on CPU
exposure count/schema; its collector also fails. Separate3.458s repair completes
16:49:21.889. Unstarted candidate launches16:51:24,finishes training17:08:43,then
its pipeline fails17:12:50 on CPU initial elapsed-time equality; its collector
also fails. Both pipeline failures remain;
allten GPU phases succeed,zero learner/frozen/replay repeats. Separate CPU
comparison/closure completes17:19:25.280; independent2.865s verification completes
**17:19:38.264**,source28b03632,status`complete_with_cpu_postprocess_repair`.
All four original services remain failed; no original passing completion exists.
Root `/x/Temp/kindle-episodic-actor-20261103/full` is closed; do not rerun it.
No Kindle GPU allocation or successor is active. Check live state before work.

The [closed bounded pair](results/2026-11-03-cdp-episodic-actor.md)
passed8192 actions/489 updates in36.677s control/38.277s candidate (+4.36%),
with33105 positive replay rows and no natural frozen episodes. All native/CPU,
legacy, tensor, replay and terminal checks pass; no failed or repeated stage.
Native e3b02163/SHA d406629f adds no parameters; disabling disagreement removes96
ensemble tensors, leaving261. Runtime,wheels,validation binary and1.008GB cache
remain. Bounded root `check` is closed; never reuse its experience or rerun it.

The [CPU replay-actor target audit](results/2026-11-03-cdp-breakout-replay-actor-targets.md) confirms no direct
return-weighted actor update on executed replay actions. Reusing existing critic
targets gives241 positive states of480; removing short-window external rewards
leaves200 and83.32% of normalized positive weight. Imagined bootstraps and
intrinsic bonuses stay fixed; this is arithmetic dependence,not a causal learning
fraction. No gameplay benefit or new loss is selected. Completed real-outcome
target accounting remains the next design question.

All17 fixtures,20 current/158 original pins,512 incoming and480 teacher alignments
pass. Approximate value recovery matches final native normalizer state to2.339e-7;
fixed sensitivity leaves the broad result unchanged. Analysis completes15:42:20.331
UTC November3/source934053ae in.436s CPU; independent recount15:43:34.993 in.202s.
No failure,model forward,emulator,GPU work,update or prior-stage repeat. Root
`/x/Temp/kindle-breakout-replay-actor-targets-20261103` is closed; do not rerun.
No Kindle service is active at closure. Keep the retained200k/400k budget points;
Breakout and the five-game goal remain unfinished. Check live state before work.

The [completed400k/R16 Breakout comparison](results/2026-11-03-cdp-breakout-budget400k.md)
raises sampled70.750→92.792 (+31.15%) andgreedy143.667→173.708 (+20.91%),all0/24
two-wall,in43m36.578s (+103.16%) for400k actions/1,599,381 frames/24,977 updates.
Initial-subtracted score/min falls35.13%/40.33%. Retain200k as the cheaper baseline
and400k as a higher-score budget point; no extension,confirmation or automatic
successor is selected. Final50k teacher CE falls.560514→.411859; guided116.0
is not unaided competence. Training maximum408/zero432+ and only2,397 native
actions at216–431 (0.59925% of all actions) show limited later-score exposure,
not verified wall boundaries or a causal diagnosis.

All446 pins,three recoveries,357 matching initial/invariant frozen tensors,
38,050 exact common200k semantic events,initial800/795 traces,full replays/videos,
guards,curves,exposure,coverage and comparison pass. Initial/100k/200k tensors
and canonical content match the retained control; full live replay/RNG equality
is not asserted. Same12M64/Auto/N8/B16/T16/H15/R16/context1,rates/teacher/expert100/
critic1/1,native447fe50/runnercf4ea238; fresh life but reused1009/cohort.

Original mainPID1034231/invocation13c5fb9a8be543deb2c279522c8498a2 and all observers
succeed; main/post/collection finish15:21:22.175/15:21:33.374/15:22:02.467 UTC
November3/source4bb0d101. Independent terminal reading and repair closure pass
once15:22:18.072 in2.689s (2.708s wrapper). Closure`complete_with_cpu_launch_repair`
preserves the original serialized-file-hash launch-wrapper failure; six-case CPU
repair retains raw integrity and exact header/payload checks. All11 repair pins
pass. No learner restart,native change or successful GPU/reader/collector repeat.
CPU co-host costs remain uncorrected. Root`/x/Temp/kindle-breakout-budget400k-20261103`
is closed; no Kindle GPU allocation remains active. Check live state before
separately selected work. Breakout and the five-game goal remain unfinished.

The [completed frozen halfway assessment](results/2026-11-03-cdp-breakout-halfway.md)
finds **22.125→70.750 sampled /33.250→143.667 greedy** from100k to200k; all eight
lane means improve in both modes. Paired lane intervals are[36.333,61.042] and
[80.292,145.875]. Initial-subtracted score/min rises1.894–1.974→3.230 sampled
and2.839–2.959→6.590 greedy. Halfway time is bracketed10m29.382s–10m55.956s;
final21m27.910s remains unchanged. All0/24 two-wall. Two checkpoints establish
progress over that interval,not a slope at200k,optimal budget or future success.

Both new frozen arms total51,048 actions/204,112 frames/52.534s,zero updates.
All413 pins,357 finite/invariant tensors per arm,full replays/videos,guards and
paired comparison pass; original final/initial evaluations are reused without
another execution.17 fixtures/two actual CLI paths pass in3.369s preflight.
ServicePID1032697/invocationc47bd45edffe4af5871bef32e1714dac succeeds
14:03:53.486 UTC November3/sourceca6bc06e; fixed terminal collection succeeds
14:04:28.611 in1.015s. No failure,build,training or prior-stage repeat.
Root `/x/Temp/kindle-breakout-halfway-20261103` is closed; do not repeat its
stages. The original200k evaluation rule is unchanged. Reused learner/cohort
limits reliability; Breakout and the five-game goal remain unfinished.

The [completed context9 comparison](results/2026-11-03-cdp-breakout-context9.md)
lowers sampled **70.750→46.583** and greedy **143.667→91.167**, all **0/24
two-wall**, in **23m30.959s (+9.55%)** for200k actions/799,606 frames/12,473
updates. Initial-subtracted score/min falls40.54%/42.60%. Retain context1 and
constant expert100/critic1/1. Prefix recomputation totals101.886s,8.169ms/update
and7.22% of training wall. Late CE.560514→.559628 barely changes; guided online
67.78→72.46 is not unaided competence. No improved state accuracy is established.

All667 pins,100k/6,223 recovery,357 matching initial/invariant frozen tensors,
47 common pre-update and800/795 initial frozen events,full replays/videos,guards,
curves,exposure and comparison pass. All five original services succeed;
main/post/collection finish **13:48:19.732 /13:48:49.527 /13:48:54.418 UTC
November3**, source55f0a46a. Fixed independent terminal reading passes once
**13:49:17.338** in1.766s wrapper. No full-run stage fails or repeats. An ad-hoc
poll rejects the unloaded recovery unit's empty invocation; original journal
identity and success pass. Earlier bounded failures/repairs remain preserved.

Context9 adds eight forward-only observed steps before16 learning steps with
12M64/Auto/N8/B16/T16/H15/R16,retained rates/teacher/constant expert100/critic1/1.
Sampling,warmup,total work and native/runner differ from447fe50; no isolated
cache-age cause or independent reliability claim. Native65dd69b4/runtime and
bounded runner are reused without a build. Keep CPU co-host times uncorrected.
Root `/x/Temp/kindle-replay-context-20261103/full` and all readers are closed;
do not repeat stages or readers. No Kindle allocation remains active at closure.
Check live state before GPU work. No extension,sweep,confirmation or automatic
successor is selected. Breakout and the five-game goal remain unfinished.

The [optional replay-state recomputation check](results/2026-11-03-cdp-replay-context.md)
passes8,192 actions/32,730 frames/485 updates in50.564s. Context9 adds eight
forward-only steps before16 learning steps; no new parameters or prefix targets.
Prefix time is3.234s/6.668ms per update. Conservative200k projection22m37.921s
passes30min; gameplay benefit remains unmeasured. First update440,all485 finite
reports,357 matching initials/invariant frozen tensors,47 common pre-update
and128 legacy events,full pixel/action/label replay and guards pass. Frozen1024
in.914s haszero updates/no natural episodes. Keep the70.750/143.667 control.

Native65dd69b4/SHA7b15cb69 passes155+5 Rust/983 packaged Python CPU checks and
strict Clippy. CPU target-shape/unused-helper failures and a failed RGB fixture
remain. Test-only694c64bd passes156+5 CPU checks and the repaired device fixture;
production/runtime unchanged,passing posterior fixture not repeated. All235
native pins/guards pass. Original bounded service fails13:08:10 UTC November3
only on its stale489-vs-declared485 queue assertion; the unstarted frozen stage
continues once. A separate CPU repair handles omitted decay-only scale fields.
All601 original/631 repair pins pass; collection completes13:11:47.928 in2.590s,
closure`complete_with_cpu_controller_repair`. Original service remains failed;
no training/frozen/replay stage repeats. Runtime/source differs447fe50,not an
isolated cause. Root `/x/Temp/kindle-replay-context-20261103` and readers are
closed. No full learner was selected at that bounded closure; the separately
selected full comparison is completed above. The five-game goal remains unfinished.

The [CPU replay-context age reconstruction](results/2026-11-03-cdp-breakout-replay-age.md)
matches all12,477 logged updates and199,632 sampled starts in.387s CPU.
Final50k unmasked contexts have median246/mean365.099/p99 1,776 updates since
their last computation;98.66% of labeled actor rows can depend on initial
context before a reset. This measures age,not latent error or a gameplay cause.
All15 fixtures and27 diagnostic/338 control pins pass; review completes
12:35:19.800 UTC November3/sourceb729ab23. No GPU work,learner,emulator,failure
or prior stage repeat. Root `/x/Temp/kindle-breakout-replay-age-20261103` is
closed. Earlier replay16k and B8/T32 results remain; this age audit alone
did not select a new buffer,prefix,loss or learner. Retain the70.750/143.667 control; goal unfinished.

The [completed Breakout H3/scale1 comparison](results/2026-11-03-cdp-breakout-open-loop.md)
lowers frozen sampled **70.750→60.875** and greedy **143.667→104.625**, all
**0/24 two-wall**, in **22m06.821s (+3.02%)** for 200k actions / 799,552 frames /
12,477 updates. Initial-subtracted score/min falls **16.76% / 29.72%**.
Retain the control without H3, constant actor/world imitation100 and critic1/1.
Late replay expert CE rises .560514→.648492; guided online67.78→84.74 is not
unaided competence. No improved forecasts or isolated loss mechanism is proven.

All644 pins,100k/6,227 recovery,357 matching initial/invariant frozen tensors,
47 pre-update events,800/795 initial frozen events,finite weighted auxiliary
reports,full replays/videos,guards,curves,exposure and comparison pass. All five
original services succeed; main/post/collection finish **12:16:22.850 /
12:16:49.658 /12:16:54.736 UTC November3**, sourceffc4a26d. Independent terminal
reading passes once **12:17:22.865** in1.404s wrapper. No full-run failure or
GPU/replay/collector/reader repeat. Earlier bounded CPU collection failure and
repair remain preserved. Keep CPU co-host timings without correction.

Nativec2d627f3/runner8946f5f3 differ from retained447fe50/cf4ea238; the explicit
imagined-bootstrap default is normalized. Reused1009/cohort does not establish
an isolated cause or independent reliability. Root
`/x/Temp/kindle-breakout-open-loop-20261103/full` and all its readers are closed.
No allocation remains active at closure; check live state before GPU work.
No extension,sweep,confirmation or automatic successor is selected.
Breakout and the five-game goal remain unfinished.

The [Breakout H3 world-supervision check](results/2026-11-03-cdp-breakout-open-loop.md)
passes8,192 actions/32,717 frames/489 updates in48.603s; conservative200k
projection21m45.264s passes30min. Frozen1,024 actions/.893s haszero updates
and no natural episodes. All602 pins,357 matching initial/invariant frozen
tensors,47 pre-update events,finite weighted auxiliary reports,full guided
replay and guards pass. Keep actor/world imitation100; only H3/scale1 world
supervision is added. Nativec2d627f3/runner8946f5f3 are reused without a build
or repeated numerical/legacy capture; gameplay benefit remains unmeasured.

Original mainPID1019894/invocation7b942f6a912842eea2e3c49f11e24cc4 succeeds
11:41:09.152 UTC November3/source63c347f9. The original collector's parent-PID
logging assertion fails; separate CPU repair passes11:43:00.131 in1.065s with
six identity fixtures/seven repair pins. Original failure and all602 pins remain;
closure`complete_with_cpu_collection_repair`. No GPU,frozen or replay repeat.
Root `/x/Temp/kindle-breakout-open-loop-20261103/native-check` and its readers
are closed. No allocation is active at closure; check live state before work.
No full learner,extension,sweep or automatic successor is selected by this check.
The five-game goal remains unfinished.

The [completed actor-only imitation-decay comparison](results/2026-11-03-cdp-breakout-actor-only-decay.md)
lowers frozen sampled **70.750→51.833** and greedy **143.667→104.750**, all0/24
two-wall, in **21m42.074s** (+1.10%). Initial-subtracted score/min falls28.07%/
28.30%. Retain constant actor/world imitation100 and the70.750/143.667 control.
The actor holds100 through100k, fades to0 by150k and stays0 for3,126 reports;
world stays100. Guided online67.78→74.68 is not unaided competence.

All649 pins,100k recovery,357 matching initial/invariant frozen tensors,paired
initial traces,full replays/videos,guards,curves,exposure and comparison pass.
The fixed pre-fade checks match12,500 transitions,6,227 reports,170 episodes,
167 resets and all357 tensors/51,502,659 elements exactly at100k. Nativec2d627f3/
runner8946f5f3 still differ from447fe50; no full-state, broad runtime or isolated
causal claim. All five original services succeed; main/post/collection finish
11:28:15.248/11:28:35.619/11:28:40.720 UTC November3, source18d493e6. Independent
terminal reading passes once11:29:04.321 in1.512s wrapper. No full-run failure,
GPU repeat or collector/reader repeat. Earlier CPU preparation failures and
co-host timings remain preserved without time correction.

Root `/x/Temp/kindle-actor-only-decay-20261103/full` and all its readers are
closed; do not repeat them. No allocation remains active at closure; check live
state before GPU work. No extension,sweep,confirmation or automatic successor
is selected. Breakout and the five-game goal remain unfinished.

The [actor-only imitation-decay qualification](results/2026-11-03-cdp-actor-only-imitation-decay.md)
passes F64 loss/derivatives and fresh Breakout8,192 actions/32,725 frames/489
updates in47.329s. Actor scale100 reaches zero at256 (234 zero-weight reports);
world stays100. All599 pins,357 matching initial/invariant frozen tensors,47
pre-update events,full pixel/action/label replay and guards pass. Legacy1,024
frozen actions reproduce128 control events with357 invariant tensors; candidate
frozen1,024 actions take.879s,zero updates/no natural episodes. Conservative200k
projection21m11s passes30min. Execution/cost are qualified; gameplay is unmeasured.

Nativec2d627f3/SHA0015dff7 at `/x/Temp/kindle-actor-only-decay-20261103/runtime`
passes149 Kindle+5 gym CPU tests,971 packaged Python checks and strict Clippy.
An original Boolean-error-message fixture failure is preserved;5c6526fc repairs
only its expectation,with no rebuild/production change. Packaging closure is
`complete_with_cpu_test_repair`. Before GPU work,349b53ea clarifies the explicit
legacy `replay_value_bootstrap=imagined` default addition. Numerical.74s/228 pins
passes10:50:21.863 UTC; original bounded service succeeds10:52:27 November3,
PID1015127/invocationcc77dbd06adf48fc9a01aec6a179b5db/source5c6526fc.
Terminal10:52:49.664 takes1.144s. No GPU stage fails/repeats. All stages/readers
are closed; do not rerun them. ReviewSHAf9b8ca51/completionSHA5097159b.

The preceding [actor optimizer attribution](results/2026-11-03-cdp-breakout-actor-optimizer.md)
finds conditional imitation contributions7,493×/10,928× imagined score-function
on two fixed final-checkpoint batches; momentum also affects the full estimate.
All13 fixtures,210 current/158 prior pins pass in1.027s CPU,zero updates/GPUwork.
F64 estimates are not measured native movement or proof of a gameplay cause.
Its root and the earlier raw-gradient capture/backward readers are closed.

Retain policy1/expert100/critic1/1 and the70.750/143.667 gameplay control,all0/24
two-wall. The optional actor-only fade preserves world imitation; the earlier
shared fade weakened both on an older recipe and regressed. No full learner,
sweep,extension or automatic successor is selected by this qualification.
No Kindle GPU allocation is active at closure; check live state before work.
The five-game goal remains unfinished.

The [completed policy-weight100 comparison](results/2026-11-03-cdp-breakout-policy100.md)
lowers sampled **70.750→38.250** and greedy **143.667→92.125**, all0/24 two-wall,
in **21m49.372s** (+1.67% training time). Initial-subtracted score/min falls
47.75%/37.48%. Retain policy1, expert100, critic1/1 and actor/critic representation
gradients off. Only policy1→100 changes, scaling score-function and entropy
together; native447fe50 and teacher stay fixed, runner5a86e224 adds12 CLI lines.
Current actor-gradient dominance is measured in the diagnostics above; reused
seed/cohort limits reliability. Slightly lower replay expert CE does not establish gameplay benefit.

All436 pins,100k/6,227 recovery,357 matching initial/invariant frozen tensors,
47 pre-update events,800/795 initial semantic events,replays/videos,guards,curves,
exposure and comparison pass. Main/post/collection succeed **10:00:29.787 /
10:00:57.364 /10:01:01.652 UTC November3**, source0c40e657; allfive original services
succeed. Independent terminal/coverage readers complete once10:02:01.134/
10:05:11.888. All52 diagnostic/436 full pins reconcile273,544 actions/1,093,592
frames. Native training actions at raw scores216–431 fall1,262→1,196,none432+;
these descriptive bins do not identify wall boundaries or a causal limitation.
Actual guidance overrides are44.545%; guided online67.68 is not frozen competence.

Root `/x/Temp/kindle-breakout-policy100-20261103/full` is closed. Bounded,recovery,
terminal and coverage readers are complete; do not repeat stages or observers.
No full-run failure or GPU repeat occurs. Two earlier bounded CPU fixture failures
and the coverage-reader provenance-path failure/CPU repair remain preserved.
CompletionSHA38776056/terminalSHA26e0da3d/coverageSHAab72765d. Keep CPU co-host
costs uncorrected. No Kindle allocation remains active at closure; check live state.
No extension,coefficient sweep,confirmation or automatic successor is selected.
Breakout and the five-game goal remain unfinished.

The [completed contact-timing CPU teacher comparison](results/2026-11-03-cdp-breakout-contact-timing-policy.md)
lowers mean185.375→125.875, with only2/8 improved lanes and zero two-wall successes
in both arms. The fixed10%/five-lane screen fails. Retain the original teacher;
no24-episode extension, integration, coefficient search or learner is selected.
All16 CPU fixture cases,211 pins, exact pixel/action/debug replay and independent
ledger recount pass. Collection13,352 actions/53,396 frames takes8.838s, replay
10.789s; controller completes09:20:01.737 UTC November3/source334972c8 in19.811s.
Recount09:20:24.966 verifies2,552 corrected near-contact decisions, zero time clamps.
No failure, repeated emulator stage, GPU work or learner update occurs. Root
`/x/Temp/kindle-breakout-contact-timing-policy-20261103` is closed; do not repeat
stages or readers. ReviewSHA17202d2d/completionSHAb915684a. No Kindle allocation is
active at closure; check live state. Breakout and the five-game goal remain open.

The preceding [saved contact-timing diagnostic](results/2026-11-03-cdp-breakout-paddle-contact-timing.md)
passes its fixed prediction screen: episode2 loss-contact MAE6.101→5.757 pixels,
successful-contact MAE2.376→2.283, with lower RMSE and seven improved lane means.
Some tails worsen. All24 pins, five new/eight reused fixtures and739-row/875-metric
recount pass; sourceb7bafe64, completion09:10:22.591 UTC November3. Its root is
closed, with no failed/repeated stage, emulator/native import, GPU work or update.
The later gameplay failure shows why this aggregate error gain was insufficient
for teacher integration. It does not diagnose the native learner's causal limit.

The [current-position](results/2026-11-03-cdp-breakout-paddle-timing.md)
and [next-position](results/2026-11-03-cdp-breakout-response-timing.md)
saved-data screens also pass; their roots are closed. All timing diagnostics
reuse development episodes and teacher states. The original runtime teacher
and retained native learner remain unchanged.

The [completed imagined-value0.3 comparison](results/2026-11-03-cdp-breakout-imagined-value03.md)
lowers sampled **70.750→35.708** and greedy **143.667→102.000**, all 0/24
two-wall, in **21m44.356s** (+1.28% training time). Initial-subtracted score/min
falls 51.16%/30.34%. Retain imagined-value1/replay-value1 and actor/critic
representation gradients off, with the retained12M64/Auto/N8/B16/T16/H15/R16/
lambda.95/encoder4e-5/behavior8e-5/dynamics4e-4/expert100/replay100k recipe.
Only imagined return fitting and its slow-value regularizer are rescaled together;
replay targets still bootstrap from imagined returns. Native447fe50 and teacher
stay fixed; runner76440d32 adds12 CLI lines. Reused seed/cohort limits reliability.

All448 pins,100k/6,227 recovery,357 matching initial/invariant frozen tensors,
47 pre-update events,800/795 initial semantic events,replays/videos,guards,curves,
exposure and comparison pass. Main/post/collection succeed **08:44:22.585 /
08:44:33.979 /08:45:10.209 UTC November3**, source44f2cb48. All five original
services succeed. Terminal/coverage readers complete once08:45:43.886/08:45:54.789;
48 diagnostic/448 full pins reconcile275,720 actions/1,102,301 frames. Native
training actions at raw scores216–431 fall1,262→404, none432+; these are descriptive
bins, not verified walls or causal coverage. Slightly better replay CE and guided
online78.46 do not outweigh worse frozen play. Actual overrides are43.987%.

Root `/x/Temp/kindle-breakout-imagined-value03-20261103/full` is closed. Recovery,
terminal and coverage readers are complete; do not repeat stages or observers.
No full-run failure or GPU repeat occurs. Five prelaunch CPU corrections and the
bounded fixture's unintended native initialization remain preserved. Completion
SHA90fcf03d/terminalSHA314d5237/coverageSHA788a17f1. Keep CPU co-host time uncorrected.
No Kindle allocation remains active at closure; check live state before work.
No extension,coefficient sweep,confirmation or automatic successor is selected.
Breakout and the five-game goal remain unfinished.

The [completed actor/critic-gradient comparison](results/2026-11-03-cdp-breakout-acgrads-value1.md) lowers sampled
70.750→54.958 and greedy143.667→117.833 in21m56.980s (+2.26% training time),
all0/24 two-wall. Initial-subtracted score/min falls24.48%/20.07%. Retain the
raw-reward control with actor_critic_gradient off and12M64/Auto/N8/B16/T16/
H15/R16/lambda.95/encoder4e-5/behavior8e-5/dynamics4e-4/replay-value1/expert100/
replay100k. Only initial imagined policy/value representation gradients change;
native447fe50, runnercf4ea238 and teacher stay exact. Active losses and slightly
lower replay expert CE do not establish gameplay benefit or an isolated cause.

All428 pins,100k/6,227 recovery,357 matched initial/invariant frozen tensors,
47 pre-update events,800/795 paired initial traces,full replays/videos,guards,
curves,exposure and comparison pass. Main/post/collection succeed07:54:00.102/
07:54:18.504/07:54:52.283 UTC November3/source286c54b. Terminal and coverage
readers complete once07:55:21.963/07:55:30.341;48 diagnostic/428 full pins
reconcile285,024 actions/1,139,517 frames. Native training actions at raw scores
216–431 fall1,262→828,none432+; bins are descriptive,not verified walls or
causal evidence. Actual44.8745% overrides and online64.28 are not competence.

Root`/x/Temp/kindle-breakout-acgrads-value1-20261103/full` is closed. Allfive
original services and recovery/terminal/coverage readers are complete; do not
repeat them. No full-run failure or GPU repeat;two pre-launch CPU corrections
remain preserved. CompletionSHAb029cb5e/terminalSHA074d990e/coverageSHAf8b8ac82.
No allocation is active at closure; check live state. No extension,coefficient
sweep,confirmation or automatic successor. Keep CPU co-host timings uncorrected.
Reused1009/cohort limits reliability. Breakout and the five-game goal remain
unfinished.

The [fixed subaction diagnostic](results/2026-11-03-cdp-breakout-subaction.md) finds no extra rescues from one
mid-action command switch: three constant commands and nine two-frame pairs
both avoid34/120 recorded losses, zero gain in every lane. Retain four-frame
control; no shorter-interval learner or automatic successor is selected. This
uses late loss states of the older104.667 greedy policy and a shared recorded
continuation, not a repeated-feedback policy or the latest143.667 control.
All302 pins,360 wrapper parity checks,120 recorded controls,eight reversed orders
and37,016-action/148,038-frame source replay pass. Eight CPU tests;35.736s analysis
completes07:10:16.312 UTC November3/sourceba52b20, controller07:10:17.145 and
independent recount07:11:01.186. No failure,GPU work or updates. Root
`/x/Temp/kindle-breakout-subaction-20261103` is closed; do not repeat stages or
readers. ReviewSHA8b92a0c2/completionSHA7e72eafb. No allocation active at closure;
check live state. Breakout and the five-game goal remain unfinished.

The [completed life-loss comparison](results/2026-11-03-cdp-breakout-life-penalty.md) lowers sampled70.750→42.542
and greedy143.667→90.875 in21m39.349s (+0.89% training time), all0/24 two-wall.
Initial-subtracted return/min falls41.21%/37.87%. Retain raw training rewards
with12M64/Auto/N8/B16/T16/H15/R16/lambda.95/encoder4e-5/behavior8e-5/dynamics4e-4/
replay-value1/expert100/replay100k and sampled collection. Only the optional
one-point life-loss aid changes; raw scores, native447fe50 and teacher stay fixed.
All1,290 losses/negative rewards independently replay; online94.72 is guided
and does not outweigh worse frozen play. Reused1009/cohort limits reliability.

All453 pins,100k/6,227 recovery,357 matched initial/invariant frozen tensors,
paired initial traces, full replays/videos, guards, curves, exposure and comparison
pass. Main/post/collection succeed06:42:06.048/06:42:24.088/06:42:57.559 UTC
November3/sourceb80194e. Independent terminal06:56:45.055 and coverage06:56:53.724
pass once;50 diagnostic/453 full pins reconcile277,592 actions/1,109,787 frames.
Native actions at raw scores216–431 barely rise1,262→1,297, with none432+.
These are descriptive score bins, not verified walls or a causal explanation.

Root`/x/Temp/kindle-breakout-life-penalty-20261103/full` is closed. All five
original services and recovery/terminal/coverage readers are complete; do not
repeat them or completed stages. No full-run failure or GPU repeat occurs.
Three earlier bounded CPU failures/corrections remain preserved. Completion
SHA4d276268/terminalSHA4e848210/coverageSHA96930bc4. No Kindle allocation is active
at closure; check live state before work. Keep CPU co-host timings uncorrected.
No penalty sweep, extension, confirmation or automatic successor is selected.
Breakout and the five-game goal remain unfinished.

The [completed replay16k comparison](results/2026-11-03-cdp-breakout-replay16k.md) lowers sampled70.750→34.000
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

The [completed lambda1 comparison](results/2026-11-03-cdp-breakout-lambda1.md) scores **45.667 sampled /
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

The [lambda1 bounded check](results/2026-11-03-cdp-breakout-lambda1.md)
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

The [fixed pixel-motion readout](results/2026-11-03-cdp-breakout-pixel-motion-readout.md)
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

The [completed Breakout crop comparison](results/2026-11-02-cdp-breakout-playfield-crop.md)
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

The [completed Breakout 25M comparison](results/2026-11-02-cdp-breakout-capacity25m.md)
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

The [bounded25M check](results/2026-11-02-cdp-breakout-capacity25m.md)
is complete:8,192 actions/489 updates in84.864s,357 actual/invariant frozen tensors,
380 pins and all guards/replays pass. Conservative200k projection37m59s fits45min;
short training costs77.95% more than12M. No stage repeats or gameplay claim.

The [saved crop-gap diagnostic](results/2026-11-02-cdp-breakout-crop-gap.md)
finds the ball visible before all103 available loss crossings; all98 nearby
dropouts occur afterward. Retain the teacher. Its14 CPU cases/17 pins pass in
.387s with zero GPU work; the saved diagnostic is complete and must not repeat.

The [posterior-bootstrap comparison](results/2026-11-02-cdp-breakout-posterior-bootstrap.md)
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

The [earlier scalar calibration](results/2026-11-02-cdp-breakout-value-calibration.md)
recovers only a small sampled-value signal (affineRMSE7.698 versus constant7.855).
It does not identify a causal critic defect or establish a gameplay correction.
Keep all original scores and completed evidence; no diagnostic should repeat.

The [completed Breakout H30/R12 comparison](results/2026-11-02-cdp-breakout-h30-r12.md) scores **58.917 sampled /
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

## Current AMD runtime and retained position

Use Auto precision on RX7900XT/RADV. BF16 and `KINDLE_DEVICE_TARGETS` are
unavailable; see the [paired precision evidence](results/2026-10-25-cdp-amd-precision.md).
The completed F16/Auto context experiments do not justify extensions.
Native learning remains on Meganeura + Blade; backend porting, optimization
and JAX parity are parked. Explicitly select centered CDP; the CLI default is
learned RGB. Keep native pooled frames and one GPU resize.

Current learning uses native447fe5013b6442c477fc7bfef7eab4122082bc54,
SHA256 `35ad0db42fc7ed5dbaf6dacddff613fdc04d83ae6e0f38548532f6ab2c7812a2`.
Breakout uses `/x/Temp/kindle-breakout-corrective-20261028/runtime` with its
exact qualified runner. Qbert uses
`/x/Temp/kindle-qbert-color-masks-20261101/runtime` and its qualified faster,
decision-equivalent teacher. Python is `/x/Code/kindle/python/.venv/bin/python`
(CPython3.13). Declarations pin exact sources, binaries, runners, teachers,
configuration and environment. Do not mix optional runtime overlays.

The latest completed guard records boot `a8648671-8577-4f42-9b46-a4345d3ac297` and kernel
`7.0.0-38-generic`, with no host recovery performed. Check live state before
GPU work; preserve guard stops on actual faults or nonfinite values. An observer
timeout does not prove the learner stopped. Do not reset/reload/reboot the GPU.

Boxing and aided Freeway pass on two compact learners; Pong passes on both
retained learners with greedy evaluation. Qbert's fixed240k/R20 recipe passes
its15k/90% gate on the development learner and
[one fresh greedy confirmation](results/2026-11-02-cdp-qbert-r20-seed18013.md),
in about26 minutes each. Its sampled means still miss15k. Breakout retains
[70.750 sampled/143.667 greedy](results/2026-11-02-cdp-breakout-replay-value1.md)
from sampled collection, and a distinct
[78.542 sampled/90.042 greedy](results/2026-11-02-cdp-breakout-greedy-training.md)
point from greedy collection. All remain zero of24 two-wall successes.
The [lower-dynamics/value1 comparison](results/2026-11-02-cdp-breakout-value1-dynamics1e4.md)
regresses both modes and is closed. No completed stage should repeat.

The full 3,188-line handover and 376-line working guide before this condensation
remain in Git at `52c2174e2bc6e3cd9ff6ae5ae9b598308e9a8f51`; the roadmap and
linked results preserve experiment details. PR31's earlier expanded dashboard
is retained at `/x/Temp/kindle-pr31-dashboard-20261102/pr-before-condensation.json`.
No artifact, checkpoint or experiment source is removed by this documentation
change. The retained original transfer/build instructions below are historical;
current active declarations take precedence.

## Preserved source and fresh-machine setup

The working branch is **`phase2-levjepa-evaluation`**, continuing PR31.
It now contains the qualified c8aeba28 library implementation directly, with
portable Git dependencies. There is no separate takeover branch to switch to.
The previous356-commit work history and detailed experiment corpus are retained
on [archive/phase2-before-cleanup-20261025](https://github.com/kvark/kindle/tree/archive/phase2-before-cleanup-20261025).
The working branch keeps current results, source and tests; no `runs/` tree,
raw experiment data or checkpoint history is committed.

| Dependency | Revision |
| --- | --- |
| [Meganeura](https://github.com/kvark/meganeura/tree/handover/cdp-qualified-20261025) | `423d373634a1b1f04ff623db2b419757f1e8d96d` |
| [Blade](https://github.com/kvark/blade/tree/handover/cdp-qualified-20261025) | `bbbc93b7d26b8196d9143944fe7d8a61d6da4583` |
| Naga fork | `13f582ddd2792befb46952498010248bd65818e6` |

These preserve the measured implementation, not a fresh upstream port. Only the
user merges PRs. Cargo fetches the dependency branches without sibling checkouts
or old absolute paths. The prior handover verified329 library source files
against the measured implementation and passed CPU workspace/Python compilation.
`tools/handover/verify_sources.py` and its manifest preserve the source receipt
and eight explicit cleanup differences: Rust formatting, actual dependency
revision reporting, and two stale CPU test repairs. Learning operations and
checkpoint architecture validation are unchanged. A separate Meganeura
portability repair records GPU commands directly on Metal/GLES, whose Blade
backends do not expose Vulkan secondary encoders; the Vulkan path is unchanged. The full CPU Rust suite passes
(127 tests;78 GPU tests ignored), as does formatting.
No GPU qualification or speed claim follows from CPU checks. Rebuilt binary
bytes can differ with paths, compiler and dependency metadata. On a new machine, rebuild instead of importing the original Linux/CPython3.14
binary. The current AMD machine already has qualified CPython3.13 packages; use
the matching retained runtime above and do not repeat its completed qualification.

Example setup (Python3.11+; original environment was3.14.4, Rust1.98.0):

```sh
git clone --branch phase2-levjepa-evaluation https://github.com/kvark/kindle.git
cd kindle
python3 -m venv python/.venv
. python/.venv/bin/activate
python -m pip install -r tools/handover/requirements.txt
cd python
maturin develop --release --locked
cd ..
```

The requirements record the source machine's versions, rather than silently
upgrading the environment. Install the target's Vulkan driver and obtain Atari
ROMs separately; ROMs and the Python virtual environment are not in the transfer.
A release build is not a successful GPU initialization check.

The original NVIDIA runtime enabled the flags below. This is historical
configuration, not a command for the current AMD machine: in particular,
`KINDLE_DEVICE_TARGETS=1` is unavailable here. Use each active declaration's exact
environment instead of copying this block.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MEGANEURA_COMMAND_CACHE=1 KINDLE_DEVICE_MEANS=1
export MEGANEURA_IMAGE_VIEWS=1 MEGANEURA_CONVOLUTION_WORDS=1
export MEGANEURA_COPY_COMPOSITION=1 KINDLE_DEVICE_TARGETS=1
export MEGANEURA_CONVOLUTION_REUSE=1
```

Select the target GPU/ICD using that machine's actual configuration. Do not copy
`MEGANEURA_DEVICE_ID=0x2c02`,5080 name assertions, old driver versions or boot IDs
onto another GPU. Declare actual device/driver expectations for the host guard.
These flags retain the qualified implementation; capability fallbacks and
performance on other GPUs still need checking.

## Transfer package

Shared destination: **`/mnt/data/kindle-handover-20261025`**.
`manifest.json` identifies every included artifact by original path, size and
SHA256. `archive-review.json` records independent verification of all304 members.
The artifact archive is392,172,204 bytes (about374MiB); it contains no build caches.
`README.md` there lists package contents and source revisions. Source Git bundles
provide an additional fallback to the pushed branches.

Extract into a new local directory, after checking the archive SHA256 against
the manifest. Never extract over an existing run. The archive includes:

- `context64/frozen-checkpoint`: recovered867-update B2/T64 weights, and the
  successful frozen-restore review. See the recovery note below.
- `breakout-b8/2m-frozen`: recovered source-control export from the warm-start
  evaluation, learner counter499939 and environment counter2027152. All346
  tensors are finite. It is a frozen derivative of the2M control; the complete
  original training logs and initial/500k checkpoints are not included.
- `diagnostics`: retained PMU captures and prepared pre-submission collector.
- `original-runtime`: original native package, build receipt and example helpers.
  Historical scripts contain absolute paths and host pins: treat as evidence,
  not ready-to-run controllers on another machine.

**Originals located:** the shared-drive cleanup moved the runs to
`/x/Artifacts/shared-drive-cleanup-20261025` on the original machine; they were
not lost. Verified original checkpoints are:

- `kindle-cdp-breakout-context64-20261023.5OA15Iew/smoke/checkpoints/4096`:
  learner867, environment4096.
- `kindle-cdp-breakout-budget-20261021.wqYChxy8/checkpoints/2000000`:
  learner499939, environment2000000.

Both originals match all346 tensors in their corresponding shared-package
frozen exports exactly. The exports have advanced environment counters5120 and
2027152; preserve their metadata rather than relabeling them as original saves.
The shared package remains usable without transferring the whole local archive.
Use the saved originals when their original counters or full run evidence are
needed. Neither version preserves replay, games, RNG or live belief.

For historical artifact paths, look under the relocated run directory above.
The package is deliberately not a copy of every profiling dump or checkpoint.

## Original arrival procedure — completed on this AMD machine

The following describes the October25 transfer protocol. Its initialization,
restore and fresh context experiments are complete. Do not execute this list
again on the current machine or treat its BF16 recipe as current AMD settings.
Use the active work above and Auto precision. A genuinely new machine needs
its own bounded host qualification, not a repetition of completed learning.

1. Build the preserved runtime and record source, compiler, Python, GPU and driver
   identity. Check available memory/disk. No backend port or optimization first.
2. Run a bounded initialization/frozen restore under `gpu_host_guard.py` or
   `run_gpu_queue.py`, using the recovered867-update checkpoint and an output
   directory created for this check. Use eight environments, native observation
   size, the published Atari protocol and sticky actions.25. Frozen evaluation
   must produce zero updates and preserve all346 tensors. A1024-action check
   suffices for this startup boundary; it is not a learning evaluation.
3. If initialization succeeds, finish the smoke review and run the declared
   fresh Breakout1009 B2/T64 life, with500k and2M checkpoints and the actual
   initial control. Eight streams;128 replay positions/update;ratio32;12M/BF16;
   centered CDP and action-effects coefficient1; no Freeway action aid. Expect
   first update at action632,124843 updates at500k,499843 at2M. Full life is
   fresh, not a continuation from the smoke or prior warm-start8M model.
4. Evaluate fixed milestones with24 natural sampled episodes each (three per
   stream), zero updates and full replay/videos. Compare to retained B8/T16
   scores58.83/98.21/1.42. Disclose different hardware/build, and the coupled
   sequence-length/independent-batch change. Review before any extension or seed
   replication. Stronger Qbert learning follows useful Breakout evidence.

See the existing runner's `--help` and the archived declarations for the complete
protocol. Do not launch the old absolute-path `run.py train` unchanged. Retain
ordinary guard checks, accounting and actual-initial/frozen comparisons; do not
turn relocation into another weeks-long qualification matrix.

## GPU fault scope

On the original host, ordinary restore repeatedly produced Xid62/PMU halt on
615.71.09 and595.91.07. One traced restore succeeded; an ASLR-preserving trace
later failed while waiting for the first world's device-local buffer fills,
before checkpoint loading/model execution. IOMMU errors followed30s later.
This localizes the wait, not the root cause. Neither hardware, driver/firmware
nor invalid runtime arguments are conclusively excluded.

No blind retries on a faulted device, and no automatic reset/reload/reboot.
The prepared collector records fill handles/capacities/offsets/sizes/values and
stops before Blade submit. Six CPU cases passed; native capture remains unrun.
Its ABI and symbols apply only to the original binary. On another healthy GPU,
start with the bounded restore above; do not inherit a historical machine-wide
stop or presume the failure reproduces there. Stop and collect evidence if it
actually faults. No training was launched during this handover.

## What not to redo

Do not restart Pong2M, the rejected B32 Freeway recipe, unchanged Breakout
warm-start extensions, the45-run old matrix, kernel geometry sweeps or JAX parity.
No concurrent/swarm learner project. Current five-game scores and limitations
are in the roadmap. User-authored `TASK.md` and `docs/cdp-claude-feedback.md`
are now tracked unchanged as historical intent/review, not current measurements.
