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

The [fresh 200k completed-return actor comparison](docs/experiments/2026-11-03-cdp-episodic-actor-200k.md)
launches **16:24:19 UTC November 3**, source783852b2. Bonus-off control trains
first, then the scale100 candidate, with no overlap or experience transfer.
Both use12M64/Auto/N8/B16/T16/H15/R16/context1, expert100 and the qualified
e3b02163 runtime. Each gets200k actions/12,477 updates and a30-minute training
cap, a retained100k/6,227 checkpoint, and final/actual-initial sampled/greedy
24-episode assessments. Original864/90%-two-wall gate and retained results stay.

Main service `kindle-episodic-actor-200k-20261103`, PID1049147, invocation
`fcc4bc49735445638173a45196bddd44`, is active. Its controller owns progress,
startup/recovery checks, phase audits, replays and comparison; the separate
`-collect` observer is active. All307 preparation pins and CPU preflight pass.
Control startup passes261 initial arrays matching bounded/retained shared
values,47 pre-update events and first update376. Its100k/6,227 recovery passes
16:32:53.148 UTC:261 finite/schema-checked tensors and file hashes, retained
read-only by checkpoint history. All307 pins and original process identity
remain verified. Control training continues; candidate has not started and no
full frozen result exists.
Root `/x/Temp/kindle-episodic-actor-20261103/full`. Check live state; do not
restart this pair or schedule another GPU job while it is active. No extension,
third learner or automatic confirmation follows. The five-game goal is unfinished.

The [closed bounded pair](docs/results/2026-11-03-cdp-episodic-actor.md)
passed8192 actions/489 updates in36.677s control/38.277s candidate (+4.36%),
with33105 positive replay rows and no natural frozen episodes. All native/CPU,
legacy, tensor, replay and terminal checks pass; no failed or repeated stage.
Native e3b02163/SHA d406629f adds no parameters; disabling disagreement removes96
ensemble tensors, leaving261. Runtime,wheels,validation binary and1.008GB cache
remain. Bounded root `check` is closed; never reuse its experience or rerun it.

The [CPU replay-actor target audit](docs/results/2026-11-03-cdp-breakout-replay-actor-targets.md) confirms no direct
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

The [completed400k/R16 Breakout comparison](docs/results/2026-11-03-cdp-breakout-budget400k.md)
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

The [completed frozen halfway assessment](docs/results/2026-11-03-cdp-breakout-halfway.md)
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

The [completed context9 comparison](docs/results/2026-11-03-cdp-breakout-context9.md)
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

The [optional replay-state recomputation check](docs/results/2026-11-03-cdp-replay-context.md)
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

The [CPU replay-context age reconstruction](docs/results/2026-11-03-cdp-breakout-replay-age.md)
matches all12,477 logged updates and199,632 sampled starts in.387s CPU.
Final50k unmasked contexts have median246/mean365.099/p99 1,776 updates since
their last computation;98.66% of labeled actor rows can depend on initial
context before a reset. This measures age,not latent error or a gameplay cause.
All15 fixtures and27 diagnostic/338 control pins pass; review completes
12:35:19.800 UTC November3/sourceb729ab23. No GPU work,learner,emulator,failure
or prior stage repeat. Root `/x/Temp/kindle-breakout-replay-age-20261103` is
closed. Earlier replay16k and B8/T32 results remain; this age audit alone
did not select a new buffer,prefix,loss or learner. Retain the70.750/143.667 control; goal unfinished.

The [completed Breakout H3/scale1 comparison](docs/results/2026-11-03-cdp-breakout-open-loop.md)
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

The [Breakout H3 world-supervision check](docs/results/2026-11-03-cdp-breakout-open-loop.md)
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

The [completed actor-only imitation-decay comparison](docs/results/2026-11-03-cdp-breakout-actor-only-decay.md)
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

The [actor-only imitation-decay qualification](docs/results/2026-11-03-cdp-actor-only-imitation-decay.md)
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

The preceding [actor optimizer attribution](docs/results/2026-11-03-cdp-breakout-actor-optimizer.md)
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

The [completed policy-weight100 comparison](docs/results/2026-11-03-cdp-breakout-policy100.md)
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

The [completed contact-timing CPU teacher comparison](docs/results/2026-11-03-cdp-breakout-contact-timing-policy.md)
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

The preceding [saved contact-timing diagnostic](docs/results/2026-11-03-cdp-breakout-paddle-contact-timing.md)
passes its fixed prediction screen: episode2 loss-contact MAE6.101→5.757 pixels,
successful-contact MAE2.376→2.283, with lower RMSE and seven improved lane means.
Some tails worsen. All24 pins, five new/eight reused fixtures and739-row/875-metric
recount pass; sourceb7bafe64, completion09:10:22.591 UTC November3. Its root is
closed, with no failed/repeated stage, emulator/native import, GPU work or update.
The later gameplay failure shows why this aggregate error gain was insufficient
for teacher integration. It does not diagnose the native learner's causal limit.

The [current-position](docs/results/2026-11-03-cdp-breakout-paddle-timing.md)
and [next-position](docs/results/2026-11-03-cdp-breakout-response-timing.md)
saved-data screens also pass; their roots are closed. All timing diagnostics
reuse development episodes and teacher states. The original runtime teacher
and retained native learner remain unchanged.

The [completed imagined-value0.3 comparison](docs/results/2026-11-03-cdp-breakout-imagined-value03.md)
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

The [completed actor/critic-gradient comparison](docs/results/2026-11-03-cdp-breakout-acgrads-value1.md) lowers sampled
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

The [fixed subaction diagnostic](docs/results/2026-11-03-cdp-breakout-subaction.md) finds no extra rescues from one
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

The [completed life-loss comparison](docs/results/2026-11-03-cdp-breakout-life-penalty.md) lowers sampled70.750→42.542
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
