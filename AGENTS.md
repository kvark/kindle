# Kindle working direction

Kindle learns while acting. Games are the first testbed; sparse explicit rewards
and human guidance are allowed. Favor minimalism, expressiveness, safety and
speed. Keep learning/inference native on Meganeura + Blade. Python is for game
adapters, reference controls and analysis. Follow `/mnt/data/GUIDELINES.md` when
available. Only the user merges PRs; commits and pushes are allowed.

## Start here

- [Machine handover](docs/handover.md): source branches, build instructions,
  shared checkpoint/evidence package and the exact unfinished experiment.
- [Roadmap](docs/kindle_single_life_dreamer_plan.md): current results and priorities.
- [PR31](https://github.com/kvark/kindle/pull/31): live status dashboard. Update its
  description when status changes; do not create a separate STATUS.md.
- [GPU incident](docs/results/2026-10-23-cdp-context64-gpu-fault.md): evidence and
  limits of the repeated initialization failures on the original machine.

## Active direction

The [completed Qbert6029 H5/R32 learner](docs/results/2026-10-29-cdp-qbert-h5-seed6029.md) finishes200k actions/799,761 frames/24,954 updates in40m22s.
Greedy15792.708/24 first pyramids and sampled15420.833/24 pass15k/90%,versus
own initials91.667/0 and103.125/0. Seed4013 remains14050/23 and13331.250/23;
one of two H5/R32 learners passes,not broad reliability. All263 pins,100k recovery,
357 finite/invariant tensors,48 recorded initial differences,full pixel/action/
label and frozen replays/videos,guards,curves,coverage,exposure and comparison pass.
Main succeeds17:52:35 UTC October29/source9920dee,postprocessing17:52:48,
collection17:53:23;all services succeed,no failures or GPU repeats. Same12M48/
Auto/N8/B16/T16/H5/R32/expert100/native447fe50,no experience reuse. Actual
overrides67,200 (33.6%);online guided scores are not frozen competence. No paired
H15/R16 control for6029. CPU co-hosting caveats remain;no time correction.
Root`/x/Temp/kindle-qbert-h5-seed6029-20261029/full`. No Qbert extension,third
learner or automatic successor selected. The five-game goal remains unfinished.

A [matched Breakout reconstruction control](docs/experiments/2026-10-29-cdp-breakout-rgb-control.md) is CPU-prepared; no GPU declaration or launch exists.
Keep12M64/B16/T16/H15/R16/expert100 and native447fe50; only centered flag and
reconstruction/future objectives change. Preflight17:29:20 UTC passes7 CLI/config
branch cases,11 audit-config cases,6 compiled scripts,132 control pins and342
shared shapes;399 tensors/14,457,556 parameters are predicted. No native import.
All263 active6029 pins and original PID/invocation stay unchanged. The1.398s
CPU co-hosting remains a timing caveat. Wait for completed Qbert6029 audits and
idle host before the8192/489-update cost check;35min full projection gate,
no full learner or automatic successor. Root`/x/Temp/kindle-breakout-rgb-control-20261029/native-check`.

The completed [Breakout kernel action readout](docs/results/2026-10-29-cdp-breakout-representation.md#a-fixed-nonlinear-readout-also-trails-the-recorded-policy) takes1.824s CPU and scores49.25% overall/47.06% near-paddle teacher agreement,
below linear50.58%/49.26% and recorded policy54.58%/53.68%. No evidence for this
replacement head; other decoders/feature precision remain unresolved.31 CPU
tests,96 pins and263 active Qbert6029 pins pass; original PID/invocation stays.
Review17:16:31 UTC October29/source7b43296. The original historical-pin failure
is retained; two exact old sources are resolved fromGit before the first fit.
No GPU work or new learner. Tests1.103s/analysis wrapper2.621s co-host with6029;
retain as a timing caveat without wall-time correction. Evidence
`/x/Temp/kindle-breakout-kernel-readout-20261029`.

The [completed H5/R32 frozen temperature assessment](docs/results/2026-10-29-cdp-qbert-h5-temperature.md) scores14025/22 first pyramids
at0.5 versus retained greedy14050/23;default sampled13331.250/23 reproduces
exactly. Both own initial arms177.083/0;all final means still miss15000. Retain
greedy forH5/R32;no sweep,extension,confirmation or automatic successor. All63
pins,357 invariant tensors per arm,6722/1138 default semantic events,replays/
videos and guards pass. Four arms135,136 actions/540,407 frames/81.325s,zero
updates;original40m24s training remains. Service succeeds16:45:24 UTC October29
/source81a0280 after2m43.706s;collection16:47:10. Qualifiede416696/runtime
`/x/Temp/kindle-action-temperature-20261029/runtime` unchanged,no build/retry.
Root`/x/Temp/kindle-qbert-h5-temperature-20261029`;no GPU allocation active at
completion. Reused development cohort;the five-game goal remains unfinished.

The [completed Qbert H5/R32 comparison](docs/results/2026-10-29-cdp-qbert-h5-r32.md) finishes200k actions/799,762 frames/
24,954 updates in40m24s,+32.67% training time. Greedy14050/23 first pyramids
improves11984.375/24 by17.24%;sampled13331.250/23 improves6759.375/24 by97.23%.
Own initials80.208/177.083 clear none;both candidate means still miss15000.
RetainH5/R32 as strongest primary development score andH15/R16 as cheaper point;
no extension,confirmation or automatic successor selected. All229 pins,100k
recovery,357 matched initial/invariant frozen tensors,paired initial traces,full
replays/videos,guards,curves,coverage,exposure and comparison pass. Main succeeds
16:31:07 UTC October29/sourcee57dd2f,postprocessing16:31:24,collection16:32:04;
all services succeed,no GPU stage repeats. Actual overrides75,030 (37.515%);
online aided returns are not competence. H15→5/R16→32 changes horizon,cadence,
sampling,warmup and total work together;no isolated effect or seed confirmation.
Native447fe50,runner,teacher fixed. Evidence`/x/Temp/kindle-qbert-h5-r32-20261029`.
No GPU allocation remains active at completion;check live state before work.
The five-game goal remains unfinished.

The [completed Qbert B8/T32 comparison](docs/results/2026-10-29-cdp-qbert-context32.md) finishes200k actions/799,710 frames/
12,469 updates in31m48s,+4.44% training time. Greedy11609.375/24 first pyramids
falls3.13% below B16/T16's11984.375/24;sampled6120.833/20 falls9.45% below
6759.375/24. Own initials80.208/177.083 clear none;both miss15000. RetainB16/T16;
no extension,confirmation or automatic successor. All229 pins,100k recovery,357
matched initial/invariant frozen tensors,paired initial traces,full replays/videos,
guards,curves,coverage,exposure and comparison pass. Main succeeds15:15:59 UTC
October29/source07ce3dc,postprocessing15:16:17,collection15:17:05;all services
succeed. B16/T16→B8/T32 also changes diversity,warmup and target rows;no isolated
memory claim. Actual overrides42.950%;online aided returns are not competence.
The.788s CPU source review remains a timing caveat. No allocation active at completion;
check live state before work. The five-game goal remains unfinished.

The [Qbert B8/T32 check](docs/results/2026-10-29-cdp-qbert-context32.md) passes8192
actions/32,756 frames/481 updates in72.797s;1024 frozen/.788s haszero updates
and no natural episodes. All41 pins,357 matching initial/invariant frozen tensors,
full guided replay and guards pass;service succeeds14:28:05 UTC October29,
collection14:28:28. Conservative200k projection32m35s passes45min. The separately
selected [fresh200k comparison](docs/experiments/2026-10-29-cdp-qbert-context32-200k.md),
12M48/Auto/N8/B8/T32/H15/R16/expert100 completes with100k recovery and four frozen arms.
B16/T16→B8/T32 preserves256 positions/update but changes temporal gradients,
sequence diversity,warmup and targets;no isolated memory claim. Native447fe50,
runner,teacher and supervision fixed. No prior experience reused. Check live state.

Frozen Qbert temperature0.5 scores12389.583/24 first pyramids,3.38% above retained
greedy11984.375/24;sampled6759.375/24 reproduces exactly. Own initials177.083/0;
mean15000 still fails. Retain0.5 as development option,not broad superiority or
global default. Four arms96,872 actions /387,366 frames /58.516s,zero updates;
original30m27s training stays. All57 pins,357 invariant tensors per arm,4204/1138
default semantic events,replays/videos/guards pass. Service succeeds14:10:15 UTC
October29/source4b24f75 after2m04.158s;collection14:10:56. Nativee416696/SHA40872a3
passes136 Rust/658 Python CPU tests/Clippy and bounded checks. Use
`/x/Temp/kindle-action-temperature-20261029/runtime`,packaging receipt retained,
857MB cache released. CPU packaging lookup failure preserved,no GPU repeat.
No active allocation,sweep,extension or successor. See action-temperature result;
check live state before work. The five-game goal remains unfinished.

Frozen modal-posterior Qbert lowers11984.375→10539.583,each24/24 first pyramids;
own initials80.208/0.000,zero pyramids. Retain sampled posteriors/greedy actions.
All53 pins,357 tensors per arm,6080/1100 default semantic events,replays/videos/
guards pass;110,112 actions/440,249 frames/67.014s,zero updates. Service exits
10:12:26 UTC October29;no active GPU allocation or successor. Nativeb76f91c/
SHA7555bf9 passes135 Rust/625 Python CPU tests and bounded checks. Original
unsynchronized test failure is preserved;corrected producer waits change only
the test. Use`/x/Temp/kindle-greedy-posterior-20261029/resumed/synchronized/runtime`;
packaging receipt retained,build cache released. See the greedy-posterior result.

Qbert H3/scale1 world supervision completes200k actions/799,709 frames/12,477
updates in30m15s. Greedy9377.083 and sampled5613.542,each23/24 first pyramids,
are below control11984.375/6759.375,each24/24. Own initials80.208/177.083,zero
pyramids; the mean15000 gate still fails. Retain the control; no extension,
retry or confirmation follows. All242 pins,100k recovery,357 matched initial/
invariant frozen tensors,both initial semantic traces,full replay/video/guard/
curve/coverage/exposure/comparison checks pass. Actual guidance87,421 actions
(43.7105%). Main exits08:41:03 UTC October29,postprocessing08:41:29;all four
services and the collector succeed. No GPU repeats or active allocation.
See `docs/results/2026-10-29-cdp-open-loop-supervision.md`; evidence remains
under `/x/Temp/kindle-open-loop-20261029/full`. Check live state before GPU work.

The frozen H3 forecast follow-up raises15-step reward-sum MAE37.499→40.793;
early2.347→2.035 improves,later45.259→49.350 worsens. All152 original control
arrays reproduce exactly. Both48,256-action captures preserve357 tensors with
zero updates;65 current/39 resolved historical pins,34 CPU tests and guards pass.
Service exits09:49:38 UTC October29/sourcea3fbd03;terminal collection passes.
Evidence `/x/Temp/kindle-open-loop-forecast-20261029` includes historical source
resolution. Retain control;no new learner/loss variant or active GPU allocation.
See the open-loop result report;candidate-state distribution/causality unmeasured.

Breakout frozen representation readouts complete20,960 actions/83,799 frames
per model in32.301s trained/32.420s initial,zero updates. Recurrent motion R²
improves.282/.051→.473/.614; trained paddle MAE4.334 CNN/9.643 recurrent pixels,
near-paddle teacher-target MAE14.848 recurrent. Useful information is present,
but linear decoding does not identify the cause of weak control. All55 pins,
357 invariant tensors per capture,matched input hashes/trajectories,readback/
counter/host checks pass;19 CPU tests and terminal collector pass. Service exits
09:02:23 UTC October29;no GPU allocation active. Native447fe50 unchanged.
Evidence `/x/Temp/kindle-breakout-representation-20261029`; see
`docs/results/2026-10-29-cdp-breakout-representation.md`. The CPU teacher-action
follow-up completes in1.333s with zero GPU work/updates: trained concatenated
features50.58% overall/49.26% near-paddle agreement trail the recorded policy
54.58%/53.68%,though better than initial38.60%/41.18%. All34 inputs and55 prior
pins stay unchanged;26 CPU tests pass. Review09:23:24 UTC October29/source26a9961.
Evidence `/x/Temp/kindle-breakout-action-readout-20261029`. No simple head/CNN
skip benefit is established; no new native architecture or learner is selected.

Optional H3 open-loop supervision (`ab80486`,default off) passes135 Rust/611
packaged Python CPU tests,strict Clippy,two guarded gradient checks and legacy
1024-action compatibility (128 matched traces,357 invariant tensors). Fresh
Qbert12M check8192 actions/32,754 frames/489 updates takes70.207s;1024 frozen/.792s
haszero updates/no completed episodes. All44 pins,357 matched initial/invariant
frozen tensors,finite auxiliary reports,full guided replay and guards pass;
service exits07:43:21 UTC October29. Conservative200k projection31m25s is below
35min;task benefit remains unmeasured. Original CPU fixture/timestamp failures
remain;no GPU repeats. Runtime `/x/Temp/kindle-open-loop-20261029/runtime`,native
SHAb4cdf26;packaging pins exact sources,both build caches released,posterior
draft parked separately. See the open-loop result/protocol. No full learner or
GPU allocation is active;check live state before work.

Qbert action-block ranking completes128 origins/2432 forecasts on48256 source
actions in95.811s,zero updates. Pairwise agreement72.96% before/57.21% after
the first pyramid; only40/128 origins distinguish candidates. Later gain over
uniform1.601 has lane interval[−.131,3.443]. All46 pins,357 invariant tensors,
128 exact prior-forecast matches,pixel trajectories and guards pass; service
exits06:52:55 UTC October29. The CPU oracle uses process copies to preserve
sticky-action memory; original snapshot failure is retained, native unchanged.
Next selected development is optional three-step open-loop world supervision,
with gradient/cost checks before a full learner. See its protocol and the
diagnosis report. No GPU allocation is active; check live state before work.

Qbert live-critic calibration completes98664 actions/394520 frames in69.704s,
zero updates. Sampled value correlation.542/R².184 beats the constant baseline
but underestimates mean realized1475.689 with1218.534; later-phase correlation
is.358. Greedy changes the policy and has larger bias. All44 pins,357 invariant
tensors per arm,12522 matching semantic events and guards pass; service exits
06:30:01 UTC on October29. No new loss/training allocation follows. Next inspect
actual alternative-action ranking. See the diagnosis report; check live state.

The October29 architecture diagnosis is within the existing five-game goal.
Retained Qbert12M frozen forecasts preserve useful15-step reward information:
cumulative MAE107.570 initial→37.499 trained, versus69.538 with unrelated actions.
Before/after-first-pyramid MAE2.347/45.259 identifies later-play weakness, with
different reward distributions. Both48256-action captures preserve357 tensors;
all39 pins/guards pass, zero updates, service exits06:18:53 UTC. Prioritize critic
calibration and actual alternative-action outcomes before assuming a multi-step
loss will solve the gap. No successor GPU allocation is selected. The separate
greedy-posterior draft remains uncommitted and has no new native/GPU validation.
See [diagnosis](docs/results/2026-10-29-cdp-learning-diagnosis.md); check live state.

Training is the priority. Backend optimization, upstream porting and JAX parity
are parked. The selected architecture is **centered CDP**, with a jointly learned
CNN, categorical RSSM and imagined actor/critic. No pretrained DINO or Tiny JEPA
encoder is required. Explicitly select CDP; the CLI default remains learned RGB.
The user's current priority is compute-constrained success on this game set.
Game-specific model sizes, image resolutions and learning hyperparameters are
allowed; one universal recipe is not a prerequisite. Prefer useful frozen
learning per GPU-hour, with short development runs before long allocations.
T64 is an experiment, not the default for subsequent development. Start from
T16 and consider lower replay ratios and smaller models using the historical
capacity evidence. Size1M already worked well for Boxing and Freeway, but was
much weaker for Breakout and Qbert. Native32/48/64 image geometry is supported;
preserve one GPU resize from native frames and verify small-object visibility.
Changing several settings to find a useful cheap recipe is allowed; disclose
the changes without claiming an isolated causal effect or five-game reliability.
Work on `phase2-levjepa-evaluation` and PR31. This branch now includes the
qualified c8aeba28 implementation with portable dependencies. Do not switch to
a separate takeover branch. Keep `runs/`, large data and archived experiment
ledgers out of Git; old evidence lives on the explicitly named archive branch.

The five-game goal is unfinished. Compact AMD Boxing and aided Freeway pass on
two learners each, within 30 minutes per learner. Pong12M48 passes its gate on both retained learners with greedy evaluation
(sampled actions pass on one of two); Qbert expert imitation improves on two seeds but
misses its full-game gate, and Breakout remains much weaker. Published
long-run scores are aspirations, not matched-budget comparisons. Do not claim
mastery or CDP superiority from short runs, finite losses or backend speed alone.
Keep detailed results in the roadmap and linked reports, not another ledger here.

Use Auto precision for current AMD learning. BF16 and `KINDLE_DEVICE_TARGETS`
remain unavailable. After three paired early updates, F16 leaves the first reward
projection unchanged while Auto changes it; raw derivatives and the mechanism
remain unmeasured. The completed F16/Auto B2/T64 lives do not justify extensions.
See `docs/results/2026-10-25-cdp-amd-precision.md` for the evidence and limits.

Breakout12M64/B16/R16/expert100 completes 200k actions /799,464 frames /12,477
updates in22m19s. Frozen sampled13.208/greedy17.875 improve4M/B8's9.458/12.208
by39.65%/46.42%, for25.72% more training time; own initials1.417/2.208.
All0/24 two-wall successes. All132 pins,100k recovery,357 bounded-matching
initial/invariant frozen tensors, full replays, guards, curves, exposure,
comparison and terminal collector pass. Main exits05:34:19 UTC,postprocessing
05:34:31 on October29. The original CPU path failure and106 pins remain preserved;
no GPU stage repeats. Retain both budget points; capacity and batch change together
on reused1009, without an isolated effect or reliability claim. Actual overrides
56.712%; online guided scores are not frozen competence. No extension,cap increase,
confirmation or automatic successor is selected. No GPU allocation is active.
See the [completed capacity result](docs/results/2026-10-26-cdp-breakout-pixel-reference.md#larger-capacity-improves-frozen-breakout-with-higher-cost); check live state before GPU work.

Breakout12M64/B16/R16/expert100 bounded check passes8192 actions/32,720 frames/
489 updates in47.623s; frozen1024/4096frames/.877s haszero updates/no natural
completed episodes. All56 pins,357 finite/invariant tensors,16,334,353 parameters,
finite supervised reports,changed CNN/reward/actor/value,3841 verified overrides,
full guided replay and guards pass. Initial values are not compared with the
4M architecture; actual12M64 shapes match the CPU derivation. Conservative200k
projection21m19s passes30min; task competence remains unmeasured. Service succeeds
04:52:02 UTC on October29/source d4200b2. No failure,GPU repeat,cap change,full
learner,extension or automatic promotion occurs. Use the matching Breakout-era
runtime; native447fe50 unchanged. No GPU allocation is active;check live state.
See [the bounded result](docs/results/2026-10-26-cdp-breakout-pixel-reference.md#larger-supervised-breakout-passes-execution-and-early-cost-checks).

Qbert guidance3840 completes200k actions/799,726 frames/12,477 updates in29m31s.
Primary greedy8187.500/24 first pyramids falls31.68% below retained1920's11984.375,
for3.04% less training time; sampled7051.042/23 rises4.31% from6759.375/24. Own
initials80.208greedy/177.083sampled,zero pyramids; both final means miss15000.
All227 pins,100k recovery,357 matching initial/invariant frozen tensors, paired
initial traces,finite reports,full replays,guards,curves,coverage,exposure,
comparison and repaired collector pass. Main succeeds04:39:46 UTC,postprocessing
04:39:55 on October29. Original CPU observer exit1 remains;replacement succeeds
with original deadlines. No GPU stage repeats or caps change. Actual overrides
59.6615%;45,727 actions at ages1920–3839 are all guided,none reaches3840,and no
cap-released native action occurs. These are exposure counts,not verified game
levels or competence. Retain1920;no extension,limit sweep,retry,confirmation or
automatic successor. The.630s co-host repair remains recorded. No GPU allocation
is active;check live state before work. See [the completed result](docs/results/2026-10-25-cdp-cheap-qbert.md#longer-guidance-lowers-primary-greedy-quality-at-the-same-short-budget).

Breakout expert100 completes200k actions/799,351 frames/24,970 updates in
17m45s. Sampled9.458/greedy12.208 improve retained expert10's4.292/5.000 for
1.92% more training time; own initials1.375/2.583. All0/24 two-wall successes.
All69 pins,100k recovery,357 matching initial/invariant frozen tensors, paired
initial traces,finite reports,full replays,guards,curves,exposure,comparison and
original collector pass. Main succeeds03:47:13 UTC,postprocessing03:47:41 on
October29; all four services terminal success. Actual overrides58.7625%;guided
scores are not frozen competence. Retain expert100 as a useful development point
and the distinct stronger sampled controls. Only expert scale10→100 changes on
reused1009; no mechanism or seed-reliability claim. No observer/GPU failure,
GPU repeat,cap change,extension,coefficient sweep,confirmation or automatic
successor. The.894s co-host Qbert analysis remains recorded. GPU idle at completion;
check live state before work. See [the completed result](docs/results/2026-10-26-cdp-breakout-pixel-reference.md#stronger-imitation-improves-frozen-breakout-at-nearly-unchanged-cost).

Qbert12M48/B16/R16/expert1000 completes200k actions/799,697 frames/12,477
updates in30m31s. Greedy10588.542/24of24 first pyramids and sampled6150/21
fall11.65%/9.02% below retained expert100, for0.21% more training time. Own
initials80.208/177.083, zero pyramids; both final means miss15000. All229 pins,
100k/6227 recovery,357 matching initial/invariant frozen tensors, paired initial
traces, finite reports, full replays, guards, curves, coverage, exposure,
comparison and original terminal collector pass. Main succeeds03:14:19 UTC,
postprocessing03:14:39 on October29; all four services are terminal success.
Actual overrides43.8025%; guided scores are not frozen competence. Only expert
scale100→1000 changes on reused4013; no mechanism or seed-reliability claim.
Retain expert100. No observer/GPU failure, repeated GPU stage, cap change,
extension, coefficient retry or automatic successor. The.668s co-host monitor
remains recorded. GPU idle at completion; check live state before work. See
[the completed result](docs/results/2026-10-25-cdp-cheap-qbert.md#stronger-imitation-lowers-both-frozen-scores-at-the-same-short-budget).

Qbert12M48/B16/R8/expert100 completes400k actions/1,599,459 frames/12,489
updates in40m35s. Greedy10042.708/23of24 first pyramids is16.20% below retained
200k/R16's11984.375/24,for33.28% more training time;sampled7456.250/22 rises
from6759.375/24. Own initials80.208greedy/177.083sampled,zero pyramids. Both
modes still miss mean15000. All228 pins,three recoveries,357 matching initial/
invariant frozen tensors,paired initial semantic traces,finite reports,full
label/pixel/action and frozen replays,guards,curves,coverage,exposure and
comparison pass. Main succeeds02:16:04 UTC,postprocessing02:16:22 on October29.
Original CPU progress-observer exit2 is preserved;the private replacement and
repaired terminal collector pass. No GPU stage repeats or caps change. Retain
200k/R16;no extension,seed confirmation or automatic successor is selected.
The experience budget and ratio change together on reused4013;no isolated causal
or reliability claim. Actual training overrides40.40975%;guided scores are not
frozen competence. No learner or GPU allocation is active. See the completed
R8 result in cheap-Qbert and its protocol;check live state before GPU work.

The compatible RGB Qbert control completes200k actions/799,695 frames/24,970
updates in28m41s. Greedy6090.625/24 first pyramids and sampled4335.417/23 are
below retained CDP's9654.167/23 and7569.792/24 in26m07s; both miss mean15000.
Own initials263.542greedy/182.292sampled,zero pyramids. The locked paired lane
learning-per-minute decision favors this CDP recipe:mean CDP-minus-RGB156.359,
95% interval[110.311,206.365],with higher absolute CDP mean and all audits passing.
This is one reused development seed/cohort with strong supervision,not five-game
superiority or independent confirmation. All195 pins,100k recovery,342 shared
CDP/399 own initial tensors,399 invariant frozen tensors,paired initial semantic
traces,full replays,guards,curves,coverage,exposure and comparison pass. Main exits
00:57:02 UTC,postprocessing00:57:28 on October29;all services terminal success.
No GPU repeat,cap change or full-run failure. Actual RGB guidance45.0985%.
Retain the favorable comparison;no extension,confirmation or automatic successor
selected. GPU idle;check live state before work. See [the completed comparison](docs/results/2026-10-29-cdp-qbert-rgb-control.md).

Qbert12M48/B16/R16/expert100 completes200k actions/799,742 frames/12,477
updates in30m27s. Greedy11984.375/24 first pyramids improves4M/B8's9654.167/23
by24.14%,for16.57% more training time;sampled6759.375/24 falls below7569.792/24.
Own initials80.208greedy/177.083sampled,zero pyramids;both final means fail15000.
All190 pins,100k recovery,357 reference/check initial and invariant frozen tensors,
finite reports,changed CNN/reward/actor/value,full label/pixel/action and frozen
replays,guards,curves,coverage,exposure and comparison pass. Main succeeds23:40:33
UTC,postprocessing23:40:50 on October28;all services terminal success. No failed
attempt,GPU repeat or cap/runtime change. Capacity and batch change together on
reused4013;no isolated capacity or seed-reliability claim. Actual overrides42.958%;
guided returns are not competence. Retain12M greedy as a promising development
budget point and4M as cheaper/stronger sampled control. No extension,confirmation
or automatic successor is selected. Use the matching Breakout-era runtime and
pinned Qbert runner;native447fe50 unchanged. See the expert100-capacity protocol
and completed cheap-Qbert result;check live state before GPU work.

Retained Breakout corrective greedy assessment completes:final5.000/initial2.583
versus sampled4.292/1.375,all0/24 two-wall successes. Original17m25s training cost
remains;zero new training/updates. All54 pins,357 invariant tensors per arm,source/
config/counters,full outcome replays/videos and guards pass. Total15,248 actions/
60,929 frames/7.601s evaluation;main succeeds23:45:31 UTC in23.812s,postprocessing
23:45:32 on October28. No failure,GPU repeat or cap change. Retain both weak modes
and stronger controls;no extension,confirmation,global default or automatic next
allocation. Both services are terminal and the host is idle;check live state. [Protocol](docs/experiments/2026-10-28-cdp-breakout-corrective-greedy.md).

Breakout encoder-rate200k completes200k actions/799,152 frames/24,970 updates
in17m41s. Raising encoder6e-6→4e-5 lowers frozen4.292→1.458 versus own initial
1.375; both0/24 two-wall successes. Training costs1.52% more. All64 pins,100k
recovery,357 matched control/check initial and invariant frozen tensors,802 paired
initial semantic events,finite reports,full replays,guards,curves,exposure and
comparison pass. Main succeeds22:45:00 UTC,replacement CPU postprocessing22:45:15
on October28. The copied CPU preparation error and proactively replaced waiting
comparison observer are retained; no GPU stage repeats or caps change. Actual
overrides67.622%; online guided returns are not competence. Retain encoder6e-6;
no extension,seed confirmation or automatic successor. Native447fe50 unchanged;
optional CLI passes348 CPU tests. The device was idle at completion. See the encoder-rate
protocol and completed pixel-reference result; check live state before GPU work.

Breakout corrective200k completes200k actions/799,234 frames/24,970 updates in
17m25s,26.02% less training time than the old forced-only cap1920 control.
Frozen4.292 versus own initial1.375 improves the old guided2.625 but stays below
unaided full18's7.083; all0/24 two-wall successes. All64 pins,100k recovery,
346 shared old/357 bounded initial and357 invariant frozen tensors,paired initial
semantic traces,finite reports,full replays,guards,curves,exposure and comparison
pass. Main succeeds22:12:00 UTC,postprocessing22:12:22 on October28. Original
CPU observer startup failure remains; replacement succeeds,no GPU stage repeats.
Actual overrides67.135%; guided mean203.275 is not competence. Multiple settings
and native/runtime change together on reused1009. Retain stronger controls;
no extension,seed confirmation or automatic successor. The device was idle at completion.
See the corrective-200k protocol and completed pixel-reference result.

Optional Breakout corrective labels pass510 CPU tests,forced-reference parity
and181,224 recorded learner-action feedback checks. Applied source2854dd9 keeps
native447fe50 unchanged. The fresh8192-action/32,725-frame/994-update check takes
36.919s;1024 frozen actions/4096 frames take.555s,zero updates/no natural episodes.
All41 pins,346 shared initial/357 invariant frozen tensors,finite reports,changed
CNN/reward/actor/value,full label/pixel/action replay and guards pass. Main succeeds
21:43:38 UTC on October28. No task competence or cap-release coverage is measured.
Use`/x/Temp/kindle-breakout-corrective-20261028/runtime` for this adapter; original
sources remain archived. Early200k projection15m01s; no full learner was selected at bounded completion.
See the Breakout corrective-label protocol and pixel-reference result. The
separately selected full comparison is completed above.

Qbert expert100/200k/R16 completes200k actions/799,703 frames/24,970 updates in
26m07s. Greedy9654.167/23 first pyramids improves100k/R32's8365.625/24 by
15.40%,for37.63% more training time; sampled7569.792/24 versus7339.583/22.
Own initials263.542/182.292,zero pyramids. Both still miss mean15000. All138 pins,
100k recovery,357 matched initial/invariant frozen tensors,paired initial traces,
full replays,guards,curves,coverage,exposure and comparison pass. Main succeeds
21:41:05 UTC,postprocessing21:41:19 on October28; all services are inactive.
Actual overrides41.383%; guided returns are not competence. Budget and ratio
change together on reused4013; no isolated effect or seed confirmation. Retain
both budget points; no extension or automatic successor. See the expert100-r16
protocol and completed result in cheap-Qbert; check live state before GPU work.

Qbert expert100 completes100k actions/399,886 frames/24,939 updates in18m59s.
Greedy8365.625/24 first pyramids improves scale10's7700/24 by8.64%; sampled
7339.583/22 improves5572.917/24 in mean but loses two pyramids. Own initials
263.542/182.292,zero pyramids. Both still miss mean15000. All111 pins,50k recovery,
357 matched initial/invariant frozen tensors,paired initial traces,full replays,
guards,curves,coverage,exposure and comparison pass. Main succeeds20:55:18 UTC,
postprocessing20:55:29 on October28; all services inactive. Actual overrides
38.397%; guided returns are not competence. Only expert10→100 changes,affecting
actor/world supervision together,on reused4013. Modest unconfirmed same-budget
gain; no extension,second expert100 learner or automatic successor selected.
The bounded check's CPU preparation schema failure remains,no GPU repeat.
See the expert-scale100 protocol and completed result in cheap-Qbert; check
live state before GPU work.

The fresh Qbert pixel-guided comparison completes 200k actions/49,939 updates
in 20m13s, but lowers unaided frozen score 809.375→633.333 and first-pyramid
coverage 16.625→11.667/21; both have 0/24 pyramids. The 5% episode rule actually
overrides 23,428 actions (11.714%). All audits, 100k recovery, 346 matched initial
tensors/semantic traces, complete pixel/action replay and comparison pass;
service exits 20:07:01 UTC. Retain unaided; no extension or confirmation.
See [the result](docs/results/2026-10-25-cdp-cheap-qbert.md#pixel-guided-training-lowers-unaided-frozen-quality).

Pong RGB32 completes500k/124,939 updates in42m19s,15.24% less training time
than RGB48, but lowers frozen14.958→2.292 and wins24/24→16/24. All numerical,
config/source/counter/346-tensor/replay/video/guard/recovery/comparison checks
pass; service exits21:02:34 UTC. Retain RGB48; no extension or confirmation.
Geometry/initialization, older control kernel and13.690s unintended integrated-
GPU inference during this run limit attribution. Full evidence is in cheap-Pong.

Qbert cap640 completes200k actions/799,403 frames/49,939 updates in20m00s.
Frozen750.000 improves whole-episode guidance633.333 but stays below unaided
809.375; all0/24 pyramids. Coverage16.208/21 versus11.667 and16.625. Actual
training overrides15,360 (7.680%);23 completed guided episodes clear a pyramid,
none of366 unguided episodes does. All audits,100k recovery,346 matched initial
and unchanged frozen tensors, semantic initial traces, full training pixel/
action replay and comparison pass; service exits21:30:21 UTC. Retain unaided.
No extension or confirmation follows capped Qbert. The applied
640-action option passes444 CPU tests; native unchanged. Use the matching
`runtime-qbert-guidance-cap` overlay. Prior isolation failures remain recorded.
See [the completed result](docs/results/2026-10-25-cdp-cheap-qbert.md#capped-guidance-recovers-some-quality-but-remains-below-unaided-training).

Pong B16 behavior rate8e-5 completes500k actions/1,999,751 frames/62,454
updates in37m38s. Frozen12.375/24 wins improves original B16's11.875 slightly,
but trails B8's14.958; all miss mean15. All audits, four recoveries,346 matched
initial/unchanged frozen tensors, semantic initial traces, replays, guards and
comparison pass; service exits22:20:58 UTC. Retain B8 as the strongest compact
Pong recipe; no extension or seed confirmation follows this small gain.
The CPU startup import failure remains recorded; no GPU stage repeated.
CLI`a2c467c` passes460 CPU tests; native unchanged. See
[the completed comparison](docs/results/2026-10-25-cdp-cheap-pong.md#behavior-rate-doubling-gives-a-small-gain-but-still-misses-the-gate).

QbertRGB48 frozen forecasts complete on15,376 recorded actions/61,462 frames
per model in50.330s trained/49.268s initial, with zero updates. One-step reward
AUC improves.500→.969; H15 reward-sum MAE22.487→12.400. All346 frozen tensors,
8 matched stream trajectories/pixel hashes, source/counter/support/guard checks
pass; service exits22:30:05 UTC. This shows predictive learning on the reused
cohort, not the cause of weak policy transfer. Retain4M48/H15; no new training
or automatic next allocation. The geometry helper passes24 CPU tests; native
unchanged. See [forecast results](docs/results/2026-10-25-cdp-cheap-qbert.md#compact-qbert-learns-reward-forecasts-despite-weak-task-performance). Check live state before GPU work.


Pong12M48/Auto/N8/B16/T16/H15/R16 completes500k actions /31,227 updates in
41m28s,16.95% less training time than4M/B8/R32. Frozen15.625/24 wins versus
own initial−20.292/0 meets mean15/90% on one learner. All audits, four recoveries,
346 frozen tensors,42 pins, recorded-action replays, guards and comparison pass;
service exits01:51:06 UTC, CPU postprocessing01:51:35 on October27.
No live pixel hashes are available in ordinary unguided logs. See the protocol
scope correction and [completed result](docs/results/2026-10-25-cdp-cheap-pong.md#larger-capacity-with-less-replay-work-reaches-the-gate).

Pong12M48 fresh4013 completes500k actions /1,999,720 frames /31,227 updates
in41m31s. Frozen14.833/24 wins versus own initial−20.542/0 narrowly misses
mean15; primary1009 scores15.625/24 wins in41m28s. Both improve their paired
4M/B8 controls14.000/14.958 with about17% less training time, but only one of
two learners passes the full gate. All numerical/config/counter checks, four
recoveries,346 frozen tensors,53 pins, replays, guards and comparison pass.
Service exits02:41:08 UTC, CPU postprocessing02:41:23 on October27. Original
progress-observer failure is retained; no GPU stage repeats. Both reused
seeds are development evidence. No extension or third seed is selected.
See [completed confirmation](docs/results/2026-10-25-cdp-cheap-pong.md#capacity-confirmation-improves4013-but-narrowly-misses-the-gate).

Breakout expert200k completes798,964 frames /49,939 updates in22m38s,
frozen4.000 versus initial1.375, below full18 control7.083/minimal4 9.542;
all0/24 two-wall successes. All audits,100k recovery,346 matched initial/frozen
tensors,46 pins, replays, guards, comparison and exposure pass. Service exits
03:16:09 UTC; postprocessing03:16:14. Retain controls; no extension/confirmation.
20,463 trusted actions (10.2315%); guided/unguided completed means24.871/2.948
are not frozen competence. See [result](docs/results/2026-10-26-cdp-breakout-pixel-reference.md#capped-native-imitation-lowers-frozen-breakout-quality).

Qbert continuation1009 completes200k actions /799,538 frames /49,939 updates
in26m38s. Frozen3137.500/17of24 first pyramids versus own initial180.208/0
improves old-teacher1083.333/2; continuation4013 scores4730.208/23 in25m23s.
Both seeds improve, neither meets mean15000;1009 also misses90% pyramids.
All audits,100k recovery,346 matched initial/unchanged frozen tensors, paired
initial semantic traces,60 pins, replays, guards, curves, coverage, exposure and
four-arm comparison pass. Main service exits04:47:41 UTC; postprocessing04:47:57
on October27. Actual1009 guidance38,400 actions (19.2%); online guided returns
are not frozen competence. Both reused development seeds; no broad reliability,
extension or third seed. See the paired continuation result in cheap-Qbert.

The Breakout cap1920 bounded check completes16,384 actions/65,473 frames/
4035 updates in114.327s;1024 frozen/4096 frames/.558s, zero updates and no
natural frozen episodes. All32 pins,346 matched initial/unchanged frozen tensors,
finite reports, CNN/reward/actor/value changes,10,125 trusted actions/3829
supervised reports, exact guided replay and guards pass. Service exits05:38:24
UTC on October27. A copied4096-action CPU exposure assumption is repaired;
original failure retained, no GPU repeat. Natural ends/late starts leave0 released
actions in this check; do not claim cap-release coverage or competence.

Breakout cap1920 completes200k actions /798,815 frames /49,939 updates in
23m33s. Frozen2.625 versus own initial1.375 is below cap640's4.000; both0/24
two-wall successes. Training costs4.02% more. All audits,100k recovery,346
matched initial/unchanged frozen tensors, paired initial semantic traces,47 pins,
full replays/videos, guards, curves, comparison and exposure pass. Main exits
06:07:00 UTC; postprocessing06:07:30 on October27. Actual guidance46,833 actions
(23.4165%); guided online mean169.700 is not frozen competence. Retain controls;
no extension or confirmation. See the completed longer-guidance result in the
Breakout pixel-reference report.

Qbert continuation greedy evaluation completes on both retained learners.
1009 scores3058.333/16of24 first pyramids versus sampled3137.500/17;4013
scores3770.833/19 versus4730.208/23. Own greedy initials85.417/263.542,0pyramids.
Both greedy arms fail mean15000/90%; retain sampled actions for Qbert.
Four arms total65,104 actions /260,268 frames /30.277s evaluation, zero new
training/updates; entire service1m17.712s. Starts06:12:49 UTC/source`62482da`,
main exits06:14:07, postprocessing06:14:10 on October27. All60 pins, source
invariance,346 invariant tensors per arm, full outcome replays/videos and
host guards pass. Original26m38s/25m23s training costs and sampled results
remain. No retry, extension, third learner or automatic successor selected.
That frozen assessment is terminal. See the completed greedy comparison in
cheap-Qbert. The five-game goal remains unfinished.

Optional native expert-world gradients pass two guarded derivative/alignment
checks and a real Qbert8192-action/1987-update check in80.263s; frozen1024/.489s,
zero updates/no natural episodes. All357 finite/invariant tensors,346 shared
initial tensors,35 pins, full guided replay and guards pass.4520 trusted actions,
1978 supervised reports; max world/actor expert-loss difference1.5e−7.
Original service fails06:31:56 UTC only at the CPU schema validator; a separate
CPU-only repair completes audits without repeating GPU work or changing pins.
129 Rust CPU tests,474 Python tests and strict Clippy pass. Native0e157df,
SHA6a9c41c; default off. Use`/x/Temp/kindle-expert-world-20261027/runtime-audited`
for future work; its packaging receipt pins native/wheel/Python source. The2.6GiB
build cache is released; wheels/test executable and all evidence remain.
Qbert4013/4M48/R16/20% with expert world gradients completes200k actions /
799,662 frames /24,970 updates in21m28s. Frozen3670.833/19of24 first pyramids
versus own initial182.292/0 improves matched R16 control3333.333/18 by10.13%
with2.32% less measured training time, but remains below5%/R32's4730.208/23.
All audits,100k recovery,357 invariant tensors,346 shared initial matches,
paired initial semantic traces,52 pins, full replays, guards and comparison pass.
Main succeeds15:21:35 UTC on October28; original CPU postprocessing fails at
15:22:04 on an eager ALE import. A separate lazy-import repair completes at
15:24:27; original failure/pins remain, no GPU repeat. Actual guidance46.088%.
Retain5%/R32 as stronger; small reused-seed R16 gain gets no extension or
confirmation. New native/runner identities limit attribution. See the completed
expert-world result in cheap-Qbert and its October28 protocol. A separate
12M48/B16/R16/20%/world-gradient check completes8192 actions/32,756 frames/489
updates in60.333s;1024 frozen/.810s, zero updates/no natural episodes. All41
pins,357 finite/invariant tensors,346 shared12M initial tensors, parameter
changes,489 supervised reports, full guided replay and guards pass.4492 trusted
actions; max world/actor expert-loss difference1.4e−7. Service succeeds15:49:49
UTC on October28. Original prelaunch first384/0 assumption and unselected-service
failure remain; reviewed first376/residual.5 passes. No GPU stage repeats.
Use`/x/Temp/kindle-qbert-expert-capacity-20261028/reviewed`. Early200k cost
projects24.55min, not competence or measured full cost. Model/batch change
together. The separate fresh200k comparison completes799,693 frames/12,477
updates in24m51s. Frozen1792.708/9of24 versus own initial177.083/0 is51.16%
below same-native4M/B8 control3670.833/19, at15.74% more training cost. All63
pins,100k recovery,357 initial/frozen tensors,finite reports,parameter changes,
full replays,guards,curves,coverage,exposure and comparison pass. Main succeeds
16:40:13 UTC,postprocessing16:40:29 on October28. Actual guidance46.411%.
Retain4M; no extension or confirmation. Model/batch change together on reused4013.
Full root`/x/Temp/kindle-qbert-expert-capacity-20261028/full`; see the completed
expert-capacity result in cheap-Qbert. That life is terminal; the separately
selected corrective-label check is below. Check live state before GPU work.

Independent corrective labels are optional and separately validated: native
447fe50/SHA35ad0db,130 Rust CPU/491 packaged Python tests,strict Clippy,a5.87s
native diagnostic and8192/994 bounded Qbert check in61.976s. Runtime
`/x/Temp/kindle-corrective-labels-20261028/runtime`;2.6GiB cache released.
Fresh200k corrective training completes799,682 frames/24,970 updates in25m57s,
frozen3633.333/17of24 pyramids versus own182.292/0 and control3670.833/19.
Training costs20.86% more. All52 pins,100k recovery,357 matching initial/unchanged
frozen tensors,paired initial traces,full label/pixel/action replays,guards,
curves,coverage,exposure and comparison pass. Main exits18:06:30 UTC,CPU
postprocessing18:07:02 on October28; no GPU repeats. Display-only caption
correction preserves original data/pins. Retain control; no extension/confirmation.
200k labels include103,712 passive states;96,288 overrides are48.144%. Late
passive teacher/learner NOOP frequencies79.447%/46.077% suggest checking wait-label
weight, not a causal diagnosis. Root`/x/Temp/kindle-qbert-corrective-full-20261028`.
See the corrective-labels protocol/completed cheap-Qbert result.

The R32/wait-weight1/100k recipe completes100k actions/399,864 frames/24,939
updates in18m58s,26.90% less training time but37.79% lower frozen mean than the
corrective200k control:2260.417/12of24 first pyramids versus3633.333/17. Own
initial182.292/0; both original15k/90% criteria fail. All55 pins,50k recovery,
357 matched initial/invariant frozen tensors,identical paired initial traces,
finite reports,equal row/weight sums,full replays/videos,guards,curves,coverage,
exposure and comparison pass. Main succeeds18:53:59 UTC,postprocessing18:54:18
on October28. Actual overrides47,072 (47.072%); guided returns are not competence.
Original CPU observer argument failure is retained; no GPU stage repeats.
Retain controls,including stronger5%/R32/world-off4730.208/23. Weight,ratio and
budget change together on reused4013/cohort; no isolated attribution,extension,
confirmation or automatic successor. Root`/x/Temp/kindle-qbert-wait-full-20261028`.
The host is idle at this milestone. See the completed wait-labels result and
check live state before GPU work.

The retained100k/wait1 greedy assessment improves2260.417/12→4965.625/21of24
first pyramids, own initial263.542/0; both original15k/90% criteria still fail.
Zero new training/updates;41,976 actions/167,838 frames/19.462s evaluation,
whole service46.404s, exits19:03:28 UTC; postprocessing19:03:31 on October28.
All47 pins/55 source-training inputs,357 invariant tensors per arm,replays/videos
and guards pass. Retain greedy as a useful18m58s development option on this one
reused learner, not a global default or independent confirmation. No extension,
learner confirmation or automatic successor. See the completed wait-greedy result.

Qbert expert scale10 completes100k actions/399,860 frames/24,939 updates in
18m55s versus scale1's18m58s. Greedy7700/24of24 first pyramids improves
4965.625/21; sampled5572.917/24 improves2260.417/12. Own sampled/greedy initials
182.292/263.542,0pyramids. Both still fail mean15000. All84 pins,50k recovery,
357 matched initial/invariant frozen tensors,paired sampled/greedy initial traces,
finite reports,full training label/pixel/action and frozen outcome replays,
guards,curves,coverage,exposure and comparison pass. Main succeeds19:42:39 UTC,
postprocessing19:42:59 on October28; observers also succeed. Only expert scale
changes, affecting actor/world supervision together; actual overrides42.147%.
Retain scale10/greedy as the stronger100k development recipe on reused4013;
no extension, second learner or successor is selected. No allocation is active
at this milestone. See the completed expert-scale10 result in cheap-Qbert and
its protocol; check live state before GPU work.

Expert10 seed1009 completes100k actions/399,823 frames/24,939 updates in19m11s.
Greedy3832.292/23of24 first pyramids versus own85.417/0 trails4013's7700/24 in
18m55s; sampled2667.708/15 versus own180.208/0 trails5572.917/24. Both greedy
learners meet90% pyramids,neither mean15000. All115 pins,50k recovery,357 finite
initial/invariant frozen tensors,346 shared legacy1009 tensors,matching sampled
initial semantic trace,full replays/videos,guards,curves,coverage,exposure and
comparison pass. Main succeeds20:16:56 UTC,postprocessing20:17:25 on October28;
observers also succeed. Actual overrides40.745%; guided returns are not competence.
Both reused development seeds; no matched100k scale1 control on1009 or broad
reliability. No extension,third learner or automatic successor selected. Host idle
at completion; retain both results and check live state before GPU work.
See`docs/experiments/2026-10-28-cdp-qbert-expert10-seed1009.md`; check live state.






Qbert R16/20% completes200k actions /799,662 frames /24,970 updates in21m59s,
13.41% less training time than5%/R32, but frozen3333.333/18of24 first pyramids
is below4730.208/23. Own initial182.292/0; both miss mean15000. All audits,
100k recovery,346 matched initial/unchanged frozen tensors, paired initial
semantic traces,47 pins, full replays, guards, curves, coverage, exposure and
comparison pass. Main exits05:23:55 UTC; postprocessing05:24:19 on October27.
Actual guidance91,913 actions (45.9565%); online guided scores are not frozen
competence. Retain5%/R32; no extension or confirmation. Probability and ratio
change together on reused4013; no isolated causal claim. See the completed
higher-exposure comparison in cheap-Qbert.

Pong12M greedy evaluation completes on both retained learners:1009/4013 each
score15.208/24of24 wins, versus own initial−20.583/−20.625 and0wins. Both pass
the original mean15/90%-win gate under changed greedy inference; original
sampled15.625/14.833 and one-of-two outcome remain. Four arms total202,768
actions/810,927 frames/111.916s of evaluation, zero new training/updates;
main service3m33s, exits05:29:46 UTC, postprocessing05:29:51 on October27.
All58 pins,346 invariant tensors per arm, full replays/videos and guards pass.
Retain greedy as a useful12M development option, not a global default or new
independent learner evidence. Both means are only.208 above15; uncertainty
intervals cross15. Original training costs41m28s/41m31s remain. No extension,
retry, third learner or successor allocation selected. That assessment is terminal. See the paired greedy result in cheap-Pong;
check live state before GPU work.

The R16/20% bounded check completes8192 actions/32,752 frames/994 updates in
52.742s, frozen1024/.524s with346 invariant tensors and no completed natural
episodes. All33 pins,346 matched initial tensors, finite reports, changed
CNN/reward/actor/value,4527 trusted actions/990 supervised reports, exact guided
replay and guards pass; service exits04:52:23 UTC on October27. No GPU stage
repeats or check experience is reused. This is execution/early-cost evidence,
not competence. Protocol remains in the continuation experiment.

Qbert continuation200k completes799,570 frames /49,939 updates in25m23s.
Frozen4730.208/23of24 first pyramids versus own initial182.292/0 improves
old-teacher200k's3153.125/17 by50.02% for26.17% more training time. It also
beats500k/R16's3752.083/21 with10.45% less time. Mean15000 still fails.
All audits,100k recovery,346 matched initial/unchanged frozen tensors, identical
initial semantic traces,56 pins, replays, guards, curves, coverage and exposure
pass; main service exits04:07:44 UTC, CPU postprocessing04:08:08 on October27.
Keep4M48/Auto/N8/B8/T16/H15/R32/coefficient1/actor0/full18/original rates,
5%episode continuation/cap1920/expert1/NOOPweight.05 as a promising development
recipe. Actual guidance34,468 actions (17.234%); guided online scores are not
unaided competence. Teacher and cap change together on reused4013; no isolated
causal effect or seed reliability claim. No extension of this completed life.
See [completed result](docs/results/2026-10-25-cdp-cheap-qbert.md#continuation-guidance-improves-frozen-learning-but-misses-the-full-gate).
Check live state before GPU work.

Qbert CPU continuation reference scores22050/24 first pyramids, all24 above15k,
with exact input/output/reset pixel replay; not native learning or held-out
confirmation. The exact456-test optional adapter is applied after terminal
Breakout audits;24,733-action parity passes, original first-pyramid AST/native
unchanged. Native bounded8192/1987 check completes32,739 frames in70.493s,
frozen1024/.531s with346 invariant tensors. All32 pins,346 matched initial
weights, finite reports, parameter changes,2987 trusted labels/1968 nonzero
expert reports, full guided replay and guards pass; service exits03:23:09 UTC.
Configuration4M48/Auto/N8/B8/T16/H15/R32/coefficient1/actor0/full18/original
rates,4013/eval4000004013,25%episodes/cap640/expert1/NOOPweight.05.
No frozen natural episodes complete in the bounded check; do not infer task
competence from it. See [result](docs/results/2026-10-25-cdp-cheap-qbert.md#native-continuation-check-passes)
and [protocol](docs/experiments/2026-10-27-cdp-qbert-continuation.md).
Check live state before GPU work.


CPU Breakout predictive reference scores166.167 across24 natural episodes,
0/24 two-wall successes; full input/output pixel-hash replay passes. Selected
after one9001 and eight reused evaluation episodes; those8 are included in24.
This reference work preceded the separately selected bounded adapter check.
55.935s affined CPU work
co-hosted with Pong; all42 active pins unchanged. See [reference result](docs/results/2026-10-26-cdp-breakout-pixel-reference.md#calibrated-paddle-response-improves-the-reference-but-misses-the-two-wall-gate).

Fresh Qbert4013 expert500k/R16 completes **500k actions /1,998,773 frames
/62,470 updates in28m20s**. Frozen **3752.083 /21 of24 first pyramids** versus
own initial182.292 /0 improves200k/R32's3153.125 /17, but costs40.88% more
training time for19.00% higher mean. Both miss the original15k/90% gate.
All audits, four recoveries,346 matched initial/unchanged frozen tensors,
identical initial semantic traces, full pixel/action replays, guards, curves,
coverage, exposure and comparison pass. All67 declared pins stay unchanged.
Service exits00:48:34 UTC on October27; postprocessing completes00:48:42.
Keep both measured budget points; no extension or seed confirmation is selected.
This completed comparison reuses4013 after its stronger result,
changes budget and ratio together, and is not independent seed confirmation.
CPU teacher prototypes remain external and unselected. See
[completed result](docs/results/2026-10-25-cdp-cheap-qbert.md#more-experience-at-r16-improves-first-pyramid-success-with-higher-cost)
and [protocol](docs/experiments/2026-10-27-cdp-qbert-expert-r16.md).
Check live state before GPU work.

The fresh Qbert4013 expert-imitation check completes 200k actions /799,532
frames /49,939 updates in **20m07s**. Frozen **3153.125 /17 of24 first pyramids**
versus own initial182.292 /0; coverage20.167 /21. Seed1009 scores1083.333 /2
in20m11s with the same recipe, so both improve but vary substantially. Retain
4M48/Auto/N8/B8/T16/H15/R32/coefficient1/actor0/full18, original rates,
5% episode guidance/cap640/expert1/NOOPweight.05. All audits, 100k recovery,
346 initial/frozen tensors, full pixel/action replay, guards, comparison and
CPU tile replay pass; all50 declaration pins stay unchanged. Training uses
11,520 trusted actions (5.760%); frozen evaluation has no guidance.
Service exits23:53:38 UTC on October26; postprocessing23:54:08, tile replay23:54:40.
Both learners miss the mean15000/90%-pyramid gate. No paired unaided Qbert4013
control exists. Both completed lives are terminal; do not extend either life.
See [completed seed check](docs/results/2026-10-25-cdp-cheap-qbert.md#expert-imitation-transfers-on4013-with-substantial-seed-variation)
and [protocol](docs/experiments/2026-10-26-cdp-qbert-expert-seed4013.md).
Check live state before GPU work.

The fresh Qbert expert-imitation comparison completes200k actions/799,509
frames/49,939 updates in20m11s. Frozen1083.333/2of24 first pyramids improves
unaided809.375/0 and capped guidance750.000/0; coverage18.625/21 versus16.625
and16.208. Own initial180.208/0. All audits,100k recovery,346 initial/frozen
tensors, matched initial traces, full pixel replay, guards and comparison pass;
service exits23:23:23 UTC.5% episode guidance/cap640 actually labels14,080
actions (7.040%); expert scale1/NOOPweight.05. Native79c9a7f/runtime
`/x/Temp/kindle-expert-20261026/runtime`. The completed fresh-seed check above
also improves over its own initial; the mean15000/90% gate still fails.
The completed1009 service is terminal; do not restart or extend it.
See [completed result](docs/results/2026-10-25-cdp-cheap-qbert.md#expert-imitation-improves-compact-qbert-but-remains-far-below-the-gate).

Optional native expert imitation passes127 Rust CPU tests/strict Clippy,
477 packaged Python tests/binding Clippy, analytic GPU gradients and an actor-
only supervised update. Fresh Qbert4096 actions/16,378 frames/963 updates take
22.662s;1024 frozen actions take.452s,346 matched initial/unchanged frozen tensors,
512 verified expert actions (38 movement),611 nonzero auxiliary reports and full
pixel replay/guards pass. Service exits22:49:20; strengthened actor-only check
exits22:51:05 UTC. Native`79c9a7f`, isolated runtime
`/x/Temp/kindle-expert-20261026/runtime`; original native retained. Scale1,
NOOPweight.05,5% episode guidance/cap640; no natural-episode competence measured.
Build cache released after packaging. The bounded checks are complete; the
completed full comparison is described above.
See [bounded result](docs/results/2026-10-25-cdp-cheap-qbert.md#native-expert-imitation-smoke-passes-task-benefit-remains-unmeasured).

The bounded guidance check completes4096/963 in22.541s and1024 frozen actions
in0.452s;512 verified overrides,346 matched initial/unchanged frozen tensors,
all finite reports and both guards pass. Original service exits failed19:40:14
UTC because the new CPU replay used a frozen-oriented reward reader. The
training-aware correction passes421 CPU tests and complete saved-action/pixel
replay; original failure retained, no GPU repeat. See cheap-Qbert report.

Pong B16 actor/critic gradients finish500k/62,454 updates in37m52s but lower
frozen11.875→−0.542 and wins24/24→10/24. Own initial−20.375/0 wins. All357-tensor/
replay/guard/recovery/curve/comparison checks and346 shared initial tensors/
semantic traces pass; service exits19:36:47 UTC. Retain disabled gradients;
no extension or confirmation. Progress-observer interruption did not restart training.

The CPU-only Qbert pixel rule clears24/24 first pyramids, mean3764.583 and
coverage21/21; complete replay passes. It misses mean15000 and is hand-coded
guidance evidence, not Kindle success. Bounded native integration passes;
the completed 200k test lowers unaided quality. See the cheap-Qbert report.

The bounded PongB16 check passes4096/466 in16.126s and1024 frozen actions in
0.436s, all357 tensors unchanged,346 shared initial tensors matching control,
all466 initial policy/value losses nonzero and both guards passing. Service
exits18:51:00 UTC. Native/CLI unchanged; this is execution evidence only.

The completed Pong greedy assessment scores14.333 versus sampled14.958, both24/24 wins and below
mean15. Own initial−20.833/0 wins; zero learning. Full config/counter/346-tensor/
replay/guard checks and source-file invariance pass; service exits18:44:25 UTC.
Retain sampled evaluation and the learner's49m55s training cost; no greedy
confirmation. Check live state before GPU work. The five-game goal is unfinished.

QbertRGB64 finishes200k/49,939 updates in22m20s,13.14% more time than RGB48,
with lower frozen718.750 versus809.375 and coverage15.083 versus16.625/21;
both0/24 pyramids. All numerical/config/counter/tensor/replay/guard/recovery/
comparison checks pass; service exits18:39:49 UTC. Retain RGB48; no extension
or confirmation. See cheap-Qbert and cheap-Pong reports for evidence/limits.

Breakout actor/critic gradients finish200k/49,939 updates in20m41s, frozen11.208
versus9.542, both0/24 two-wall successes. All357-tensor/recovery/replay/guard
checks and matched shared initial weights/traces pass; main service exits
18:05:14 UTC. Original CPU comparison import failure is retained; separate
repair passes without GPU repetition. Modest unconfirmed gain, older-kernel
control; no extension, seed confirmation or global default change selected.

The bounded actor/critic check passes4096/963 in22.474s and1024 frozen actions
in0.489s, all357 tensors unchanged, all963 initial policy/value losses nonzero,
changed CNN/reward weights and346 shared initial tensors matching control.
Both guards pass; service exits17:34:23 UTC. The exact313-test CLI patch is
applied; native unchanged. Full-run recovery/frozen audits use the observed
357-tensor schema including11 frozen actor copies. This is execution evidence,
not task competence or gradient correctness.

Pong minimal6 completes500k/124,939 updates in44m21s,11.17% less training time,
but frozen10.375 is below full18's14.958; both24/24 wins and below mean15. All
audits, four recoveries, replays, guards and comparison pass; service exits
17:28:27 UTC. Retain full18; no extension or conditional4013 confirmation.

Qbert4M48/B8/T16/H30/R32 completes200k/49,939 updates in26m43s,35.34% more
training time than H15. Frozen582.292 versus809.375 and coverage9.583 versus
16.625/21 are worse; both0/24 pyramids. Retain H15. Final training/frozen/replay/
guard/comparison checks and matched346 initial tensors/semantic traces pass.
The declared100k recovery was never written: pinned queue saved every200k.
Original watcher failures remain; completion is explicitly protocol-incomplete.
See`docs/experiments/2026-10-26-cdp-qbert-long-imagination.md`. No GPU repeat.

PongB16/R40's500k target is interrupted by a host restart:265,440 persisted
actions/41,416 finite reports, saved200k/31,192 updates. Separate frozen saved/
initial scores−9.708/−20.375, both0/24 wins. Full frozen checks pass after a
CPU checker correction; original failures and torn tails remain. No resume.
New boot a8648671 uses kernel7.0.0-38/AMD module9BDBC866F55BF296F1B7FA2;
same RX7900XT/Mesa26.0.8/native binary. Host scan and512-action frozen restore
pass with346 unchanged tensors. No host recovery. Qbert's H15 control used
7.0.0-34; retain that kernel change as a comparison confound.

Breakout coefficient.01 completes200k/49,939 updates in21m07s, but lowers
frozen9.542→5.042, both0/24 two-wall successes. All audits/comparison pass after
CPU-only repair of an omitted replay manifest. Original service failure at
13:43:51 UTC remains recorded; no GPU stage repeats. Retain coefficient1.

PongH5 completes500k/124,939 updates in38m07s,23.65% less training time than H15,
but frozen8.667 is below14.958; both24/24 wins and below mean15. All audits,
matched initial tensors/traces and comparison pass; service exits13:18:17 UTC.
Retain H15; no extension or conditional4013 confirmation selected.

Qbert's5% training aid completes200k in20m17s, but lowers frozen score809.375→
678.125 and coverage16.625→15.125, both0/24 pyramids. All audits/comparison and
matched initial tensors/traces pass; service exits12:24:16 UTC. Retain unaided
recipe; no extension or confirmation selected.

PongB16 doubled rates finish500k in38m32s, with frozen9.125 versus11.875 for
original rates; both24/24 wins and below mean15. All audits/comparison and
matched initial tensors/traces pass; service exits11:59:15 UTC. Retain original
rates; no extension or seed confirmation selected.

Qbert12M48/B16 completes200k in32m19s, scores959.375/0of24 pyramids and coverage
16.292/21, versus4M/B8's809.375/0of24 and16.625/21 in19m45s. All audits and
comparison pass; service exits11:15:13 UTC. No extension or confirmation.

PongB16 completes500k/62,454 updates in38m45s,22.37% less training time than B8,
but frozen11.875 is below14.958; both24/24 wins and below mean15. All audits and
comparison pass; service exits10:28:51 UTC. No extension or confirmation.
Qbert500k/R16 remains weaker than its200k/R32 control, with0/24 pyramids.

Original compact runtime/evidence: `/x/Temp/kindle-cheap-20261025`. Native
`bfb8c31`, frozen CPU replay`5043b7a`; its retained runner adds guidance limits,
with runner SHA256`50ba7759d4ff108d724e831d83fda69c9d4f4dce3da660cebf394f995cac98da`.
These hashes describe completed original-runtime runs. Expert imitation uses
the isolated79c9a7f runtime above; do not mix overlays. Each new declaration
pins its exact source/runtime hashes. The handover
locates the build receipt. Native 32/48/64 geometry and the
intermediate 4M preset pass bounded training/frozen checks, including 64 restore.
The CLI exposes imagination length for fresh runs; defaults retain 15 and
restores reject overrides. Do not restart completed training, old queues, broad
seed campaigns or unchanged long extensions. Keep single-actor learning ahead
of swarms/concurrent learners.

Qbert25M48/B16/R16 completes200k actions /799,730 frames /12,477 updates in
42m51s,40.72% more time than12M. Greedy11984.375→9773.958 (−18.44%),both24/24;
sampled6759.375→8678.125 (+28.39%),24→22 pyramids. Own initials9.375/177.083
clear none;both final means miss15000. Retain12M/R16;preserve sampled gain
separately,no extension/confirmation/successor. All273 pins,100k recovery,
357 matching25M bounded/R8 initial and invariant frozen tensors,paired initial
traces,full replays/guards/curves/coverage/exposure/comparison pass. Only model
size changes from12M;geometry/initialization differs,no12M initial equality.
Actual guidance41.6855%. Main succeeds13:38:47 UTC October29/source5d9d93e,
postprocessing13:38:55,collection13:39:32;all services succeed. CPU preparation
failure and2.290s co-host diagnosis retained,no GPU repeat. No active allocation.
See25M/R16 result;the five-game goal remains unfinished.

The13:17:13 UTC CPU-only KL-floor diagnosis finds late mean dynamics KL5.491
Qbert/11.357 Breakout,above floor1;per-state clipping/derivatives unmeasured.
All9 analysis inputs/273 active pins/main invocation unchanged;2.290s low-priority
co-host work is a timing caveat. No additional learner selected;see diagnosis.

Qbert25M48/B16/R16 bounded check passes8192 actions /32,757 frames /489 updates
in100.299s;1024 greedy frozen/4096 frames take1.202s,zero updates/no natural episodes.
All53 pins,357 finite/invariant frozen tensors,357 matching25M/R8 initial tensors,
derived geometry/parameter changes/finite expert reports/full guided replay/guards
pass. Only exposed model size changes from12M/R16;update schedule/supervision/
native447fe50/control runner/teacher fixed. No12M initial equality. Conservative
200k projection44m54s narrowly passes45min;execution/early cost,not competence.
Actual short-check guidance55.457%. Service succeeds12:33:10 UTC October29/source
`a25513b`,collection12:33:56. No failure/GPU repeat/full learner or promotion.
See completed25M/R16 result;check live state before GPU work.

Qbert25M48/B16/R8 completes200k actions /799,672 frames /6239 updates in26m24s,
13.28% less training time. Greedy5817.708/24 pyramids falls51.46% below retained
11984.375/24;sampled4533.333/23 falls32.93% below6759.375/24. Own initials9.375/
177.083 clear none;mean15000 fails. Retain12M/R16;no extension/confirmation or
successor. All226 pins,100k/3114 recovery,357 matching25M bounded initial/invariant
frozen tensors,full replays/videos,guards,curves,coverage,exposure/comparison pass.
Capacity12M→25M and replay16→8 together;no isolated effect/12M initial equality.
Native447fe50/control runner/teacher/supervision fixed. Actual guidance45.751%;
guided online mean15207.979 is not competence. Main succeeds12:21:32 UTC October29,
postprocessing12:22:03,collection12:22:22;all services succeed,no failure/GPU repeat.
See completed25M/R8 result;no GPU allocation active,check live state before work.

Qbert25M48/B16/R8 bounded check passes 8192 actions / 32,753 frames / 245 updates
in 63.119s; 1024 frozen / 4096 frames take 1.148s, zero updates/no natural episodes.
All 44 pins, 357 finite/invariant frozen tensors, derived geometry/31,636,273
parameters, update checks, full guided replay and guards pass. Capacity12M→25M
and replay16→8 change together; expert100/all weights1/world gradients/all-state
labels/native447fe50 remain fixed. No cross-model initial equality is claimed.
Conservative 200k projection 28m15s passes 35min; this is early cost, not competence.
Actual short-check guidance54.089%. Service succeeds11:32:49 UTC October29/source
`aeee122`; collection11:33:20. No full learner or automatic promotion selected.
See the completed25M/R8 result; check live state before GPU work.

Qbert NOOP-weight.25 completes200k actions/799,700 frames/12,477 updates in
30m12s. Greedy9756.250/24 pyramids and sampled6120.833/22 fall18.59%/9.45%
below retained weight1's11984.375/24 and6759.375/24;own initials80.208/177.083
clear none. Mean15000 fails. All230 pins,100k recovery,357 initial/frozen tensors,
paired initial traces,full replays,guards,curves,coverage,exposure and comparison
pass. Main succeeds11:18:14 UTC October29,postprocessing11:18:33;all services/
collector succeed. Retain weight1;no extension,confirmation or further sweep.
A1.000s CPU recount finds later passive wait agreement95.044%→86.137%,hop
55.968%→58.100%;changing states limit interpretation. Native447fe50 unchanged;
no GPU allocation active. See the completed NOOP-weight report;check live state.

Qbert NOOP-weight.25 bounded check passes8192 actions/32,754 frames/489 updates
in70.160s;1024 frozen/.783s,zero updates/no natural episodes. All45 pins,357
matched initial/invariant frozen tensors,weighted-loss counts,full guided replay
and guards pass;service succeeds10:29:24 UTC October29/source75eca29. Native
447fe50/control runner/teacher unchanged;matching Breakout-era runtime required.
Conservative200k projection31m24s passes35min;no task benefit measured. Original
CPU import assertion failure is preserved,corrected before GPU launch. No full
learner or GPU allocation is active. See the NOOP-weight result/protocol and
check live state before work.

## Experiment practice

- Use small development runs and meaningful checks; avoid turning preparation
  into another backend qualification project. Automate monitoring and audits.
- Keep batched inference with eight independent Atari environments. Report
  aggregate actions and emulator frames separately, plus wall-clock time.
- Retain explicit config, seeds, native/source identity, learning curves and
  frozen natural-episode evaluation against actual-initial controls. Record
  exploration aids, budgets, cutoffs and failed attempts. Confirm promising
  changes across seeds; selected/reused development seeds are not held-out tests.
- Checkpoints do not preserve replay, live belief, games or RNG. A restored
  extension is a warm start, not uninterrupted learning or an independent seed.
- Preserve visual detail: native max-pooled frames with one GPU resize. Do not
  reintroduce CPU downscale followed by upscale.
- Put compact results in docs/results and update PR31. Logs, weights, videos,
  build caches and profiling dumps belong outside Git. Keep only useful
  checkpoint milestones; release disposable build caches after packaging.

## GPU operation

GPU work is authorized unless the device is genuinely faulted. Known allocation
warnings alone are not an approval gate. Use the existing host guard and stop on
real faults or numerical failures. No automatic driver reset/reload/reboot.
Separate NVML polling is unnecessary for the current experiment.

Historical boot24f26399 on595.91.07 remained faulted. A later boot9461079a
was observed during handover; no GPU health check was performed by this task.
Repeated PMU halts occurred on595 and615; the latest captured wait was in the
first world's device-local initialization fills, before checkpoint loading.
This is localization, not proof of a buffer bug, firmware cause or bad hardware.
A debugger run with ASLR disabled once succeeded; that is not a fix.

On that host, use the prepared pre-submission capture after recovery, not another
blind restore. On a different healthy GPU, do not inherit the old boot/device
assertions or presume it is wedged. Start with bounded initialization/restore,
record the new device/driver, and stop if the fault reproduces. The ABI-specific
GDB collector is for the original binary only, not an arbitrary rebuild.

## Historical material

Original run/checkpoint archives were relocated to
`/x/Artifacts/shared-drive-cleanup-20261025` on the original machine. The4096-action
context smoke and2M Breakout originals are present and match all346 tensors in
the shared handover exports; see the handover for exact paths and counters.

The old incident/optimization ledgers and detailed experiment data remain on
`archive/phase2-before-cleanup-20261025`. Current result reports are retained here. Historical “next” paragraphs are not active instructions.
User-authored TASK.md and docs/cdp-claude-feedback.md are preserved as context;
subsequent CDP decisions and current measured results supersede their older
architecture/performance assumptions. Do not merge upstream PRs on the user's
behalf or modify neighboring worktrees during handover.
