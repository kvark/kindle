# CDP handover — October25,2026

The [200k Qbert H3 comparison](results/2026-10-29-cdp-open-loop-supervision.md#full-comparison-retains-the-control)
is complete:200,000 actions/799,709 frames/12,477 updates in30m15s. Greedy mean
9377.083 and sampled5613.542,each23/24 first pyramids,are below the retained
control11984.375/6759.375,each24/24. Own initials80.208/177.083,zero pyramids.
The original15000 mean gate fails. Retain control; no extension or confirmation.
All242 pins,100k recovery,357 initial/frozen tensors,initial semantic traces,
full replays/videos,guards,curves,coverage,exposure and comparison pass. Main
service exits08:41:03 UTC October29;postprocessing08:41:29. Allfour services and
the terminal collector succeed,with no GPU repeats. Evidence:
`/x/Temp/kindle-open-loop-20261029/full/qbert-open-loop-h3-200k-seed4013`.
No GPU allocation is active; check live state before new work.

The next [Breakout representation preparation](experiments/2026-10-29-cdp-breakout-representation.md)
tests accessible ball/paddle position and motion information in the retained
12M CNN and recurrent state,against its actual initial model on identical
recorded states. CPU corpus preparation under
`/x/Temp/kindle-breakout-representation-20261029` aligns20,960 actions and
954/509/601 examples across separate train/validation/test lanes. It constructs
no agent. Native capture is not yet allocated: implement CPU checks,freeze a
declaration and pass a bounded capture first. Use the retained447fe50 runtime
specified by that protocol; no rebuild or new learner is selected.

Optional H3 supervision (`ab80486`,default off) passes135 Rust/611 packaged
Python CPU tests,strict Clippy,both guarded numerical checks and1024-action
legacy compatibility. Fresh Qbert12M8192 actions/32,754 frames/489 updates take
70.207s;1024 frozen/.792s haszero updates/no completed episodes. All44 pins,
357 matching initial/invariant frozen tensors,finite auxiliary reports,full
guided replay and guards pass;service exits07:43:21 UTC October29. Conservative
200k projection31m25s passes35min;task benefit remains unmeasured. Original CPU
timestamp/fixture failures remain without GPU repetition. Use isolated runtime
`/x/Temp/kindle-open-loop-20261029/runtime`,native SHAb4cdf26,and its pinned
`native-check/atari_vector.py`. Packaging receipt is at that root;both build
caches are released. No full learner or active GPU allocation. See the
[completed result](results/2026-10-29-cdp-open-loop-supervision.md) and
[protocol](experiments/2026-10-29-cdp-open-loop-supervision.md);check live state.
The separate posterior draft is parked as described below.

Qbert actual action-block comparison completes128 origins/2432 modeled plans
on48256 recorded actions/192980 frames in95.811s,zero updates. Ranking agreement
is72.96% before/57.21% after the first pyramid;40/128 origins are informative.
Later gain over uniform1.601 has interval[−.131,3.443]. All46 pins,357 invariant
tensors,128 exact prior-forecast matches,pixel trajectories and guards pass;
service exits06:52:55 UTC October29. The original CPU snapshot failure is retained;
process-isolated branches pass complete replay/order checks without changing ALE.
Evidence `/x/Temp/kindle-action-ranking-20261029`,
[result](results/2026-10-29-cdp-learning-diagnosis.md).
Next selected work is the [bounded open-loop supervision implementation](experiments/2026-10-29-cdp-open-loop-supervision.md).
No full learner or GPU allocation is selected yet; check live state. The separate
greedy-posterior draft is parked at
`/x/Temp/kindle-open-loop-20261029/parked-posterior` (patch, exact files and hash
receipt); exclude it from an experimental runtime unless explicitly qualified.

Live Qbert critic calibration completes four frozen arms,98664 actions/394520
frames in69.704s,zero updates. Sampled predicted mean1218.534 versus realized
1475.689,correlation.542/R².184,beats the constant baseline; after-first-pyramid
correlation is.358. Greedy behavior has larger bias and a different target policy.
All44 pins,357 invariant tensors per arm,12522 exact semantic events and guards
pass; service exits06:30:01 UTC October29. No changed loss or training allocation
is selected. Next inspect actual alternative-action ranking; evidence is in
`/x/Temp/kindle-critic-calibration-20261029` and the
[diagnosis report](results/2026-10-29-cdp-learning-diagnosis.md).

The October29 learning diagnosis completes both48256-action frozen Qbert12M
captures in207.599/207.409s, zero updates. Fifteen-step reward-sum MAE falls
107.570 initial→37.499 trained; unrelated actions give69.538. Later-play errors
are substantially larger, but forecasts remain useful through15 steps. All39
pins,357 invariant tensors, recorded trajectories/pixel hashes and guards pass;
service exits06:18:53 UTC. No new loss or training allocation is selected.
Next diagnostic direction: critic calibration and actual alternative-action
outcomes. The optional greedy-posterior draft is separate and now parked above;
its CPU tests passed, but no new native runtime was built or GPU-tested.
See [result and source audit](results/2026-10-29-cdp-learning-diagnosis.md).
No GPU allocation is active; check live state before work.

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
See the [completed capacity result](results/2026-10-26-cdp-breakout-pixel-reference.md#larger-capacity-improves-frozen-breakout-with-higher-cost); check live state before GPU work.

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
See [the bounded result](results/2026-10-26-cdp-breakout-pixel-reference.md#larger-supervised-breakout-passes-execution-and-early-cost-checks).

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
is active;check live state before work. See [the completed result](results/2026-10-25-cdp-cheap-qbert.md#longer-guidance-lowers-primary-greedy-quality-at-the-same-short-budget).

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
check live state before work. See [the completed result](results/2026-10-26-cdp-breakout-pixel-reference.md#stronger-imitation-improves-frozen-breakout-at-nearly-unchanged-cost).

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
[the completed result](results/2026-10-25-cdp-cheap-qbert.md#stronger-imitation-lowers-both-frozen-scores-at-the-same-short-budget).

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
selected. GPU idle;check live state before work. See [the completed comparison](results/2026-10-29-cdp-qbert-rgb-control.md).

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
allocation. Both services are terminal and the host is idle;check live state. [Protocol](experiments/2026-10-28-cdp-breakout-corrective-greedy.md).

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

**Receiving-host update:** [AMD takeover](results/2026-10-25-cdp-amd-takeover.md).
RX7900XT/RADV cannot run the saved BF16 configuration. A labelled F16-input copy
passes restore with all346 tensors unchanged. The fresh2M B2/T64 life and audits
are complete: frozen500k/final/initial scores14.71/10.04/1.42, all0/24 two-wall
successes. Training took11h40m; that service has exited. Paired early updates
show precision-dependent differences, and Auto's short training/restore passes.
The fresh [500k Auto comparison](experiments/2026-10-25-cdp-amd-auto.md) also
completed: frozen15.67 versus F16's14.71, all0/24 two-wall successes, in2h53m.
All audits pass. Both services have exited; do not restart or extend these lives.
The user now prioritizes cheaper game-specific recipes; use the roadmap.
[Cheap Freeway screens](results/2026-10-25-cdp-cheap-freeway.md) finish200k actions
in about8 minutes but remain weak (unassisted/aided frozen0.00/0.25, initial0.00).
[Smaller native recipes](results/2026-10-25-cdp-cheap-runtime.md) add4M capacity
and32/48 image geometry; GPU training/frozen checks and retained64 restore pass.
The [200k Boxing screen](results/2026-10-25-cdp-cheap-boxing.md) completes in6m03s
but fails: frozen−3.875 versus initial0.00, both10/24 wins, all audits pass.
The [Pong4M/RGB48 screen](results/2026-10-25-cdp-cheap-pong.md) completes in10m22s:
frozen−17.125 versus initial−20.375, both0/24 wins, all audits pass.
Fresh Boxing2017,1M/RGB32/500k/R32 now passes: frozen76.8333 versus initial2.875,
24/24 wins,124939 updates in28m22s; all audits pass and the service exits. The fixed
recipe also passes fresh confirmation4013:79.1667 versus initial−0.625,24/24 wins,
500k/124939 updates in28m13s, all audits pass and service exits. Two useful learners
do not establish broad reliability. Details and curves are in the Boxing report.
[Qbert4M/RGB48/R32](results/2026-10-25-cdp-cheap-qbert.md) finishes200k in19m45s:
frozen809.375 versus180.2083,0/24 first pyramids. Mean coverage improves5.83→16.63
of21 cubes; all audits pass, but the task gate fails. No extension.
[Breakout4M/RGB64/R32](results/2026-10-26-cdp-cheap-breakout.md) finishes200k in
22m30s: frozen6.2083 versus1.375,0/24 two-wall successes, all audits pass. A CPU
replay verifies one12061-action delay before FIRE after a lost life. No extension.
[Breakout's1% native mixture](results/2026-10-26-cdp-breakout-unimix.md) completes
200k in22m29s: frozen7.0833 versus initial1.375, longest selected no-FIRE gap447
versus control12115. The baseline weights with only frozen mixture score6.0833
with gap724 in9.8s and zero updates. All three policies remain0/24 two-wall
successes. All audits pass and services exit; no extension.
[Compact aided Freeway](results/2026-10-25-cdp-cheap-freeway.md#compact-rgb32-with-the-successful-update-budget)
completes500k/124939 updates in28m17s: frozen31.0417 versus actual initial0.00,
24/24 versus0/24 crossing successes, aid disabled in evaluation. All audits pass
and service exits. Constant-UP CPU reference scores21.6667,0/24. The successful
1M/RGB32/R32 recipe's focused confirmation follows below; no isolated attribution.
[Pong500k/R32](results/2026-10-25-cdp-cheap-pong.md#r32-and500k-actions) completes
in48m53s: frozen11.0 versus initial−20.375,23/24 versus0/24 wins. All audits pass
and service exits; the mean15 gate fails. No unchanged extension.
[Freeway seed4013 confirmation](results/2026-10-25-cdp-cheap-freeway.md#fixed-recipe-seed4013-confirmation)
also passes:32.000 versus initial0.00,24/24 crossing successes,500k/124939 updates
in28m50s. All audits pass and service exits. Two compact learners now pass within
30 minutes each. The [retained Breakout2M frozen check](results/2026-10-26-cdp-retained-breakout-auto.md)
scores67.5833 with0/24 two-wall successes,21,976 actions in14.127s and zero updates.
All346 source/copied/frozen tensors match; all audits pass and service exits.
Only precision metadata changes to Auto; the warm start below is separately declared.
[Qbert coefficient100](results/2026-10-25-cdp-cheap-qbert.md#coefficient100-does-not-improve-the-fixed-budget-result)
completes200k/49939 updates in20m12s: frozen809.375,0/24 pyramids. Score matches
the control but mean coverage falls16.625→13.708 of21 cubes. All audits pass and
service exits; no extension.
The [H5 Breakout warm start](results/2026-10-26-cdp-breakout-short-imagination.md)
completes50k new actions/12439 updates in8m02s but regresses67.5833→27.2083,
both0/24 two-wall successes. The frozen source trace matched H15 before learning;
all audits pass and service exits. Retain the source; no extension.
The [Pong4013 confirmation](results/2026-10-25-cdp-cheap-pong.md#fixed-recipe-seed4013-confirmation)
completes500k/124939 updates in50m14s: frozen7.333 and22/24 wins versus own
initial−20.333 and0/24. All audits and curves pass; service exits successfully.
Both compact Pong learners win most matches but miss mean15. No extension.
The [six-action Qbert test](results/2026-10-25-cdp-cheap-qbert.md#six-action-choices-run-faster-but-reduce-frozen-coverage)
completes200k/49939 updates in18m21s: frozen605.208 versus own initial148.958,
0/24 pyramids. Coverage10.167/21 and score are lower than the full18 control.
All audits/curves pass and service exits; no extension. The separate CPU Breakout
pixel follower scores11.333 and a fixed two-frame interception variant20.292,
both0/24 two-wall successes with complete replay. Neither supplies training guidance.
The [compact Breakout H5 result](results/2026-10-26-cdp-breakout-compact-h5-budget.md)
retains a45-minute cutoff at487,736 actions/121,873 updates: frozen8.458 versus
initial1.375,0/24 two-wall successes. Its shared prefix costs18.06% less wall time
than H15, but the larger-budget recipe still fails the task. Numerical,
checkpoint/accounting and frozen replay checks pass; the original controller/
guard retain their time-budget failure. Both services exit; no extension.
The [Pong coefficient.01 primary](results/2026-10-25-cdp-cheap-pong.md#smaller-bonus-improves-the-fixed-budget-result)
completes500k/124939 updates in49m55s: frozen14.9583 and24/24 wins versus own
initial−20.375 and0/24. This improves the coefficient1 control's11.0/23 wins,
but still misses mean15. All audits/curves pass and its service exits; no extension.
The [coefficient.01 confirmation](results/2026-10-25-cdp-cheap-pong.md#smaller-bonus-seed4013-confirmation)
finishes500k/124939 updates in49m58s:14.0 and24/24 wins versus initial−20.333.
Its coefficient1 control scores7.333/22 wins. All audits/curves/comparisons pass;
service exits08:27:53 UTC. Both smaller-bonus seeds miss mean15; no extension.
The [four-action Breakout result](results/2026-10-26-cdp-breakout-minimal-actions.md)
completes200k/49939 updates in20m59s,6.64% less training time than full18.
Frozen9.542 versus own initial1.167 improves the full18 control's7.083, but
still0/24 two-wall successes. All audits/comparison pass; service exits09:01:55.
The [Qbert500k/R16 result](results/2026-10-25-cdp-cheap-qbert.md#more-experience-at-r16-does-not-improve-the-result)
finishes in26m53s: frozen719.792 versus initial180.208,0/24 pyramids and14.125/21
coverage, below the200k/R32 control809.375/16.625. All audits/comparison pass;
service exits09:33:03 UTC. No extension or confirmation selected.
The [bounded B16 check](results/2026-10-25-cdp-cheap-pong.md#b16-bounded-check-passes-with-lower-early-wall-cost)
passes16,384 actions/2,002 updates and ordinary restore, using72.224s versus
94.054s for B8. All346 initial tensors match and restored tensors stay unchanged;
service exits09:38:03 UTC. The23.21% early cost reduction is not learning evidence.
The [full-budget PongB16 result](results/2026-10-25-cdp-cheap-pong.md#b16-at-the-fixed500k-budget)
completes500k/62,454 updates in38m45s,22.37% less training time than B8. Frozen
11.875 versus initial−20.375 retains24/24 wins but falls below B8's14.958 and
misses mean15. All four recoveries, frozen audits, curves and comparison pass;
service exits10:28:51 UTC. No extension or seed confirmation selected.

The [bounded Qbert capacity check](results/2026-10-25-cdp-cheap-qbert.md#larger-capacity-at-rgb48-bounded-check)
passes16,384 actions/65,481 frames/2,002 updates in149.387s, versus93.813s for
4M/B8. Actual14,123,473 parameters, all reports/config/counters/finite checks,
selected parameter changes and ordinary restore/346-tensor invariance pass.
Service exits10:35:06 UTC. This is execution/early-cost evidence, not competence.

The [completed Qbert capacity/batch screen](results/2026-10-25-cdp-cheap-qbert.md#larger-capacity-at-the-fixed200k-budget)
scores959.375 versus own initial165.625, with0/24 pyramids. At200k actions,
12M/B16 takes32m19s versus4M/B8's19m45s; coverage16.292/21 is below16.625/21.
All audits and completed comparison pass; service exits11:15:13 UTC. The score
increase does not justify63.70% more wall time for this compute-constrained
screen. No extension or confirmation selected.

The [completed Pong doubled-rate test](results/2026-10-25-cdp-cheap-pong.md#doubling-b16-learning-rates-lowers-the-frozen-score)
scores9.125 versus the B16 control's11.875, both24/24 wins and below mean15.
Training38m32s is essentially unchanged from38m45s. All audits, initial-tensor/
semantic-trace equality and completed comparison pass; service exits11:59:15 UTC.
Retain the original rates; no extension or seed confirmation selected.

The [completed Qbert training-aid test](results/2026-10-25-cdp-cheap-qbert.md#persistent-training-aid-does-not-improve-frozen-coverage)
scores678.125 versus809.375 without aid; mean cube coverage15.125 versus16.625,
both0/24 pyramids. Training20m17s versus19m45s, with10,039 verified overrides.
All audits and completed comparison pass; service exits12:24:16 UTC. Retain the
unaided recipe; no extension or confirmation selected.

The [completed PongH5 test](results/2026-10-25-cdp-cheap-pong.md#shorter-imagination-saves-time-but-lowers-the-frozen-score)
finishes500k/124,939 updates in38m07s,23.65% less training time than H15. Frozen
8.667 is below14.958, both24/24 wins and below mean15. All audits, matched initial
tensors/traces and comparison pass; service exits13:18:17 UTC. Retain H15; no
extension or conditional4013 confirmation is selected.

The [completed smaller-bonus Breakout test](results/2026-10-26-cdp-breakout-minimal-actions.md#smaller-bonus-lowers-the-frozen-score)
finishes 200k actions/49,939 updates in 21m07s, but lowers frozen score
9.542→5.042, both 0/24 two-wall successes. All audits and comparison pass after
copying the omitted, verified replay manifest and rerunning only the unchanged
CPU audit. The original service failure at 13:43:51 UTC and incomplete watcher
receipt are retained; no GPU work repeats. Retain coefficient 1; no extension.

The [Pong B16/R40 run is interrupted](results/2026-10-25-cdp-cheap-pong.md#host-restart-interrupts-b16r40)
after a host restart:265,440 complete logged actions/41,416 finite reports,
saved200k/31,192-update checkpoint. Its separate frozen assessment scores−9.708
versus initial−20.375, both0/24 wins. All frozen checks pass after a CPU-only
checker-import correction; original failures/unsealed tails remain. The500k
quality question is unresolved. No resumed training or extension.

The [completed Qbert H30 test](results/2026-10-25-cdp-cheap-qbert.md#longer-imagination-costs-more-and-lowers-frozen-coverage)
finishes200k actions/799,441 frames/49,939 updates in26m43s,35.34% more training
time than H15. Frozen582.292 versus809.375 and coverage9.583 versus16.625/21
are worse; both0/24 first pyramids. Retain H15. Final training/frozen/replay/
guard/comparison checks pass, including346 matched initial tensors and identical
semantic initial traces. The declared100k recovery was never written because the
pinned queue saved every200k. Original watcher failures remain; separate CPU
completion records `protocol_complete=false`. Service exits16:26:25 UTC. No GPU
repeat, extension or confirmation for H30.

The [completed six-action Pong test](results/2026-10-25-cdp-cheap-pong.md#six-actions-save-time-but-lower-frozen-quality)
finishes500k/124,939 updates in44m21s,11.17% less training time than full18.
Frozen10.375 versus own initial−20.042 is below full18's14.958; both24/24 wins
and below mean15. All numerical/config/counter/tensor/replay/guard checks,
four recoveries and curves/comparison pass; service exits17:28:27 UTC. Dimensions
and initial weights differ, and the control used the older kernel. Retain
full18; no extension or conditional4013 confirmation is selected.

The [bounded Breakout actor/critic check](experiments/2026-10-26-cdp-breakout-actor-critic-gradients.md)
passes4096 actions/963 updates in22.474s and1024 frozen actions in0.489s. All357
tensors preserve frozen invariance; all963 initial policy/value loss reports
are nonzero, CNN/reward weights change and346 shared initial tensors match the
control. Both guards pass; service exits17:34:23 UTC. This verifies execution,
not useful task learning or gradient correctness. Native unchanged; the exact
313-test CLI patch is applied as runner`fbdaee3`.

The [completed Breakout actor/critic test](results/2026-10-26-cdp-breakout-minimal-actions.md#actorcritic-gradients-give-a-modest-unconfirmed-gain)
finishes200k/799,187 frames/49,939 updates in20m41s, with frozen11.208 versus
9.542 and own initial1.167; both0/24 two-wall successes. All357-tensor/frozen,
100k recovery, numerical/replay/guard checks and matched346 shared initial
weights/semantic traces pass. Main service exits18:05:14 UTC. A separate CPU
comparison import failure is retained; corrected comparison passes without
repeating GPU work. This is a modest unconfirmed gain, with an older-kernel
control; no extension, seed confirmation or global default change is selected.

The [completed QbertRGB64 test](results/2026-10-25-cdp-cheap-qbert.md#rgb64-costs-more-and-lowers-frozen-quality)
finishes200k/799,444 frames/49,939 updates in22m20s,13.14% more training time
than RGB48. Frozen718.750 versus809.375 and coverage15.083 versus16.625/21 are
lower; both0/24 first pyramids. All numerical/config/counter/346-tensor/replay/
guard/100k recovery/curve/coverage/comparison checks pass; service exits18:39:49
UTC. Retain RGB48; no extension or confirmation. Geometry, parameter shapes,
initialization and old/new kernel/module limit attribution.

The [completed Pong greedy-action assessment](results/2026-10-25-cdp-cheap-pong.md#greedy-evaluation-does-not-improve-the-retained-pong-policy)
scores14.333 versus sampled14.958, both24/24 wins and below mean15. Own initial
greedy−20.833/0 wins. Final/initial77,640/20,288 actions and310,513/81,145 frames
in29.564/7.757s, zero learning,24 natural matches each. All346 frozen tensors,
source files, config/counters, complete replays/videos and both guards pass;
service exits18:44:25 UTC. Retain sampled evaluation; no greedy confirmation.
The retained learner's49m55s training cost remains. The five-game goal is unfinished.

The [bounded PongB16 actor/critic check](experiments/2026-10-26-cdp-pong-actor-critic-gradients.md)
passes4096/466 updates in16.126s and1024 frozen actions in0.436s. All357 tensors
remain unchanged,346 shared initial tensors match the control, all466 initial
policy/value reports are nonzero and both guards pass. Service exits18:51:00
UTC. Native/CLI unchanged; this is execution evidence, not useful learning.

The [completed PongB16 actor/critic comparison](results/2026-10-25-cdp-cheap-pong.md#b16-actorcritic-gradients-lower-frozen-quality)
finishes500k/62,454 updates in37m52s but lowers frozen11.875→−0.542 and wins
24/24→10/24. Own initial−20.375/0 wins. All357-tensor/replay/guard/recovery/
curve/comparison checks,346 shared initial tensors and semantic traces pass;
service exits19:36:47 UTC. Retain disabled gradients; no extension or confirmation.
The separate progress observer was reattached without restarting training.

The [CPU-only Qbert pixel rule and prepared adapter](results/2026-10-25-cdp-cheap-qbert.md#pixel-only-cpu-rule-clears-the-first-pyramid)
clear24/24 first pyramids, mean3764.583 and21/21 coverage. Collection and exact
replay each take about30s; no agent or learning. This hand-coded candidate
still misses mean15000 and does not establish unaided Kindle transfer.
The optional episode-guidance adapter passes CPU tests and full pixel/action
replay. The bounded native check completes4096/963 in22.541s and1024 frozen
restore in0.452s;512 verified overrides,346 matched initial/unchanged frozen
tensors, all finite reports and both guards pass. Its service fails19:40:14 UTC
only in the new CPU reader, which incorrectly rejects training intrinsic rewards.
A training-aware correction passes421 tests and the saved trajectory replay;
original failure retained, no GPU repeat.

The fresh Qbert pixel-guided comparison completes 200k actions/49,939 updates
in 20m13s, but lowers unaided frozen score 809.375→633.333 and first-pyramid
coverage 16.625→11.667/21; both have 0/24 pyramids. The 5% episode rule actually
overrides 23,428 actions (11.714%). All audits, 100k recovery, 346 matched initial
tensors/semantic traces, complete pixel/action replay and comparison pass;
service exits 20:07:01 UTC. Retain unaided; no extension or confirmation.
See [the result](results/2026-10-25-cdp-cheap-qbert.md#pixel-guided-training-lowers-unaided-frozen-quality).

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
See [the completed result](results/2026-10-25-cdp-cheap-qbert.md#capped-guidance-recovers-some-quality-but-remains-below-unaided-training).

Pong B16 behavior rate8e-5 completes500k actions/1,999,751 frames/62,454
updates in37m38s. Frozen12.375/24 wins improves original B16's11.875 slightly,
but trails B8's14.958; all miss mean15. All audits, four recoveries,346 matched
initial/unchanged frozen tensors, semantic initial traces, replays, guards and
comparison pass; service exits22:20:58 UTC. Retain B8 as the strongest compact
Pong recipe; no extension or seed confirmation follows this small gain.
The CPU startup import failure remains recorded; no GPU stage repeated.
CLI`a2c467c` passes460 CPU tests; native unchanged. See
[the completed comparison](results/2026-10-25-cdp-cheap-pong.md#behavior-rate-doubling-gives-a-small-gain-but-still-misses-the-gate).

QbertRGB48 frozen forecasts complete on15,376 recorded actions/61,462 frames
per model in50.330s trained/49.268s initial, with zero updates. One-step reward
AUC improves.500→.969; H15 reward-sum MAE22.487→12.400. All346 frozen tensors,
8 matched stream trajectories/pixel hashes, source/counter/support/guard checks
pass; service exits22:30:05 UTC. This shows predictive learning on the reused
cohort, not the cause of weak policy transfer. Retain4M48/H15; no new training
or automatic next allocation. The geometry helper passes24 CPU tests; native
unchanged. See [forecast results](results/2026-10-25-cdp-cheap-qbert.md#compact-qbert-learns-reward-forecasts-despite-weak-task-performance). Check live state before GPU work.


Pong12M48/Auto/N8/B16/T16/H15/R16 completes500k actions /31,227 updates in
41m28s,16.95% less training time than4M/B8/R32. Frozen15.625/24 wins versus
own initial−20.292/0 meets mean15/90% on one learner. All audits, four recoveries,
346 frozen tensors,42 pins, recorded-action replays, guards and comparison pass;
service exits01:51:06 UTC, CPU postprocessing01:51:35 on October27.
No live pixel hashes are available in ordinary unguided logs. See the protocol
scope correction and [completed result](results/2026-10-25-cdp-cheap-pong.md#larger-capacity-with-less-replay-work-reaches-the-gate).

Pong12M48 fresh4013 completes500k actions /1,999,720 frames /31,227 updates
in41m31s. Frozen14.833/24 wins versus own initial−20.542/0 narrowly misses
mean15; primary1009 scores15.625/24 wins in41m28s. Both improve their paired
4M/B8 controls14.000/14.958 with about17% less training time, but only one of
two learners passes the full gate. All numerical/config/counter checks, four
recoveries,346 frozen tensors,53 pins, replays, guards and comparison pass.
Service exits02:41:08 UTC, CPU postprocessing02:41:23 on October27. Original
progress-observer failure is retained; no GPU stage repeats. Both reused
seeds are development evidence. No extension or third seed is selected.
See [completed confirmation](results/2026-10-25-cdp-cheap-pong.md#capacity-confirmation-improves4013-but-narrowly-misses-the-gate).

Breakout expert200k completes798,964 frames /49,939 updates in22m38s,
frozen4.000 versus initial1.375, below full18 control7.083/minimal4 9.542;
all0/24 two-wall successes. All audits,100k recovery,346 matched initial/frozen
tensors,46 pins, replays, guards, comparison and exposure pass. Service exits
03:16:09 UTC; postprocessing03:16:14. Retain controls; no extension/confirmation.
20,463 trusted actions (10.2315%); guided/unguided completed means24.871/2.948
are not frozen competence. See [result](results/2026-10-26-cdp-breakout-pixel-reference.md#capped-native-imitation-lowers-frozen-breakout-quality).

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
See the expert10-seed1009 protocol; check live state before GPU work.






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
See [completed result](results/2026-10-25-cdp-cheap-qbert.md#continuation-guidance-improves-frozen-learning-but-misses-the-full-gate).
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
competence from it. See [result](results/2026-10-25-cdp-cheap-qbert.md#native-continuation-check-passes)
and [protocol](experiments/2026-10-27-cdp-qbert-continuation.md).
Check live state before GPU work.


CPU Breakout predictive reference scores166.167 across24 natural episodes,
0/24 two-wall successes; full input/output pixel-hash replay passes. Selected
after one9001 and eight reused evaluation episodes; those8 are included in24.
This reference work preceded the separately selected bounded adapter check.
55.935s affined CPU work
co-hosted with Pong; all42 active pins unchanged. See [reference result](results/2026-10-26-cdp-breakout-pixel-reference.md#calibrated-paddle-response-improves-the-reference-but-misses-the-two-wall-gate).

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
[completed result](results/2026-10-25-cdp-cheap-qbert.md#more-experience-at-r16-improves-first-pyramid-success-with-higher-cost)
and [protocol](experiments/2026-10-27-cdp-qbert-expert-r16.md).
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
See [completed seed check](results/2026-10-25-cdp-cheap-qbert.md#expert-imitation-transfers-on4013-with-substantial-seed-variation)
and [protocol](experiments/2026-10-26-cdp-qbert-expert-seed4013.md).
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
See [completed result](results/2026-10-25-cdp-cheap-qbert.md#expert-imitation-improves-compact-qbert-but-remains-far-below-the-gate).

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
See [bounded result](results/2026-10-25-cdp-cheap-qbert.md#native-expert-imitation-smoke-passes-task-benefit-remains-unmeasured).

New boot`a8648671-8577-4f42-9b46-a4345d3ac297`, kernel7.0.0-38, AMD module
`9BDBC866F55BF296F1B7FA2`; same RX7900XT/Mesa26.0.8/native binary. Host scan,
512-action restore/346-tensor invariance and natural frozen evaluation pass.
No host recovery performed. Qbert's refreshed queue uses this identity; its H15
control used7.0.0-34, so record the kernel change as a comparison confound.
Check live service state before GPU work. Whole-episode and capped Qbert
guidance, Pong RGB32, Pong behavior-rate learning and their postprocessing are
complete, including the fixed frozen Qbert forecast comparison. The bounded expert checks also complete as described above. Do not restart
completed services.

Original compact native source`bfb8c31`; its retained runner supports per-episode
Qbert guidance limits, SHA256
`50ba7759d4ff108d724e831d83fda69c9d4f4dce3da660cebf394f995cac98da`.
Use `runtime-qbert-guidance-cap` for that retained runtime. The expert candidate
uses the isolated79c9a7f runtime described above. Frozen replay remains
`5043b7a` and the current guidance replay supports both bounded and whole-episode
kinds. The selected declaration pins exact source hashes. Native SHA256
`f2606ac28f5f293b73ccdb3f408278c6c584e50ccc73d52fd605a39b231609a5`.
Evidence, wheels and build receipt: `/x/Temp/kindle-cheap-20261025`;
`original-native` preserves the prior AMD binary. Both installed and source-tree
extensions now use the new native build; old completed queues are historical.
Use these reports for current host, precision, results and evidence details.
The historical handover steps below describe its origin.

The goal remains strong, reliable learning on **Boxing, Pong, Freeway, Breakout
and Qbert**, then native games and transfer. Training is the priority; another
backend optimization/JAX-parity campaign is not the next step.

Read this file, the [short roadmap](kindle_single_life_dreamer_plan.md), and the
[Breakout experiment](experiments/2026-10-23-cdp-breakout-context64.md). The
[PR31 description](https://github.com/kvark/kindle/pull/31) is the live dashboard.
Historical reports retain failed attempts and limitations; their old next steps
are not instructions to restart those queues.

## Source you actually need

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
bytes can differ with paths, compiler and dependency metadata. Rebuild on the
new machine rather than importing the original Linux/CPython3.14 binary.

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

The measured runtime enabled:

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

## First action on the new machine

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
