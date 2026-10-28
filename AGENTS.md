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
See the corrective-labels protocol/completed cheap-Qbert result. The separate
R32/wait-weight1 check passes4096 actions/16,381 frames/963 updates in43.238s,
1024 frozen/.495s,36 pins,357 initial/frozen tensors,equal row/weight sums,full
replay and guards; exits18:15:27 UTC. No natural frozen episodes. Early17m36s/
100k projection supports a separately selected fresh100k/24,939-update comparison,
50k recovery and fixed24-episode final/own-initial evaluation under a30-minute
training cap. Same runtime; wait weight/ratio/budget change together on reused4013.
Full root`/x/Temp/kindle-qbert-wait-full-20261028`; one fresh life starts18:29:15
UTC/source`00eed19` on October28, service`kindle-qbert-wait1-r32-100k-20261028`,
PID189477/invocation`00d3435a0afe4e2688a0d7c2b43aa1a6`. Startup55 pins,357
matching initial tensors and429 finite supervised reports pass.50k recovery and
terminal audits are automated. Original CPU progress-observer argument failure
is retained; corrected observer runs separately without repeating GPU work.
No frozen outcome yet; no extension or automatic successor. See the wait-labels
protocol and check live state before GPU work.



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
