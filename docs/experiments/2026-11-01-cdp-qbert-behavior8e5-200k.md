# Fresh Qbert actor/critic rate comparison

Select one fresh200k learner after the [bounded check](../results/2026-11-01-cdp-qbert-behavior8e5.md)
and its original collector complete. The8,192-action/733-update check takes47.940s;
the10%-margin200k projection21m27s passes35min. All496 inputs,357 matching initial/
invariant frozen tensors,47 pre-update events,733 finite reports,full guided
replay and host guards pass. Main15:14:46/collection15:15:08 UTC November1/source
b735644. Execution and early cost are established;gameplay benefit is unmeasured.

Relative to retained10007 H5/R24,change actor/critic learning rate **4e-5→8e-5**.
Keep centeredCDP/12M48/Auto/N8/B16/T16/H5/R24,world/base/encoder4e-5,dynamics4e-4,
greedy collection,sampled posteriors,actor-unimix0,disagreement1,expert100/unit
weights/world gradients,all-state corrective labels,20% episode continuation
guidance/cap1920,full18/sticky.25/repeat4,native pooled frames and one GPU48resize.
No new loss,model change or native build. Faster behavior adaptation may help at
fixed replay work,but actor and critic effects are not separated. More replay
work and shorter rollouts lowered this control's primary score;that motivates
testing a different factor without establishing a causal diagnosis.

Use native447fe50/SHA35ad0db and exact runneraba4300d. Use qualified teacher
overlay`/x/Temp/kindle-qbert-color-masks-20261101/runtime`,teacherSHAfa261c53;
its130 CPU tests,472 exact snapshot decisions/states and full200k label/pixel/
action replay already pass. Source bytes and CPU cost differ from the retained
control. Compare actual training/acting/learning time with this disclosed;
do not claim isolated native speed gains. Preserve original runtimes/evidence.

Start fresh at development seed10007/evaluation4000010007. Reuse no old weights,
experience,belief,games,RNG or bounded-check data. Fix200,000 actions/18,715
updates,first update376,residual credit.75 and35-minute training cap. Retain the
R24 cadence and update-clock warmup. Audit100k/9,340 and automatically preserve
all four checkpoint files before replacement,with source/copy hashes. This
retains weights/metadata,not uninterrupted replay/belief/game/RNG recovery.
Use the fixed200k final checkpoint;no checkpoint selection.

Greedy is primary against retained **14,339.583/24 first pyramids** in28m33.661s;
sampled **13,761.458/24** is secondary. Assess final and actual-initial weights
in both modes,first3 natural episodes per8 lanes=24 per arm. Keep the original
mean15,000/90%-first-pyramid gate. Guidance/labels off,zero updates/truncations,
600k-action/300s cap per arm;retain excess episodes and partial tails. Require357
matching initial/invariant frozen tensors,matched pre-update and actual-initial
semantic traces,finite reports,changed CNN/reward/actor/value,full training
pixel/action/label replay,frozen outcome/frame/reset replays/videos,host guards,
curves,coverage,exposure and fixed comparison. Ordinary frozen logs lack live
pixel hashes;online guided returns are not unaided competence. Reused seed/cohort
limits independence and reliability claims,even if this learner passes.

CPU qualification passes15:15:20 UTC in3.395s/3.506s wrapper:five CLI paths,
28 config,five teacher-identity,eight mode,16 header,four observer,11 recovery,
six retention andtwo actual observer/collector integration cases;51 paths,
17 controllers,496 bounded/442 control inputs and357 schema tensors. Zero
native/emulator constructions. Reuse previous plot-runtime qualification.
All source review and checks pass without failures;do not repeat qualification.
Main/audit/training-replay caps5400/1800/1200s;post/collection6000/6300s. Keep the
original observers/collector and preserve any failures without GPU repetition.

Root`/x/Temp/kindle-qbert-behavior8e5-20261101/full`,trial
`qbert-12m-48-b16-h5-r24-encoder4e5-behavior8e5-greedy-expert100-200k-seed10007`,
service`kindle-qbert-behavior8e5-200k-20261101`. Require idle host and completed
bounded evidence before declaration/launch. No extension,retry,cap increase,
rate sweep,confirmation learner or automatic successor is selected.
