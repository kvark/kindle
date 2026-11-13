# Denoising stages of the retained Breakout world model

Use the retained qualified reward32/sigma_data.05 model to distinguish its
learned high-noise prediction from the final three-step sampler output. The
older stage trace used the weaker sigma_data.5 model;do not rerun it orimport its
findings as evidence about this checkpoint. Root
`/mnt/data/kindle-breakout-reward32-stage-trace-20261109`.

Reuse exact retained source8dbf9ab5/native-cpu-test executable786ab275e07e80ea23139b0916d669006b1745ea6aa7a4e98ae5464a511a1ebd,
checkpoint `/mnt/data/kindle-breakout-denoising-reward32-20261109/train/checkpoint-2048`
andmatching runner. No rebuild,production API,network,loss,sampler orweight change.
Archive exact probe/implementation sources from its retained source.tar,not the
newer mutable worktree. Native test listing is CPU-only andmust identify the
existing ignored frozen_sampler_stages test. Same AMD/Auto settings as control.

Trace27 actual-history one-step origins5..31 ×four draws83001/83003/83009/83023
×eight lanes. The probe receives only prefix images strictly beforethe target,
prior actions andone submitted incoming command per lane. Target RGB/expected
predictions enter only outer parity/scoring checks. Export allthree absolute
quantized denoised RGB estimates atsigmas5/.28308183/.002,plusfinal RGB:108 traces,
324 denoiser calls. Every finalbatch must match cached control teacher RGB bit-
exactly (108 arrays/864 images). Four additional normal API origin16 one-step
queries must match20 cached model arrays,including observed-denoising andevents.
Then135 frozen tensors must match source checkpoint exactly,withage2048 unchanged.
This is frozen diagnosis,no new training ordecoder policy selection.

Before GPU,pin checkpoint/native/source/inputs/probe/runner/scripts andfinish
CPU preparation. Source RGB/actions andexpected teacher/origin arrays must match
qualified closure hashes/layout. Keep the original three-step sigma grid,no
noise orprecision change. Use existing host guard,300s cap,no NVML/reset/reload.
Observe exact invocation/MainPID/queue;one job only,no timeout restart. Require
successful terminal state andnine sealed guard files before CPU analysis.
Analysis scripts must use new filenames andseparate declarations after the
native declaration;never edit a pinned script. No build cache created.

Score eachstage atall27 origins/four draws:unchanged strict ball detection and
1px matching,paddle/brick/pixel metrics,false/ambiguous detections,motion relative
tothe actual previous ball,andcontrast creation/erasure onactual sprite pixels.
Keep allunsupported cases explicit. RGB-derived sprite masks are analysis only,
never model inputs. Quantify same-row stage gains/losses anddraw-dependent results;
do not count draw repetitions as new learners. Stage final must reproduce the
retained teacher140/149/140/140 of198. Retain per-case rows andfixedlane0/draw0
stage images. No relaxed detector,peak-only location orbest-stage substitution.

This doesnot assess a one-stage autoregressive rollout:later histories in this
trace are actual,andallstage estimates belong tothe original three-step path.
Any sampler intervention requires a separate declaration. Eventheads arefrozen;
only normal final-origin events areparity-checked. Do not claim new event
calibration,natural-game competence orworld-model proof. Full physics/events,
changed-brick memory,late/tunnel/second-wall/above-top,untouched families andfresh
confirmation remain required. Fullres protocol failure remains failed;all prior
learners/assessments stayclosed. No extension,collector orautomatic successor.

Completed: [result](../results/2026-11-09-breakout-reward32-stage-trace.md).
Later stages improve ball matches114/114/107/111→140/149/140/140;stage2→3 loses
no correct cases. Weak creation ispresent alreadyathigh noise. Native parity/
frozen checks,105 declaration/147 closure pins,nine guardfiles pass;service
terminal,no learning/build. No sampler intervention orsuccessor selected;
full goal remains unproven,andfullres protocol failure staysfailed.
