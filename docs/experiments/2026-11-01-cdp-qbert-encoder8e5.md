# Bounded Qbert encoder8e-5 check at the retained 200k recipe

The completed300k comparison lowers both frozen scores despite improving replay
imitation fit. More replay, shorter imagination,25M capacity and a higher behavior
rate also failed to improve the primary10007 result. Retain the200k/12M/H5/R24
recipe and the qualified faster teacher.

Select one bounded check changing only **encoder4e-5→8e-5** as a learner setting.
Earlier Qbert H15/R16 comparisons improve at6e-6→2e-5 and2e-5→4e-5, but they
use different collection and imagination/replay settings from this recipe.
Breakout8e-5 has mixed results. None of this proves a Qbert8e-5 benefit; this
is one development point, not a rate sweep or a universal rate recommendation.

Keep centered CDP/12M/RGB48/Auto/N8/B16/T16/full BPTT/microbatch16/H5/R24,
base4e-5/dynamics4e-4/behavior unset (base4e-5),warmup1000,greedy collection,
sampled posteriors,actor-unimix0,action-effects1,expert100/unit weights/world
gradients and all-state corrective labels. Keep20% episode continuation
guidance/cap1920,full18/sticky.25/repeat4,native pooled frames and one GPU resize.
Use native447fe50/SHA35ad0db,runneraba4300d and qualified faster teacherfa261c53
from`/x/Temp/kindle-qbert-color-masks-20261101/runtime`;no build or native edit.
Reuse learner10007/evaluation4000010007 for development, with fresh weights,
replay,games,beliefs and RNG. No previous experience is reused.

Fix8,192 actions/733 updates,first update376/residual credit.75,then1,024 greedy
frozen actions withzero updates. Require357 finite/control-matching initial
tensors,matching pre-update semantic events,357 invariant frozen tensors,
exact configuration/counters,finite supervised reports,changed CNN/reward/actor/
value,full training pixel/action/corrective-label replay and both host guards.
CPU qualification must exercise both actual CLI paths,valid and invalid audit
configurations,the357 tensor schemas,the packaged teacher and update schedules.

Training/frozen/audit/service caps remain180/60/180/480s. For any future200k
learner, require the maximum of the bounded estimate
`training_seconds * 200000 / 8192 * 1.10` and the completed fast-teacher
behavior8e-5 timing reference`1252.376230151 * 1.10` to fit**1,800s (30min)**.
That reference uses a different behavior rate and trajectories; it is a cost
estimate, not isolated native speed or a guarantee of completion.

A passing check establishes execution and early cost only. Any separately
selected full comparison must start fresh at200k/18,715 updates,retain the
100k/9,340 four-file recovery,and assess final/actual-initial greedy-primary
and sampled24-natural-episode arms under the original mean15k/90%-first-pyramid
gate. Compare against retained14,339.583 greedy/13,761.458 sampled,both24/24,
with own initials1.042/171.875. Report time and own-initial gain per minute.
Qualified teacher bytes and CPU cost differ from the original200k control.

No full learner is selected by this protocol. Preserve failures;no restart of
completed stages,cap increase,extension,rate sweep,automatic promotion or
check-experience reuse. The previous300k pipeline is terminal and its separate
CPU-repaired closure is authoritative; its failed original postprocessing and
collection services remain failed. Check live state before launch.

Root`/x/Temp/kindle-qbert-encoder8e5-20261101/native-check`,trial
`qbert-12m-48-b16-h5-r24-encoder8e5-greedy-expert100-check-seed10007`,service
`kindle-qbert-encoder8e5-check-20261101`. The five-game goal remains unfinished.

CPU qualification passes18:28:48 UTC November1 in1.175s (1.211s wrapper):
both actual CLI paths,19 audit-config cases,357 schemas,442 retained control
inputs,16 packaged files and8192/100k/200k update schedules. No native module,
agent or environment is constructed. The timing-reference projection is
22m57.614s;the bounded estimate and gameplay remain unmeasured. Preparation
initially stops on a redundant text-replacement assertion;original source and
failure receipt remain. A separate repair verifies the six completed files
unchanged and finishes preparation without repeating a GPU or qualified stage.
