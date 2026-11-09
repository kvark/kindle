# Rehearsal while expanding the accurately fitted recurrent world

The recurrent contact model fits110/110 supported future balls onfour draws,
butthe completed contact-first curriculum replaces that data withbroad replay
andends26/17/20/20 of110 contacts,withpoor broad forecasts. The subsequent
pixel-denoiser comparisons also fail sustained rollouts. Select one different
replay allocation:interleave contact rehearsal andbroad training,retaining the
existing recurrent world/objective. No old learner orstage isrestarted.
Root `/mnt/data/kindle-breakout-world-contact-rehearsal-20261109`.

Restore the original contact checkpoint2048,all204 tensors/moments/config/age,
not the failed curriculum's final. Existing native85dbe150/SHA5b112866 andmatching
qualified runner unchanged,no build. Keep RGB64/12M/B8/T32/H16/full BPTT/Auto,
seed41001,allrates4e-4,mixing.01,observed/future motion RGB CE100,KL1/.1/free-nat1,
observed reward/continuation1.68 weights/8,852,675 scalars. Checkpoints omit live
belief/replay/full RNG;this is a warm start,not uninterrupted learning,a fresh
learner orindependent seed confirmation. No config override orfrozen subset.

Exactly6,144 newupdates,age2048→8192. Odd newsteps use the first3,072 rows ofthe
closed curriculum's4/2/1/1 broad plan. Even steps use the eight original contact
windows,one pereach source environment. The24 source streams remain distinct;
onlyRGB/actions/rewards/terminals enter learning. The contact batch must equal
the old declared batch exactly,including reset/outcome treatment. RAM positions,
lives andcontroller labels remain selection/analysis-only. No new collection,
emulator action/frame,policy,critic,CDP,teacher orbackend primitive.

CPU independently recounts1,572,864 replayed arrivals,759,269 valid suffix targets,
3,570 valid suffix terminals,6,078 positive suffix rewards,4,053 observed terminal
rows,19,195 observed positive rows,52,988 context resets,3,922 source resets.
Unique arrivals:32,820 broad of32,850 available plus256 contact. Replayed counts
are not new experience. Compare to the retained6144-broad-update curriculum and
fresh H16/H0 controls fromcached outputs. Rehearsal changes data exposure/order
andoptimization history;equal newupdate counts do not isolate forgetting ormake
a matched-experience comparison. No claim that contacts forgetting isthe sole cause.

Keep original fitting/development source families,selection lists and384 dense
final origins identical. Reuse the closed curriculum's384 warm-initial dense
exports,with204 inherited tensors/config exact and32 contact arrays checked on
the newrestored learner. Do not repeat its dense initial assessment. Preserve
initial/4096/6144/final8192 checkpoints,all6144 eleven-loss rows,every source/mask/
weighted-total identity,periodic ordinary four-draw cohorts andall68 changed
weights. Final contact assessment retains all1..16 endpoints andallfour draws,
including every paddle/brick/pixel/event regression. Keep the original24 contact
fitting gates;this retention diagnostic cannot replace full-corpus gates.

One service,two serial guarded jobs,2400s main/600s restore caps,existing AMD host
guard,no NVML/reset/reload. Require terminal success/two guards18 sealed files,
204 own frozen tensors,18 development restore arrays and32 contact restore arrays
exact. CPU verify source data/plan/config/seed-family separation/import closure
beforeGPU. Archive immutable scripts/source/native/runner/data/checkpoint pins;
new CPU analysis orrepairs use new filenames/declarations. Reuse passed native
values/derivatives/masks/restore,no new backend qualification project.

Fixed broad dense gates stay observed>=90%,one-step>=89/109,eight-step>=28/107.
Missing balls fail;unsupported detector endpoints stay unscored. No bestdraw,
checkpoint,threshold orfavorable-only comparison. Ifallthree broad gates andall
original contact fitting gates pass,review first andseparately declare final-only
84 frozen action/oracle groups withoriginal energy/event/precision thresholds.
Otherwise retain the complete failed dense/contact evidence without an automatic
GPU successor. No old oracle/collector/model assessment repeats.

Report actual new learning/export/restore costs plusinherited186.289s native/
191.500s loop. Cached actual-initial evidence,selected/reused development families
andone warm-start trajectory are not untouched generalization orreliability.
Precise all-horizon physics,correct/calibrated rewards andlife/game-over timing,
persistent removed-brick memory,late/tunnel/second-wall/above-top,untouched
families andfresh confirmation remain required. The full world-model goal stays
active andunproven;no fitting orintermediate pass completes it.
