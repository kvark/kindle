# Denoiser learning rate1e-3

The goal **prove the world model training on Breakout** remains unproven.
The fixed-budget rate comparison fails its fitting gate.
Final ordinary matches across four draws are[0, 0, 0, 0]/110 andprivileged
actual-history one-step matches are[0, 0, 0, 0]/198. Observed sigma.5 denoising
ball accuracy is[0.375, 0.0, 0.25, 0.125],with eight supported balls per draw. The rate1e-4
control scores0/110,0/198 and0/8,respectively. Reused clips/seed make this a
development comparison,not held-out reliability or a complete world model.

The [protocol](../experiments/2026-11-09-breakout-denoising-rate1em3.md) changes
only learning rate1e-4→1e-3 for both image/event heads. Actual training histories,
ordered targets/native noise stream,seed41001,RGB64/trainB32/inferB8,Auto,
LaProp epsilon1e-8/AGC.3,context-motion scale100/radius4/mean1,45 parameters/
1,787,843 scalars/135 saved tensors,v2 format andthree Euler samples remain.
This is a fresh learner,not a restored extension. No collector,actor/critic,
object/RAM labels,extra inference images or native rebuild.

| Check | Result |
|---|---|
| Training | 2,048 calls/updates;65,536 targets;224 unique arrivals;588 source-prefix positive targets |
| Initial parity | 135 tensors and20 per-draw forecast arrays exact |
| First actual update,before update2 | Four forward losses unchanged;90 optimizer moments exact;45 changed weights |
| First image/event update-norm ratios | 10.000001277/9.999995322 |
| Frozen/restore | 135 exact tensors;20 exact per-draw restore arrays |
| Call/loop time | 153.982s/161.896s;rate1e-4 call157.905s |
| Mean image loss,first/last64 updates | 0.809515→0.048756;control.848643→.124014 |

All130 declared/72 CPU analysis pins,
four CPU runner checks and2,048 finite six-component reports pass. Native
source879ab527/binary9657af72 and its completed guarded weighted-loss value/
derivative qualification are reused. Runner/protocol sourcee775384f. Matching
runtime/runner/wheel,config,plan andsource archives stay pinned outside Git.
Shared-host activity limits timing attribution. New emulator actions/frames0/0.

| Final ordinary horizon | Draw0 | Draw1 | Draw2 | Draw3 |
|---|---|---|---|---|
|1|0/8|0/8|0/8|0/8|
|4|0/6|0/6|0/6|0/6|
|8|0/8|0/8|0/8|0/8|
|16|0/8|0/8|0/8|0/8|

| Update | Draw0 h1/h4/h8/h16 accuracy |
|---|---|
|0|0.000/0.000/0.000/0.000|
|509|0.000/0.000/0.000/0.000|
|1021|0.000/0.000/0.000/0.000|
|1533|0.000/0.000/0.000/0.000|
|2048|0.000/0.000/0.000/0.000|

All endpoint/regression andphysical/event metrics remain in `review.json` and
`train/rollout-*.json`. Missing balls fail;18 unsupported ball endpoints per task/
draw remain excluded explicitly. Privileged forecasts receive only actual past
images andone incoming command,not the target image. They are not ordinary
rollouts. Observed sigma.5 denoising remains distinct from RSSM reconstruction.
Evaluated suffixes have no positive reward/ending events,so this check cannot
establish event timing or natural calibration. Static bricks alone do not prove
changed-world memory. Raw ball candidate counts stay in `pixel-review.json`.

Only required cached actual-initial RGB is decoded;old final scores are read
from cached JSON withno old final RGB/model query. The old rate1e-4 allocation,
mixed-history/uniform/broad denoisers andall earlier failures/repairs stay closed.
No unchanged extension,broad learner,collector or automatic successor is selected.

Full physics/event precision,natural calibration,persistent removed-brick memory,
late/tunnel/second-wall/above-top coverage,untouched families andfresh confirmation
remain required. The existing full oracle lacks above-top endpoints. Neither
this development gain nor a fitting pass can complete the goal. Actor/critic,
CDP comparison andfive-game work stay parked.

Root `/mnt/data/kindle-breakout-denoising-rate1em3-20261109` retains source/native
identities,config,ordered plan,losses,all checkpoints,guard receipts,CPU reviews,
`individual-endpoints.csv` (1,376 rows) and`contact-lane0-causal.png` (16 exact
fixed lane0/draw0 frames from the same RGB64 camera).

The rate change does not repair the declared task. Although last64 image loss
is about61% lower than the rate1e-4 control,no ordinary/privileged final ball
matches are gained. One final ordinary draw has one coherent candidate at the
wrong position;other ordinary endpoints andall privileged ones havezero raw
ball-color candidates. Keep native default1e-4;no working denoiser recipe is
established by either learning rate.

The separately selected frozen conditioning check makes16 queries of this
current final checkpoint in1.377s,withno training or emulator actions/frames.
Twenty baseline per-draw arrays match the previously saved current-model
forecasts exactly;all135 saved tensors remain exact. The variants cyclically
shift only historical RGB across lanes,keeping noise andcommands fixed.

| Past RGB shifted | Input RMSE,last four | Forecast output RMSE,range over four draws |
|---|---|---|
| All past frames | .031053 | .017104–.017758 |
| Older frames,latest unchanged | .027288 | .010063–.010291 |
| Latest frame,older unchanged | .014821 | .009140–.010154 |

Thus both older andlatest images influence predictions at the image level.
The result contradicts a complete disconnection from history;it does not
establish correct ball-specific conditioning,motion learning or a unique error
cause. The shifted histories can conflict with unchanged commands;there isno
emulator branch or counterfactual physics/calibration score for them. Every
draw/lane response andevent-head difference remains in
`frozen-context-review.json`,withraw arrays in`frozen-context/queries.npz`.

An exact target join provides a second distinction:for source arrival15,
observed noisy-current denoising matches3/8,0/8,2/8,1/8 balls,whileone-step
forecasts fromthe same preceding actual histories match0/8 on all four draws.
The former task receives a noisy target image;the latter starts from noise.
This does not isolate sigma,sampler accuracy or one causal reason for failure.
The ball remains the unresolved fitting limitation despite active conditioning.

Closure verifies259 pins in0.606s,three sealed guards/27 files
andtwo terminal successful services. Frozen review verifies148 declared/
10 CPU pins. No new build/cache,native fixture or old-model query.
