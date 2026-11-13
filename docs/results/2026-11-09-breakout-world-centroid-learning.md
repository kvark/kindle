# Broad world learning with pixel-derived position supervision

Fixed final8192 passes **0/3 original broad dense gates**. The intermediate screen fails;do not adopt this recipe orallocate its conditional oracle assessment.

The full Breakout world-model goal remains active andunproven. Correct native execution,smaller auxiliary losses andconditional coordinate accuracy cannot replace ordinary RGB physics,event calibration orfull-game coverage.

[Protocol](../experiments/2026-11-09-breakout-world-centroid-learning.md).
Root `/mnt/data/kindle-breakout-world-centroid-learning-20261109`.

## Ordinary decoded RGB forecasts

First fixed draw,within one resized pixel. Missing/diffuse predicted balls fail;supports vary byhorizon andrelated origins are not independent episodes. No bestcheckpoint,draw,tolerance orauxiliary-coordinate replacement.

|Split/model|Observed|1|4|8|16|32|
|---|---:|---:|---:|---:|---:|---:|
|Development: Actual initial|0/112|0/109|0/109|0/107|0/114|0/95|
|Development: Retained H0|108/112|94/109|33/109|14/107|3/114|0/95|
|Development: Original H16|99/112|84/109|47/109|25/107|10/114|4/95|
|Development: RGB1 + position1000|85/112|73/109|35/109|15/107|5/114|5/95|
|Fitting: Actual initial|0/163|0/177|0/180|0/176|0/171|0/179|
|Fitting: Retained H0|159/163|153/177|73/180|34/176|14/171|5/179|
|Fitting: Original H16|154/163|130/177|86/180|61/176|21/171|19/179|
|Fitting: RGB1 + position1000|134/163|119/177|50/180|34/176|12/171|15/179|

Unchanged original observed>=90%,one>=89/109,eight>=28/107 gates: `{'observed_ball': False, 'one_step_ball': False, 'eight_step_ball': False}`.
All384 actual-initial/final dense origins andsource/target/command/reset/outcome arrays match the retained controls. Allhorizon/paddle/brick/RGB/changed-pixel/event metrics andallraw forecast/features remain. No source/model/oracle recollection orold-model GPUquery.

## Learned position head diagnostic

The linear head istrained natively onobserved andcausal imagined states. After learning,a separate CPU affine calculation applies its stored F32 weights tothe saved F32 features inF64. Safetensor matrix shape[2560,2] ischecked. Native-head readback parity isnot measured. The diagnostic isconditioned ontrue visible ball support;the head always supplies xy andhas no presence output. It isnot acomplete forecast,proper distribution score,replacement instrument ornative RGB success.

|Stage/split|State|Supported|Within1px|Euclidean RMSE (pixels)|Median error|
|---|---|---:|---:|---:|---:|
|initial/fitting|observed 0|163|0|21.350|18.744|
|initial/fitting|prior 1|177|0|22.184|19.191|
|initial/fitting|prior 4|180|0|22.170|17.488|
|initial/fitting|prior 8|176|0|21.141|17.051|
|initial/fitting|prior 16|171|0|20.605|16.758|
|initial/fitting|prior 32|179|0|20.347|17.562|
|initial/development|observed 0|112|0|21.785|18.129|
|initial/development|prior 1|109|0|22.767|20.390|
|initial/development|prior 4|109|0|22.738|20.830|
|initial/development|prior 8|107|0|23.116|18.542|
|initial/development|prior 16|114|0|21.234|18.251|
|initial/development|prior 32|95|0|19.881|17.208|
|final/fitting|observed 0|163|29|2.434|2.078|
|final/fitting|prior 1|177|31|3.417|1.864|
|final/fitting|prior 4|180|26|3.848|2.134|
|final/fitting|prior 8|176|15|4.360|2.604|
|final/fitting|prior 16|171|16|7.295|3.895|
|final/fitting|prior 32|179|3|10.665|7.355|
|final/development|observed 0|112|15|2.437|1.819|
|final/development|prior 1|109|17|2.430|1.902|
|final/development|prior 4|109|16|4.751|2.098|
|final/development|prior 8|107|6|7.220|3.451|
|final/development|prior 16|114|6|10.098|4.294|
|final/development|prior 32|95|1|12.227|9.277|

Every initial/final fitting/development/horizon error andout-of-image coordinate stays inhead-review.json/head-endpoints.csv. Pixel-derived target centroids andthe fixed scoring detector differ slightly;target audit andunsupported extra labels remain inqualification evidence. Aweak readout doesnot prove information absence,and astrong conditional xy readout doesnot prove object presence,event calibration oraccurate imagery.

## Additional original contact-family diagnostic

These eight clips are not this learner fitting data,butare reused development evidence,not untouched generalization. Fixedfour draws/all1..16 endpoints arepreserved. The separate original contactfit trained directly onthese clips andscored110/110 each;it isnot a matched-budget/generalization control.

|Draw|Matches /110 supported|
|---|---:|
|0|17/110|
|1|18/110|
|2|16/110|
|3|19/110|

## Recipe,supervision andscope

Fresh8,192 updates,RGB64/12M/B8/T32/H16/full BPTT/Auto,seed41001,allrates4e-4,mixing.01,KL1/.1/free-nat1,observed reward/continuation1. RGB weight1 plusobserved/future centroid1000. One shared linear recurrent-feature2560→xy head adds5,122 weights:70 logical/8,857,797 scalars/210 checkpoint tensors. This changes auxiliary architecture/objective andRGB scale together versusoriginalH16;no isolated cause ormatched-compute claim. Initial210 tensors match qualified freshzero-step weights,not a qualifier warm start.

Targets use red excess over1.5×max(other channels),region[4/64,28/64,60/64,56/64],pixel-center xy,mass>1e-6. They encode game-specific color/region supervision derived fromRGB,not emulator/object-state input or an unguided general representation. Extra mass-valid unsupported labels remain explicit. Observed/future losses average fullBxT/BxH withdetached valid/reset masks;future images never correct prior states. Ordinary RGB/events/brick gates stay.

Original16 fitting/eight reused development streams andexact sealed4/2/1/1 plan:2,097,152 replayed arrivals,975,941 valid suffix targets,9,570 suffix terminals,16,008 positive suffix rewards.34,440 observed positive rows,10,909 observed terminals,75,808 context resets,10,488 source resets;32,849 of32,850 fitting arrivals sampled. Replayed counts are not new experience. Fitting/development seed families staydisjoint,but41001/development families are reused,not fresh confirmation. No actor/critic/CDP,pretrained encoder,new emulator action/frame orcollection.

## Verification,cost andremaining requirements

Qualification/source449361ae/nativeee44ec16b509c601d6beefafeb03630aa9be9a3681e632a444dbd67b2b788712 reused exactly,no rebuild oroverlay.139 Rust CPU/strict core+binding Clippy/19 packaged Python checks andnative masked F64 derivatives precede the learner;earlier fixture API/prefix/style failures remain inqualification withcorrected checks. This learner usesmatching runtime/runner.

All8,192 fifteen-loss rows finite,total/source/mask identities pass,all70 weights change. Observed/future centroid valid fractions stay0..1 andfuture validity never exceeds its reset mask.210 actual-initial/own frozen tensors,34 finite forecast exports,18 development/32 contact restore arrays andv22 config/ages pass.149 native declaration/140 CPU analysis pins verified. Allnative stages stop successfully,two guards/18 sealed files,no native retry orrepair.

Learning-call interval 766.210s;training/checkpoint/forecast loop 879.901s. Loop excludes initial construction/denseinitial exports,final contact restore diagnostic,separate final restore andCPU analysis. Qualification separatelycost20.353s;its1,249,533,952-byte disposable buildcache isalready released. Evaluation every2048 differs fromold1024;shared-host wrapper times do not isolate kernel speed orlearning efficiency. No old learning cost ishidden.

Checkpoints0/2048/4096/final8192,allfour-draw curves0/2048/4096/6144/8192,raw arrays/features,position diagnostics andfixedlane0/draw0 lossless clip remain outside Git. Native checkpoints omit replay/live belief/full RNG;later restores wouldbe warm starts,not uninterrupted learning orindependent learners. Invocation `25f16bdc03a542cf979ef38ec116b22b`.

Precise all-horizon physics,correct/naturally calibrated rewards andlife/game-over timing,persistent removed-brick memory,late/tunnel/second-wall/above-top,untouched families andfresh confirmation remain required. No fitting/implementation/intermediate score completes thefull goal. No automatic learner/extension/collector/successor. Actor/critic,CDP comparison andfive-game work stayparked;prior jobs stayclosed andfullres staysprotocol-failed.

Closure verifies311 pins,one terminal successful service/two guards/18 sealed files. Broad0/3;conditional oracle assessment isnot allocated.
