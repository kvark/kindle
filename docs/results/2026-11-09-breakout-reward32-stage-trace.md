# Retained Breakout denoiser: later stages help ball recovery

The goal **prove the world model training on Breakout** remains unproven.
This [frozen trace](../experiments/2026-11-09-breakout-reward32-stage-trace.md)
shows the retained sigma_data.05/reward32 model already under-renders the new
ball atits first high-noise estimate. Later stages improve strict ball recovery.
It doesnot support switching toone stage,claiming optimal three-step sampling
orproving a loss/architecture cause. Actual-history one-step contexts only.

| Denoised estimate | Strict matches /198, four draws | Median new-pixel change projection | Median old-pixel erasure projection |
|---|---|---|---|
|sigma5|114/114/107/111|0.653–0.670|0.967–0.973|
|sigma.283|139/149/139/138|0.721–0.770|0.997–1.000|
|sigma.002 /served|140/149/140/140|0.730–0.778|0.999–1.000|

Stage1→2 creates131 correct cases andloses
12 acrossfour reused draws (net+119).
Stage2→3 creates4 andloses0.
These are matched source/draw cases,not independent learners. Where detected,
median motion projection is.966–.971 atstage1 and.978–.986 atstage3 (ideal1).
Creation isweaker than erasure atallstages. This ispartial learned motion evidence,
not accurate rendered world dynamics orfull physics coverage.

Unweighted image MSE increases5.06075643e-05→6.58823379e-05
(+30.18%) while strict ball matches improve.
Thus this pixel metric alone is a poor physics-quality proxy here. It isnot
the native weighted/preconditioned training loss,so this comparison doesnot
establish that the training objective caused the errors. Allstage paddle/brick/
pixel metrics,false detections,candidate ambiguity and2,592 individual endpoints
are retained. Contrast projections useactual changed sprite pixels foranalysis
only;no detector change,object labels orpeak-only surrogate forecast.

Same retained native source8dbf9ab5/executable SHA
786ab275e07e80ea23139b0916d669006b1745ea6aa7a4e98ae5464a511a1ebd,
checkpoint `/mnt/data/kindle-breakout-denoising-reward32-20261109/train/checkpoint-2048`.
No rebuild,production API,loss,network,sampler,precision orweight change. Reuse
qualified probe/control sources fromtheir immutable source.tar;newer worktree
full-resolution code isnot used. Older sigma_data.5 trace staysclosed;its scores
are not imported as this model's evidence.

Native108 traces/324 denoiser calls andfour normal origin16 one-step queries
complete in3.630972225s,including restore/readback/serialization on
shared host,not isolated kernel time. All108 finalbatch arrays/864 images and20
origin model arrays match cached controls bit-exactly;stage3 also equals served
final RGB bitwise.135 frozen tensors exact,age2048 unchanged. No learning,new
emulator action/frame ornew policy/decoder. One successful terminal service,
host guard/nine sealed files pass.27 fixedlane0/draw0 lossless panels andallrows
remain outside Git. No build cache created.

No ordinary autoregressive one-stage rollout isassessed:intermediate estimates
use actual histories alongthe original three-step path. Normal final-origin
reward/positive/continuation outputs onlyareparity-checked,not new event
calibration. Four draws reuse onelearner/clipset. Full physics/events,natural
calibration,changed-brick memory,late/tunnel/second-wall/above-top,untouched
families andfresh confirmation remain required. Retain the original sampler/model;
no sampler intervention,learner,extension,collector orautomatic successor selected.
Fullres protocol failure remains failed. No result completes the full goal.

Artifacts `/mnt/data/kindle-breakout-reward32-stage-trace-20261109`,invocation
9dd85b41fdac403db6eaf4c76bf2887f.

Closure verifies147 pins in0.561s,one successful terminal service/nine sealed guard files.135 tensors remain exact,no learning/build/collector orsuccessor.
