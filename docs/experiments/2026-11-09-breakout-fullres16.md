# Full-resolution residual pixel predictor

Select one fresh spatial-architecture comparison after wider U-Net channels
regressed at the fixed target budget. Small-ball visibility remains weak despite
conditional precision where detected. Test preserving spatial features at64×64;
this motivates a complete alternative recipe,not an established resampling cause.
Root `/mnt/data/kindle-breakout-fullres16-20261109`.

Add optional full_resolution=false/omitted,preserving exact U-Net paths/v1–v10.
True requiresv11 andconditional-fullres16-residual0p1-history4 for selectedwidth16.
After the usual88→16 input projection,applyseven16→16 ConvNorm blocks,each added
to its input withfixed residual scale.1,then16→3 output convolution. Allnine
pixel convolutions are5×5 atnative side;no strided feature convolution,upsampling
orU-Net concatenation. Maximum theoretical receptive field37×37. Train/sample
parameters share names/shapes;event graph/heads unchanged. Width32 combination
is CPU-checked butnot selected/qualified for native learning here.

Pixel parameters81,459 versus606,707,event heads1,181,136,total1,262,595 versus
1,787,843. This changes feature resolutions,channel allocation,skip topology and
parameter budget together;do not isolate one cause. Compute isnot matched;record
wall time. The normal native maxpool/frame-to-RGB64 resize remains unchanged.
No CPU resize,object label,emulator state,pretrained encoder orbackend primitive.
Conv/norm/SiLU/add/scale use existing native operations/derivative mechanisms.

Keep actual histories,RGB64,B8/trainB32,Auto,rate.001,seed41001,context-motion100,
residual/half-max,.05 data scale,default noise mean−1.2,reward32,three Euler steps,
2,048 five-frame calls/updates,65,536 targets,224 unique arrivals,588 positive
exposures andnative training Gaussian/target order. No direct-frame,target-motion,
width32 orhistory augmentation. No new data/game actions/frames,actor/critic/CDP
orretained-agent change. Four-frame conditioning doesnot solve unobserved-state
memory;full-world gates remain.

Run33 focused Rust CPU/67 packaged configuration-restore/four runner checks and
strict release test-target Clippy. Check allnine pixel convolution output sizes
stay native side,event graph identity,train/sample parameter sharing/counts,
legacy defaults andv11/architecture rejection before GPU. Reuse existing primitive
derivative mechanisms;bounded finite native learning/event parity/frozen restore
qualifies this selected shape/topology usage. No claim ofnew independent full-
network float64 derivatives,no extra backend project.

Before main,a guarded default U-Net fixture must match135 initial/frozen/first-
update tensors,20 arrays/six losses to cached reward32;one qualification update
is separate. Pin source/native/runner/data/scripts before GPU. Main135 tensors
must be finite,45 weights/1,262,595 scalars.57 event tensors plus15 unchanged
input/output projection tensors must match control initially. Main first/final
57 event tensors andtwo first event losses exact. Pixel trunk shapes/initial
RGB differ;retain own initial/final RGB/events. All45 weights change;save every
loss/checkpoint andordinary ages0/509/1021/1533/2048,initial/final actual-history
one-step forecasts. Require135 own frozen tensors/20 restored arrays exact,
v11/full_resolution/architecture retained. No privileged head-only stage.

Fixed final2048 gates unchanged: all supported ordinary1/4/8/16 balls within1px
and>=90% of110 ordinary endpoints per draw;actual-history one-step>=90% of198.
Missing predictions fail;18 unsupported per task/draw unscored. Retain1,376 rows,
motion,pixel ambiguity,paddle/brick/event regressions andfixedlane0/draw0 lossless
frames. No best-age,tolerance orscope change. Four draws share one learner;
one causal positive/no terminals cannot establish calibration orreliability.

One learner,existing AMD host guard,300s defaultfixture/900s train/300s restore.
Finish fixture terminal audit before main;two terminal services/three sealed
guards/27 files before CPU review. Stop realfaults/nonfinite output;no reset,
reload,NVML ortimeout restart. Preserve failures/separate repairs,retain native/
wheel/CPU executable/source/checkpoints andrelease own disposable build cache.
No extension,collector,old-model repeat orautomatic successor. Full physics/
events,natural calibration,changed-brick memory,late/tunnel/second-wall/above-top,
untouched families andfresh confirmation remain required. No fitting score can
complete the full world-model orfive-game goal.
