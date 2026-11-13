# Direct Breakout next-frame learning: bounded fitting result

The goal **prove the world model training on Breakout** remains unproven.
The [direct-frame trial](../experiments/2026-11-09-breakout-direct-frame.md)
produces6/6/6/6 of110 ordinary ball matches and
63/63/63/63 of198 actual-history one-step matches.
0/24 individual gates pass;overall fitting fails.
Reject this tested recipe; retain clean-history reward32. It does not improve both tasks consistently.

| Final2048 task | Stochastic control | Direct candidate |
|---|---|---|
| Ordinary generated history,110 supported per draw |35/38/34/33|6/6/6/6|
| Actual-history one-step,198 supported per draw |140/149/140/140|63/63/63/63|
| Ordinary horizon1,support8 |6/6/5/6|3/3/3/3|
| Ordinary horizon4,support6 |3/3/2/2|0/0/0/0|
| Ordinary horizon8,support8 |2/6/4/3|1/1/1/1|
| Ordinary horizon16,support8 |0/1/0/0|0/0/0/0|

| Event task | Own initial reward RMSE | Stochastic trained | Direct trained |
|---|---|---|---|
|rollout|0.423461862|3.17931973e-05|3.50777527e-05|
|teacher|0.408190652|0.0314884793|0.0314894771|

Optional deterministic_frame=false remains omitted/default and preserves the
original denoising path/v1–v7. True usesv8,sampler direct-history4-quantized and
observed task causal-single-pass-not-observed-denoising. No parameter,graph,
loss expression orbackend primitive changes. Direct inputs have four past RGB
frames/four actionplanes,zero noisy-current channels andconstant ln5/4 plane.
Conditioners cin0,cout.05,cskip0 produce target(next−lastactual)/.05. Decoder
adds the latestpast image andquantizes once. Inference uses one graph call per
future frame instead ofthree Euler denoises; no future Gaussian is sampled.
Training consumes original Gaussian draws butthey do not alter direct inputs.

This changes the learning task andsampler together. It does not isolate random
noise alone. B32 repeats eight original targets four times;direct duplicates no
longer get different noisy-current inputs. A deterministic predictor also cannot
represent all unresolved transition uncertainty. Actual next image is a training
label andinput only to the separate privileged event classifier;it never enters
direct pixel conditioning. Target-independence/algebra andunchanged graphs are
CPU checked. Existing qualified graph derivatives are reused.

The API observed-image output is now a causal forecast from preceding frames,
not observed denoising/reconstruction. Its legacy observed_denoising saved-array
field remains for artifact compatibility with explicit new task metadata. Own
initial RGB is retained,not claimed equal to the old stochastic initial. Four
seed outputs are bit-identical,initial/final/restore;these are one deterministic
prediction,not independent confirmation. No object label orpretrained encoder.

Keep RGB64,B8/trainB32,Auto,rate.001,seed41001,context-motion100,residual frame,
sigma_data.05,reward_limit32,2,048 calls/updates,65,536 target exposures over224
unique reused arrivals and588 positive exposures. Half-max/default-noise-mean
fields remain serialized butdo not affect direct inputs. No history corruption,
new data,emulator actions/frames,actor/critic/CDP orretained-agent change.
All57 first/final event weights/moments are bit-exact tocontrol;scalar forecast
event differences arise fromgenerated image inputs,not different head learning.

Sourcef756f278/native105e12079324564bbc70ab190f769e4872fc843abbc796b78ae34342c6a0a631.
27 Rust CPU/54 packaged config-restore/four runner checks andstrict release
test-target Clippy pass. Initial build.sh stopped during package installation
because this environment has no pip;install.log preserves that failure. A
separate package_v2.py safely extracts the retained wheel to an isolated runtime;
54 packaged checks pass there. No GPU was launched before repair/qualification.
Defaultfalse fixture matches135 initial/frozen/first-update tensors,20 arrays
andsix losses before main launch;its one update is separate fromthe learner.
Main135 initial tensors match;57 first event tensors/two event losses match.
All45 weights change andlosses/tensors remain finite.135 own initial/final frozen
tensors/20 restored arrays match;v8/direct task preserved after restore.

Learn-call interval162.021262s;loop167.317788s on shared
host. Calls include packing/native execution/readback;loop includes intermediate
exports/checkpoints,not all initial/final assessments. No isolated speed claim.
Native Meganeura+Blade perform learning/inference;Python supplies corpus/analysis.
Own792,670,537-byte apparent disposable cache released after
packaging/assessment;wheel,native,CPU executable,source andcheckpoints retained.

All supported ordinary1/4/8/16 endpoints must be within1px andall16>=90%;
actual-history one-step>=90%. Missing predictions fail;18 unsupported per task/
draw unscored. No tolerance,budget,best-age selection orscope change. Retain
all1,376 endpoint rows,motion,pixel ambiguity,paddle/brick/events andfixed
lane0/draw0 lossless frames. One causal reward1 source example across four
identical draws is not fresh evidence;there are no terminal examples or
natural calibration. Full physics/events,changed-brick memory,late/tunnel/
second-wall/above-top,untouched families andfresh confirmation remain required.
No fitting score can complete the full world model orfive-game objective.

Root `/mnt/data/kindle-breakout-direct-frame-20261109`,fixture invocation
67875c91f55f4524a6a9cf761a417d41,main2a384ddf3ee34bfbaeeead7f41de34a8.
No extension,collector,old-model rerun orautomatic successor selected.

Closure verifies270 pins in0.730s,three sealed guards/27 files andtwo terminal successful services. Default135-tensor/20-array/six-loss parity,57 first/final event tensors,135 own frozen tensors and20 restore arrays allpass. Own disposable build cache released;native,wheel,source,checkpoints andoriginal packaging failure retained.
