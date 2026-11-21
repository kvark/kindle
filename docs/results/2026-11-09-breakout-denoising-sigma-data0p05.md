# Sigma_data0.05: learned one-step motion, incomplete rollouts

The goal **prove the world model training on Breakout** remains unproven.
The [fixed-budget data-scale test](../experiments/2026-11-09-breakout-denoising-sigma-data0p05.md)
improves ordinary matches to[35, 38, 34, 33]/110 andprivileged actual-history one-step
to[140, 149, 140, 140]/198 across four diffusion draws,versus1/1/0/2 and11/17/16/14 with
sigma_data.5. It still fails the declared fitting gates. This is one development
learner onreused clips/seed,not a broad reliability ormastery result.

The new model advances detected balls instead ofcopying old positions. On the
detected/current/previous supported pairs,median movement projection is.979–.986
(copy0,correct step1) andmedian next-position error.334–.363 RGB64 pixels.
No detected supported one-step ball lieswithinone pixel ofits previous position.
Most remaining one-step failures areabsence;longer rollouts also misplace balls.

| Draw | Correct next /198 | Detected /198 | Detected within1px ofprevious | Median next error ondetected/previous-supported pairs | Median movement projection |
|---|---|---|---|---|---|
|0|140|143|0|0.363|0.979|
|1|149|151|0|0.363|0.986|
|2|140|140|0|0.363|0.980|
|3|140|142|0|0.334|0.984|

All864 one-step rows,including unsupported cases,remain in`motion-review.json`.
Projection isdot(predicted−previous,target−previous)/squared_norm(target−previous).
The detected/previous-supported subsets have136/144/134/138 cases;conditional
precision must not be confused with recall over all198 supported next frames.

| Control | Ordinary correct /110 | One-step correct /198 |
|---|---|---|
| New own actual-initial | [1, 2, 1, 1] | [0, 0, 0, 0] |
| New final | [35, 38, 34, 33] | [140, 149, 140, 140] |
| Sigma_data.5 final | [1,1,0,2] | [11,17,16,14] |
| Plain last-frame copy | 2 | 0 |

Thus this recipe learns beyond its own initial/copying behavior. The coefficient
change affects both optimization andsampling parameterization;this comparison
does not isolate one training-only cause or establish a universal scale recipe.

| Final ordinary horizon | Draw0 | Draw1 | Draw2 | Draw3 |
|---|---|---|---|---|
|1|6/8|6/8|5/8|6/8|
|4|3/6|3/6|2/6|2/6|
|8|2/8|6/8|4/8|3/8|
|16|0/8|1/8|0/8|0/8|

On the same source-arrival16..31 subset with110 supported endpoints,privileged
one-step predictions score[77, 86, 77, 77],versus ordinary[35, 38, 34, 33]. The histories
andnoise schedules differ:privileged queries reset their sampling seed at each
origin,while ordinary rollout advances it. This is descriptive evidence ofa
longer-forecast gap,not a pure causal estimate ofhistory-error amplification.

Only sigma_data changes from.5 to.05. Internal whole-image std on224 unique
training arrivals is.47237,frame-difference std/RMS.03537;the previous residual
model retained the absolute-image scale. Configured scale changes EDM c_in/
c_out/c_skip consistently in training andinference. Past-image normalization
remains fixed.5;log-sigma conditioning,noise amplitudes/distribution/RNG,actual
history,coordinate anchor,quantization,Euler scheme andsigma grid remain.
Changing coefficients changes residual targets andeffective image-space loss
weighting;aggregate MSE levels are not matched across scales.

Same fresh2,048 calls/updates,65,536 targets,224 unique arrivals,588 positive
source-prefix targets,32,768 forced sigma5 exposures,seed41001,RGB64/trainB32/
inferB8,Auto,rate1e-3,LaProp epsilon1e-8/AGC.3,context-motion100,residual/half-max
modes andthree Euler evaluations. No count recount,collection,new emulator
act/frame,object/RAM labels,pretrained encoder,new weights orbackend primitive.
Learning/inference stay native Meganeura+Blade;Python prepares/analyzes only.

| Verification | Result |
|---|---|
| Source/native | 7871f08b /af13a03eada23d60aa483e111f68a63d6e16c208daa76ec80ffdc8d7a61474a1 |
| CPU | 18 Rust,402 packaged,4 runner checks;strict core Clippy |
| Graph/weights | Complete graph nodes/loss source unchanged;45 parameters/1,787,843 scalars/135 tensors |
| Native loss math | Qualified weighted-loss value/derivative receipt reused;new affine coefficients independently checked against F64 |
| New native default-scale v4 initial | 135 initial/frozen tensors and20 per-draw first-step arrays exact;guard review finishes before learner,enforced bywrapper |
| New learner initial | 135 tensors exact;own forecast controls retained because new coefficients intentionally change predictions |
| First update,before update2 | Six finite reports/correct total,45 changed parameters,135 finite tensors;event losses/57 weights/moments exact |
| Final event-head state | All57 weights/moments bit-exact tocontrol |
| Frozen/v5 restore | 135 exact tensors and20 exact per-draw output arrays |
| Declared/CPU analysis inputs | 136/93 pins |
| Call/loop time | 175.803s/184.417s;control160.819s/185.468s |

Time is shared-host elapsed,including native CPU/GPU/readback andexports;no
isolated kernel speed claim. Optional sigma_data defaults.5 andis omitted when
default,preserving v1/v2/v3/v4 via the original coefficient function. Non-default
requiresv5;inconsistent headers/bad scales reject beforeGPU. The unchanged
history scale isexplicitly tested. Keep.05 as development evidence,not a new
global default or proven full-game setting.

Observed sigma.5 denoising matches[6, 6, 7, 6]/8;this noisy-target task isdistinct
fromRSSM reconstruction andfuture prediction. Missing balls fail;18 unsupported
per task/draw remain explicit. Every physical/event regression,1,376 endpoints,
16 exact fixed lane0/draw0 frames andloss row remains. Evaluated suffixes lack
positive rewards/endings,soevent timing/natural calibration remain unproved.
Static wall accuracy alone doesnot establish changed-brick memory.

An optional read-only console summary raises TypeError when multiplying an
unscored(None) horizon metric byzero support. Its original source/output are
retained as`inspection-original.py/.log`;the separate CPU-only
`inspection-repair.json` handleszero support explicitly,leaving metrics/gates
unchanged. The main saved evaluation andall native checks already pass;no
learner/model query isrepeated andno comparison isrelabeled.

No old trained model/final RGB isqueried;only required cached initial outputs
aredecoded for compatibility. All previous trials/failures/repairs remain closed
andretained. No extension,broad learner,collector orautomatic successor selected.
Actor/critic,CDP comparison andfive-game work stay parked. Original full physical/
event precision,natural calibration,brick memory,late/tunnel/second-wall/above-
top coverage,untouched families andfresh confirmation remain required. Full
oracle lacks above-top endpoints. This development gain cannot complete the goal.

Root `/mnt/data/kindle-breakout-denoising-sigma-data0p05-20261109` retains all
source/native/wheel/validation executable,config/plan,checkpoints/losses,guards,
CPU reviews,`individual-endpoints.csv`,`motion-review.json` and
`contact-lane0-causal.png`.

Closure verifies269 pins in1.002s,three sealed guards/27 files
andtwo terminal successful services. Own2,821,882,240-byte disposable build
caches are released;native,wheel,validation executable,source andweights remain.
