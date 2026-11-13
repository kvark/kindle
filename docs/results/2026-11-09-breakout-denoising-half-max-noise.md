# Half-maximum-noise denoising: small one-step gains, failed fitting

The goal **prove the world model training on Breakout** remains unproven.
The [declared fitting check](../experiments/2026-11-09-breakout-denoising-half-max-noise.md)
fails. Final ordinary matches are[1, 1, 0, 2]/110 andprivileged actual-history
one-step matches[11, 17, 16, 14]/198 across four diffusion draws. The residual control
scores0/110 and0/198 on every draw. High-noise emphasis yields limited next-
position matches in this development recipe;it does not establish a useful
world model,robust improvement or the unique cause of earlier failures.

| Draw | One-step correct /198 | Ball detected /198 | Detected within1px ofprevious | Median previous/target error ondetected pairs | Median movement projection |
|---|---|---|---|---|---|
|0|11|48|34|0.214/2.236|0.000|
|1|17|49|30|0.414/2.236|0.000|
|2|16|48|30|0.417/2.236|0.000|
|3|14|51|32|0.405/2.236|0.000|

Control detections135/135/135/128 all match previous positions andnone match
next positions. New detections48/49/48/51 are fewer,with11/17/16/14 correct
next positions. Median movement projection remains0 among detected pairs;
copying andabsence still dominate. Projection isdot(predicted−previous,
target−previous)/squared_norm(target−previous),wherecopy0 andideal movement1.
All864 one-step rows,including unsupported frames,remain in`motion-review.json`.

The four draws share one learned checkpoint andthe same reused fitting clips;
they are not independent training seeds. Matched source endpoints have
intersection1 andunion36 across draws.

| Source lane | Supported one-step endpoints | Correct ondraw0/1/2/3 |
|---|---|---|
|0|25|3/4/2/4|
|1|25|0/1/2/1|
|2|24|0/0/0/0|
|3|26|1/3/3/2|
|4|22|2/1/2/1|
|5|26|2/0/1/0|
|6|25|3/5/3/2|
|7|25|0/3/3/4|

Only optional half_max_noise changes. The first orlast half ofB32,alternating
by update parity,isforced to sigma5;the other half retains the original clipped
lognormal law. Every clip hasfour noisy copies:two forced-high andtwo original-
law every update. All original sigma normals are still consumed before pixel
noise. Target order,pixel-normal draws andoptimizer-update ordering remain.
Exactly32,768 valid target exposures areforced at5,plus any natural clipping
among the other half. The original nominal population mass at5 is
0.00961098 (about.96%);this is a
formula calculation,not a measured count ofthe old run's realized sigmas.

The sigma weighting ofthe joint image objective changes. At a fixed sigma,
withpositive original-law coverage andsigma supplied to the network,the ideal
conditional denoising mean remains. Finite-capacity optimization changes.
Aggregate image-MSE averages across these different mixtures are not matched
losses andmust not be used as evidence ofbetter physics. No new object/RAM
labels,pretrained encoder,motion rule orbackend primitive. Learning/inference
stay native Meganeura+Blade;Python prepares andanalyzes only.

Same fresh2,048 calls/updates,65,536 targets,224 unique arrivals,588 positive
source-prefix targets,rate1e-3,seed41001,RGB64/trainB32/inferB8,Auto,LaProp
epsilon1e-8/AGC.3,context-motion scale100,residual coordinates,actual four-frame
training histories,45 parameters/1,787,843 scalars/135 tensors andthree Euler
sampling steps. No old learner extension,collection or emulator action/frame.

| Check | Result |
|---|---|
| Source/native | ccf09d7f /d3b46c074691287fbb1d998be6c381def6f0ed9fb7cf1fdcb8fb53d7ae268ea2 |
| CPU | 16 Rust,394 packaged,4 runner checks;strict core Clippy |
| Graph/loss | Complete nodes/weights/source unchanged;weighted-loss native value/derivative receipt reused |
| New native fresh pre-mixture v3 initial | 135 initial/frozen tensors;20 first-step per-draw output arrays exact;no learning |
| New learner initial | 135 tensors and20 full per-draw forecast arrays exact |
| First update,before update2 | Six finite reports/correct total,45 changed weights;event losses and57 event weights/moments exact |
| Final event-head control | All57 weights/moments remain bit-exact to residual control |
| Frozen/v4 restore | 135 tensors and20 per-draw output arrays exact |
| Declared/CPU analysis pins | 134/93 |
| Call/loop time | 160.819s/185.468s;control158.881s/167.465s |

Legacy qualification guard/review finishes before learning,enforced bywrapper.
Optional half_max_noise defaultsfalse andis omitted fromlegacy config;v1/v2/v3
behavior remains. True requiresv4;inconsistent flags/headers reject beforeGPU.
Inference sampler isunchanged. Shared host/filesystem activity limits timing
attribution;call timing includes native CPU/GPU/readback,not isolated kernels.
No default change isjustified bythis failed gate.

| Final ordinary horizon | Draw0 | Draw1 | Draw2 | Draw3 |
|---|---|---|---|---|
|1|1/8|1/8|0/8|2/8|
|4|0/6|0/6|0/6|0/6|
|8|0/8|0/8|0/8|0/8|
|16|0/8|0/8|0/8|0/8|

Observed sigma.5 matches[5, 3, 5, 4]/8,versus control4/2/3/2;this noisy-target task
is distinct from RSSM reconstruction andfrom next-frame generation. Own initial
ordinary/one-step scores remainzero;plain last-frame copy scores2/110 ordinary
and0/198 one-step. Missing balls fail;18 unsupported per task/draw stay explicit.
Raw ordinary detections are4/2/1/3,with all remaining supported endpoints absent.
All physical/event metrics,regressions,1,376 endpoints and16 exact fixed lane0/
draw0 frames remain. Evaluated suffixes havezero positive reward/ending events
andcannot establish natural calibration or event timing. Static walls do not
prove changed-brick memory. Motion/coordinate measurements are analysis only.

No old trained model/final RGB isqueried;only required cached actual-initial
arrays aredecoded. All prior trials/failures/repairs remain retained andclosed.
No extension,broad learner,collector or automatic successor isselected.
Actor/critic,CDP comparison andfive-game work stay parked. Full physical/event
precision,natural calibration,persistent brick memory,late/tunnel/second-wall/
above-top coverage,untouched families andfresh confirmation remain required.
The full oracle lacks above-top endpoints. This limited gain cannot complete
the full world-model goal.

Root `/mnt/data/kindle-breakout-denoising-half-max-noise-20261109` retains source,
native/wheel/validation executable,config/plan,all loss rows/checkpoints,guard/
review receipts,`motion-review.json`,`event-control-review.json`,
`individual-endpoints.csv` and`contact-lane0-causal.png`.

Closure verifies261 pins in1.118s,three sealed guards/27 files
andtwo terminal successful services. Own2,814,996,842-byte disposable build
caches are released;native,wheel,validation executable,source andweights remain.
