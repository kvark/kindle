# Pixel-derived world centroid supervision: native qualification

Optional standalone observed/future centroid supervision passes implementation
qualification. This doesnot establish useful learning or Breakout competence.
The full world-model goal remains active andunproven.

One linear feature2560→xy head adds5,122 weights:70 logical/8,857,797 scalars/
210 checkpoint tensors at12M. Ordinary RGB/action inputs andinference output
shapes stay. Detached RGB color/region targets are game-specific supervision;
there isno emulator/object-state input orunguided representation claim. Targets
are within1px on14,114/14,125 fitting,8,511/8,518 supplement and7,014/7,020 reused
development supported frames;1,096/454/515 extra mass-valid unsupported frames
remain explicit. Label validity isnot oracle ball presence orabove-brick coverage.

Observed andcausal future position losses share the linear head. Future masks
preserve terminal arrivals andzero later reset crossings;targets/weights are
stopped andlosses average fullBxT/BxH. CPU causal ancestry/gradient tests verify
future images/masks/later actions never enter earlier states. Gradients reach
past encoder/posterior,GRU,prior andhead. No newbackend primitive oractor/critic.

139 Rust world CPU tests pass,40 GPU tests ignored. Strict core/binding Clippy
and19 packaged Python tests pass. Native masked F64 values/derivatives,detached
pixel/keep parameters andexactzero masked gradients pass. Full-shape fresh
RGB64/B8/T32/H16 update has15 finite reports andcorrect weighted total.67 of70
weights change initially,including both new head weights andencoder/GRU/prior;
three reward hidden weights retain their first-step initialization. All210 own
initial/trained frozen tensors,204 shared initial tensors and32 actual-initial/
trained restore model arrays match exactly. Checkpointv22 binds config/objective/
geometry andrejects incompatible paths beforeGPU. Other inference graphs stay.

Initial CPU fixture compilation failed onincorrect reference-module/Graph-clone
APIs;the first corrected fixture also used anincorrect prior prefix. Strict
Clippy caught two fixture style issues. Original source/logs remain;separate
corrected checks pass. No GPU job orhost-guard audit repeats for these repairs.
The original native qualification itself completes successfully in20.353s,
including primitive subprocess,restore/readback andsource checks onshared host.
One guard/nine sealed files pass;one worldoptimizer update,no mainlearner update,
emulator action/frame,reset/reload/NVML orhost recovery.

Source449361ae/nativeee44ec16b509c601d6beefafeb03630aa9be9a3681e632a444dbd67b2b788712;
CPU test2d1e397ae1be40740d19011c5056b10711be0b0bc930fd2fe5a32d597dcf65ac.
Root `/mnt/data/kindle-breakout-world-centroid-20261109`,invocation
`d4a9ba97a764454aaf29b015f984c45c`. Wheel,native,test executable,source,failed
fixtures,logs andcheckpoints are retained;own disposable build cache isreleased.

Fresh RGB1/observed+future centroid1000 learning requires its separate declaration,
not aqualification warm start. No learner isallocated bythis receipt. Low
coordinate losses cannot replace unchanged RGB physics/event/brick evaluation.
Precise full-game forecasts,natural event calibration,removed-brick memory,late/
tunnel/second-wall/above-top,untouched families andfresh confirmation remain open.
Actor/critic,CDP comparison andfive-game work stayparked;prior jobs stayclosed.

Closure verifies190 pins andone terminal successful service/nine sealed guard files;1,249,533,952 bytes ofown build cache released.
