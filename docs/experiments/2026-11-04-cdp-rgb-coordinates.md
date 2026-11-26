# Fixed coordinate channels inside the learned RGB encoder

Select one optional native architecture test within the existing five-game goal.
Breakout's retained 200k recipe scores143.667 greedy/70.750 sampled, with zero
two-wall successes. Earlier fixed readouts find position error, but do not prove
information absent. Better own-replay imitation fitting repeatedly fails to
produce better frozen play. Test whether explicit image position makes the
existing CNN easier to train; do not infer the causal limitation in advance.

The [CoordConv paper](https://arxiv.org/abs/1807.03247) supplies the architectural
idea of coordinate channels. Its results are motivation, not a prediction for
this CDP recipe. No object detector, teacher coordinates or emulator state enters
the model input.

## Native implementation and qualification

Add default-off `rgb_coordinates`. Inside the learned RGB encoder, append x then
y planes to every image immediately before the first convolution. For resized
pixel center (x,y), use ((x+.5)/side−.5, (y+.5)/side−.5). Coordinates refer to the
resized image, including when an explicit crop is configured. They are constants
in the native graph, not parameters or replay entries. Preserve the existing
single GPU resize, RGB values, observation/replay dimensions, later CNN widths,
RSSM and actor/critic. First-layer weights grow by 2×25×32=1,600 at12M; widened
initialization can change later initial values. Do not claim matching baseline
initial tensors after changing architecture.

Require learned RGB; reject feature/video encoders. Keep legacy serialization
unchanged when disabled. Restore retains the recorded option; fresh CLI overrides
on restore/evaluation are rejected. Old native packages must reject requested
or restored coordinate models before emulator or GPU construction. RGB decoder
targets remain the original three channels. Existing frame-pair/crop options
remain distinct; no combined recipe is selected here.

Qualify config/default/serialization and32/48/64 geometry on CPU. Check exact
parameter increments, unchanged replay/latent dimensions and backward graphs.
Use independent F64 CNN output/loss and finite differences for RGB, x-plane,
y-plane and later-layer weights on one guarded native64 fixture. Preserve any
failure; investigate before proceeding, without loosening checks after results.
Run appropriate Rust/Python CPU suites and strict Clippy; package an isolated
wheel/test executable with exact source hashes. Verify a short frozen legacy
restore on the new native binary, with zero updates and unchanged tensors.
No backend optimization, dependency change or old experiment repeat.

## Bounded learning decision

After native qualification, declare one fresh8,192-action check and1,024-action
frozen restore. Keep the retained centeredCDP12M64/Auto/N8/B16/T16/H15/R16,
base/encoder4e-5, dynamics4e-4, behavior8e-5, critic1/1, expert100/unit weights/
world gradients,20% episode guidance/cap1920/all-state labels, sampled collection/
posteriors, latent/actor mixing.01, replay100k and full18/sticky.25/repeat4/noops0.
Only coordinate channels are enabled; use fresh1009, without experience reuse.
Native/runner sources differ from retained447fe50 and must be recorded.

Require489 finite updates at the declared cadence, correct architecture/schema,
changed CNN/reward/continuation/actor/value, full pixel/action/label replay,
guards and357 invariant frozen tensors. Freeze exact commands and inputs before
launch. Training cap180s, frozen cap60s; conservative projection
elapsed×200000/8192×1.1 must remain within1,800s. Report actual cost separately
from baseline47.690s; one short observation is not a sustained speed claim.

Passing selects no automatic full learner. A separately declared200k comparison
would require100k recovery, actual-initial/final sampled/greedy24-episode arms,
all audits and paired own-initial gain/min comparisons against the retained
recipe. Keep original mean864/90%-two-wall gate. No sweep, extension, seed
campaign or successor follows from this protocol. One GPU allocation at a time;
check live identity, stop actual faults, and never reset/reload/reboot the GPU.

Evidence root: `/x/Temp/kindle-rgb-coordinates-20261104`.
