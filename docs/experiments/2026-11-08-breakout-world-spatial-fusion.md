# Breakout: nonlinear fusion of image history and recurrent features

The contact-first curriculum fails broad physical/event gates and loses most
precise contact fitting. The earlier optional spatial decoder computes local
history and global RSSM features independently, concatenating them only before
a linear output convolution. Its logits are additive across those two feature
paths; the final sigmoid still combines them nonlinearly. The global state may
already encode history. This observation is not an expressiveness impossibility
or an established cause of failure. Select a distinct learned interaction test.

Default-off `rgb_spatial_fusion` requires the standalone categorical plain RGB
spatial decoder without image feedback. After concatenating the global decoder
field and history CNN field, add a5x5 stride1 ConvNorm to16 channels, including
per-pixel channel RMS and SiLU, then the final5x5 RGB output convolution. At12M,
the joint input has32 global plus8 history channels. Output projection input
narrows40→16. Add14,232 scalar weights net:80 logical/8,871,955 weights/240
checkpoint tensors. Compared with spatial v10,228 common tensors/76 weights
should match initialized values; the output kernel changes shape and three
fusion parameters are new.201 common tensors/67 weights should match H0.

Keep RGB64/12M/RSSM feature2560 plus24,576 image-history scalars. History's effective
decoder receptive field grows17→21 pixels. Fusion, output width, capacity, work,
initialization and gradient paths change together; no isolated nonlinear-effect
claim follows. Full world state includes image history, which legacy RSSM feature
exports omit. Event heads still read RSSM only. No peak-memory/speed benefit is
claimed. No object/RAM labels, external encoder, actor/critic or new backend op.

Observed reconstruction reads only the two preceding images, zeroed across
observed resets. Imagined history shifts in generated images with full BPTT.
Future images/outcomes/reset labels remain stopped targets/masks and never
correct state or clear imagined history. Terminal arrivals remain scored; mask
after reset crossings and average over full BxH. Reuse the qualified history,
Conv5/RMS/SiLU/upsample/categorical/motion-loss primitives. No image feedback or
future event loss is enabled. Persist the fusion architecture in v12; old natives
must reject it before GPU creation. Defaults and v1-v11 stay unchanged.

CPU checks cover defaults/config/header/objective/scope rejection, world geometry,
parameter counts, unique inputs, shared global/history ancestry and gradient
connectivity. F64 finite-difference checks cover the new concatenation→ConvNorm→
output composition at40 fusion inputs/16 outputs, with elementwise and full
directional probes. Packaged Python rejection/regression checks and strict core/
binding Clippy pass before native allocation. Preserve the initial CPU count
fixture failure (8,882,355 expected versus8,871,955 actual): the preset depth is16,
not32. The count-only correction changes no production architecture. Preserve the first
strict Clippy layout failure too; boxing the new optional CPU descriptor fixes
its enum-size warning without changing the GPU graph or parameter count.

Separately declare one900s guarded native qualification: composed fusion F64
values/raw derivatives, frozen parameters/no optimizer step; actual decoder joint
logit interaction under independent global/history perturbations and lane
isolation; full-shape one update, eleven finite losses/total/mask, required changed
families including all fusion parameters,228 spatial/201 H0 common initial tensors,
240 own initial/frozen tensors and18 exact initial/trained restore arrays. Reject
v12 under retained v11 native with GPU loader disabled. No old model forecast or
successful native fixture is repeated.

Only after reviewed qualification, declare fresh8,192 updates from0 (3600s) and
final restore (600s). Original full8,192-row4/2/1/1 plan,16 fitting/eight development
streams, RGB64/B8/T32/H16/full BPTT/Auto, seed41001, rates4e-4, observed/future motion
RGB100, KL1/.1/free-nat1 and observed reward/continuation1 stay. Recount975,941 valid
future rows/9,570 future terminals/16,008 positive future rewards. No new collection/
emulator actions/frames. Retain actual initial/2048/4096/final,384 dense initial/
final origins, all reports/exposures,240 frozen tensors and18 restored arrays.
Reused seed/families remain development, not untouched generalization or reliability.

After reviewed learning/restore, separately declare only the new final's84 frozen
oracle groups (1800s), history16 primary/32 secondary, eight draws and all1/4/8/16/32
endpoints. Reuse cached oracle/H0/categorical H16/spatial arrays without old-model
queries. Gates stay:observed>=90%,one>=89/109,eight>=28/107,category-balanced
survival-weighted16/32 ball energy<=.12712666431732942,serve NOOP false balls<=10%,
ending32 Brier<=.10,ending16<=.01,RIGHT16 ending<=.10,nonending category means<=5%,
freeflight8 ball presence>=90% on each command. Keep every individual alarm and
all action/horizon/reward/paddle/brick regressions; no easier slice or best draw.

Full causal physics,event timing/calibration,persistent brick memory,late/tunnel/
second-wall coverage,untouched families and fresh confirmation remain required.
Existing oracle has no above-top-brick endpoint. Intermediate gates cannot complete
the goal. No unchanged extension, repeated stage, sweep or automatic successor.
Root `/mnt/data/kindle-breakout-world-spatial-fusion-20261108`; AMD/Auto/existing
host guard, stop real faults/nonfinite state, no reset/recovery/NVML. No GPU work
has run at protocol declaration time.
