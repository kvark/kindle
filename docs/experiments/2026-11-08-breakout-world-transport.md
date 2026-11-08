# Breakout: learned pixel transport with an RGB fallback

The [fixed-event phase](../results/2026-11-08-breakout-world-fixed-events.md)
preserves perception and improves some event measures but fails precise forecasts
and ending timing. The [fusion](../results/2026-11-08-breakout-world-spatial-fusion.md)
and [pixel residual](../results/2026-11-08-breakout-world-pixel-residual.md) variants
also fail. These results do not isolate a failure cause. Select a distinct
image-prediction bias, rather than another loss-weight or seed comparison.

Default-off `rgb_spatial_transport` retains categorical RSSM, two-image causal
history and the fusion decoder. From its existing16-channel fused feature map,
a new1x1 convolution produces35 fields:17 horizontal offset logits,17 vertical
offset logits and one blending logit. Offsets are every integer−8..8 at the
native model grid. Softmax across offsets at each pixel, shared by all three
colors. Horizontally mix zero-padded copies of the latest image's probabilities,
then vertically mix that result. The horizontal field at the intermediate source
row participates in the vertical mix; this is an explicitly ordered separable
transport, not arbitrary joint289-offset probabilities.

Sigmoid blending chooses transported RGB versus the existing sigmoid RGB head.
The latter can generate new content or motion beyond the transport window.
Convert the mixed probabilities to logits after clipping to1e-6..1−1e-6.
This keeps uniformly spread small-object mass above the probability floor more
often than a1/255 clip; finite black/white values and derivatives must be checked
on Auto. Fallback identity is clipped at the extremes. Use ordinary named D3 initialization and zero biases;
no flow/alpha bonuses, pretrained weights, object labels or emulator-state input.
Use existing slice/concat/reshape/softmax/elementwise operations only. No backend
primitive, dependency, actor/critic or event-head change.

New head adds560 weights/35 biases:82 logical/8,872,550 scalars/246 tensors at
RGB64/12M. Preserve all old parameter names and named initialization. RSSM
feature2560 plus24,576 image-history scalars stay. Main reconstruction sees two
earlier images; forecasts seed from the observed prefix and shift generated RGB
with full BPTT through both transport and blend. True future RGB/outcome/reset
labels remain stopped targets/masks only, never history corrections. Head
predictions remain RSSM-only. Transport may still blur or be ignored; no physical
accuracy, sampling calibration or event improvement is assumed.

Require fusion, standalone `motion_event_rollout`,positive H and no residual or
image feedback. Bind flag/objective/geometry/scope to v17; old native must reject
before GPU. Preserve defaults/v1-v16 and old report layouts. Existing13 reports
and all scales stay; do not repeat the earlier Python fixed-event layout bug.

Before allocation, CPU tests cover direction/zero-padding/color sharing, finite
values, blend/clamp behavior, actual F64 values/derivatives and finite differences
through history/masks/RGB, graph ancestry, unique names, new parameter geometry,
header/scope and Python rejection/CLI paths. Strict core/test/binding Clippy and
isolated package checks. Reuse unchanged loss/GRU/filter primitive receipts;
do not turn this into backend qualification.

Separately declare native composition qualification (900s): new transport values/
derivatives versus the fixed F64 fixture, known shifts/boundaries/lane isolation,
finite black/white cases and history-gradient connectivity. One full-shape update
must have13 finite reports/correct totals/masks,246 initial/frozen tensors,
all240 common v13 initial tensors exact,16 cached nonimage initial arrays exact,
required changed families including both new parameters,and18 initial/trained
restore arrays. Retain actual-initial images; do not call an old model again.

Only after reviewed qualification, separately declare fresh8,192 updates (3600s)
and final restore (600s). Same16 fitting/eight development streams, original full
4/2/1/1 plan and sealed exposures; no new collection/recount. RGB64/12M/B8/T32/
H16/full BPTT/Auto,seed41001,rates4e-4,mixing.01,observed/future RGB100,KL1/.1/
free-nat1,all event scales1. Keep actual initial/2048/4096/final,384 dense origins,
all reports/curves and exact final restore. Reused seed/families are development.
Changing decoder composition/parameterization changes gradients and optimizer
history; no isolated causal error attribution or independent-seed claim.

After reviewed learning/restore, separately declare only the new final84 frozen
groups (1800s),history16 primary/32 secondary,eight draws,1/4/8/16/32 endpoints.
Reuse oracle/H0/fusion/v13 arrays; no old-model query or completed-stage repeat.
All gates stay:observed>=90%,one>=89/109,eight>=28/107,primary category-balanced
survival-weighted16/32 ball energy<=.12712666431732942 (H0.1695022191),serve-NOOP
false balls<=10%,ending32 Brier<=.10,ending16<=.01,RIGHT16 risk<=.10,every nonending
category mean<=5%,freeflight8 presence>=90% each command. Retain individual alarms
and all action/horizon reward/paddle/brick regressions;no best draw/checkpoint.

Full precise physics,event timing/calibration,persistent removed-brick memory,
late/tunnel/second-wall coverage,untouched families and fresh confirmation remain
required. Existing oracle has no above-top-brick endpoints. Intermediate gates
cannot finish the goal. No unchanged extension,sweep or automatic successor.
Root `/mnt/data/kindle-breakout-world-transport-20261108`;AMD Auto/host guard,
check live identities,stop real faults/nonfinite state,no reset/recovery/NVML.
No GPU allocation or full learner at selection.
