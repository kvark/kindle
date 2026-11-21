# Pixel-derived position supervision for the isolated recurrent world

The completed contact rehearsal retains110/110 onfour contact draws,butbroader
fitting/development eight-step matches are13/176 and3/107. Test an explicit
foreground-position learning signal alongside RGB. This is a new supervised
world objective,not a restart ofthe closed whole-agent centroid100 experiment.
The latter trained observed nonlinear readouts withactor/critic;this proposal
trains a shared linear readout onobserved andcausal imagined world states.
Root `/mnt/data/kindle-breakout-world-centroid-20261109`.

Reuse the existing detached RGB color-centroid target calculation:channel0,
background_scale1.5,normalized region[4/64,28/64,60/64,56/64],valid red-excess
mass>1e-6. Coordinates use pixel centers in[-.5,.5]. This encodes game-specific
knowledge ofa color/region. No object/RAM state enters model inputs;targets
are derived fromactual RGB. This is extra supervision,not evidence foran
unguided universal representation. There is no new encoder orbackend primitive.

CPU audit:targets match within1pixel on14,114/14,125 detector-supported fitting,
8,511/8,518 supplement and7,014/7,020 reused development arrivals;no supported
row has invalid centroid. Median errors .183/.220/.193 pixels. Centroid-valid
butdetector-unsupported rows1,096/454/515 remain explicit;mass validity isnot an
oracle ball-presence label. No detector/gate changes orabove-brick coverage claim.

Default-off existing color_centroid_supervision becomes supported only forplain
categorical standalone motion_cross_entropy withpositive suffix. Reject other
geometry/objectives/phases beforeGPU. Checkpointv22 binds the architecture and
settings;ordinary restore preserves them. Old defaults andwhole-agent centroid
head stay. One linear feature2560→2 head adds5,122 scalars/two logical weights:
70 weights/8,857,797 scalars/210 weights+optimizer tensors atRGB64/12M. No hidden
MLP inthis standalone head;the existing whole-agent nonlinear head isunchanged.
Inference graphs/ordinary RGB/reward/continuation outputs are unchanged inshape.

Observed centroid loss averages squared xyerror over fullBxT,withdetached
mass-valid mask. Future head reads the causal prior belief at each H16 transition;
future RGB creates only detached targets,not states. Multiply validity byexisting
stopped reset-crossing suffix mask,then average fullBxH,not valid count. Addboth
raw losses withthe configured centroid scale. The15 reports keep the first11
motion reports andappend observed_loss/valid,future_loss/valid. Keep allreset/
terminal/outcome semantics andfull BPTT. Future state must not depend onfuture
RGB,reset weights orlater actions. Gradients must reach past encoder/posterior,
GRU,prior andlinear head;target/mask gradients arezero.

Selected phase:CPU/build/package andone guarded native qualification only.
Reuse existing primitive receipts. CPU tests cover F64 masked values/derivatives,
reset zeros,target/mask detachment,full geometry,shared parameters,inference
identity,causal ancestry,gradient connectivity,constructor/header restrictions
andreport accounting. Preserve initial fixture compilation/prefix-name failures
withsource/logs andseparate repaired checks. Strict core/binding Clippy and
packaged affected Python checks required. Retain wheel/native/test executable/
source;release disposable build cache after packaging.

BeforeGPU,separately pin source/native/executable/runner/config/data/scripts.
One900s native qualification stage checks the small masked centroid values and
derivatives against F64,exact zero masked rows anddetached target/keep parameters.
One full-shape fresh update thenchecks all210 tensors/15 finite reports,total
identity andvalid masks;both new head weights andshared encoder/GRU/prior must
change. Require204 common initial tensors and32 actual-initial contact model
arrays exact tocached old reference. Require210 own frozen tensors and32 trained
restore arrays exact. No old model isqueried;no game actions/frames. AMD Auto,
existing host guard,no NVML/reset/reload. Stop realfaults/nonfinite state.

Only after reviewed qualification,separately declare a fresh bounded learning
check. Candidate recipe isRGB1 plusobserved/future centroid1000,RGB64/12M/B8/
T32/H16,8,192 updates,allrates4e-4,mixing.01,KL1/.1/free-nat1,observed events1.
Use the original16 broad fitting/eight development streams andsealed4/2/1/1 plan.
This changes auxiliary head/objective andRGB weight together;no isolated cause
claim. Reused41001/seed families aredevelopment,not fresh confirmation. All
ordinary RGB physics/event/brick gates stay;coordinate losses cannot replace
forecasts. Require actual-initial/final dense origins,complete curves andfrozen
restore. No learner orautomatic successor isallocated bythis qualification.

Full precise all-horizon physics,event timing/natural calibration,changed-brick
memory,late/tunnel/second-wall/above-top,untouched families andfresh confirmation
remain required. A low coordinate loss orpassed implementation doesnot establish
a good world model. Actor/critic,CDP comparison andfive-game work stayparked.

Qualification completes: [native result](../results/2026-11-09-breakout-world-centroid.md),190 closure pins/nine sealed files. Build cache1,249,533,952 bytes released. Fresh [learning declaration](2026-11-09-breakout-world-centroid-learning.md) isseparate;qualifier checkpoint isnot resumed.
