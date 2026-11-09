# Replay-derived color attention on the world RGB loss

Position10,000 improves its conditional coordinate head but weakens decoded RGB
(development77/112 observed,70/109 one-step,14/107 eight-step). Test loss attention
on the actual decoder pixels rather than another auxiliary readout. Prior motion
attention weights any changing pixel; this option adds explicit color/region
knowledge. It is not a new encoder, object-state input or general visual prior.

Root `/x/Temp/kindle-world-color-attention-20261109`. The selected stage is CPU,
build/package and one guarded native qualification. No learner is allocated yet.

For centered replay pixels x, let q=max(x.red+.5 −1.5×max(x.green+.5,x.blue+.5),0)
inside region[4/64,28/64,60/64,56/64], zero outside. Multiply the existing motion
weights by(1+1000q), shared across all three channels, and renormalize the combined
weights to mean one in each image. Apply these detached weights to observed and
causal imagined RGB cross-entropy. Retain the ordinary full-row denominator and
stopped reset masks. Future images and derived weights never feed prior beliefs.
No mass validity or oracle presence label is required. This can emphasize false
color fragments; accurate presence, trajectories and imagery remain requirements.

Default-off `rgb_color_attention` reuses the color/region settings schema; scale
is attention strength, not an auxiliary coefficient. Accept only standalone plain
categorical motion_cross_entropy with a positive suffix and no position head.
Reject Gaussian/spatial/previous-frame/latent-only combinations before GPU.
Checkpointv23 binds the flag; legacy headers cannot silently enable it. No new
model parameters, reports or inference path:68 logical weights/8,852,675 scalars/
204 weights+moments at12M,11 reports. Whole-agent construction/restore reject it.
Default graphs and old position supervision remain unchanged when disabled.

CPU checks: independent F64 color/motion attention values and RGB-logit gradients,
detached replay/motion/reset inputs, exact reset-zero gradients, mean-one weights,
causal state ancestry, unchanged parameters and inference, default serde/header
compatibility and constructor guards. Retain initial fixture failures separately:
inferred unsized strings, then an ideal F64 reciprocal instead of the graph's
F32 averaging constant. Corrected reference must use the actual F32 constants;
do not count failed invocations as passes. Run affected world/config checks and
strict core/binding Clippy, plus packaged affected Python checks.

Pin source/native/executable/runner/config/data/scripts before GPU. Reuse the
existing eight recorded contact clips for qualification only; no new collection.
One900s guarded stage runs the new native primitive value/derivative/reset test
and a fresh full RGB64/12M/B8/T32/H16/Auto optimizer update. RGB reconstruction100,
rates4e-4,mixing.01,KL1/.1/freeNat1,observed reward/continuation1,seed41001.
Require all204 fresh tensors and32 initial inference arrays to match the retained
old actual-initial control,11 finite reports and correct total/mask accounting.
Require encoder, recurrent core, prior and decoder updates, own204 frozen tensors
and32 trained restore arrays exact. Qualification is not a useful-learning score.

Use the matching newly packaged native and runner, host-only guard, no NVML,
reset/reload/reboot or automatic retry. Stop real faults and numerical failures.
Retain wheel/native/test executable/source/logs and release the disposable build
cache after qualification. Preserve every original failure and pinned script;
later CPU repairs use new filenames and declarations.

Only after reviewed qualification, separately declare a fresh8,192-update broad
learning check against the retained original H16 RGB100 control, with the exact
same16 fitting/eight development streams and4/2/1/1 plan. The intended difference
is this RGB attention; no position head. Keep original fixed final observed≥90%,
one≥89/109,eight≥28/107 and all physical, event, paddle and brick metrics.
No detector/tolerance/checkpoint/draw substitution. If the intermediate screen
passes, any84-group oracle assessment needs a separate declaration.

Full precise physics,natural event calibration,persistent removed-brick memory,
late/tunnel/second-wall/above-top coverage,untouched families andfresh confirmation
remain required. No implementation or fitting result completes the goal.
Actor/critic,CDP comparison andfive-game work remain parked;prior jobs stay closed.
