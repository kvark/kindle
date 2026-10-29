# Optional native pixel-centroid supervision

The completed fixed linear/nonlinear readouts leave substantial Breakout ball
position error,while the RGB64 pixels retain precise detections. More replay
fitting lowers imitation error without improving frozen control;the earlier
higher encoder rate also harms Breakout. Select an optional native auxiliary
objective to test a direct positional learning signal. This is a new training
intervention motivated by those diagnostics,not a proven causal fix.

Keep centered CDP,learned CNN,categorical RSSM and imagined actor/critic. From
each current replay RGB image,compute a detached color-excess centroid inside
an explicit normalized rectangular region. For the future Breakout check,use
red excess max(R−1.5*max(G,B),0),x[8/160,152/160),y[90/210,185/210),and normalized
pixel centers. Images remain native max-pooled frames with exactly one GPU
resize. No CPU image conversion,emulator RAM,position input to the actor,new
teacher action rule or live inference side channel is added.

The auxiliary head is one hidden layer with the model's usual units and two
linear coordinate outputs from the posterior feature. Targets and presence
weights are stopped;only the prediction path updates its head and CNN/RSSM.
Coordinates are full-image normalized pixel centers minus.5. Use the sum of
two squared errors,masked when color mass is at most1e-6,and average over all
replay rows,including absent rows with zero contribution. Report raw loss and
valid fraction separately. The scale is explicit. No actor/critic parameters
are trained by this objective. Existing inference graphs omit the unused head.

Represent the option as `color_centroid_supervision` with scale,foreground
channel,background multiplier and normalized x0/y0/x1/y1 region. Default is off
and omitted from legacy serialization. Validate finite positive coefficients,
RGB observations,channel bounds and a nonempty in-bounds pixel-center region.
Keep old checkpoint/config behavior. The feature is available only in a
separately packaged runtime after checks;do not mix earlier experiment overlays.

First require CPU config/legacy tests,graph target detachment and CNN/posterior
gradient connectivity,absence from inference,and unchanged default graph
identity. Then require guarded native F64 target/loss/gradient checks including
blank,off-region,mixed-color and fractional/multiple mass cases. Verify zero
target gradient and exact zero prediction gradient when absent. A separately
declared bounded real-learning/frozen check must preserve shared initial tensors,
finite metrics,changed CNN/reward/actor/value parameters,replay alignment,
invariant frozen tensors and host guards. Record added parameters and cost.

No full learner,budget extension or automatic promotion is selected here. A
future task comparison must start fresh,retain fixed final/actual-initial
unaided natural-episode evaluations and the original two-wall gate,with all
configuration/seed/runtime/source/exposure changes disclosed. These pixel-derived
targets add an explicit game-specific color/region prior. Their supervised loss
and detection precision are not proof of unaided task competence.

## CPU implementation checks

The optional native target/loss,training head,raw loss/valid-fraction reporting,
Python capability flag,`--breakout-centroid-scale` CLI and accounting checks are
implemented. The CLI permits fresh Breakout CDP with native frames,requires
explicit native support,and rejects restore-time overrides. Existing default
and inference graphs retain their structure;the auxiliary head adds five tensors.

CPU checks pass139 Kindle and5 kindle-gym Rust tests,683 Python checks with
zero native-construction attempts,and strict workspace Clippy. The Python
prepackaging suite uses the retained native only for default configuration;
all constructors are replaced by rejecting stubs. Ten native API construction
cases are excluded from that count. Actual packaged-native validation remains
separate. The first Rust compilation fails on an inferred integer type in the
ignored GPU-test fixture;explicit usize indices repair it. Original and corrected
build logs remain under`/x/Temp/kindle-color-centroid-20261029`;no GPU work repeats.
Native numerical checks,packaging and a real bounded learner remain pending.

## Original numerical check stops at test readback

The first guarded release test executes once and fails in.72s with a CPU
`read_buffer: F32 view exceeds buffer capacity` assertion. The test mistakenly
uses `Session::read_output(length)` as an output-index API. No target/loss/
derivative comparison completes;no numerical pass or GPU fault is inferred.
The original75-pin declaration,binary,source snapshot,logs and guard failure
remain under`gradients`. No learner or checkpoint is created.

Select a corrected validation using `read_output_by_index` with explicit target,
mask and scalar buffer lengths. Only the test body changes;the production target
and loss source must remain byte-identical. Use a separate declaration,binary
and evidence directory with the same120-second cap and all original numerical
assertions. This is a second execution of the isolated derivative fixture after
a diagnosed validator defect,not a passing closure of the first check. No
training is repeated and no full learner is selected.

The second fixture also stops in.72s,after target,mask,F64 loss and derivative
assertions pass,because its final valid-fraction check requires exact floating
equality:the F32 reduction returns.49999997 instead of.5. Its source,binary,
declaration and failure remain in`gradients-corrected`. Select a third isolated
fixture with a one-F32-epsilon tolerance for this mean only. Exact per-row masks
and absent gradients,and all earlier loss/derivative tolerances,remain unchanged.
The production target/loss source remains byte-identical;no training has run.

## Numerical pass and bounded learning selection

The third isolated fixture passes all F64 target/loss/derivative checks,exact
absent-row zero gradients,detached targets,unchanged prediction parameters and
host guards. All83 declaration pins remain unchanged. The earlier two failed
fixture executions remain recorded;zero learner stages have run or repeated.
Review`gradients-tolerant/review.json` SHA256
`eb0c4bef75bf820e895b32f1a16399faf8f732cdad9997a229b548a28ca9f136`.

Native2974160 is packaged at`/x/Temp/kindle-color-centroid-20261029/runtime`,
SHA256`e7ba0d4b208fa15be5790615034fb36ab727e4bd02b041114775b42b8e888e90`.
The exact packaged Python suite passes683 checks with ten native-construction
cases deselected and zero native-construction attempts. Earlier144 Rust CPU
tests and strict workspace/binding Clippy remain recorded in the build receipt.

Select one fresh8192-action/489-update check followed by1024 frozen greedy
actions. Keep retained12M64/Auto/N8/B16/T16/H15/R16/full18/expert100/world-gradient/
corrective-label recipe,seed1009/evaluation4000001009,20% episode overrides,
cap1920,all action weights1 and original rates. Add centroid scale100,selected
before observing learner metrics. Its one-hidden-layer256-unit head adds five
tensors/656386 parameters,for16990739 trainable parameters. Require actual
checkpoint/schema verification,357 identical shared initial tensors,identical
pre-update semantic traces,362 unchanged frozen tensors,finite and nonzero
auxiliary metrics,changed head/CNN/reward/actor/value weights and complete guided
pixel/action/label replay. The target and prediction use the same replay row;
this adds no outgoing-action label alignment. Keep the native one-resize path.

Use existing host guards and240s training/60s frozen/180s CPU audit/480s service
caps. Measure early cost without treating it as a full-run time guarantee.
This separately selected check does not select a full learner,repeat earlier
training or permit reuse of its experience. No natural-episode competence is
expected from the short frozen budget. The native and runner versions differ
from retained447fe50 and must remain explicit even if initial checks match.
