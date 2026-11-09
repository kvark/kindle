# Bounded reward support: stable learned scalars, unfinished world model

The goal **prove the world model training on Breakout** remains unproven.
The [bounded-support learner](../experiments/2026-11-09-breakout-denoising-reward32.md)
removes the enormous scalar-tail error on the reused fitting data and improves
from its own initial predictions. All78 denoiser weights/moments and all final
RGB arrays remain bit-exact to the retained control. Thus this is an isolated
reward-representation result; it does not improve the failed physics gates.

| Event input task | Rows | Old trained reward RMSE | New own-initial RMSE | New trained RMSE |
|---|---|---|---|---|
|ordinary|512|65.6043668|0.417886046|3.17931973e-05|
|forecast|864|1592.51626|0.408339442|0.0314884793|
|Actual-image classifier|224|1803.23116|Not queried|0.0578802619|

The smaller support is already a strong structural prior before learning; the
own-initial comparison prevents calling that prior alone a training gain. In
the one exported causal positive case,arrival5/lane1,true reward1, the four
draws predict about.327–.332 initially and.990713–.990715 after learning. Positive
probability rises about.502–.503→.99294. These four draws share one training
example, not four independent confirmations. Ordinary windows contain only zeros.

Recognition is still incomplete: the actual-image classifier predicts.267701
with positive probability.270793 for the other reward1 example,arrival4/lane4.
The second actual positive reaches.990707/.992934. The actual-image task supplies
the true next image and is privileged, not a world forecast. There are no
terminal examples; natural event frequency/calibration remain untested. A
supplementary exact key over five RGB frames/four consumed action IDs finds
224 distinct inputs and no conflicting labels; that does not rule out learned
feature aliasing, partial observability or inadequate optimization.

Training now uses255 symmetric symlog-spaced bins between exactly−32 and+32,
including zero, consistently for two-hot targets and categorical means. Input
rewards outside the configured support reject before an update; no silent
clipping or posthoc restricted-mass decoder. All fitting labels0/1 fit the bound.
The native logits/support reconstruct every decoded value. The
[earlier diagnosis](2026-11-09-breakout-denoising-event-logits.md) showed tiny
probability on enormous values dominated error, including real-image inputs;
its diagnostic central contribution is not used as this model's prediction.

Only reward_limit=32 changes the learning recipe. Optional defaultNone preserves
the exact original support/decoder and checkpointv1–v6; explicit bounds requirev7.
This is a game-specific prior, not a guarantee that future data lie within32;
later datasets must pass the range guard. No new parameters, graph nodes, loss
expression or backend primitive. Pixel/state inputs, sigma_data.05, log-noise
mean−1.2,history scale.5, actual histories,RGB64/trainB32/inferB8,Auto,rate.001,
context-motion100,residual/half-max modes,three Euler evaluations,seed41001,
target/pixel-normal order and2,048 updates/65,536 targets stay unchanged.
Native Meganeura+Blade perform learning/inference; Python prepares/analyzes only.

Physics remains35/38/34/33 of110 ordinary matches and140/149/140/140 of198
actual-history one-step matches;0/24 individual fitting gates pass. This is
proved by78 exact final denoiser tensors and all cached RGB equality, as well
as the unchanged physical assessment. Missing/unsupported conventions and all
ball/paddle/brick/event regressions stay intact. No best-age or threshold change.

| Verification | Result |
|---|---|
| Source/native |8dbf9ab5 /60b9b67244d740fcc39c7b26b42b0a42bd1e9a77b6a1e01f1e4c64b963d97e29 |
| CPU |25 focused Rust,50 packaged config/restore,four runner checks;strict release test-target Clippy |
| DefaultNone v5 fixture |135 initial/frozen/first-update tensors,20 full initial arrays,six losses exact before learner launch |
| New own initial |135 tensors/16 non-reward output arrays exact;own new scalar forecasts retained |
| First main update |45 changed weights/135 finite tensors,six valid losses,exact image loss and78 denoiser tensors before update2 |
| Final pixel state |78 denoiser tensors and all cached ordinary/teacher RGB exact |
| New v7 freeze/restore |135 tensors/20 restored arrays exact |
| Frozen head stage |200 batches/1,600 rows,12 own cached event arrays exact;raw logits/support retained |
| Learn-call/loop intervals |161.420s/169.299s |
| Declared/physical CPU inputs |140/125 pins |

The default fixture's one qualification update is separate from the main budget.
Call timing includes packing/native CPU/GPU/readback; loop timing includes
periodic exports, not initial/final assessment. Shared host, no isolated speed
claim. All ages0/509/1021/1533/2048 and2,048 finite loss rows remain. Teacher
exports retain already computed event outputs/labels without another query.
The fixed post-restore stage additionally captures the two privileged positives.

Four guards/36 sealed files and two terminal successful services cover default
fixture,learner,restore andhead. Main invocation9705ac45f1cd4d468252c5a0dac7db8c;
default invocation6bda97876a3f4d3fa134378d329b1cb8. The observer's terminal metadata
retains a literal two audited main guards; all three actual main audits ran and
passed. terminal-metadata-clarification.json preserves the original and states
the correct three plus one fixture, with no native/audit repeat. Own
792,664,189-byte apparent build cache is released; native/wheel/source/executable,
all checkpoints and raw evidence remain. Existing native loss derivatives are
reused because the graph is unchanged; support/target math is independently
checked against F64.

Retain reward_limit32 as a Breakout development setting, not a new global default
or proof of event mastery. This is one reused learner seed/data, no fresh or
held-out confirmation. No learner,extension,collection orautomatic successor
selected here. All prior trials/failures/repairs stay closed. Full physical/event
precision,natural calibration,changed-brick memory,late/tunnel/second-wall/above-
top,untouched families andfresh confirmation remain required. The full oracle
lacks above-top endpoints. Actor/critic,CDP comparison andfive-game work stay
parked. No fitting score completes the full objective.

Root `/mnt/data/kindle-breakout-denoising-reward32-20261109` retains
config/plan/source/native/wheel,checkpoints/losses,guards,physical reviews,
event-review.json,head-review.json,all1,600 logit rows,1,376 physical endpoints,
and16 exact fixedlane0/draw0 frames. Scalar output changes are learned under
the bounded representation; source RGB is unchanged.

Closure verifies306 pins in1.097s, four sealed guards/36 files
and two terminal successful services. Head135 tensors remain exact and v7
restore/head metadata retain reward_limit32. Own build cache is absent.
