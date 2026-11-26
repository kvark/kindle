# Bounded reward support, isolated from pixel learning

The [frozen logit diagnosis](../results/2026-11-09-breakout-denoising-event-logits.md)
reconstructs native reward predictions and identifies enormous scalar-support
tails as the dominant error, even on actual images. Select one fresh matched-
budget learner with reward support bounded to±32. This is a new representation
and training test, not a posthoc clipped decoder or an automatic successor.
Root `/mnt/data/kindle-breakout-denoising-reward32-20261109`.

Add optional reward_limit, defaultNone/omitted. None retains the exact original
TwoHotBins constructor/encoding/decoding and checkpointv1–v6. A finite limit in
[.01,1e6] requiresv7. Construct255 symmetric symlog-spaced scalar values between
exactly−limit and+limit, including exact zero. Use them consistently for two-hot
training labels and decoded categorical means. Reject input rewards outside the
configured range before an update; do not silently clip them. All retained
fitting rewards0/1 fit±32. This explicit game-specific prior limits future use:
any later dataset must satisfy the range guard. No claim of whole-game coverage.

No graph/parameter/loss-expression change. Keep sigma_data.05/log-noise mean−1.2,
actual histories,RGB64/trainB32/inferB8,Auto,rate.001,LaProp epsilon1e-8/AGC.3,
context-motion100,residual/half-max modes,three Euler steps,seed41001,exact target/
noise order and2,048 calls/updates/65,536 targets. The new label mapping changes
event-head learning; the pixel objective and inputs stay identical. Require all
78 denoiser weights/moments bit-exact at first/final update to the retained
control. Event-head parity is not expected. Preserve all physical gates; matching
pixel arrays merely reuses the control's failed physics result.

Run25 focused Rust CPU checks,50 packaged config/restore checks,four runner
checks and strict release test-target Clippy. Independently check support nodes,
two-hot mean/decoded roundtrips,range rejection,old headers and identical graph/
parameters. Reuse qualified native loss derivatives, since the graph is unchanged.
Before the main learner, one guarded defaultNone v5 fixture must match135 initial/
frozen/first-update tensors,20 full initial arrays andsix losses exactly to the
cached control. Its one qualification update is separate. Pin all inputs/native/
source/runner before GPU and finish the fixture's terminal guard review before
launching the learner, enforced by wrapper.

New initial135 tensors must match. Initial RGB/positive/continuation arrays
(16 per draw-set) match control; scalar rewards intentionally differ. Retain own
initial reward forecasts so a smaller support alone cannot be called learning.
Before update2 require six finite losses/correct sum,45 changed weights/135 finite
tensors,and78 first denoiser tensors exact. Preserve every loss/checkpoint and
ages0/509/1021/1533/2048. Final78 denoiser tensors and all cached teacher/ordinary
RGB must match. Require135 frozen tensors and20 restored output arrays exact.

Extend teacher exports to retain already computed reward/positive/continuation
outputs and target labels at arrivals5..31,initial/final. This adds no query and
changes no prediction. It includes the positive source event5/lane1; event4/lane4
remains outside this causal API's exports. Compare initial/final ordinary-zero
scalar RMSE and one-step positive/zero reward error,positive probabilities/Brier,
ending metrics and all source/draw/horizon cases. Two positives in training and
one exported positive across four draws do not establish reliability/calibration.
No modal or restricted-mass diagnostic readout counts as a learned result.

After restore, run one fixed frozen head-only stage on the new candidate using
the qualified collector:200 batches/1,600 rows, actual/forecast/ordinary inputs,
12 own cached event arrays exact and135 frozen tensors exact. This captures
both actual-image positive examples while keeping their privileged status
explicit. Retain raw logits/support and all rows. No new RGB generation or
additional learning. This declared assessment has a300s cap; four sealed guards
(36 files), not three, are required across fixture/train/restore/head stages.

Keep original fitting gates: every supported1/4/8/16 ordinary ball within one
pixel and>=90% of110 endpoints per draw,actual-history one-step>=90% of198;
missing predictions fail and18 unsupported per task/draw stay explicit. Retain
1,376 endpoints,all ball/paddle/brick/event regressions and fixedlane0/draw0 frames.
No best-age selection or tolerance change. No new data,emulator actions/frames,
object labels,pretrained encoder,actor/critic change or backend primitive.

Existing AMD host guard,caps300s fixture/900s learner/300s restore,one learner,
Auto,no reset/reload/NVML. Stop on real faults/nonfinite data. Terminal services/
three sealed guards precede CPU review. Release own disposable build cache after
packaging. No extension,broad learner,collection or automatic successor. Preserve
failures and separately declared repairs. Full physics/events/natural calibration,
changed-brick memory,late/tunnel/second-wall/above-top,untouched families and fresh
confirmation remain required. The full objective remains unproven.

Completed: the [result](../results/2026-11-09-breakout-denoising-reward32.md)
reduces ordinary zero-reward RMSE.417886 initial→.0000317932 trained, and the
causal positive prediction about.33→.990714. Retain32 for development, not as a
global default or reliability claim. One privileged positive remains missed;
physics is bit-identical and0/24 gates still fail. All default/isolation/frozen/
restore/head checks pass, four guards/36 files and306 closure pins verified,
both services terminal, own cache released. Original observer terminal count2
is preserved with a CPU metadata clarification tothree actual main guards plus
one fixture. No native/audit repeat or successor selected. Full goal unproven.
