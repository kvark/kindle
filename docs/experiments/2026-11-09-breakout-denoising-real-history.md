# Breakout denoising with actual training history

The [past-motion importance trial](../results/2026-11-09-breakout-denoising-context.md)
fails even the eight-clip fitting task. Test whether mixed generated training
history impedes fitting before balls can be rendered. This is one fresh bounded
learner, not an extension or a full-goal proof. Root
`/mnt/data/kindle-breakout-denoising-real-history-20261109`.

Reuse native source879ab5275726b8528c7eed8699ef20a2b594ce7d and qualified
binary9657af725307ac6ecf5f85456b5cfcbc660940c337cf5133d4107704e83a80de
from `/mnt/data/kindle-breakout-denoising-context-20261109`. No rebuild or
native fixture repeat. Require its completed weighted-loss value/derivative
and host-guard reviews before constructing the learner; pin those receipts,
matching runtime/runner/wheel bytes and current source archive.

Change only training-history delivery. Expand each former (offset,length)
block to (offset+i,5), i=0..length-5. Each five-arrival call supplies four actual
past frames and one next target; it performs one update. The original target
order is exact, with2,048 calls/updates,65,536 targets,224 unique arrivals and
588 positive source-prefix targets. Reuse the closed exposure counts; do not
recount or collect episodes. Native training RNG and optimizer persist across
calls, and prefix loading/save/predict do not consume that RNG. Thus splitting
preserves the noise stream and update order. Both image and event heads now
see real past context on every update; neither receives future controls/images
at prediction. There is no gradient unroll through history in either mode.

Keep seed41001,RGB64/trainB32/inferB8,Auto,LR1e-4,LaProp epsilon1e-8,AGC.3,
context-motion scale100/radius4/mean1,45 weights/1,787,843 scalars/135 tensors,
v2 format and three Euler evaluations. Eight32-arrival source clips and four
noisy copies per lane are unchanged. No object/RAM labels or emulator action/
frame; no actor/critic. Four CPU checks cover ordered-plan equality, lane/action
alignment and future-image exclusion for ordinary and privileged forecasts.

Before update2, require135 exact actual-initial tensors and20 per-draw initial
arrays equal to the retained control, six finite first reports and45 changed
weights. Compare all four first loss components within1e-6 and all135 first
trained tensors within rtol1e-5/atol1e-7; report the exact-tensor count separately.
The first actual update counts toward2,048. Export ages0/509/1021/1533/2048,
ordinary16-step and privileged actual-history one-step at5..31, draws
83001/83003/83009/83023. Target4 is excluded by the observed-denoising API.
Each draw must match every supported1/4/8/16 ball within one pixel and at least
90% of all110 ordinary endpoints; the privileged gate separately requires90%
of198. Missing balls fail;18 unsupported endpoints per task/draw stay explicit.
Observed sigma.5 denoising is distinct from RSSM reconstruction. Retain all
physical/event endpoints, regressions, fixed lane0/draw0 clip and135 exact
frozen tensors. Separate restore must reproduce all20 per-draw model arrays.
Only required cached actual-initial RGB is decoded; no old final RGB/model query.

Use existing host guard,900s train/300s restore caps, stop on real faults or
nonfinite data; no reset/reload/NVML. Inspect active jobs first; do not disturb
other users. Host currently has a separate user's trainer, while target AMD
VRAM usage is about0.6GB/20GB; record contention as a timing limitation.
Declare successfully before launching; terminal success and sealed guard audits
must precede CPU analysis. No automatic extension, broad learner or successor.

Passing this fitting task would not prove the world model. Preserve full
physics/event/calibration/brick-memory gates, late/tunnel/second-wall/above-top
coverage, untouched families and fresh confirmation. Current oracle has no
above-top endpoint. Read the result before selecting any further allocation.

The [completed result](../results/2026-11-09-breakout-denoising-real-history.md) fails:0/110 ordinary and0/198 privileged one-step on each draw. All135 first trained tensors match the control exactly. The allocation is closed;no unchanged extension is selected.
