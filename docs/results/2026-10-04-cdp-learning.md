# Small Dreamer-CDP: better online learning at lower cost

October 4, 2026. [Protocol](../experiments/2026-10-04-cdp.md) ·
[Compact data, every episode and curves](2026-10-04-cdp-learning.json) ·
[Numerical qualification and retained failures](2026-10-04-cdp-qualification.md).

**All three paired Seaquest seeds favor CDP.** Its final online score is70.9%
higher, with13.6% less wall time and35.9% less world-training time. This supports
the small CDP candidate on this development screen, not Atari-wide superiority,
frozen competence or the published Crafter/XL result. The declared frozen
state/forecast diagnostics are still running; the architecture decision follows
their review. Learned RGB remains the default for now.

## Learning and cost

Six fresh runs, seeds1009/2017/3019,200,000 actual actions and49,939 updates each.
Size1M/N8/B8/T16/H15/R32, full18 actions, sticky.25, repeat4, no no-ops, aids or
pretraining. Same learned RGB64 CNN/RSSM/behavior recipe; CDP replaces pixel
reconstruction with detached-target cosine prediction and the declared split
learning rates. This tests that complete package, not one isolated component.

| Method | Final online mean [seed-bootstrap 95% CI] | Mean run time | World train / full update |
| --- | ---: | ---: | ---: |
| RGB Dreamer |318.1 [278.0,379.6]|29m23s|12.787 /33.670ms|
| Dreamer-CDP |543.6 [442.4,627.2]|25m22s|8.194 /28.914ms|

Scores are the last50 completed-episode mean per learner, equally weighted
across seeds. Paired CDP-minus-RGB is **+225.5 [62.8,330.4]**. Individual RGB
scores are278.0/379.6/296.8; CDP561.2/442.4/627.2. Three-seed percentile-bootstrap
intervals are coarse; do not turn their positive lower bound into a broad
significance or mastery claim. All1,247 completed episodes and48 unfinished
stream tails remain in the JSON/raw logs. No selected seed or episode was removed.

![Three-seed online learning versus actual actions and wall time](2026-10-04-cdp-learning.svg)

CDP uses86.35% of paired RGB run time [85.85%,86.87%], or about1.16x throughput.
Construction adds5.35s RGB /5.30s CDP, separately from these run times. The smaller
world objective does not remove other costs: imagined behavior still takes
about14.1ms/update, roughly half of CDP's update time. This is a changed-learning
comparison, not a pure unchanged-objective kernel speedup.

CDP has804,785 parameters versus RGB688,004. Its dense latent predictor is larger
in parameter count than this small convolutional decoder, but cheaper to train.
Both paths use one GPU RGB64 resize from native frames; neither is an
RGB64-upscaled LeVJEPA pipeline. CDP is online JEPA-style learning inside Dreamer,
not a pretrained causal-video encoder. No detached visualization decoder is kept.

## Audits and diagnostics

All six guards/seals, independent transition/counter ledgers and finite final
checkpoints pass. Zero learner debt, retries, new allocation warnings or actor
work remains in the learning queue. The automation independently reconstructs
every episode from transitions before producing the curves. Across451,266
Vulkan samples, minimum estimated headroom is9,423,355,904 bytes and maximum
estimated usage7,123,107,840 bytes. These are not physical free/peak VRAM or GPU
utilization measurements. No separate NVML polling or host recovery.

The first excluded frozen-collector smoke stopped at09:53 UTC on the exact known
allocation warning, before any diagnostic result. Its child was reaped; no new
recorded Xid/hang. Kernel source time predates journal receipt; no cause is
assigned. The broad snapshot hit its output limit, but the streaming warning
and sealed evidence remain. The remaining short qualification checks used the
user's existing two-exact-warning/120-second permission. Both pass without a
new warning:384 actions and four32-update readouts each, zero actor updates,
all250 CDP /292 RGB saved tensors unchanged, three matching real frame/transition
traces and exact h1 deterministic-prior alignment. These768 actions are excluded
diagnostic experience, not online training or competence evidence.

The six full frozen jobs use ordinary guards, with no allocation-warning
exception:32,768 random actions/model, fixed trajectory splits and four GPU
readouts/model. Their results and independent saved-array audits are pending.
Current/predicted player coordinates, action controls, persistence, reward
events and latent spread address different questions; none alone proves a
complete or useful game-state model. No bullet/readable-full-state claim is made.

## Reproducibility and limits

Production implementation `bee4e64`, native library
`32353ffb5d4516aa9281e94004f7b7ca2f126c9d29bfd62e33c36f78c44f502a`,
Meganeura13b19d33/Bladee349cddf, RTX5080/580.178.04. Numerical qualification includes
independent F64 cosine derivatives and1,300 CDP /1,524 upstream RGB comparisons;
gradient isolation/common-gradient optimizer parity is not bitwise stochastic
trajectory equivalence. Historical Phase2 RGB/Tiny runs are not reused controls
for this changed backend. The authors' Crafter/XL result and unmeasured speed
claims are not reproduced by this small Seaquest experiment.

Raw local evidence: [experiment directory](../../runs/cdp-evaluation-20261004.olfQrV),
[complete learning summary](../../runs/cdp-evaluation-20261004.olfQrV/learning-summary.json),
[six successful guards](../../runs/cdp-evaluation-20261004.olfQrV/learning-queue/result.json).
Those workspace artifacts are not public downloads. The compact JSON and SVG
above are committed. No new learning campaign, Phase3 or swarms start here.
