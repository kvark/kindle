# Breakout playfield crop: optional native input experiment

Select optional native crop support and one bounded execution/cost check after
the negative [25M comparison](../results/2026-11-02-cdp-breakout-capacity25m.md).
Keep the retained 12M model and 64×64 output. The candidate rectangle is
**x=0, y=32, width=160, height=168** in the native160×210 frame: full width and
rows32–199. The GPU reads this rectangle during its existing antialiased resize;
Python continues to supply the original pooled frame. There is no extra image
resize, CPU pixel transform, network parameter or teacher change.

The inspected retained greedy video frame contains the roof edge, bricks,
ball and paddle within this rectangle. Removing the top display also removes
score/life information; preserving task-relevant information across all states
is not established by one image. Vertical scale increases from64/210 to64/168
while horizontal scale stays64/160. This tests a different input allocation at
the same tensor size, not a proved resolution bottleneck or promised speedup.

## Preceding source review

The retained447fe50 source does not reveal an expert-label timing mismatch.
The runner computes labels from current frames before acting; vector ingestion
stores the pending label on the arrival frame. `SequenceBatch::expert_targets`
uses frame t+1's label for state t and masks last/reset boundaries. Both the
actor replay features and world expert targets use the same time-major rows.
The GPU handoff copies the first B×(T−1) posterior features from imagination's
time-zero block. Existing alignment/passive-label tests cover these boundaries;
no old GPU test or trajectory is rerun for this source review. It does not prove
the absence of every implementation error or identify the cause of weak play.

The completed scale1 coverage analysis already shows only13 of52 guided
episodes reaching the1920-action cap, followed by1,286 native actions. Most
guided episodes end before release. This does not select longer guidance;
reuse that completed evidence without recounting the trajectories.

## Implementation and qualification

Add optional `observation_crop` to checkpoint configuration, absent by default.
Only learned RGB accepts it. Reject empty/overflowing rectangles, source-frame
overruns and combinations with the fixed native color-centroid supervision.
Fresh CLI `--image-crop X Y WIDTH HEIGHT` requires explicit native support;
restore uses recorded crop geometry and rejects overrides. Record the crop in
the preprocessing description. Default full-frame descriptions stay identical.
Cropped checkpoints use a distinct input-architecture tag, so older native
loaders reject them even if their configuration parser ignores unknown fields.

Use the original upload and resident-buffer paths, preserving row stride and
pixel format. CPU checks cover pixel addresses, bounds, legacy configuration
defaults, serialization and runner rejection/configuration. Build and package
the native extension, run relevant CPU checks and strict Clippy, and retain
source/dependency/compiler/native/test-binary hashes and every failed attempt.
Do not claim GPU qualification from compilation or CPU tests.

Before learner construction, run one separately guarded GPU geometry test:
RGB32/48/64, selected Atari crop, odd subrectangle, full-frame rectangle and
single edge pixel; compare an independent scalar cropped-image reference with
CPU-upload and strided resident RGB/RGBA/BGRA sources. Include unaligned source
offsets, padding and inactive-stream clearing. No model updates or game actions.
Verify the new default path against a completed retained1024-action frozen
capture without executing its old runtime again. Require matching semantic
events and357 invariant checkpoint tensors; any explicit added default fields
must be identified. Pin the exact capture before executing the new native arm.

## Bounded candidate

Only after CPU, geometry and default compatibility pass, select the concrete
fresh8,192-action/489-update command with crop0/32/160/168. Keep12M64/Auto/N8/
B16/T16/fullBPTT/microbatch16/H15/R16, encoder4e-5,behavior8e-5,dynamics4e-4,
replay-value1/imagined bootstrap,expert100/unit weights/world gradients/all-state
corrective labels,20% episode guidance/cap1920,actor-unimix.01,action-effects1,
replay100k,warmup1000,full18/sticky.25/repeat4/noops0,seed1009 and the retained
teacher. No checkpoint/replay/belief/game/RNG experience is transferred.

Keep a120-second training cap and1,024-action/60-second frozen restore,zero
updates and guidance during restore. Require finite reports and tensors,
changed CNN/reward/actor/value,exact configuration/counters,357 initial tensors
matching the same preset's qualified initialization,357 frozen invariant tensors,
full native-input/action/label outcome replay and host guards. Cropping changes
the model inputs; do not require initial action/latent traces to equal full-frame
control. No natural episode or gameplay benefit is required from this short check.

The conservative200k projection is1.1×bounded training seconds×200000/8192;
it must fit30 minutes before considering a full comparison. Resolve and pin
all commands,source inputs,schemas and audit/collector limits before launch.
No full learner,cap increase,retry,crop sweep or automatic successor is selected
here. A full comparison would retain own actual-initial controls,24 natural
episodes per sampled/greedy arm and the original90%-two-wall gate.

Root `/x/Temp/kindle-playfield-crop-20261102`. Check live state before every
GPU allocation and do not overlap jobs. Stop on real faults/nonfinite values;
no driver recovery. Preserve failures and inspect completed stages before any
separate CPU repair. The five-game goal remains unfinished.

The first CPU build at sourcec275c1e fails before running tests: the existing
gradient diagnostic helper still calls the three-argument RGB constructor.
The original declaration,24.924s failed compile receipt/log and exact helper
source remain retained. A separate correction explicitly passes `None` for
that unchanged full-frame fixture. No GPU work or completed test stage repeats;
resume qualification in a separate attempt, reusing only the compiler cache.
