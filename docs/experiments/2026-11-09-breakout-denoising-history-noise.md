# Corrupted image history for Breakout rollouts

Select one fresh, bounded test of history robustness. The matched-noise
[diagnosis](../results/2026-11-09-breakout-denoising-paired-history.md)
loses38–46 actual-history successes when replacing real past frames with generated
ones. Cached reward32 static-pixel RMS grows.004957/.011580/.018333/.032345 at
horizons1/4/8/16 over31,255 static pixels. That motivates a conditioning-noise
experiment; it does not establish the cause of object-motion failures.
Root `/mnt/data/kindle-breakout-denoising-history-noise-20261109`.

Reuse the unchanged qualified reward32 native60b9b67244d7 and its exact runner
from `/mnt/data/kindle-breakout-denoising-reward32-20261109`. No build, new native
implementation, backend qualification or legacy fixture is needed. Keep config
and parameter initialization exact: RGB64,B8/trainB32,Auto,seed41001,rate.001,
context-motion100,residual/half-max modes,sigma_data.05,log-noise mean−1.2,
reward_limit32,three Euler steps. Same224 unique target arrivals,2,048 calls/
updates,65,536 targets,588 positive target exposures and native RNG draw order.
No new emulator action/frame, object label, actor, critic or CDP change.

Call1 stays entirely clean: require135 first-update tensors andsix losses exact
to reward32 before update2. Calls2..2048 corrupt only four past RGB frames of
copies1/3 (lanes8..15/24..31);copies0/2 stay clean. Sample one independent spatial
float32 Gaussian texture per original lane with NumPy SeedSequence([48001,call]),
std.02 in centered RGB units. Share it across the four past frames and both
corrupted copies; clip to[−.5,.5]. Each16-lane sigma half contains eight clean and
eight corrupted examples;32,752 targets get corrupted prefixes. Current target,
reward,terminal and actions remain byte-identical. Target pixels cannot affect
prefix corruption. CPU checks must prove reproducibility, clean first call,
unchanged targets/labels, balanced halves and preserved static history.

This also changes residual anchors, past-derived motion importance and event
conditioning. These are coupled effects of corrupted history, not isolated
mechanisms. Correlated additive textures are an approximation to rollout error;
they do not directly simulate missing/misplaced balls, blur or brick forgetting.
Keep original fitting gates and original full-world requirements unchanged.

Before GPU declare exact source/native/runner/data/control/script hashes and
finish CPU checks. Require135 initial tensors and20 initial output arrays exact,
45 parameters changed andfinite losses,save all learning rows and checkpoints at
0/509/1021/1533/2048. Final weights/RGB may differ. Save own initial/final ordinary
and actual-history one-step RGB/reward/positive/continuation outputs; verify135
frozen tensors and20 restored arrays exactly. No privileged head-only stage.

Evaluate fixed final2048 (no best-age choice): every supported1/4/8/16 ordinary
ball within one pixel and>=90% of110 ordinary endpoints per draw; actual-history
one-step>=90% of198 per draw. Missing predictions fail,18 unsupported per task/
draw stay explicit. Retain1,376 endpoints,ball/paddle/brick/pixel/event regressions,
false balls,motion projections and fixedlane0/draw0 frames. Compare own initial
and cached reward32; never rerun completed control models. The one causal
positive across four draws is one fitting event, not fresh confirmation or
natural calibration. No terminal examples exist.

One learner with existing host guard,900s train/300s restore caps. Stop real
faults/nonfinite output; no reset/reload/NVML. Observe authoritative systemd
invocation/MainPID/command. Finish train+restore terminal review and two sealed
host audits (18 files) before CPU assessment. No build cache is created. Retain
all failures separately; no automatic successor or extension. Keep reward32
unless this trial supports advancement. Full physics/events/natural calibration,
changed-brick memory,late/tunnel/second-wall/above-top,untouched families andfresh
confirmation remain required; a tiny fitting pass cannot complete the goal.
