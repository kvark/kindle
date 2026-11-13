# Smaller training batches at the same target budget

The retained reward32 denoiser still misses ball formation on fitting examples.
Its B32 batches repeat eight source contexts four times, averaging their noisy
gradients into one update. Test B8 with four consecutive micro-updates per old
B32 call:8,192 updates and65,536 targets, rather than2,048 updates. This is a new
bounded training-geometry comparison, not an unchanged extension or automatic
successor. The frozen noisy-input attenuation has closed negatively.
Root `/mnt/data/kindle-breakout-denoising-batch8-20261109`.

Fresh constructor, seed41001, exact qualified native60b9b672/source8dbf9ab5 and
matching reward32 runner. No rebuild. Only config train_batch_size changes32→8;
inferenceB8,RGB64,learning rate.001,sigma_data.05,half-max noise,past-motion100,
residual formulation,reward support32 andthree Euler stages remain unchanged.
No training from any restored or manually perturbed checkpoint. Four actual
history frames for every T5 call; all eight lanes retained, no tiled duplicates.

Repeat each old source offset four consecutive times. CPU checks require exact
per-source exposure counts against the B32 plan:224 unique source arrivals,
65,536 targets,588 positive exposures and32,768 forced-max-noise targets. Half
of each source's appearances remain forced-max. Native RNG interleaving changes
with batch geometry; do not claim paired noise or isolate optimizer update count
from averaging, variance, Adam age and event-head learning. Match targets, not
wall time or FLOPs. Count actual cost and all regressions.

Require135 initial tensors and20 initial model arrays exact to the cached control,
finite six losses with one update/eight targets per call,all45 parameters changed,
and135 frozen tensors/20 restored arrays exact. No first/final parameter parity
with B32 is expected. Retain all8,192 loss rows and checkpoints at2036/4084/6132/
8192, matching control target-exposure milestones509/1021/1533/2048. Evaluate
all four normal16-step rollouts and108 actual-history one-step queries; retain
RGB,event,paddle,brick,pixel,false-detection and motion metrics,not just ball score.

Keep the fixed strict detector and24 gates:ordinary supported1/4/8/16 exact,
all16 >=90% of110 per draw,actual-history one-step >=90% of198. Missing ball
fails; unsupported endpoints unscored. Four reused draws share one learner and
selected clips. Improvement remains fitting evidence only, not natural event
calibration, generalization, removed-brick memory or game mastery. No target
coordinates/object labels/emulator state enter training. No game actions/frames.

Pin data, source snapshot, native, runner and scripts before GPU. Treat pinned
scripts as immutable; new CPU analysis uses new filenames/declarations. One
service,two serial guarded stages,900s train/300s restore caps,existing AMD guard,
no NVML or reset. Stop numerical/real host faults. No automatic learner,extension,
collector or successor. Full physics/events,natural calibration,changed-brick
memory,late/tunnel/second-wall/above-top,untouched families andfresh confirmation
remain required. Actor/critic,CDP comparison andfive-game work stayparked.

Completed: [mixed fitting/rollout result](../results/2026-11-09-breakout-denoising-batch8.md). Both guarded stages pass;215 closure pins. Retain B32 for ordinary rollouts;B8 retained as fitting evidence. No successor selected.
