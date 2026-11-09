# Training with a causal generated latest-history frame

Select one fresh bounded learner to address generated-history instability.
The [target-motion trial](../results/2026-11-09-breakout-target-motion.md) improves
one-step matches but reduces ordinary matches to8–10/110,with77–80 ambiguous
red regions. Additive history noise andgrouped noisy-target denoised histories
also failed. This test uses actual causal native forecasts as training prefixes;
it does not reuse target-denoised outputs orrestart a completed learner.
Root `/mnt/data/kindle-breakout-causal-prefix-20261109`.

Reuse the exact qualified reward32 native60b9b67244d7/source8dbf9ab5 andmatching
runner from `/mnt/data/kindle-breakout-denoising-reward32-20261109`.
No build,native change,new derivative check orlegacy fixture. Same config:
RGB64,B8/trainB32,Auto,rate.001,seed41001,context-motion100,residual/half-max,
sigma_data.05,default noise mean−1.2,reward32,three Euler steps. Target-motion
anddirect-frame options are absent/off.2,048 five-frame calls/updates,65,536
target exposures,224 unique arrivals,588 positive exposures andnative training
Gaussian/target order stay matched. No new emulator actions/frames,object labels,
actor/critic/CDP orbackend primitive.

Calls1..512 use actual histories. Before call513,andthen every64 updates,
freeze the current model age andquery26 one-step forecasts for source arrivals
5..30. For arrivalp,pass actual observations strictly beforep andthe submitted
command atp,seed49001+32*age+p. Store only causal prior RGB as generated history;
observed noisy-frame reconstructions are not substitutes.24 refreshes atages
512..1984 yield624 native forecast queries/4,992 imagined cache frames. They
are extra inference,not new emulator experience orSGD updates. Preserve every
cache,age,source index andseed. Cache queries must not change learner age;
135 tensors before/after the first refresh must match exactly. Native prediction
uses local seeded RNG anddoes not consume the training Gaussian RNG.

For1,426 eligible calls after warmup with target arrivalT>=6,replace latest
prefix RGB (row3) ofcopies1/3,lanes8..15/24..31,withcached forecastT−1.
Earlier three prefix frames stay actual;copies0/2 stay actual. Both16-lane sigma
halves therefore contain eight actual/eight generated-latest histories. Targets,
rewards,terminals andall actions remain unchanged.22,816 exposures use generated
latest frames;short arrivals4/5 remain fully actual because the forecast API
requires five observed prefix frames. Preserve cache float32 RGB without CPU
resize orpixel-roundtrip. Initial training remains clean even before caches exist.

This tests one corrupted history frame with three actual fallback frames,not
training on four fully generated histories. It changes residual anchors,past-
derived importance andevent conditioning together. Final event weights need
not match control. Generated pixels are detached;there is no gradient through
sampling. Stale caches are atmost63 updates old. Full closed-loop behavior and
unobserved-state memory remain separate requirements.

Before GPU,pin all source/native/runner/data/control/scripts andpass seven CPU
runner checks: exact target budget,causal prefixes/commands,seeds/frozen age,
prior-vs-observed output selection,warmup/refresh/short-history boundaries,
clean halves andunchanged target/labels. Reuse retained native qualification.
Main135 initial/first-update tensors,six first losses and20 initial arrays must
match control. At509,before augmentation,135 tensors and20 rollout arrays must
also match exactly. Retain own initial/final ordinary andteacher RGB/events,
all losses,checkpoints andages0/509/1021/1533/2048. Verify135 own frozen tensors,
20 restored arrays andv7 config exactly. Validate every training row's cache age
andthe22,816 augmented exposures during CPU closure.

Keep fixed final2048 gates: all supported ordinary1/4/8/16 balls within1px and
>=90% of110 ordinary endpoints per draw;actual-history one-step>=90% of198.
Missing predictions fail;18 unsupported per task/draw stay unscored. Retain
1,376 endpoints,motion,pixel ambiguity,paddle/brick/events andfixedlane0/draw0
frames. No best-age orthreshold change. Four draws share one learner;one exported
causal positive andno terminals cannot establish reliability/calibration.

One learner,existing host guard,900s train/300s restore. Observe exact systemd
invocation/MainPID/queue;no timeout restart. Require terminal service,two sealed
guards/18 files before CPU review. Stop realfaults/nonfinite data;no reset/reload/
NVML. No build cache created. Preserve failures/separate repairs;no extension,
collector,old-model repeat orautomatic successor. Full physics/events,natural
calibration,changed-brick memory,late/tunnel/second-wall/above-top,untouched
families andfresh confirmation remain required. A fitting pass cannot finish
the full world-model orfive-game goal.
