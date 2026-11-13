# Bounded learning with generated denoised history

The [matched-noise diagnosis](../results/2026-11-09-breakout-denoising-paired-history.md)
finds real-history77/86/77/77 versus recursive40/41/35/33 correct balls of110.
Select a fresh fixed-budget learner using the same sigma_data.05 model, but
group existing targets into short training sequences. This tests robustness to
imperfect history now that the coefficient change learns one-step motion. The
older generated-history trial used sigma_data.5 and did not learn that motion;
it does not answer this interaction. No old trial is restarted or extended.
Root `/mnt/data/kindle-breakout-denoising-generated-history-20261109`.

Use existing native7871f08b/SHAaf13a03e, unchanged config/model/optimizer. No
build or new backend qualification. Same fresh seed41001,45 parameters/1,787,843
scalars,RGB64/trainB32/inferB8,Auto,rate.001,context-motion100,residual/half-max
modes,sigma_data.05/history scale.5,three-step sampler and2,048 updates/65,536
targets. Replace2,048 five-arrival calls with513 calls: first five,511 eight,
last seven arrivals. Each four-arrival history starts from real images; up to
three subsequent inputs are denoised estimates from earlier noisy targets,
using the existing native learn loop. These are not fully sampled causal
forecasts and still contain noisy target information. No claim of eliminating
all train/inference mismatch is made.

Expand grouped calls back to single target indices on CPU and require identical
target order/exposures,224 unique targets and588 positive prefix exposures to the
control. Training RNG draws retain order because the native loop draws the same
sigma/pixel normals once per SGD update. History, residual anchors, motion
importance and event conditioning now differ after a generated frame; disclose
these coupled changes. No new labels, object/RAM inputs, target-based importance,
pretrained encoder, actions/frames or collection. Both model learning and
inference stay native Meganeura+Blade.

Pin declaration, config/plan/data, source archives, exact native/matching runner
and prior completed evidence before allocation. Fresh135 initial tensors and20
full per-draw initial rollout arrays must exactly match the sigma_data.05 control
before learning. First single-update call counts in budget and must preserve
all45 weight changes, finite135 tensors, six valid losses, event losses and57
event weights/moments before update2. Later event state may change because its
history changes; do not require or claim final event parity. Require all final
tensors finite,135 frozen tensors and20 restored output arrays exact.

Retain ages0/509/1021/1533/2048,all513 averaged-loss rows and their2,048 updates,
draws83001/83003/83009/83023,ordinary16-step and actual-history one-step outputs,
all physical/event regressions,1,376 endpoints and fixedlane0/draw0 frames.
Inherit unchanged fitting gates: every supported1/4/8/16 ordinary ball within
one pixel and>=90% of110 endpoints per draw; actual-history one-step>=90% of198.
Missing predictions fail;18 unsupported per task/draw remain explicit. Keep
own-initial/copy controls and sigma_data.05 actual-history-training scores.
Learning can improve or regress; no best-age substitution or seed reliability.

Use existing AMD host guard with900s learner/300s restore caps, one learner,
stop on real faults/nonfinite data, no reset/reload/NVML. Successful terminal
services and sealed guards precede CPU review. No extension, collector, broad
run or automatic successor. Preserve failures and separately declared repairs.
Even a fitting pass leaves original full physics/events/natural calibration,
changed-brick memory, late/tunnel/second-wall/above-top, untouched families and
fresh confirmation required. Actor/critic, CDP comparison and five-game work
remain parked.

Completed: the [result](../results/2026-11-09-breakout-denoising-generated-history.md)
regresses to13/9/10/10 ordinary matches of110 and59/60/55/59 one-step matches of198;
all0/24 fitting gates. Retain the actual-history-training sigma_data.05 control.
All native stages/restore/frozen checks pass and the service is terminal.
Original CPU plan-schema and observer audit-path failures are preserved, with
separately declared CPU repairs. All149 closure pins pass; no model work repeats.
No extension, collector or successor selected. The full goal remains unproven.
