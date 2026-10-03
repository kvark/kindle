# Fixed-target Tiny diagnostic

The direct-policy online screen completes without useful forecasts. It cannot
separate missing information from a poorly fitted predictor or changing targets.
This diagnostic addresses that distinction; it does not replace online learning
evidence or add another representation family. The active feasibility goal stays
open until its evidence supports a scoped conclusion.

## Frozen collection — declared before execution

Use all three final direct-policy Tiny checkpoints, seeds1009/2017/3019, without
selection by score. Restore the exact trained encoder, not the original weights
named by its pretraining provenance. Collect uniform-random Seaquest trajectories
with native pixels, sticky.25, full18 actions, repeat4, no reset noops and no
learning. Training seeds9101/9109/9127, validation10103/10111, test11113/11117/11131;
each has4,096 actions. Thus32,768 diagnostic actions per checkpoint,98,304 total,
additional to all prior interaction budgets. Whole seeds belong to one split.

Retain every transition and terminal target, plus separate reset arrivals;
future targets may not cross an episode reset. The 16-arrival encoder chunk
boundary is not an episode boundary. Save production7x7x64 features, actions,
rewards, boundaries, chunk phase and RAM position labels. RAM is diagnostic-only
and never enters the actor or its replay. Source maps/limitations are retained.
No downscale/upscale preprocessing or teacher substitution.

An initial128-action/trajectory plumbing test on checkpoint1009 adds1,024
diagnostic interactions, excluded from fitting. It must preserve every checkpoint
tensor's bytes (keys, shapes, dtypes, values) and learner counter before full collection. Full collection has a
15-minute deadline per checkpoint and no automatic retry. Expected retained
corpus size~1.2GiB, plus frozen-after checkpoint checks; current disk headroom
is7.8GiB. No unrelated history is deleted to make room.

## Native fitting — declared design, not launched yet

Reuse the native GPU `RegressionProbe`, not a CPU learner. Its existing two-layer
128-hidden MLP, batch64, Adam1e-3 and training-only F64 normalization remain.
Support full3,136-component targets by widening the diagnostic output limit;
qualify that dimension with independent value/raw-gradient checks before use.
Do not project away latent components to fit the existing64-output limit.

For each frozen encoder, fit one current-state readout (positions/reward event/
terminal) and two action-conditioned residual predictors, horizons1 and15.
Predict all latent components. Inputs are current/previous latent features,
the proposed action sequence and encoder phase, never future images or labels.
Prediction origins are every fourth action, fixed independently of labels;
current-state readouts retain every observed action outcome. Use2,048 updates
per head, seed20261003, validation-only checkpoint selection every128 updates.
Three encoders × three heads is9 diagnostic fits, not9 RL runs. Fitting is
conditional on successful collection and wide-output qualification.
The serial fitting queue has a30-minute deadline per encoder and stops on the
first failure without retry. The wide-output numerical test has a120-second
deadline. GPU fitting never overlaps collection; CPU preparation may overlap
frozen collection, whose wall time is not a matched throughput measurement.

Compare held-out prediction against persistence and unrelated-action controls,
report raw and training-scale-normalized errors by trajectory and chunk-boundary
crossing. Decode real/predicted/persisted future latents with the frozen state
readout to expose object/reward information rather than only aggregate feature
MSE. Report event counts and training fit separately from held-out fit; use whole
test trajectories for uncertainty, not independent correlated-frame resampling.
The fixed head seed does not measure probe-initialization variability. RAM
decodability is a diagnostic, not proof of control sufficiency or bullet detail.

A successful fixed-target fit would rule out a blanket claim that these latents
cannot support prediction; it would not prove the online RSSM/objective learns
them efficiently. Failed heads require distinguishing optimization fit from
information loss. No larger unchanged online queue or Phase3 starts automatically.

Ordinary host guards, expected RTX5080/driver580.178.04 and ≥2GiB sampled Vulkan
estimated headroom remain mandatory. Every new allocation warning stops work.
No NVML polling, driver recovery or concurrent GPU job. Latest Meganeura main
was rechecked October3 at21:40:6268ea5, unchanged; current13b19d33 includes dV fix.

## Retained collection smoke failures

The first smoke stopped after128 actions: a new collection trajectory reset the
actor without closing the previous one. Artificial recording cutoffs are now
passed to the actor and saved separately from real ALE terminal/truncation labels.
The second collected all1,024 actions with zero updates, then rejected unequal
safetensors file hashes. A CPU audit confirms every world/behavior/slow-value
tensor is bitwise unchanged; serialization/header identity was the wrong check.
The corrected check compares all tensor bytes and retains before/after file
hashes as provenance. Neither failure recorded a GPU/kernel fault or recovery.
Both outputs remain under `runs/fixed-tiny-latents-20261003.5uHqSE` and neither
is fitting data. A fresh corrected1,024-action smoke is declared separately;
these extra interactions are not silently folded into the fitting corpus.
The corrected smoke passes in18.885s: all696 saved tensors unchanged, zero
updates, eight complete trajectories, ≥8.736GiB sampled estimated headroom.
Total excluded smoke interactions are2,176 (128+1,024+1,024).

CPU dimension tests, strict workspace Clippy and1,042 Python tests pass. The
first CPU-build service exited before compilation because its PATH lacked Cargo;
the explicit build PATH succeeds. This is not a native/GPU failure.
