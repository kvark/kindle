# Breakout: separate the world-loss gradients

The fixed-weight measurement is complete. It measures the existing objectives and one local optimizer sensitivity; it does not prove a physical failure mechanism, qualify a new recipe or complete the world-model goal. No learner or successor is selected.

[Protocol](../experiments/2026-11-08-breakout-world-loss-gradients.md).
Artifacts: `/mnt/data/kindle-breakout-world-loss-gradients-20261108`.

At this reused checkpoint, weighted future-RGB gradients exceed KL gradients on every measured prior/core condition. The native component sum agrees with the joint gradient; loss scaling changes the hypothetical normalized fresh prior gradient substantially. These are local parameter-space measurements, not proof that lowering RGB improves training.

## Scope

Use the retained v13 suffix-event final at8,192 updates, with fusion, posterior main readouts, motion RGB100, KL1/.1, observed/future reward/continuation1 and H16 free RGB100. Residual copying and image feedback are off.80 logical weights/8,871,955 scalar weights; original240-tensor checkpoint includes optimizer moments. Read only three already retained fitting-plan batches at0,2047,8191; four fixed posterior/suffix noise pairs per batch. These are reused fitting conditions, not independent episodes or untouched development.

Select observed RGB, configured KL, observed events, future RGB, future events and the original joint scalar as diagnostic output0. All remaining production inputs/outputs remain. Seventy-five native forward/backward steps, including one exact joint repeat per batch; no configured optimizer, no actual weight update, new collection or repeated frozen physics assessment. This is a new measurement, not a rerun of the earlier predictive-energy noise comparison.

## Raw gradient balance

The table gives medians of norms across12 fixed conditions after applying the actual scalar coefficients. KL and event groups already include their configured coefficients. Norms are descriptive; vectors can cancel, parameter families differ in size, and raw scale does not determine the optimizer update. All per-condition/per-parameter values, pairwise directions and four-draw variation remain in `gradient-review.json`.

| Parameter family | Observed RGB×100 | KL1/.1 | Observed events | Future RGB×100 | Future events |
|---|---:|---:|---:|---:|---:|
| representation.encoder |630.16|1.3931|0.0071815|582.79|0.0052861|
| representation.posterior |547.81|1.6083|0.010076|749.82|0.0062138|
| dynamics.core |315.04|7.5662|0.064909|4543.6|0.085986|
| dynamics.prior |0|15.597|0|834.26|0.011363|
| decoder |18070|0|0|81855|0|
| reward |0|0|0.14245|0|0.13224|
| continuation |0|0|0.040915|0|0.093352|

Across the12 measured conditions, the weighted future-RGB/KL norm ratio ranges22.199–87.212 in the prior and336.248–883.791 in the core. These are single-draw parameter-space norm ratios, not fractions of error or proof that KL should dominate.

## Shared gradient directions

Each cell below shows median and full minimum..maximum cosine over the same12 conditions. A negative value means local opposition in parameter space, not that one objective is globally harmful or responsible for a measured physical error. Zero-gradient pairs are unsupported.

| Family | Observed RGB / future RGB | KL / joint |
|---|---|---|
| representation.encoder |-0.7161 [-0.8164, 0.8554]|0.1960 [0.0603, 0.4788]|
| representation.posterior |-0.3823 [-0.6092, 0.6610]|0.2172 [0.1377, 0.4566]|
| dynamics.core |0.0249 [-0.0157, 0.0619]|0.1420 [0.0582, 0.1856]|
| dynamics.prior |no common gradient|0.0075 [-0.2648, 0.2093]|
| decoder |0.3528 [-0.0327, 0.5041]|no common gradient|
| reward |no common gradient|no common gradient|
| continuation |no common gradient|no common gradient|

## One predeclared local sensitivity

Compare RGB coefficient100 against1 for both observed and future RGB, keeping every other component fixed. Apply the previously qualified F64 whole-parameter AGC/LaProp reference with the same anchor moments and hypothetical next age8,193. This is not an actual optimizer step, a sweep, fresh-training behavior or bitwise native replay (including subnormal handling). The table separates the newly normalized gradient from the momentum update direction. A local change does not prove improved forecasting or identify a useful coefficient.

| Family | Fresh-gradient cosine, median [min,max] | Momentum-update cosine, median [min,max] |
|---|---|---|
| representation.encoder |0.9350 [0.8854, 0.9937]|0.9865 [0.9511, 0.9988]|
| representation.posterior |0.9734 [0.9392, 0.9906]|0.9844 [0.9723, 0.9985]|
| dynamics.core |0.9654 [0.9138, 0.9759]|0.9927 [0.9816, 0.9954]|
| dynamics.prior |0.4707 [0.1893, 0.6122]|0.8877 [0.8567, 0.9146]|
| decoder |0.9960 [0.9854, 1.0000]|0.9992 [0.9954, 1.0000]|
| reward |1.0000 [1.0000, 1.0000]|1.0000 [1.0000, 1.0000]|
| continuation |1.0000 [1.0000, 1.0000]|1.0000 [1.0000, 1.0000]|

## Verification and limits

Three new CPU checks and strict core/test Clippy pass. Original helper-API compile failure/source remains; a separate repair uses the actual deep-clone and owned-path APIs. Only a cfg(test) registration/new test module changes core source; production graph/API, checkpoint formats and Python runtimes remain unchanged. Source `c46fa6e34400a6c3093e0b8c4916183d7ea8fcec`; test executable SHA256 `103452871eb061821c2bc1328a4a554308f00c617181e58476bd3df2e2284bd9`.

Native gradients reconstruct the weighted component sum with maximum normalized residual1.82489e-06, below the predeclared2e-4 bound for every family/condition. All three joint repeats are bitwise exact. All12 before/after snapshots match the anchor's80 logical weights, optimizer age remains0, and the original240-tensor checkpoint stays unchanged. Absent parameter gradients are explicitly represented and checked as zeros (408 checks). All selected inputs have unique names, matched hashes and finite native losses/gradients; production reset, reward, discount and suffix masks are retained.

Native measurement440.689s includes graph/session construction, forward/backward work and buffer exports; CPU review21.279s. One successful guard/nine sealed files, no host recovery/NVML or unfinished child. Native forward/backward work, not learner updates, is measured; Rust test compilation, pinning, CPU figures and closure are separate costs. All75 raw buffers, losses, input hashes, layouts and logical snapshots remain retained.

Four draws and one reused checkpoint provide limited local evidence. The actual straight-through derivatives are biased derivatives of a stochastic surrogate. No gradient norm, cosine, hypothetical update or reproducibility check certifies useful learned physics. This does not close the causal attribution among representation, dynamics, loss fitting, sampling and data coverage. No new learner or coefficient is selected.

All original physical/event gates remain. Precise all-horizon physics, collision/reward/life-loss timing and calibration, persistent removed-brick memory, late-game/tunnel/second-wall coverage, untouched families and fresh learner confirmation remain required. The current oracle has no above-top-brick endpoints. The goal remains active and unproven.

The first CPU figure call failed on a Matplotlib scale keyword; a separately declared CPU repair uses the scale-specific API. Its original script/log/declaration and first runtime metadata remain preserved. Native measurements and gradient analysis are unchanged and never rerun.
