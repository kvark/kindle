# Breakout action-feature learning

The fixed fresh comparison is complete. The full world-model goal remains unproven.

| Model | Observed | 1 action | 4 actions | 8 actions | 16 actions | 32 actions |
|---|---:|---:|---:|---:|---:|---:|
|Actual initial|0/112|0/109|1/109|0/107|0/114|1/95|
|Retained one-hot|103/112|102/109|69/109|53/107|22/114|8/95|
|Action features|106/112|101/109|66/109|53/107|25/114|11/95|

Intermediate development gates: 3/3. These are observed >=.90, one-step >=89/109 and eight-step >=28/107. They do not establish mastery.

Native calls 1841.013s; loop 1998.783s. Retained one-hot control: 2203.088s / 2369.265s.

Fresh 16,384 updates, RGB64/1M/B8/T32/H16/full BPTT/Auto/41001; fixed LEFT/RIGHT/FIRE table, RGB100, KL1/.1/freeNat1, mixing.01, event losses1, rates4e-4. Same original two-cycle plan; no new game actions or frames. Development origins and seed are reused. Source/runtime differs from the historical learner; cached default-off initial controls match, without proof of all old/new trained trajectories being identical.

All 16,384 finite 13-report rows, first qualified report, total/mask identities, plan/counts, 240 initial/frozen tensors, 80 changed parameters, 18 restore arrays, 384 dense initial origins and 58 finite forecast exports pass. Detailed per-horizon, paddle/brick/event regressions remain in learning-review.json and raw exports.

This completed learning allocation selected no automatic physical-branch query or successor. A separately reviewed frozen assessment is declared below. Full distributional physics/action responses, natural event calibration, persistent removed-brick memory, late/tunnel/second-wall/above-top coverage, untouched families and fresh confirmation remain required.

Artifacts: `/x/Temp/kindle-world-action-features-learning-20261109`.

## Fitting forecasts and limits

| Model | Observed | 1 action | 4 actions | 8 actions | 16 actions | 32 actions |
|---|---:|---:|---:|---:|---:|---:|
|Retained one-hot|146/163|154/177|107/180|84/176|37/171|20/179|
|Action features|147/163|159/177|114/180|90/176|43/171|24/179|

The final development paddle is within one pixel in 71/128 one-step cases,
versus56/128 for one-hot, but45/126 versus46/126 at eight steps. These are point
metrics on recorded trajectories. They do not establish counterfactual control,
collision laws, event calibration or persistent brick memory.

The separately pinned cached-curve analysis uses all nine checkpoints and both
32-origin natural-curve groups, with exact input/target arrays. Dots are the
first fixed draw; bands span four correlated draws, not confidence intervals or
independent learners. No early checkpoint replaces the declared final. Native
call time is16.435% lower and the measured training/evaluation loop is15.637%
lower in this pair. Source revisions and run conditions differ; no isolated
feature-table or kernel speedup, time-to-threshold guarantee or reliability claim.

The fixed natural-development row0/draw0 image comparison shows recognizable
short forecasts and drifting/diffuse long forecasts. This single row is an
illustration, not a selected success case or the aggregate result. Images and
curves remain outside Git: `curve-images-16384-v1.png` and
`learning-efficiency-v1.png` under the artifact root.

## Execution and next assessment

Scientific selection7da749d7;native source1331b702/SHA15ea18b5 with the exact
runtime-v2/runner documented in the [native qualification](2026-11-09-breakout-world-action-features-native.md).
Two terminal native jobs and two host guards pass,18 sealed files. All223
execution,167 CPU audit,24 CPU-stage and175 closure pins verify. The separately
pinned post-run curve analysis verifies80 inputs; the fixed image has four.
Both construct no model or emulator and make no new optimizer update.

Training invocation `2a0f0114f88a4163b9f5fc9d8e46fc4c`, observer
`fb8170cb74e540dba3f507f6b70299ba`, and automatic CPU finisher
`9f8c030c8ca3492f9ca15d40f38387ca` are inactive/successful. The initial declaration
writer NameError remains with its traceback; the separately named writer repair
succeeds before GPU execution. No GPU stage fails or repeats. The earlier
semantics verifier and strict Clippy failures remain in their own reports.

After reviewing this complete3/3 result and the limitations above, select the
separate [frozen chosen-action assessment](../experiments/2026-11-09-breakout-world-action-features-branches.md).
It queries only the new final checkpoint, with no training, new emulator data,
old-model query, budget extension or acceptance change. The complete Breakout
world-model goal remains active and unproven.

The subsequent frozen assessment is now closed and fails the broader gates:
primary .224881281067 versus one-hot .172365704612. See its linked report and
the [saved-survival diagnostic](2026-11-09-breakout-world-action-features-survival.md).
The three dense passes above remain intermediate evidence only.
