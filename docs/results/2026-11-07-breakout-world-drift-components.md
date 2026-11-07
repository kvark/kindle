# Retained recurrent mismatch is mostly shared across sampled futures

The [CPU-only decomposition](../experiments/2026-11-07-breakout-world-drift-components.md)
is closed. In the two retained state-query checkpoints, 82–86% of empirical
recurrent mismatch at8/16 is in the mean difference across four draws, rather
than dispersion around that mean. A finite-draw correction still leaves75–82%.
This favors examining shared learned transitions/representation before a sampling
suppression experiment. It does not identify a causal physical-error fraction,
prove a particular correction, or describe the newest recurrent-energy checkpoint.

## Fixed analysis

Reuse all saved [state-substitution traces](2026-11-07-breakout-world-state-swap.md):
coverage/latent-overshooting checkpoints,32 natural plus10 terminal-aligned
development origins,four fixed draws,all32 horizons. No new model query,update,
emulator action/frame or GPU construction. Original masks exclude padding,
reset crossings and unavailable teacher states; terminal arrivals remain valid.

At a fixed origin/horizon, delta_d=h_prior,d-h_teacher,d. The exact empirical
identity is mean(delta²)=mean_delta²+mean((delta-mean_delta)²),averaging components
and draws. The paired covariance decomposition of centered variance also agrees.
The unbiased estimate of squared population mean delta subtracts centered variance
divided by3 for four draws. Negative finite-sample estimates are retained.

|Model / horizon|Origins|Empirical mean share|Corrected mean share|Prior / teacher entropy,nats|
|---|---:|---:|---:|---:|
|Coverage8|32|86.31%|81.75%|47.12 /30.52|
|Coverage16|32|84.91%|79.88%|47.30 /30.28|
|Latent overshooting8|32|84.46%|79.28%|51.38 /30.15|
|Latent overshooting16|32|81.59%|75.46%|51.81 /29.36|

All four empirical fractions exceed the predeclared2/3 triage threshold.
At horizon1,the paired recurrent states are identical,as required by the shared
observed prefix. Natural32 has30 supported origins; ending32 has none and remains
untested. Every supported horizon/origin and all categorical statistics are saved.

The categorical distributions remain broad: across-draw slot disagreement at8 is
69.78%/50.85% prior/teacher for coverage and72.35%/51.63% for latent overshooting.
High disagreement is not itself nuisance noise: the learned code,partial
observability and stochastic teacher history also contribute. Entropy uses the
intended mixture in F64,not bitwise native probability exports. Internal units
and categorical codes are not invariant cross-model quality measures.

Four reused draws are limited. Prior/teacher branches share random numbers;
teacher states are stochastic and use future images. The correction's estimator
property does not turn latent-state mismatch into physical forecast error.
Neither the empirical nor corrected share proves that lowering sampling entropy
cannot help. These observations favor the next investigation,not world competence.

## Evidence and next question

Root `/mnt/data/kindle-breakout-world-drift-components-20261107`,source base
`5225e61c7de12832ff65b9f31b8c402664dd5177`. Exact new script/protocol/declaration
and original source/data receipts are retained.106 input/four output hashes,
the independent algebra fixture,6,144 per-origin algebra identities and2,360
supported origin-time rows pass. Analysis takes2.139s; no native work repeats.

Preserve the first declaration failure: an old data receipt points at the live
corpus runner changed byab68b0f7. The corrected declaration verifies that original
member against its old expected hash inside the previously pinned source archive,
retains those bytes and records the explicit substitution. It never changes the
historical receipt/hash or claims the current runner is the old one. No analysis
result was produced by the failed declaration. `closure-review.json` preserves
original failure versus corrected CPU completion; original native-query failures
also remain failed. All state/image/data artifacts are unchanged.

Code review identifies a concrete missing combination. Motion attention fixes
observed ball reconstruction,but the earlier direct future-image objectives use
uniform MSE/cross-entropy; the motion objective explicitly rejects H>0. The
[new motion-weighted rollout protocol](../experiments/2026-11-07-breakout-world-motion-rollout.md)
will test direct causal future RGB supervision with the already useful generic
motion weighting. This is not established by the variance statistic. Pointwise
future-image pressure previously failed and can blur uncertain futures; external
trajectory,distribution and event-timing gates must decide. No unchanged extension,
sampling sweep,CDP comparison or actor/critic is selected. Full goal remains open.
