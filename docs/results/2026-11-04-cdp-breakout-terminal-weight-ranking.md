# Terminal weighting changes ranking as well as probability levels

The fixed matched-arrival cohort shows no demonstrated posterior ranking gain
from terminal weight10. Its one-step prior ranking is lower, with a paired lane
interval below zero. The analytic inverse of the training weight reduces false
ending indications but leaves higher intermediate-life error than the control.
It preserves every ranking and cannot repair the prior discrimination change.

This CPU-only check uses the saved [weight10 arrival outputs](2026-11-04-cdp-breakout-terminal-weight-arrivals.md),
following the [fixed protocol](../experiments/2026-11-04-cdp-breakout-terminal-weight-ranking.md).
It adds no model execution, learning, emulator replay or GPU work.

## Ranking actual endings against intermediate life losses

Lower continuation ranks a game ending first. Each AUC uses all24 terminal ×96
intermediate-life pairs. Intervals use5,000 paired resamples of eight whole lanes,
seed20261104, recomputing pooled AUC with lane multiplicities. The2,304 pairs are
not independent observations. Posterior is primary; prior is secondary.

| State | Control AUC | Weight10 AUC | Candidate−control | Paired lane95% interval |
|---|---:|---:|---:|---:|
| Observed arrival posterior |.952691|.946615|−.006076|[−.040365,.028646]|
| One-step predicted arrival prior |.791667|.696615|−.095052|[−.188379,−.013455]|

The posterior interval does not establish equivalence. Its162 changed pair orders
include88 previously correct pairs becoming wrong and74 wrong pairs becoming
correct. The prior changes639 pairs:429 become wrong and210 become correct.
Neither comparison has exact ties. The outputs therefore are not a strict
monotone transformation of the control, even though posterior AUC stays close.
The prior result concerns this reused cohort, not broad learner reliability.

## A fixed inverse reduces the probability shift but leaves the tradeoff

For conditional continuing probability p, gamma=.9969969987869263 and terminal
weight w=10, the conditional optimum of the weighted soft-target BCE is
q=gamma*p/(p+w*(1-p)). Its inverse to the unweighted soft target is
f(q)=gamma*w*q/(gamma+(w-1)*q). No parameter or threshold is fitted.

This relationship assumes an ideal conditional fit; it does not guarantee that
the learned head is calibrated. The expression is applied unchanged on[0,1].
Nine intermediate-life posterior outputs and five prior outputs exceed gamma;
those use the declared algebraic extrapolation. No terminal output exceeds gamma.
All2,304 pair orders in each mode, including ties, remain unchanged after the
inverse. The already fixed below-.5 cutoff is diagnostic, not an acting rule.

| State and outcome | Control MSE | Raw weight10 MSE | Inverse MSE | Below.5: control / raw / inverse |
|---|---:|---:|---:|---|
| Posterior,24 endings |.644558|.054219|.319645|5 /22 /12|
| Posterior,96 intermediate lives |.003178|.099410|.026632|0 /11 /3|
| Prior,24 endings |.911006|.423453|.731082|0 /12 /2|
| Prior,96 intermediate lives |.001530|.153795|.022044|0 /23 /2|

The inverse's posterior terminal MSE remains below the control by.324913,
interval[−.467115,−.182808], while intermediate-life MSE remains above it by.023454,
interval[.007621,.041880]. Prior differences are−.179924
[−.251285,−.098661] at endings and+.020514[.008547,.036105] at intermediate lives.
Relative to raw weight10, the inverse increases terminal error and decreases
intermediate-life error in both modes; all four corresponding intervals exclude
zero on this cohort. Full group means, lane differences and intervals are saved.

## Meaning for the architecture

The targeted terminal loss changes both output levels and class ordering.
Observed posterior states retain strong separation on this diagnostic, while
predicted prior states separate the two outcomes less well. This narrows the
questions for future work to what predicted states preserve and how imagined
behavior uses those predictions. It does not identify whether dynamics, features,
head fitting or another training interaction causes the difference.

The outcome-conditioned cohort contains20% game endings, excludes ordinary
ongoing frames and follows the retained control's trajectories. It cannot measure
population calibration or the candidate's own failure-state distribution.
Models use different native versions and learned states. Posterior outputs are
offline F64 estimates; prior estimates were checked against native capture.
The inverse changes saved diagnostic outputs only, not the learned actor.

Retain the existing recipe's143.667 greedy/70.750 sampled over weight10's
120.542/66.375, all0/24 two-wall. No correction, loss variant, weight sweep,
replacement head, learner or automatic successor is selected. The original
864/90%-two-wall and five-game goal remain unfinished.

## Execution and evidence

Source`db4840bd80a8f23408f7dba7dd90eccd0edf4606` is committed before analysis.
All24 CPU tests pass, including conditional-optimum round trip, endpoints,
monotonicity, invalid inputs, exact AUC cases, lane multiplicities and blocked
native/emulator imports. Preflight verifies1,085 prior pins and44 prior artifacts
in3.837s. The new declaration has1,125 pins, including protocol and analysis code.

One analysis, PID1144889, takes2.060s on one CPU core at nice10 and completes
07:27:01.291657 UTC November4; its completion manifest is written07:27:03.312692.
Independent4.073s CPU closure completes07:27:22.450704, verifying16 final artifacts,
all1,125 pins,13,824 pair orders,240 transformed outputs, group errors, bootstrap
intervals and process identity. No failure, repair, tolerance change or repeated
stage occurs. Native checkpoints and prior captures are unchanged. No learner or
GPU allocation is active at closure; check live state before subsequent work.

Root`/x/Temp/kindle-terminal-weight-ranking-20261104`:

| Artifact | SHA256 |
|---|---|
|`declaration.json`|`996441c8c74efe16e82f449123f233a65c96d38ea78bae31f7cb22c09b84b72e`|
|`analysis.json`|`681cdd15dcc001c921afe66762da9554cd1df81ad73e9d4fba636384b9d07669`|
|`completion.json`|`3762c669865ce4959f7ec859a7a1d177747b26ff3b275569ca1f2bbf114224a0`|
|`terminal-summary.json`|`fb92af3a554868de762a0acb1968759000811e381c635a72b2d4a1bd105857e3`|
|`closure-execution.json`|`e8ed7c1fd99d7f67c04a2891038aab23a616230959f4de0543cc55d2c969b189`|
