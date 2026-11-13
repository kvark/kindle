# Breakout still predicts continuation after observing most game endings

On the same24 recorded game endings, the final model's continuation head output
falls from **.952850 before seeing the arrival frame to .759636 after observing
it**, against a target of zero. Twenty individual squared errors decrease and
four increase; all eight lane means decrease. Only **5 of24 observed arrivals**
produce continuation below.5, versus zero before observation.

This establishes both useful observation feedback and a substantial remaining
error on the observed terminal states. It narrows the problem beyond forecasting
alone, without isolating representation, rare-example fitting, stochastic state
draws or optimization as its cause. The preceding
[replay audit](2026-11-04-cdp-breakout-terminal-replay.md) found terminal targets
present but only.1683% of replay rows, with some evidence of fitting.

The [fixed diagnostic](../experiments/2026-11-04-cdp-breakout-arrival-head.md)
completes43,424 source actions /173,660 emulator frames in **50.457s**, with zero
updates and all357 tensors unchanged. Retained gameplay remains70.750 sampled /
143.667 greedy in21m28s, all0/24 two-wall. The original864/90%-two-wall and
five-game goal remain unfinished.

## Paired predicted and observed arrivals

| Fixed cohort | Continuation before observation | After observation | Target | Prior MSE | Posterior MSE |
|---|---:|---:|---:|---:|---:|
|24 game endings|.952850|.759636|0|.911006|.644558|
|96 intermediate life losses|.979477|.979466|.996997|.001530|.003178|

The paired posterior-minus-prior terminal MSE difference is **−.266448**, with
a fixed lane-bootstrap95% interval **[−.401298,−.154655]**. This interval describes
the reused eight-lane cohort, not independent learners or population calibration.
No individual errors tie within1e-6. Intermediate-life mean continuation changes
little, while its squared error increases; there is no general forecast-quality
improvement claim. Intermediate life losses retain their nonterminal target.

The same recorded action is forecast immediately before each selected arrival.
After that action executes, the model observes the recorded frame and its
posterior feature is exported **before reset**. Exact source review confirms
that terminal flags do not enter the posterior network: the observation update
precedes replay insertion and reset bookkeeping. All120 deterministic state
vectors match exactly between predicted and observed arrivals. The categorical
states use distinct diagnostic-prior and live-posterior draws, so observation
conditioning and categorical draws both change.

## Offline head readout and its qualification

The qualified native runtime exports both state types but has no direct
posterior-continuation binding. The observed-state result therefore uses an
**offline F64 reference** of the unchanged saved head: Linear2560→256,
RMSNorm(epsilon1e-4, learned scale), SiLU, Linear256→1 and sigmoid. This analysis
does not act or learn. Native acting, model sources and checkpoint are unchanged.

The reference agrees with all120 native prior continuations to maximum absolute
error **1.178e-7**, below the fixed1e-4 limit. Native and reference terminal prior
means are.9528502077 and.9528502110. The primary paired difference uses reference
outputs for both state types. All features are finite and have the expected2048
deterministic entries and32 one-hot categorical groups of16. Independent
einsum/reduction arithmetic reproduces all240 reference outputs within1e-12.

Thirteen CPU tests pass, including an independent scalar head calculation,
extreme logits, invalid parameters/features, categorical/reset mismatch checks,
separate targets, all real endpoint joins and entrypoint checks before native
construction. Exact native graph/binding sources and the five saved parameter
tensors are pinned. Prior-reference agreement qualifies the implementation on
captured prior states; observed outputs remain reference estimates rather than
direct native posterior readbacks.

## Implication and limits

Seeing the realized ending provides useful information, but this final model
still assigns high continuation to most selected terminal arrivals. Improving
terminal-state fitting is now a concrete learning question. The diagnosis does
not establish that simply increasing a loss weight, changing replay sampling or
extending recurrent context would improve frozen gameplay. The whole continuation
loss also covers abundant nonterminal examples; no new coefficient or learner is
selected here.

These windows were selected because a life loss occurred. The result does not
measure population calibration or general terminal recall. Scalar posterior
histories differ from the original vector policy's draws; all saved scalar
one-step predictions reproduce exactly. Training replay states differ from this
frozen cohort. No individual terminal from the replay audit is identified here.
The measured gap does not isolate a transition-model defect, prove absence of
information, or replace the original full-game gate with terminal classification.

## Execution and closure

Native447fe50/SHA35ad0db and the retained replay-value1 checkpoint remain fixed.
All633 preparation /634 execution pins pass, including606 prior inputs and22
prior final artifacts. Every recorded input/output pixel, action, reward, frame,
boundary, reset and stream hash matches. All120 saved one-step predictions and
120 repeated endpoint queries reproduce. Diagnostic queries preserve live
features, actor probabilities and counters. The capture advances43,424 actions
with zero updates; all357 frozen arrays remain finite and byte-identical.

Source`8754f65d`, mainPID1102754/invocation`c114e5e2e92442ebbf08136138683b22`
launches03:27:30 UTC November4 and succeeds **03:28:26.630**. Capture wall time
includes scalar re-observation, exports and checkpoint validation; it is not
isolated GPU kernel time. CPU analysis takes **.320s** on one low-priority core;
its wrapper takes.414s. Independent **3.691s CPU closure** completes
**03:28:57.049**, verifies the original service/journal and host guards, recounts
all pairs and the bootstrap interval, and checks every tensor again.

All52 final artifact hashes verify. Preparation, capture, analysis and original
closure pass without failures, repairs, host recovery or repeated stages.
Root `/x/Temp/kindle-breakout-arrival-head-20261104` is closed. No learner or GPU
allocation is active; do not restart its capture or collectors. Check live state
before future work. No successor allocation is selected.

| Artifact | SHA256 |
|---|---|
| Declaration |`aa9338388dfd6aaa3061d7cf700c21ccb058d2ecf69e726fd89c8197c24fff0c`|
| Frozen capture |`709171fd7e9577730ab81afb8499ce8fe691e4b0530650edc2dfed4df19e0506`|
| Analysis |`5c99095a07663510ad376e999f70b4f5466d1fca18209d55568bff5c00fe9f0e`|
| Main completion |`523232eb27ce85d229d257ca6584323615c9a5e6b199c21da621ca93328e6107`|
| Independent terminal summary |`7253123c7f57dfb307c60577e7dd419eaa7c8c4f497d6837b4bb4f812c87fc77`|
