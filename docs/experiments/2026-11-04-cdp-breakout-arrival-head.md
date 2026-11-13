# Breakout continuation before and after observing the same arrival

The [terminal replay audit](../results/2026-11-04-cdp-breakout-terminal-replay.md)
shows terminal targets are present and partly fitted. The frozen one-step prior
still predicts high continuation at selected endings. Compare its predicted
arrival state with the posterior after observing that same arrival, using the
unchanged final model and the existing continuation head.

Use the exact retained replay-value1 checkpoint, native447fe50/SHA35ad0db and
corrective runtime. Reuse all120 endpoint identities from the ending-feedback
diagnostic:24 game endings and96 intermediate life losses, across eight lanes.
Do not select examples by their predictions. Re-observe the complete recorded
43,424-action/173,660-frame cohort once, with fixed actions and full pixel,
transition, reset and stream parity. This is a new state-export diagnostic;
no completed capture, analysis or collector is restarted.

At each endpoint, require exact reproduction of its saved one-step reward,
continuation and value predictions. Export the one-step prior feature through
the existing forecast_states API; its duplicate first/final outputs must match.
Repeat the native behavior query and verify unchanged live features, policy
probabilities and counters. After executing the recorded action and observe,
capture the arrival posterior feature before any reset. Terminal flags do not
enter the posterior network; source review must confirm that ordering.

No posterior-continuation binding exists in this runtime. Evaluate only the
saved continuation head on the exported features as offline F64 analysis:
Linear(2560,256), RMSNorm(epsilon1e-4, learned scale), SiLU, Linear(256,1), sigmoid.
Pin the exact native graph sources and five checkpoint parameter tensors.
Qualify the reference against independent scalar fixtures and invalid inputs.
Require absolute error at most1e-4 against all120 native prior continuations;
otherwise the diagnostic fails, with no tolerance relaxation or GPU repeat.
The posterior output is an offline reference estimate, not a direct native
posterior readback. Native acting and learning remain unchanged.

Validate finite2560-wide features and32 one-hot groups of16 categorical entries.
Record maximum difference between the2048 deterministic prior/posterior entries;
require at most2e-4 to detect an ordering/reset mismatch. Report categorical
disagreement without interpreting it as calibrated uncertainty. The diagnostic
prior draw and live posterior draw are distinct, as in the existing runtime.

Primary: on the24 game endings, report prior/native/reference and posterior
continuation means, squared errors against zero, paired posterior-minus-prior
reference MSE difference, eight lane means and a fixed5,000-resample lane
bootstrap95% interval using seed20261104. Retain all individual pairs and counts
below.5. Report the same descriptive means/errors on96 intermediate life losses,
whose target remains F32(1−1/333). Do not pool both targets into one headline.

This outcome-conditioned cohort cannot establish population calibration, seed
reliability, optimal action ranking or a gameplay gain. A posterior improvement
would narrow the issue toward predicting useful arrival features; it would not
isolate the transition model, stochastic draws, latent alignment or optimization.
Weak posterior predictions would motivate inspecting arrival representation and
head fitting. Neither result automatically selects a new loss or learner.

Pin previous evidence and all current scripts before one guarded180s capture.
Require zero updates, all357 finite invariant tensors, correct counters and host
guards without recovery. Analyze on one low-priority CPU core under60s; independently
recount saved rows and verify head arithmetic, original service and artifacts.
Keep original failures if any stage fails; no completed-stage repeats.

Root `/x/Temp/kindle-breakout-arrival-head-20261104`. Retain70.750 sampled/
143.667 greedy in21m28s, all0/24 two-wall. The original864/90%-two-wall and
five-game goal remain unfinished. No successor GPU job is selected.
