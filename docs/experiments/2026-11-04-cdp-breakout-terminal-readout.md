# Fixed CPU readout of ending information in saved Breakout states

The continuation10 learner regresses, and its matched arrival diagnostic does
not establish better ending recognition. Before another loss or sampler change,
test whether a simple supervised readout can recover ending information from
the already captured latent states. This is an offline diagnostic, not native
policy training or a replacement head selected for acting.

Use the exact120 arrivals from the completed scale1 and continuation10 captures:
24 game endings and96 intermediate life losses,15 arrivals per lane. Preserve
every origin, outcome and prior/posterior feature; use each model's own saved
native-qualified F64 head output as its baseline. No model execution, emulator
replay, new experience or checkpoint change. Block native and emulator imports.

Primary representation is the retained scale1 posterior. Candidate posterior
and both one-step priors are fixed secondary views. Fit each view independently;
never combine latent coordinates from different models. Leave one ofeight lanes
out at a time:105 training rows,15 test rows, no held-out labels in fitting or
preprocessing. Report all120 out-of-lane predictions. This is one reused,
outcome-conditioned cohort, not training-seed confirmation or population terminal
calibration. Grouped cross-validation reduces within-lane leakage but does not
remove all cohort dependence.

Use one fixed class-balanced linear ridge readout, with no parameter search.
Targets are0 at game endings and F32(1−1/333) at continuing life losses. Training
weights n/(2*n_class) give equal total weight to both classes. Standardize each
feature using training-only mean/population deviation; ignore columns with
deviation≤1e-6 and divide active features by sqrt(active feature count). Include
an unregularized intercept through weighted centering. Minimize the weighted
sum of squared errors plus **1.0*||coefficients||²**, using the dual linear solve.
Clip output to[0,1] for reported prediction errors; retain raw outputs and folds.

Report per-class MSE, their equal-weight mean, terminal and continuing recalls,
balanced accuracy, and rank AUC using negative continuation. Classify as terminal
only below gamma/2; ties continue. Baselines are each saved head, constant gamma
and constant gamma/2. Report training fits separately from held-out predictions.
Store coefficients and preprocessing for independent normal-equation and
prediction checks without refitting.

Primary evidence for useful linear information requires lower held-out balanced
MSE than both the saved head and balanced constant, plus a negative upper bound
for readout-minus-head balanced MSE under5,000 lane-bootstrap resamples,
seed20261104. This is a fixed diagnostic screen, not an architecture or gameplay
promotion. Report all secondary views without switching the primary. Negative
results cannot establish absence of information or rule out other decoders.

Before fitting, pin both completed captures/analyses and terminal manifests,
validate all120 joins/targets/features, and test the ridge against an independent
small primal solve, training-only preprocessing, malformed inputs, class metrics
and the full command boundary. Run once on one CPU core at nice10, native/emulator
imports blocked, under60s. Independently verify all saved folds satisfy the
weighted normal equations, reproduce predictions/metrics/intervals, and retain
original failures. No completed fit, capture, replay or collector repeats.

Root `/x/Temp/kindle-breakout-terminal-readout-20261104`. Retain scale1 and70.750
sampled/143.667 greedy. No new GPU allocation, learner, loss, sampler or automatic
successor is selected. Original864/90%-two-wall and five-game goal remain unfinished.
