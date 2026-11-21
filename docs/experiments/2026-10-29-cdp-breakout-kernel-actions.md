# Nonlinear teacher-action readouts from saved Breakout features

The completed linear readouts trail the recorded policy. Select one CPU-only
RBF kernel ridge follow-up to check nonlinear accessibility of teacher actions.
Keep the same initial/trained CNN, recurrent and concatenated features,
954/509/601 train/validation/test lane split, every-eighth-row selection,
four canonical actions, label/input alignment checks and metrics from the
[linear protocol](2026-10-29-cdp-breakout-action-readout.md).

Normalize with training means and standard deviations, dividing by square root
of feature width. Set the squared bandwidth to the median positive off-diagonal
training squared distance. Use exp(-squared-distance/bandwidth), centered with
training kernel means and an unpenalized intercept. For each one-hot target,
select ridge penalty from 1e-4,1e-3,1e-2,.1,1,10,100 by validation MSE, exactly
the earlier linear selection criterion. Do not refit validation data, tune the
bandwidth, select on test scores or add feature variants. Report all six fits.

Use a direct augmented-system comparison to check the centered solver, an XOR
case to check nonlinear expressiveness, and checks for test-data independence,
lane isolation and forbidden native imports. Reuse tested label/metric helpers.
Pin protocol, source, tests, original features, labels and completed linear
results before running. Keep all 263 active Qbert6029 pins unchanged and record
its original PID/invocation before and after. One affined CPU, low priority,
one BLAS thread and a60-second analysis cap; no environment, native agent, GPU
work, learner update or checkpoint writes. Retain CPU co-hosting time as a
Qbert training-time caveat without subtracting it from measured wall time.

Teacher agreement is not teacher correctness or closed-loop task performance.
The original vector-policy draws differ from scalar feature re-observation;
this is not an exact actor-head replacement. The development learner/cohort
is reused and the two test lanes do not establish learner-seed reliability.
A negative kernel result does not exclude other nonlinear readouts. No native
architecture/loss change or new learner is selected by this diagnostic alone.
Evidence root: `/x/Temp/kindle-breakout-kernel-readout-20261029`.
