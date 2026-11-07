# Frozen-state readouts for the Breakout world diagnosis

Select a CPU-only analysis of existing coverage,ST-energy and conditional-score-
function exports. No GPU/emulator construction,new forecasts,learner or collector.
The world-model parameters/objectives remain unchanged. This is diagnostic
evaluation with labels,not a new model loss or a passing competence criterion.
Root `/mnt/data/kindle-breakout-world-score-function-20261107/readout-diagnosis`.

Use the retained256 fitting/128 development dense exports with one fixed draw.
Verify identical source locations,images,actions/validity and labels across the
three models. Fit separate readouts per model,using fitting data only. Development
families remain disjoint from fitting,but already used for development decisions;
this is not an untouched test or independent-episode reliability assessment.

Fixed ridge coefficient10 after fitting-only feature centering/unit-standardization;
drop features with standard deviation<1e-6. Use dual ridge,one intercept equal to
the fitting target mean. No hyperparameter/draw/checkpoint selection. Test encoder
and posterior features against current ball position/velocity/lives,then causal
prior features at1/4/8/16/32 against corresponding actual future labels.

RAM positions/lives are evaluation references only,imperfect and never model
inputs/targets. Coordinates are reported in original source pixels,not equated
to resized detector centers. Velocity is the difference between adjacent submitted-
action arrivals,not emulator-frame velocity. Require finite endpoints and exclude
>32px jumps as respawn/slot discontinuities. Score natural-terminal arrivals;
exclude future reset crossings/incomplete horizons using the original validity.
Report supported counts,per-axis errors,life-count error and last-life confusion.

A successful prior-state readout can show information available to that readout
when RGB prediction is poor; it does not identify a complete sufficient state or
prove accurate stochastic forecasts. A failed linear readout does not prove
information absent,or distinguish nonlinear coding,limited readout fitting,
representation aliasing and true dynamics failure. Comparing differently fitted
readouts is not a CDP-superiority or causal architecture claim. Retain source,
label/input hashes,coefficients and all cases,including unfavorable results.


## Closed CPU diagnosis

See the [result](../results/2026-11-07-breakout-world-readout-diagnosis.md).
All three models' source images/labels/locations/validity match;eight feature
input pins,63 coefficient/prediction files and a separate30-pin source/action
audit are retained. The latter reconstructs controls from24 unchanged corpus
files and identical runners/configs; native call-argument buffers were not saved.
No GPU/emulator,new forecast or model update. Useful short-term motion readouts
do not rescue precise future positions,and linear/generalization limitations
prevent an information-absence or complete-state conclusion. No readout learner
or loss is selected for Kindle. Full world goal remains unproven.
