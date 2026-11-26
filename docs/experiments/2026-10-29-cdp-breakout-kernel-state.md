# Nonlinear position and motion readouts from saved Breakout features

The fixed RGB64 pixel decoder preserves much more precise ball positions than
the learned-feature linear readouts recover. Select one CPU-only nonlinear
regression diagnostic on the existing initial/trained captures. It tests whether
the already checked RBF decoder recovers finer state information before choosing
an encoder or training-objective change. Earlier action-kernel results remain
negative and are not rerun.

Keep all2,064 selected rows and the954/509/601 train/validation/test lane split.
Fit CNN,recurrent and concatenated features for all six original targets:
ball x/y,paddle x,horizontal/vertical displacement and teacher response-target x.
For each target,exclude missing training/validation labels separately;never
exclude a test row from prediction. Reuse the exact centered RBF solver from
`probe_breakout_kernel_actions.py`: training-only normalization and median
positive squared-distance bandwidth,with an unpenalized intercept. Select the
penalty from1e-4,1e-3,1e-2,.1,1,10,100 by validation MSE only. No validation
refit,bandwidth sweep,test selection or decoder revision after reading results.

Report every initial/final feature variant,selected penalties,valid-target
support,MAE/RMSE/R² overall,near-descending and per test lane. Save every
prediction and its lane/index identity. Compare to the retained linear results
on identical support; also compare ball/paddle positions with the fixed pixel
decoder on the exact common rows,disclosing its misses and extras. No ranking
or architecture change is selected from a favorable subset.

Check masking,validation/test independence,lane isolation and a subprocess that
forbids native imports,alongside the existing direct-system kernel checks.
Pin source,tests,protocol,captures,corpus and prior results before analysis.
One low-priority CPU core,one BLAS thread,120-second analysis cap. No environment,
agent,native extension,GPU work,learning update or checkpoint write. Keep any
failure;no automatic repeat or cap increase. Check live state before execution.

Teacher pixel detections are reference labels,not independent physical truth.
The bottom-screen crop and reused weak-policy cohort limit coverage; held-out
lanes are not independently trained learners. Scalar captured categorical draws
differ from the original vector policy. Nonlinear accessibility does not prove
the actor uses the information;failure of this decoder does not prove information
absent. No new learner,loss or native architecture is selected by this diagnostic.

Evidence root:`/x/Temp/kindle-breakout-kernel-state-20261029`.
