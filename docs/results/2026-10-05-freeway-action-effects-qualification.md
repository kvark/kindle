# Freeway action-effects bonus: qualified, not yet a learning result

The changed bonus subtracts each predictor's all-action average before taking
ensemble disagreement. Exact finite-action evaluation is batched on GPU; only
the selected action's scalar bonus enters the existing reward path. Predictor
training, parameters, encoder, coefficient, optimizer and acting policy are
unchanged. There is no privileged input or game-specific directional rule.

114 CPU Rust tests,1,137 Python tests, formatting and strict workspace/Python
Clippy pass. Four guarded GPU tests pass: independent F64 values, state-offset
and action-permutation invariance; repeated/novel states with all actions
observed; replay/imagination alignment; and zero-scale update equivalence.
The repeated-state fixture now observes every action because the bonus's
reference itself depends on their predictions. Its reduction bounds are not
relaxed.

Three old saved models also pass the new full-network GPU/F64 readout:
native-F32 maximum errors3.07e-8/3.09e-8/3.27e-8, unchanged3e-6 bound. Default
cooperative errors4.39e-5/4.67e-5/4.54e-5 are retained. UP–DOWN preference agrees
at all768 states; best-action identity agrees97.3/93.0/94.5%. Production precision
is unchanged. All346 model/optimizer tensors/model remain exact. These are
readouts of old policies with a new bonus, not evaluations of a newly trained actor.

Excluded seed9001 smoke:2,048 actions/451 finite updates in15.26s,31.66ms/update,
zero rewards. Its1,024-action frozen restore performs zero updates and preserves
all346 tensors. All nine guards and retained-log audits pass; no host recovery
or NVML polling. Vulkan estimated headroom checks pass; peak physical VRAM and
GPU utilization are not measured.

This bonus costs more: retained original visual-target smoke was12.36s and
25.32ms/update. The new smoke takes23% longer end to end and25% longer per update. This is a descriptive small smoke, not a matched speed
benchmark. The bounded learning test is worthwhile because the transformed
bonus has much clearer action contrast, not because it is a speed optimization.

Next: the [declared three-seed32,768-action screen](../experiments/2026-10-05-freeway-action-effects.md).
Do not extend automatically. Freeway remains unsolved.

Evaluation-tooling follow-up: the score/replay reader now accepts a frozen
disagreement-trained configuration only with zero updates and zero stored
intrinsic rewards. Negative/nonfinite scales, host visitation, shaped scores
and training-mode exceptions remain rejected; extrinsic-only training audits
are unchanged.1,161 Python tests pass, including24 new cases. The existing
1,024-action frozen smoke replays exactly on CPU, with no new GPU context or
learning. No native rebuild, runner/wrapper change or active-run setting change.

[Compact evidence](2026-10-05-freeway-action-effects-qualification.json).
Artifacts: `runs/freeway-action-effects-20261005.z9KUOhCS`.
