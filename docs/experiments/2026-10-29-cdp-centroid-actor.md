# Actor decisions on the fixed Breakout state corpus

Select a frozen diagnostic within the five-game goal. The
[centroid state readouts](../results/2026-10-29-cdp-centroid-representation.md)
show better recurrent ball-position access but much smaller improvements in
the teacher's desired paddle target. Measure the actual learned actor on those
same saved recurrent states before selecting another loss or learner.

Evaluate the control and centroid100 final actor weights using a CPU F64
reference for the native three-layer affine/RMSNorm/SiLU MLP, epsilon1e-4,
followed by softmax and actor-unimix .01 over all18 actions. Native actor,
network and distribution sources are identical between447fe50 and2974160;
check the pinned Meganeura RMSNorm implementation as well. Python performs
reference analysis only; native learning and inference remain unchanged.

Before using the reference on saved features, qualify it against native policy
probabilities on the first128 recorded actions of lane0 for each checkpoint.
Use each checkpoint's original runtime. These measurements add probabilities
not saved by the earlier feature captures. Require maximum absolute probability
error at most1e-4 and identical greedy actions on all128 states, fixed before
execution. Repeated readback must preserve state, probabilities and counters.
Compare the16 previously saved feature rows exactly, verify every recorded
pixel/action/reward/reset/frame and preserve all357/372 checkpoint tensors.
Each check has a120s cap and a host guard. Stop on failure without automatic
retry or tolerance changes; never repeat completed GPU stages for a CPU defect.

Use the saved2,064 feature rows and the original teacher labels. Readout test
lanes6–7 supply601 rows, including136 near-paddle descending states. Compare
both native actors' exact full18-action choices, mean teacher probability,
cross-entropy, entropy, per-class confusion, opposite LEFT/RIGHT commands,
directional NOOPs and missed FIRE commands. Report all rows, near-paddle rows
and each test lane. Full18 aliases are not silently collapsed to teacher classes.
The original vector policy's recorded decisions remain a distinct baseline:
the saved features use scalar posterior draws.

Fit the existing linear and fixed RBF teacher-action readouts to the candidate
features, using the original lanes0–3 training /4–5 validation split, fixed
penalty grids, training-only normalization and bandwidth. Reuse the saved
control readout predictions. Preserve every class, variant and prediction;
no test-based choice, validation refit or head replacement follows. CPU analysis
has a120s cap. A single600s service owns both bounded qualifications and analysis.

Use fresh output paths and pin exact source, checkpoints, feature captures,
teacher rows and prior completions before execution. CPU tests must check the
reference against independent scalar arithmetic, shape/finite checks, stable
softmax and error accounting without native/emulator imports. No new learner,
full gameplay evaluation, environment trajectory or native build is selected.
The centroid full learner remains `closed_protocol_failed` regardless of this
diagnostic's outcome.

These are teacher-agreement measurements on reused control trajectories, not
teacher correctness, causal attribution or closed-loop competence. The corpus
does not cover the centroid policy's long timeout. Different training histories,
native versions and scalar states limit comparisons. A better readout would not
by itself justify inserting it into the live policy. The five-game task gates
remain unchanged; no automatic successor is selected.

Evidence: `/x/Temp/kindle-centroid-actor-20261029`.
Service: `kindle-centroid-actor-20261029`.
